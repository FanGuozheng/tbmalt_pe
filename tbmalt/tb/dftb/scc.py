"""DFTB calculator.

implement pytorch to DFTB
"""
import torch
import tbmalt.common.maths as maths
from tbmalt.tb.electrons import fermi
from tbmalt.tb.properties import mulliken
from tbmalt.common.maths.mixer import Simple, Anderson
# from tbmalt.tb.properties import Properties
from tbmalt.tb.properties import dos, pdos, band_pass_state_filter
from tbmalt.common.batch import pack
from tbmalt.tb.electrons import Gamma
from tbmalt.tb.coulomb import Coulomb
from tbmalt.common.units import AUEV
Tensor = torch.Tensor


class Scc:
    """Self-consistent charge density functional tight binding method.

    Arguments:
        system: Object contains geometry and orbital information.
        skt: Object contains SK data.
        parameter: Object which return DFTB and ML parameters.

    Examples:
        >>> from ase.build import molecule as molecule_database
        >>> from tbmalt.common.structures.system import System
        >>> from tbmalt.io.loadskf import IntegralGenerator
        >>> from tbmalt.tb.sk import SKT
        >>> from tbmalt.tb.dftb.scc import SCC
        >>> from tbmalt.common.parameter import DFTBParams
        >>> molecule = System.from_ase_atoms([molecule_database('CH4')])
        >>> sktable = IntegralGenerator.from_dir('./slko/auorg-1-1', molecule)
        >>> skt = SKT(molecule, sktable)
        >>> parameter = DFTBParams()
        >>> scc = SCC(molecule, skt, parameter)
        >>> scc.charge
        >>> tensor([[4.3054, 0.9237, 0.9237, 0.9237, 0.9237]])

    """

    def __init__(self, system: object, skt: object, parameter: object,
                 coulomb=None, periodic=None, properties=[], **kwargs):
        self.system = system
        self.skt = skt
        self.params = parameter
        self.coulomb = coulomb
        self.periodic = periodic
        if self.periodic:
            self.mask_pe = periodic.mask_pe

        self.ham, self.over = skt.H, skt.S
        self._init_scc(**kwargs)

        self._scc()

        # if 'dipole' in properties:
        self.dipole = self._dipole()
        # if 'homo_lumo' or 'gap' in properties:
        self.homo_lumo = self._homo_lumo()
        self.cpa = self._cpa()

    def _init_scc(self, **kwargs):
        """Initialize parameters for (non-) SCC DFTB calculations."""
        # basic DFTB parameters
        self.scc = self.params.dftb_params['scc']
        self.maxiter = self.params.dftb_params['maxiter'] if self.scc == 'scc' else 1
        self.mask = torch.tensor([True]).repeat(self.ham.shape[0])
        self.atom_orbitals = self.system.atom_orbitals
        self.size_system = self.system.size_system  # -> atoms in each system

        # intial charges
        self.qzero = self.system.get_valence_electrons(self.ham.dtype)
        self.charge = kwargs.get('charge') if self.scc == 'xlbomd' else self.qzero.clone()
        self.nelectron = self.qzero.sum(axis=1)

        # get the mixer
        self.mix = self.params.dftb_params['mix']

        if self.mix in ('Anderson', 'anderson'):
            self.mixer = Anderson(self.charge, return_convergence=True)
        elif self.mix in ('Simple', 'simple'):
            self.mixer = Simple(self.charge, return_convergence=True)

        if self.system.periodic:
            assert self.coulomb is not None
            assert self.periodic is not None
            self.distances = self.periodic.periodic_distances
            self.u = self._expand_u(self.skt.U)
        else:
            self.distances = self.system.distances
            self.u = self.skt.U

        if self.scc in ('scc', 'xlbomd'):
            self.method = kwargs.get('gamma_method', 'read')
            self.gamma = Gamma(self.u, self.distances, self.system.numbers,
                               self.periodic, method=self.method).gamma
        else:
            self.gamma = torch.zeros(*self.qzero.shape)

        self.inv_dist = self._inv_distance(self.system.distances)

        # replace the ewald summation for non-periodic systems
        if self.periodic:
            if not self.mask_pe.all():
                _invr = torch.clone(self.inv_dist)
                _invr[self.mask_pe] = self.coulomb.invrmat
                self.coulomb.invrmat = _invr

        self.shift = self._get_shift()

    def _scc(self, ibatch=[0]):
        """SCF for non-periodic-ML system with scc.

        atomind is the number of atom, for C, lmax is 2, therefore
        we need 2**2 orbitals (s, px, py, pz), then define atomind2
        """
        for iiter in range(self.maxiter):
            # get shift and repeat shift according to number of orbitals
            shift_ = self._update_shift()
            shiftorb_ = pack([ishif.repeat_interleave(iorb) for iorb, ishif in
                              zip(self.atom_orbitals[self.mask], shift_)])
            shift_mat = torch.stack([torch.unsqueeze(ishift, 1) + ishift
                                     for ishift in shiftorb_])

            # H0 + 0.5 * S * G
            this_size = shift_mat.shape[-1]   # the new shape
            fock = self.ham[self.mask, :this_size, :this_size] + \
                0.5 * self.over[self.mask, :this_size, :this_size] * shift_mat

            # calculate the eigen-values & vectors via a Cholesky decomposition
            epsilon, eigvec = maths.eighb(fock, self.over[self.mask, :this_size, :this_size])

            # calculate the occupation of electrons via the fermi method
            occ, nocc = fermi(epsilon, self.nelectron[self.mask])

            # eigenvector with Fermi-Dirac distribution
            c_scaled = torch.sqrt(occ).unsqueeze(1).expand_as(eigvec) * eigvec
            self.rho = c_scaled @ c_scaled.transpose(1, 2)  # -> density

            # calculate mulliken charges for each system in batch
            q_new = mulliken(self.over[self.mask, :this_size, :this_size],
                             self.rho, self.atom_orbitals[self.mask])

            # last mixed charge is the current step now
            if self.scc == 'nonscc':
                self.charge = q_new
            else:
                if iiter == 0:
                    self.epsilon = torch.zeros(*epsilon.shape)
                    self.eigenvector = torch.zeros(*eigvec.shape)
                    self.nocc = torch.zeros(*nocc.shape)
                    self.density = torch.zeros(*self.rho.shape)
                self.qmix, self.converge = self.mixer(q_new)
                self._update_scc(self.qmix, epsilon, eigvec, nocc)

                if (self.converge == True).all():
                    break  # -> all system reach convergence

        return self.charge

    def _update_scc(self, qmix, epsilon, eigvec, nocc):
        """Update charge according to convergence in last step."""
        self.charge[self.mask] = qmix
        self.epsilon[self.mask, :epsilon.shape[1]] = epsilon
        self.eigenvector[self.mask, :eigvec.shape[1], :eigvec.shape[2]] = eigvec
        self.nocc[self.mask] = nocc
        self.density[self.mask, :self.rho.shape[1], :self.rho.shape[2]] = self.rho
        self.mask = ~self.converge

    def _dipole(self):
        """Return dipole moments."""
        return torch.sum((self.qzero - self.charge).unsqueeze(-1) *
                         self.system.positions, 1)

    def _homo_lumo(self):
        """Return dipole moments."""
        # get HOMO-LUMO, not orbital resolved
        _mask = torch.stack([
            ieig[int(iocc)] - ieig[int(iocc - 1)] < 1E-10
            for ieig, iocc in zip(self.eigenvalue, self.nocc)])
        self.nocc[_mask] = self.nocc[_mask] + 1
        return torch.stack([
            ieig[int(iocc) - 1:int(iocc) + 1]
            for ieig, iocc in zip(self.eigenvalue, self.nocc)])

    def _onsite_population(self):
        """Get onsite population for CPA DFTB.

        sum density matrix diagnal value for each atom
        """
        ao = self.system.atom_orbitals
        nb = self.system.size_batch
        ns = self.system.size_system
        acum = torch.cat([torch.zeros(ao.shape[0]).unsqueeze(0),
                          torch.cumsum(ao, dim=1).T]).T.long()
        denmat = [idensity.diag() for idensity in self.density]
        # get onsite population
        return pack([torch.stack([torch.sum(denmat[ib][acum[ib][iat]: acum[ib][iat + 1]])
                    for iat in range(ns[ib])]) for ib in range(nb)])

    def _cpa(self):
        """Get onsite population for CPA DFTB.

        J. Chem. Phys. 144, 151101 (2016)
        """
        onsite = self._onsite_population()
        nat = self.system.size_system
        numbers = self.system.numbers

        return pack([1.0 + (onsite[ib] - self.qzero[ib])[:nat[ib]] / numbers[ib][:nat[ib]]
                     for ib in range(self.system.size_batch)])

    def _update_charge(self, qmix):
        """Update charge according to convergence in last step."""
        self.charge[self.mask] = qmix
        self.mask = ~self.converge

    def _inv_distance(self, distance):
        _inv_distance = torch.zeros(*distance.shape)
        mask = distance.ne(0.0)
        _inv_distance[mask] = 1.0 / distance[mask]
        return _inv_distance

    def _get_shift(self):
        """Return shift term for periodic and non-periodic."""
        if not self.system.periodic:
            return self.inv_dist - self.gamma
        else:
            return self.coulomb.invrmat - self.gamma

    def _update_shift(self):
        """Update shift."""
        return torch.stack([(im - iz) @ ig for im, iz, ig in zip(
            self.charge[self.mask], self.qzero[self.mask], self.shift[self.mask])])

    def _expand_u(self, u):
        """Expand Hubbert U for periodic system."""
        shape_cell = self.distances.shape[1]
        return u.repeat(shape_cell, 1, 1).transpose(0, 1)

    @property
    def ini_charge(self):
        """Return initial charge."""
        return self.qzero

    @property
    def eigenvalue(self):
        """Return eigenvalue."""
        return self.epsilon * AUEV

    @property
    def fermi(self):
        """Fermi energy."""
        return self.homo_lumo.sum(-1) / 2.0

    @property
    def dos_energy(self, unit='eV', ext=1, grid=1000):
        """Energy distribution of (P)DOS.

        Arguments:
            unit: The unit of distribution of (P)DOS energy.

        """
        self.unit = unit
        e_min = torch.min(self.eigenvalue.detach()) - ext
        e_max = torch.max(self.eigenvalue.detach()) + ext

        if unit in ('eV', 'EV', 'ev'):
            return torch.linspace(e_min, e_max, grid)
        elif unit in ('hartree', 'Hartree'):
            return torch.linspace(e_min, e_max, grid) * AUEV
        else:
            raise ValueError('unit of energy in DOS should be eV or Hartree.')

    @property
    def pdos(self):
        """Return PDOS."""
        energy = torch.linspace(-1, 1, 200)
        return pdos(self.eigenvector, self.over, self.eigenvalue, energy)

    @property
    def dos(self):
        """Return energy distribution and DOS with fermi energy correction."""
        sigma = self.params.dftb_params['sigma']
        energy = self.dos_energy
        energy = energy.repeat(self.system.size_batch, 1)  # -> to batch

        # make sure the 1st dimension is batch
        if self.unit in ('eV', 'EV', 'ev'):
            return dos((self.eigenvalue),
                       energy, sigma)  # , mask=self.band_filter)
        elif self.unit in ('hartree', 'Hartree'):
            return dos(self.eigenvalue,
                       energy, sigma)  # , mask=self.band_filter)

    @property
    def band_filter(self, n_homo=torch.tensor([3]), n_lumo=torch.tensor([3]),
                    band_filter=True) -> Tensor:
        """Return filter of states."""
        if band_filter:
            n_homo = n_homo.repeat(self.eigenvalue.shape[0])
            n_lumo = n_lumo.repeat(self.eigenvalue.shape[0])
            return band_pass_state_filter(self.eigenvalue, n_homo, n_lumo, self.fermi)
        else:
            return None
