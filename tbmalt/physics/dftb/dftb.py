"""DFTB calculator.

implement pytorch to DFTB
"""
from typing import Literal, Dict, List, Optional
import torch
from torch import Tensor
import tbmalt.common.maths as maths
from tbmalt import Basis, Geometry, SkfFeed, SkfParamFeed, hs_matrix
from tbmalt.physics.dftb.repulsive import Repulsive
from tbmalt.common.maths.mixer import Simple, Anderson
from tbmalt.physics.properties import mulliken, dos, pdos, band_pass_state_filter
from tbmalt.common.batch import pack
from tbmalt.physics.fermi import fermi
from tbmalt.data.units import _Hartree__eV
from tbmalt.ml.skfeeds import VcrFeed, TvcrFeed
from tbmalt.physics.electrons import Gamma
from tbmalt.structures.periodic import Periodic
from tbmalt.physics.coulomb import Coulomb


class Dftb:
    def __init__(self,
                 geometry: Geometry,
                 shell_dict: Dict[int, List[int]],
                 path_to_skf: str,
                 repulsive: bool = False,
                 skf_type: Literal['h5', 'skf'] = 'h5',
                 basis_type: str = 'normal',
                 periodic: Periodic = None,
                 mixer: str = 'Anderson',
                 **kwargs):
        self.skf_type = skf_type
        self.shell_dict = shell_dict
        self.geometry = geometry
        self.repulsive = repulsive
        self.mixer_type = mixer

        self.interpolation = kwargs.get('interpolation', 'PolyInterpU')
        self.batch = True if self.geometry.distances.dim() == 3 else False
        hsfeed = {'normal': SkfFeed, 'vcr': VcrFeed, 'tvcr': TvcrFeed}[basis_type]

        _grids = kwargs.get('grids', None)
        _interp = kwargs.get('interpolation', 'PolyInterpU')

        self.basis = Basis(self.geometry.atomic_numbers, self.shell_dict)
        self.h_feed = hsfeed.from_dir(
            path_to_skf, shell_dict, vcr=_grids, skf_type=skf_type,
            geometry=geometry, interpolation=_interp, integral_type='H')
        self.s_feed = hsfeed.from_dir(
            path_to_skf, shell_dict, vcr=_grids, skf_type=skf_type,
            geometry=geometry, interpolation=_interp, integral_type='S')
        self.skparams = SkfParamFeed.from_dir(
            path_to_skf, geometry, skf_type=skf_type)

    def init_dftb(self, **kwargs):
        self._n_batch = self.geometry._n_batch if self.geometry._n_batch \
            is not None else 1
        self.dtype = self.geometry.positions.dtype if \
            not self.geometry.isperiodic else torch.complex128
        self.qzero = self.skparams.qzero

        self.nelectron = self.qzero.sum(-1)

        if self.geometry.isperiodic:
            self.periodic = Periodic(self.geometry, self.geometry.cell,
                                     cutoff=self.skparams.cutoff, **kwargs)
            self.coulomb = Coulomb(self.geometry, self.periodic, method='search')
            self.distances = self.periodic.periodic_distances
            self.u = self._expand_u(self.skparams.U)
            self.max_nk = torch.max(self.periodic.n_kpoints)
        else:
            self.distances = self.geometry.distances
            self.u = self.skparams.U  # self.skt.U
            self.periodic, self.coulomb = None, None

        # if self.method in ('Dftb2', 'Dftb3', 'xlbomd'):
        self.method = kwargs.get('gamma_method', 'read')
        self.short_gamma = Gamma(
            self.u, self.distances, self.geometry.atomic_numbers,
            self.periodic, method=self.method).gamma

        self.atom_orbitals = self.basis.orbs_per_atom
        self.inv_dist = self._inv_distance(self.geometry.distances)

    def _inv_distance(self, distance):
        """Return inverse distance."""
        inv_distance = torch.zeros(*distance.shape)
        inv_distance[distance.ne(0.0)] = 1.0 / distance[distance.ne(0.0)]
        return inv_distance

    def _update_shift(self):
        """Update shift."""
        return torch.stack([(im - iz) @ ig for im, iz, ig in zip(
            self.charge[self.mask], self.qzero[self.mask], self.shift[self.mask])])

    def __call__(self, hamiltonian, overlap, iiter):
        # calculate the eigen-values & vectors via a Cholesky decomposition
        epsilon, eigvec = maths.eighb(hamiltonian, overlap)

        # calculate the occupation of electrons via the fermi method
        self.occ, nocc = fermi(epsilon, self.nelectron[self.mask])

        # eigenvector with Fermi-Dirac distribution
        c_scaled = torch.sqrt(self.occ).unsqueeze(1).expand_as(
            eigvec) * eigvec
        self.rho = c_scaled @ c_scaled.transpose(1, 2)  # -> density

        if iiter == 0 or not self.batch:
            self._density = self.rho.clone()
            self._occ = self.occ.clone()
            self._epsilon = epsilon.clone()
            self._eigvec = eigvec.clone()
        elif iiter > 0 and self.batch:
            self._epsilon[self.mask, :epsilon.shape[-1]] = epsilon
            self._eigvec[self.mask, :eigvec.shape[-1],
                         :eigvec.shape[-1]] = eigvec.clone()

        # calculate mulliken charges for each system in batch
        return mulliken(overlap, self.rho, self.basis.orbs_per_atom[self.mask])

    def __hs__(self, hamiltonian, overlap, **kwargs):
        """Hamiltonian or overlap feed."""
        multi_varible = kwargs.get('multi_varible', None)

        if self.geometry.isperiodic:
            hs_obj = self.periodic
        else:
            hs_obj = self.geometry
        if hamiltonian is None:
            self.ham = hs_matrix(
                hs_obj, self.basis, self.h_feed, multi_varible=multi_varible)
        else:
            self.ham = hamiltonian
        if overlap is None:
            self.over = hs_matrix(
                hs_obj, self.basis, self.s_feed, multi_varible=multi_varible)
        else:
            self.over = overlap

    def _next_geometry(self, geometry, **kwargs):
        """Update geometry for DFTB calculations."""
        if (self.geometry.atomic_numbers != geometry.atomic_numbers).any():
            raise ValueError('Atomic numbers in new geometry have changed.')

        self.geometry = geometry

        if self.geometry.isperiodic:
            self.periodic = Periodic(self.geometry, self.geometry.cell,
                                     cutoff=self.skparams.cutoff, **kwargs)
            self.coulomb = Coulomb(self.geometry, self.periodic, method='search')
            self.max_nk = torch.max(self.periodic.n_kpoints)
        # calculate short gamma
        if self.method in ('dftb2', 'dftb3'):
            self.short_gamma = Gamma(
                self.u, self.distances, self.geometry.atomic_numbers,
                self.periodic, method=self.method).gamma

        self.atom_orbitals = self.basis.orbs_per_atom
        self.inv_dist = self._inv_distance(self.geometry.distances)

    def _get_shift(self):
        """Return shift term for periodic and non-periodic."""
        if not self.geometry.isperiodic:
            return self.inv_dist - self.short_gamma
        else:
            return self.coulomb.invrmat - self.short_gamma

    def _update_scc_ik(self, epsilon, eigvec, _over, this_size,
                       iiter, ik=None, n_kpoints=None):
        """Update data for each kpoints."""
        if iiter == 0:
            if n_kpoints is None:
                self.epsilon = torch.zeros(*epsilon.shape)
                self.eigenvector = torch.zeros(*eigvec.shape, dtype=self.dtype)
            elif ik == 0:
                self.epsilon = torch.zeros(*epsilon.shape, n_kpoints)
                self.eigenvector = torch.zeros(
                    *eigvec.shape, n_kpoints, dtype=self.dtype)

        if ik is None:
            self.epsilon[self.mask, :epsilon.shape[1]] = epsilon
            self.eigenvector[
                self.mask, :eigvec.shape[1], :eigvec.shape[2]] = eigvec
        else:
            self.epsilon[self.mask, :epsilon.shape[1], ik] = epsilon
            self.eigenvector[
                self.mask, :eigvec.shape[1], :eigvec.shape[2], ik] = eigvec

    def _expand_u(self, u):
        """Expand Hubbert U for periodic system."""
        shape_cell = self.distances.shape[1]
        return u.repeat(shape_cell, 1, 1).transpose(0, 1)

    @property
    def init_charge(self):
        """Return initial charge."""
        return self.qzero

    @property
    def homo_lumo(self):
        """Return dipole moments."""
        # get HOMO-LUMO, not orbital resolved
        return torch.stack([
            ieig[int(iocc) - 1:int(iocc) + 1]
            for ieig, iocc in zip(self._epsilon, self.nocc)])

    @property
    def cpa(self):
        """Get onsite population for CPA DFTB.

        J. Chem. Phys. 144, 151101 (2016)
        """
        onsite = self._onsite_population()
        nat = self.geometry.n_atoms
        numbers = self.geometry.atomic_numbers

        return pack([1.0 + (onsite[ib] - self.qzero[ib])[:nat[ib]] / numbers[
            ib][:nat[ib]] for ib in range(self.geometry._n_batch)])

    @property
    def eigenvalue(self, unit='eV'):
        """Return eigenvalue."""
        sca = _Hartree__eV if unit == 'eV' else 1.0
        return self._epsilon * sca

    @property
    def charge(self):
        return self._charge

    @property
    def total_energy(self):
        return self.electronic_energy + self.repulsive_energy

    @property
    def band_energy(self):
        """Return H0 energy."""
        return (self._epsilon * self._occ).sum(-1)

    @property
    def H0_energy(self):
        """Return H0 energy."""
        return (self.ham * self._density).sum((-1, -2))

    @property
    def coulomb_energy(self):
        """Calculate Coulomb energy (atom resolved charge)."""
        _q = self.charge - self.qzero
        deltaq = _q.unsqueeze(1) * _q.unsqueeze(2)
        return 0.5 * (self.shift * deltaq).sum((-1, -2))

    @property
    def electronic_energy(self):
        """Return electronic energy."""
        return self.H0_energy + self.coulomb_energy

    @property
    def repulsive_energy(self):
        """Return repulsive energy."""
        return self.cal_repulsive().repulsive_energy

    @property
    def dipole(self):
        """Return dipole moments."""
        return torch.sum((self.qzero - self._charge).unsqueeze(-1) *
                         self.geometry.positions, 1)

    @property
    def deltaq(self):
        """Delta chrage."""
        return self.charge - self.qzero

    def cal_repulsive(self):
        return Repulsive(self.geometry, self.skparams, self.basis)

    @property
    def density(self):
        return self._density

    @property
    def energy_weighted_density(self):

        mask = self._occ.ne(0).unsqueeze(1).expand_as(self._eigvec)
        _eig = torch.zeros(self._eigvec.shape)
        _eig[mask] = self._eigvec[mask]

        dm1 = _eig @ _eig.transpose(1, 2)

        _eps = torch.zeros(self._occ.shape)
        _mask = self._occ.ne(0)
        _eps = self._occ * self._epsilon
        shift = torch.min(_eps) - 0.1
        _eps[_mask] = _eps[_mask] - shift
        c_scaled = torch.sqrt(_eps).unsqueeze(
            1).expand_as(self._eigvec) * self._eigvec #* sign

        dm2 = c_scaled @ c_scaled.transpose(1, 2)
        return dm2 + dm1 * shift

    @property
    def shift_orbital(self):
        return self.shift_orb

    @property
    def shift_mat(self):
        return self._shift_mat

    @property
    def U(self):
        return self.skparams.U


class Dftb1(Dftb):
    """Density-functional tight-binding method with 0th correction."""

    def __init__(self,
                 geometry: Geometry,
                 shell_dict: dict = None,
                 basis: object = None,
                 repulsive=True,
                 skf_type: str = 'h5', **kwargs):
        self.method = 'Dftb1'
        self.maxiter = kwargs.get('maxiter', 1)
        super().__init__(
            geometry, shell_dict, basis, repulsive, skf_type, **kwargs)
        super().init_dftb(**kwargs)

    def __call__(self,
                 charge: Tensor = None,  # -> Initial charge
                 geometry: Geometry = None,  # Update Geometry
                 hamiltonian: Tensor = None,
                 overlap: Tensor = None,
                 **kwargs):
        if geometry is not None:
            super()._next_geometry(geometry, **kwargs)
        self.shift = self._get_shift()

        self.mask = torch.tensor([True]).repeat(self._n_batch)
        super().__hs__(hamiltonian, overlap, **kwargs)

        # self.gamma = self.inv_dist - self.short_gamma

        if charge is not None:
            d_q = charge - self.qzero
            self._shift = torch.bmm(d_q.unsqueeze(1), self.shift)
            self.shift_orb = pack([
                ishif.repeat_interleave(iorb) for iorb, ishif in
                zip(self.atom_orbitals, self._shift)])
            self._shift_mat = torch.stack([torch.unsqueeze(ishift, 1) + ishift
                                           for ishift in self.shift_orb])

            self.hamiltonian = self.ham + 0.5 * self.over * self._shift_mat
            self.qm = super().__call__(self.hamiltonian, self.over, iiter=0)
        else:
            self.qm = super().__call__(self.ham, self.over, iiter=0)

        self._charge = self.qm


class Dftb2(Dftb):
    """Self-consistent-charge density-functional tight-binding method."""

    def __init__(self,
                 geometry: object,
                 shell_dict: Dict[int, List[int]],
                 path_to_skf: str,
                 repulsive: bool = True,
                 skf_type: Literal['h5', 'skf'] = 'h5',
                 basis_type: str = 'normal',
                 periodic: Periodic = None,
                 mixer: str = 'Anderson',
                 **kwargs):
        self.method = 'Dftb2'
        self.maxiter = kwargs.get('maxiter', 60)
        super().__init__(geometry, shell_dict, path_to_skf,
                         repulsive, skf_type, basis_type, periodic, mixer, **kwargs)
        super().init_dftb(**kwargs)
        self.converge_number = []

    def __call__(self,
                 charge: Tensor = None,  # -> Initial charge
                 geometry: Geometry = None,  # Update Geometry
                 hamiltonian: Tensor = None,
                 overlap: Tensor = None,
                 **kwargs):
        """Perform SCC-DFTB calculation."""
        self.mixer = globals()[self.mixer_type](self.qzero, return_convergence=True)
        self.shift = self._get_shift()

        if geometry is not None:
            super()._next_geometry(geometry)

        self.mask = torch.tensor([True]).repeat(self._n_batch)
        super().__hs__(hamiltonian, overlap, **kwargs)
        self._charge = self.qzero.clone() if charge is None else charge

        # Loop for DFTB2
        for iiter in range(self.maxiter):
            if self.mask.any():
                self._single_loop(iiter)
            else:
                break

    def _single_loop(self, iiter):
        """Perform each single SCC-DFTB loop."""
        # get shift and repeat shift according to number of orbitals
        shift_ = self._update_shift()
        shiftorb_ = pack([ishif.repeat_interleave(iorb) for iorb, ishif in
                          zip(self.atom_orbitals[self.mask], shift_)])
        shift_mat = torch.stack([torch.unsqueeze(ishift, 1) + ishift
                                 for ishift in shiftorb_])
        if self.geometry.isperiodic:
            shift_mat = shift_mat.repeat(torch.max(
                self.periodic.n_kpoints), 1, 1, 1).permute(1, 2, 3, 0)

        # For molecule, the shift_mat will be [n_batch, n_size, n_size]
        # For solid, the size is [n_batch, n_size, n_size, n_kpt]
        this_size = shift_mat.shape[-2]
        fock = self.ham[self.mask, :this_size, :this_size] + \
            0.5 * self.over[self.mask, :this_size, :this_size] * shift_mat

        if not self.geometry.isperiodic:
            d_q = self._charge[self.mask] - self.qzero[self.mask]
            self._shift = torch.bmm(d_q.unsqueeze(1), self.shift[self.mask])

            self.shift_orb = pack([
                ishif.repeat_interleave(iorb) for iorb, ishif in
                zip(self.basis.orbs_per_atom[self.mask], self._shift)])

            _shift_mat = torch.stack([torch.unsqueeze(ishift, 1) + ishift
                                     for ishift in self.shift_orb])

            this_size = _shift_mat.shape[-1]   # the new shape
            overlap = self.over[self.mask, :this_size, :this_size]
            self.hamiltonian = self.ham[self.mask, :this_size, :this_size] + \
                0.5 * overlap * _shift_mat

            # single loop SCC-DFTB calculation
            self.qm = super().__call__(self.hamiltonian, overlap, iiter)
            self.qmix, _mask = self.mixer(self.qm)
            self._charge[self.mask] = self.qmix
            self._density[self.mask, :self.rho.shape[1], :self.rho.shape[2]] = self.rho
            self._occ[self.mask, :self.occ.shape[-1]] = self.occ
            if iiter == 0:
                self._shift_mat = _shift_mat.clone()
            else:
                self._shift_mat[self.mask, :this_size, :this_size] = _shift_mat

            self.mask = ~_mask
            self.converge_number.append(_mask.sum().tolist())

        else:
            self.ie, eigvec, nocc, density, q_new = [], [], [], [], []
            self._mask_k = []

            # Loop over all K-points
            for ik in range(self.max_nk):

                # calculate the eigen-values & vectors
                iep, ieig = maths.eighb(
                    fock[..., ik], self.over[self.mask, :this_size, :this_size, ik])
                self.ie.append(iep), eigvec.append(ieig)
                self._update_scc_ik(
                    iep, ieig, self.over[..., ik],
                    this_size, iiter, ik, torch.max(self.periodic.n_kpoints))

                iocc, inocc = fermi(iep, self.nelectron[self.mask])
                nocc.append(inocc)
                iden = torch.sqrt(iocc).unsqueeze(1).expand_as(ieig) * ieig
                irho = (torch.conj(iden) @ iden.transpose(1, 2))  # -> density
                density.append(irho)

                # calculate mulliken charges for each system in batch
                iq = mulliken(self.over[self.mask, :this_size, :this_size, ik],
                                  irho, self.atom_orbitals[self.mask])

                _q = iq.real
                q_new.append(_q)

            nocc = pack(nocc).T
            self.rho = pack(density).permute(1, 2, 3, 0)
            if iiter == 0:
                self.nocc = torch.zeros(*nocc.shape)
                self._density = torch.zeros(*self.rho.shape, dtype=self.rho.dtype)

            q_new = (pack(q_new).permute(2, 1, 0) * self.periodic.k_weights[
                self.mask]).sum(-1).T
            self.qmix, _mask = self.mixer(q_new)
            epsilon = pack(self.ie)
            if iiter == 0:
                self._epsilon = epsilon
                self._charge = self.qmix
                self._nocc = nocc
                self._density = self.rho
            else:
                self._epsilon[:epsilon.shape[0], self.mask, :epsilon.shape[-1]] = epsilon
                self._charge[self.mask, :self.qmix.shape[-1]] = self.qmix
                self._nocc[self.mask, :nocc.shape[-1]] = nocc
                self._density[self.mask, :self.rho.shape[1], :self.rho.shape[2]] = self.rho
            self.mask = ~_mask

    def _onsite_population(self):
        """Get onsite population for CPA DFTB.

        sum density matrix diagnal value for each atom
        """
        nb = self.geometry._n_batch
        ns = self.geometry.n_atoms
        acum = torch.cat([torch.zeros(self.atom_orbitals.shape[0]).unsqueeze(0),
                          torch.cumsum(self.atom_orbitals, dim=1).T]).T.long()
        denmat = [idensity.diag() for idensity in self.density]

        # get onsite population
        return pack([torch.stack([torch.sum(denmat[ib][acum[ib][iat]: acum[ib][iat + 1]])
                    for iat in range(ns[ib])]) for ib in range(nb)])

    @property
    def dos_energy(self, unit='eV', ext=1, grid=1000):
        """Energy distribution of (P)DOS.

        Arguments:
            unit: The unit of distribution of (P)DOS energy.

        """
        self.unit = unit
        e_min = torch.min(self._epsilon.detach()) - ext
        e_max = torch.max(self._epsilon.detach()) + ext

        if unit in ('eV', 'EV', 'ev'):
            return torch.linspace(e_min, e_max, grid)
        elif unit in ('hartree', 'Hartree'):
            return torch.linspace(e_min, e_max, grid) * _Hartree__eV
        else:
            raise ValueError('unit of energy in DOS should be eV or Hartree.')

    @property
    def pdos(self):
        """Return PDOS."""
        energy = torch.linspace(-1, 1, 200)
        return pdos(self.eigenvector, self.over, self._epsilon, energy)

    @property
    def dos(self):
        """Return energy distribution and DOS with fermi energy correction."""
        sigma = self.params.dftb_params['sigma']
        energy = self.dos_energy
        energy = energy.repeat(self.system.size_batch, 1)  # -> to batch

        # make sure the 1st dimension is batch
        if self.unit in ('eV', 'EV', 'ev'):
            return dos((self._epsilon - self.fermi.unsqueeze(1)),
                       energy, sigma)  #, mask=self.band_filter)
        elif self.unit in ('hartree', 'Hartree'):
            return dos(self._epsilon - self.fermi.unsqueeze(1) * _Hartree__eV,
                       energy, sigma)  #, mask=self.band_filter)

    @property
    def band_filter(self, n_homo=torch.tensor([3]), n_lumo=torch.tensor([3]),
                    band_filter=True) -> Tensor:
        """Return filter of states."""
        if band_filter:
            n_homo = n_homo.repeat(self._epsilon.shape[0])
            n_lumo = n_lumo.repeat(self._epsilon.shape[0])
            return band_pass_state_filter(self._epsilon, n_homo, n_lumo, self.fermi)
        else:
            return None


class Dftb3(Dftb):
    """Density functional tight binding method with third order.
    """

    def __init__(self):
        pass
