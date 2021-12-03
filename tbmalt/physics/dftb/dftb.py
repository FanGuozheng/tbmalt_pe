"""DFTB calculator.

implement pytorch to DFTB
"""
from typing import Literal, Optional
import torch
from tbmalt import Basis, Geometry, SkfParamFeed
import tbmalt.common.maths as maths
from tbmalt import SkfFeed, hs_matrix
from tbmalt.ml.skfeeds import VcrFeed, TvcrFeed
from tbmalt.physics.electrons import fermi
from tbmalt.physics.properties import mulliken
from tbmalt.common.maths.mixer import Simple, Anderson
from tbmalt.physics.properties import dos, pdos, band_pass_state_filter
from tbmalt.common.batch import pack
from tbmalt.physics.electrons import Gamma
from tbmalt.structures.periodic import Periodic
from tbmalt.physics.coulomb import Coulomb
from tbmalt.common.units import AUEV
Tensor = torch.Tensor


class Dftb:
    """Density functional based tight binding method template class.

    The `Dftb` is designed to calculate high-throughput DFTB calculations, the
    input will be transfered to batch to avoid a lot of `if` or `else`. To
    make sure the DFTB results could be easily used by other framework, the
    output will be transfered to single if the input is single.

    Arguments:
        geometry: `Geometry` object in TBMaLT.
        shell_dict : The angular momenta of each shell.
        path_to_skf: Path to Slater-Koster files.
        hamiltonian: Hamiltonian Tensor.
        overlap: overlap Tensor.
        skf_type: The type of input Slater-Koster files.
        basis_type: The type of basis.
        mixer: Type of mixers.

    """

    def __init__(self,
                 geometry: Geometry,
                 shell_dict: dict,
                 path_to_skf: str,
                 hamiltonian: Optional[Tensor] = None,
                 overlap: Optional[Tensor] = None,
                 skf_type: Literal['h5', 'skf'] = 'h5',
                 basis_type: Literal['normal', 'vcr', 'tvcr'] = 'normal',
                 mixer: Literal['Simple', 'Anderson'] = 'Anderson',
                 **kwargs):
        # General parameters
        self.mixer_type = mixer
        self.geometry = geometry
        self.batch = True if self.geometry.distances.dim() == 3 else False
        self.basis = Basis(geometry.atomic_numbers, shell_dict)

        hsfeed = {'normal': SkfFeed, 'vcr': VcrFeed, 'tvcr': TvcrFeed}[basis_type]
        _grids = kwargs.get('grids', None)
        _interp = kwargs.get('interpolation', 'PolyInterpU')

        # Max iterative SCC steps
        maxiter = kwargs.get('maxiter', 60)
        self.maxiter = maxiter if self.method in ('Dftb2', 'Dftb3') else 1

        # Create periodic related routine
        self.skparams = SkfParamFeed.from_dir(
            path_to_skf, geometry, skf_type=skf_type)
        if self.geometry.isperiodic:
            self.periodic = Periodic(self.geometry, self.geometry.cell,
                                     cutoff=self.skparams.cutoff, **kwargs)
            self.coulomb = Coulomb(self.geometry, self.periodic, method='search')
            hs_obj = self.periodic  # TEMP CODE!!!
        else:
            hs_obj = self.geometry

        # initialize hamiltonian and overlap
        multi_varible = kwargs.get('multi_varible', None)
        if hamiltonian is None:
            h_feed = hsfeed.from_dir(
                path_to_skf, shell_dict, vcr=_grids, skf_type=skf_type,
                geometry=geometry, interpolation=_interp, integral_type='H')
            self.ham = hs_matrix(
                hs_obj, self.basis, h_feed,  multi_varible=multi_varible)
        else:
            self.ham = hamiltonian
        if overlap is None:
            s_feed = hsfeed.from_dir(
                path_to_skf, shell_dict, vcr=_grids, skf_type=skf_type,
                geometry=geometry, interpolation=_interp, integral_type='S')
            self.over = hs_matrix(
                 hs_obj, self.basis, s_feed, multi_varible=multi_varible)
        else:
            self.over = overlap

    def init_dftb(self, **kwargs):
        self.mask = torch.tensor([True]).repeat(self.ham.shape[0])
        self.atom_orbitals = self.basis.orbs_per_atom
        self._n_batch = self.geometry._n_batch  # -> atoms in each system

        # intial charges
        self.qzero = self.skparams.qzero
        charge = kwargs.get('charge', None)
        self.charge = charge if charge is not None else self.qzero.clone()
        self.nelectron = self.qzero.sum(axis=1)

        assert self.mixer_type in ('Anderson', 'Simple')
        self.mixer = globals()[self.mixer_type](
            self.qzero, return_convergence=True)

        if self.geometry.isperiodic:
            self.distances = self.periodic.periodic_distances
            self.u = self._expand_u(self.skparams.U)
        else:
            self.distances = self.geometry.distances
            self.u = self.skparams.U  # self.skt.U
            self.periodic, self.coulomb = None, None

        if self.method in ('Dftb2', 'Dftb3', 'xlbomd'):
            self.method = kwargs.get('gamma_method', 'read')
            self.gamma = Gamma(
                self.u, self.distances, self.geometry.atomic_numbers,
                self.periodic, method=self.method).gamma
        else:
            self.gamma = torch.zeros(*self.qzero.shape)

        self.inv_dist = self._inv_distance(self.geometry.distances)

        # replace the ewald summation for non-periodic systems
        if self.geometry.isperiodic:
            if not self.periodic.mask_pe.all():
                _invr = torch.clone(self.inv_dist)
                _invr[self.periodic.mask_pe] = self.coulomb.invrmat
                self.coulomb.invrmat = _invr

            self.max_nk = torch.max(self.periodic.n_kpoints)

        self.shift = self._get_shift()

    def _inv_distance(self, distance):
        """Return inverse distance."""
        inv_distance = torch.zeros(*distance.shape)
        inv_distance[distance.ne(0.0)] = 1.0 / distance[distance.ne(0.0)]
        return inv_distance

    def __call__(self, hamiltonian, overlap, iiter):
        # calculate the eigen-values & vectors via a Cholesky decomposition
        epsilon, eigvec = maths.eighb(hamiltonian, overlap)

        # calculate the occupation of electrons via the fermi method
        occ, nocc = fermi(epsilon, self.nelectron[self.mask])

        # eigenvector with Fermi-Dirac distribution
        c_scaled = torch.sqrt(occ).unsqueeze(1).expand_as(eigvec) * eigvec
        self.rho = c_scaled @ c_scaled.transpose(1, 2)  # -> density
        if iiter == 0:
            self._density = self.rho.clone()

        # calculate mulliken charges for each system in batch
        return mulliken(overlap, self.rho, self.basis.orbs_per_atom[self.mask])

    def _expand_u(self, u):
        """Expand Hubbert U for periodic system."""
        shape_cell = self.distances.shape[1]
        return u.repeat(shape_cell, 1, 1).transpose(0, 1)

    def _get_shift(self):
        """Return shift term for periodic and non-periodic."""
        if not self.geometry.isperiodic:
            return self.inv_dist - self.gamma
        else:
            return self.coulomb.invrmat - self.gamma

    def _update_shift(self):
        """Update shift."""
        return torch.stack([(im - iz) @ ig for im, iz, ig in zip(
            self.charge[self.mask], self.qzero[self.mask], self.shift[self.mask])])

    def _update_scc(self, qmix, epsilon, eigvec, nocc):
        """Update charge according to convergence in last step."""
        self.charge[self.mask] = qmix
        self.nocc[self.mask] = nocc
        self.density[self.mask, :self.rho.shape[1], :self.rho.shape[2]] = self.rho

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

    @property
    def mulliken_charge(self) -> Tensor:
        return self.charge.squeeze(0)

    @property
    def dipole(self) -> Tensor:
        """Return dipole moments."""
        return torch.sum((self.qzero - self.charge).unsqueeze(-1) *
                         self.geometry.positions, 1).squeeze(0)

    @property
    def homo_lumo(self) -> Tensor:
        """Return dipole moments."""
        # get HOMO-LUMO, not orbital resolved
        _mask = torch.stack([
            ieig[int(iocc)] - ieig[int(iocc - 1)] < 1E-10
            for ieig, iocc in zip(self.eigenvalue, self.nocc)])
        self.nocc[_mask] = self.nocc[_mask] + 1
        return torch.stack([
            ieig[int(iocc) - 1:int(iocc) + 1]
            for ieig, iocc in zip(self.eigenvalue, self.nocc)]).squeeze(0)

    @property
    def onsite_population(self) -> Tensor:
        """Get onsite population for CPA DFTB.

        sum density matrix diagnal value for each atom
        """
        ao = self.geometry.atom_orbitals
        nb = self.geometry.size_batch
        ns = self.geometry._n_batch
        acum = torch.cat([torch.zeros(ao.shape[0]).unsqueeze(0),
                          torch.cumsum(ao, dim=1).T]).T.long()
        denmat = [idensity.diag() for idensity in self.density]

        # get onsite population
        return pack([torch.stack(
            [torch.sum(denmat[ib][acum[ib][iat]: acum[ib][iat + 1]])
             for iat in range(ns[ib])]) for ib in range(nb)]).squeeze(0)

    @property
    def cpa(self):
        """Get onsite population for CPA DFTB.

        J. Chem. Phys. 144, 151101 (2016)
        """
        onsite = self._onsite_population()
        nat = self.geometry._n_batch
        numbers = self.geometry.numbers

        return pack([1.0 + (onsite[ib] - self.qzero[
            ib])[:nat[ib]] / numbers[ib][:nat[ib]]
            for ib in range(self.geometry.size_batch)]).squeeze(0)

    @property
    def ini_charge(self):
        """Return initial charge."""
        return self.qzero.squeeze(0)

    @property
    def eigenvalue(self):
        """Return eigenvalue."""
        return self.epsilon.squeeze(0) * AUEV

    @property
    def fermi(self):
        """Fermi energy."""
        return self.homo_lumo.sum(-1).squeeze(0) / 2.0

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
        return pdos(
            self.eigenvector, self.over, self.eigenvalue, energy).squeeze(0)

    @property
    def dos(self, sigma: float = 0.1):
        """Return energy distribution and DOS with fermi energy correction."""
        energy = self.dos_energy
        energy = energy.repeat(self.geometry.size_batch, 1)  # -> to batch

        # make sure the 1st dimension is batch
        if self.unit in ('eV', 'EV', 'ev'):
            return dos((self.eigenvalue),
                       energy, sigma).squeeze(0)  # , mask=self.band_filter)
        elif self.unit in ('hartree', 'Hartree'):
            return dos(self.eigenvalue,
                       energy, sigma).squeeze(0)

    def band_filter(self, n_homo=torch.tensor([3]), n_lumo=torch.tensor([3]),
                    band_filter=True) -> Tensor:
        """Return filter of states."""
        if band_filter:
            n_homo = n_homo.repeat(self.eigenvalue.shape[0])
            n_lumo = n_lumo.repeat(self.eigenvalue.shape[0])

            return band_pass_state_filter(
                self.eigenvalue, n_homo, n_lumo, self.fermi)


class Dftb1(Dftb):
    """Density functional tight binding method with first order."""

    def __init__(self,
                 geometry: object,
                 shell_dict: dict,
                 path_to_skf: str,
                 hamiltonian: Optional[Tensor] = None,
                 overlap: Optional[Tensor] = None,
                 skf_type: Literal['h5', 'skf'] = 'h5',
                 basis_type: Literal['normal', 'vcr', 'tvcr'] = 'normal',
                 mixer: Literal['Simple', 'Anderson'] = 'Anderson',
                 **kwargs):

        self.method = 'Dftb2'
        self.geometry = geometry

        self.basis = Basis(geometry.atomic_numbers, shell_dict)

        self.dtype = self.geometry.positions.dtype if \
            not self.geometry.isperiodic else torch.complex128

        super().__init__(geometry, shell_dict, path_to_skf,
                         hamiltonian, overlap, skf_type, basis_type,
                         mixer, **kwargs)

        super().init_dftb(**kwargs)

        self._scc()

    def _scc(self, iiter=0):
        """"""
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

        # calculate the eigen-values & vectors via a Cholesky decomposition
        if not self.geometry.isperiodic:
            epsilon, eigvec = maths.eighb(
                fock, self.over[self.mask, :this_size, :this_size])
            self._update_scc_ik(epsilon, eigvec, self.over, this_size, iiter)
            occ, nocc = fermi(epsilon, self.nelectron[self.mask])

            # eigenvector with Fermi-Dirac distribution
            c_scaled = torch.sqrt(occ).unsqueeze(1).expand_as(eigvec) * eigvec
            self.rho = c_scaled @ c_scaled.transpose(1, 2)  # -> density

            if iiter == 0:
                self.nocc = torch.zeros(*nocc.shape)
                self.density = torch.zeros(
                    *self.rho.shape, dtype=self.rho.dtype)

            # # calculate mulliken charges for each system in batch
            q_new = mulliken(self.over[self.mask, :this_size, :this_size],
                             self.rho, self.atom_orbitals[self.mask])
        else:
            self.ie, eigvec, nocc, density, q_new = [], [], [], [], []
            self._mask_k = []

            # Loop over all K-points
            for ik in range(self.max_nk):

                # calculate the eigen-values & vectors
                iep, ieig = maths.eighb(
                    fock[..., ik],
                    self.over[self.mask, :this_size, :this_size, ik])
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
                self.density = torch.zeros(
                    *self.rho.shape, dtype=self.rho.dtype)

            q_new = (pack(q_new).permute(2, 1, 0) * self.periodic.k_weights[
                self.mask]).sum(-1).T
            # self.qmix, self.converge = self.mixer(q_new)

            epsilon = pack(self.ie)

        self._update_scc(q_new, epsilon, eigvec, nocc)


class Dftb2(Dftb):
    """Self-consistent charge density functional tight binding method.

    Arguments:
        geometry: Object contains geometry and orbital information.
        skt: Object contains SK data.

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

    def __init__(self,
                 geometry: object,
                 shell_dict: dict,
                 path_to_skf: str,
                 hamiltonian: Optional[Tensor] = None,
                 overlap: Optional[Tensor] = None,
                 skf_type: Literal['h5', 'skf'] = 'h5',
                 basis_type: Literal['normal', 'vcr', 'tvcr'] = 'normal',
                 mixer: Literal['Simple', 'Anderson'] = 'Anderson',
                 **kwargs):

        self.method = 'Dftb2'
        self.geometry = geometry

        self.basis = Basis(geometry.atomic_numbers, shell_dict)

        self.dtype = self.geometry.positions.dtype if \
            not self.geometry.isperiodic else torch.complex128

        super().__init__(geometry, shell_dict, path_to_skf, hamiltonian,
                         overlap, skf_type, basis_type, mixer, **kwargs)

        super().init_dftb(**kwargs)

        self._scc()

    def _scc(self):
        """"""
        for iiter in range(self.maxiter):
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

            # calculate the eigen-values & vectors via a Cholesky decomposition
            if not self.geometry.isperiodic:
                epsilon, eigvec = maths.eighb(
                    fock, self.over[self.mask, :this_size, :this_size])
                self._update_scc_ik(epsilon, eigvec, self.over, this_size, iiter)

                occ, nocc = fermi(epsilon, self.nelectron[self.mask])

                # eigenvector with Fermi-Dirac distribution
                c_scaled = torch.sqrt(occ).unsqueeze(1).expand_as(eigvec) * eigvec
                self.rho = c_scaled @ c_scaled.transpose(1, 2)  # -> density

                if iiter == 0:
                    self.nocc = torch.zeros(*nocc.shape)
                    self.density = torch.zeros(
                        *self.rho.shape, dtype=self.rho.dtype)

                # calculate mulliken charges for each system in batch
                q_new = mulliken(self.over[self.mask, :this_size, :this_size],
                                 self.rho, self.atom_orbitals[self.mask])
                self.qmix, self.converge = self.mixer(q_new)

            else:
                self.ie, eigvec, nocc, density, q_new = [], [], [], [], []
                self._mask_k = []

                # Loop over all K-points
                for ik in range(self.max_nk):

                    # calculate the eigen-values & vectors
                    iep, ieig = maths.eighb(
                        fock[..., ik],
                        self.over[self.mask, :this_size, :this_size, ik])
                    self.ie.append(iep), eigvec.append(ieig)
                    self._update_scc_ik(
                        iep, ieig, self.over[..., ik], this_size, iiter,
                        ik, torch.max(self.periodic.n_kpoints))

                    iocc, inocc = fermi(iep, self.nelectron[self.mask])
                    nocc.append(inocc)
                    iden = torch.sqrt(iocc).unsqueeze(1).expand_as(ieig) * ieig
                    irho = (torch.conj(iden) @ iden.transpose(1, 2))
                    density.append(irho)

                    # calculate mulliken charges for each system in batch
                    iq = mulliken(
                        self.over[self.mask, :this_size, :this_size, ik],
                        irho, self.atom_orbitals[self.mask])

                    _q = iq.real
                    q_new.append(_q)

                nocc = pack(nocc).T
                self.rho = pack(density).permute(1, 2, 3, 0)
                if iiter == 0:
                    self.nocc = torch.zeros(*nocc.shape)
                    self.density = torch.zeros(
                        *self.rho.shape, dtype=self.rho.dtype)

                q_new = (pack(q_new).permute(2, 1, 0) *
                         self.periodic.k_weights[self.mask]).sum(-1).T
                self.qmix, self.converge = self.mixer(q_new)
                epsilon = pack(self.ie)

            # Update charge, convergence
            self._update_scc(self.qmix, epsilon, eigvec, nocc)
            self.mask = ~self.converge
            if (self.converge == True).all():
                break


class Dftb3(Dftb):
    """Density functional tight binding method with third order.
    """

    def __init__(self):
        pass
