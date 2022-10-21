"""DFTB calculator.
implement pytorch to DFTB
"""
from typing import Literal, Dict, List, Union

import numpy as np
import torch
from torch import Tensor
import tbmalt.common.maths as maths
from tbmalt import Basis, Geometry, SkfFeed, SkfParamFeed, hs_matrix
from tbmalt.physics.dftb.repulsive import Repulsive
from tbmalt.common.maths.mixer import Simple, Anderson
from tbmalt.physics.properties import mulliken, mulliken_orb_to_atom,\
    dos, pdos, band_pass_state_filter
from tbmalt.common.batch import pack
from tbmalt.physics.fermi import fermi
from tbmalt.data.units import _Hartree__eV
from tbmalt.ml.skfeeds import VcrFeed, TvcrFeed
from tbmalt.physics.dftb.shortgamma import ShortGamma
from tbmalt.structures.periodic import Periodic
from tbmalt.physics.coulomb import Coulomb
from tbmalt.data.units import energy_units
from tbmalt.physics.filling import fermi_search, fermi_smearing
from tbmalt.common import split_by_size
from tbmalt.physics.force import DftbGradient


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
                 temperature: Union[float, Tensor] = 300.0,
                 **kwargs):
        self.skf_type = skf_type
        self.shell_dict = shell_dict
        self.geometry = geometry
        self.repulsive = repulsive
        self.mixer_type = mixer
        self.temperature = temperature
        self.spin_params = kwargs.get('spin_params', None)
        self.plusu_params = kwargs.get('plusu_params', None)
        self.merge_onsite_orbs = kwargs.get('merge_onsite_orbs', True)
        self.onsite_orbital_resolved = kwargs.get('onsite_orbital_resolved', False)
        self.orbital_resolved = kwargs.get('orbital_resolved', False)
        self.force = kwargs.get('force', None)

        self.interpolation = kwargs.get('interpolation', 'PolyInterpU')
        self.batch = True if self.geometry.distances.dim() == 3 else False
        hsfeed = {'normal': SkfFeed, 'vcr': VcrFeed, 'tvcr': TvcrFeed}[basis_type]

        _grids = kwargs.get('grids', None)
        _interp = kwargs.get('interpolation', 'PolyInterpU')

        self.basis = Basis(self.geometry.atomic_numbers, self.shell_dict)
        self.atom_orbitals = self.basis.orbs_per_atom
        self.mask_atom_block = (self.basis.on_atoms.unsqueeze(-1) -
                                self.basis.on_atoms.unsqueeze(-2)) == 0
        self.mask_l_block = (self.basis.on_shells.unsqueeze(-1) -
                             self.basis.on_shells.unsqueeze(-2)) == 0
        self.mask_onsite_block = self.mask_atom_block * self.mask_l_block

        self.h_feed = hsfeed.from_dir(
            path_to_skf, shell_dict, vcr=_grids, skf_type=skf_type,
            geometry=geometry, interpolation=_interp, integral_type='H',
            merge_orbs=self.merge_onsite_orbs,
            onsite_orbital_resolved=self.onsite_orbital_resolved)
        self.s_feed = hsfeed.from_dir(
            path_to_skf, shell_dict, vcr=_grids, skf_type=skf_type,
            geometry=geometry, interpolation=_interp, integral_type='S',
            merge_orbs=self.merge_onsite_orbs,
            onsite_orbital_resolved=self.onsite_orbital_resolved)
        self.skparams = SkfParamFeed.from_dir(
            path_to_skf, geometry, skf_type=skf_type, repulsive=repulsive)

    def init_dftb(self, **kwargs):
        self._n_batch = self.geometry._n_batch if self.geometry._n_batch \
            is not None else 1
        self.dtype = self.geometry.positions.dtype if \
            not self.geometry.is_periodic else torch.complex128
        self.qzero = self.skparams.qzero
        self.numbers_orbs, self.l_shell, self.orbs_per_shell, self.qzero_orbs,\
            self.mask_orbital_resolved = self._qzero_orbs()
        self._charge_orbs = self.qzero_orbs.clone()

        if self.spin_params is None:
            self.nelectron = self.qzero.sum(-1)

        # Initialization for SDFTB
        else:
            self.qzero_shell = self._qzero_shell()
            self.w_params = self._gather_w()
            self.q_potential = torch.zeros(self.qzero_shell.shape)
            self.q_potential_orbs = torch.zeros(self.qzero_orbs.shape)
            self.unpaired_electrons = self.spin_params['unpaired_electrons']\
                if 'unpaired_electrons' in self.spin_params.keys() else None

            self.nelectron = torch.zeros((*self.qzero.shape[:-1], 2))
            self.nelectron[..., 0] = self.qzero.sum(-1) / 2
            self.nelectron[..., 1] = self.qzero.sum(-1) / 2
            if self.unpaired_electrons is not None:
                _mask = self.unpaired_electrons != 0
                self.q_potential[_mask] = self.qzero_shell[_mask] /\
                                          self.qzero_shell[_mask].sum(-1) *\
                                          self.unpaired_electrons[_mask]
                self.nelectron[..., 0] = self.nelectron[..., 0] + self.q_potential.sum(-1) / 2
                self.nelectron[..., 1] = self.nelectron[..., 1] - self.q_potential.sum(-1) / 2

                self.q_potential_orbs[_mask] = self.qzero_orbs[_mask] /\
                                          self.qzero_orbs[_mask].sum(-1) *\
                                          self.unpaired_electrons[_mask]

            # self._charge_orbs_u = self._charge_orbs.clone()
            # self._charge_orbs_d = self._charge_orbs.clone()

        # Initialization for DFTB+U
        if self.plusu_params is not None:
            if 'type' not in self.plusu_params.keys():
                self.plusu_params['type'] = 'fll'
            self.plusu_orbs, self.plusu_l = self._gather_plus_u()
            self.alpha_u = 0.5

        if self.geometry.is_periodic:
            self.periodic = Periodic(self.geometry, self.geometry.cell,
                                     cutoff=self.skparams.cutoff, **kwargs)
            self.coulomb = Coulomb(self.geometry, self.periodic, method='search')
            self.distances = self.periodic.periodic_distances
            self.u = self._expand_u(self.skparams.U)
            self.max_nk = torch.max(self.periodic.n_kpoints)

            # self.ksampling = Ksampling(
            #     self.geometry._n_batch, self.geometry.atomic_numbers, **kwargs
            # )
            # self.kpoints = self.ksampling.kpoints
            self.kpoints = self.periodic.kpoints
            self.n_kpoints = self.periodic.n_kpoints
            self.k_weights = self.periodic.k_weights
            self.phase = self.periodic.phase
            self.max_nk = torch.max(self.periodic.n_kpoints)

        else:
            self.distances = self.geometry.distances
            self.u = self.skparams.U  # self.skt.U
            self.periodic, self.coulomb = None, None

        if self.orbital_resolved:
            pos = pack([ipos.repeat_interleave(iorb, dim=0) for iorb, ipos in
                        zip(self.atom_orbitals, self.geometry.positions)])
            self.orbs_numbers = pack([ipos.repeat_interleave(iorb, dim=0) for iorb, ipos in
                        zip(self.atom_orbitals, self.geometry.atomic_numbers)])
            self.distance_orbs = torch.cdist(pos, pos, p=2)
            # Not real orbital resolved U, just repeat over l
            self.u_orbs = self._expand_u_orbs(self.u)
        else:
            self.u_orbs = None
            self.orbs_numbers = None
            self.distance_orbs = None

        # if self.method in ('Dftb2', 'Dftb3', 'xlbomd'):
        mask = self.periodic.mask_central_cell if self.geometry.is_periodic else None
        gamma = ShortGamma(
            self.u, self.geometry.atomic_numbers,
            self.distances, self.periodic,
            orbital_resolved=self.orbital_resolved,
            u_orbs=self.u_orbs, orbs_numbers=self.orbs_numbers,
            distance_orbs=self.distance_orbs,
            mask_central_cell=mask)
        self.short_gamma = gamma.short_gamma
        self.short_gamma_pe = gamma.short_gamma_pe
        self.short_gamma_pe_on = gamma.short_gamma_pe_on

        dist = self.distance_orbs if self.orbital_resolved else self.geometry.distances
        self.inv_dist = self._inv_distance(dist)

    def _inv_distance(self, distance):
        """Return inverse distance."""
        inv_distance = torch.zeros(*distance.shape)
        inv_distance[distance.ne(0.0)] = 1.0 / distance[distance.ne(0.0)]
        return inv_distance

    def _update_shift(self):
        """Update shift."""
        return torch.einsum(
            '...i,...ij->...j',
            self.charge[self.mask] - self.qzero[self.mask], self.shift[self.mask])

    def _qzero_orbs(self):
        """Return m resolved charge."""
        qzero_orbs = torch.zeros(self.basis.orbital_matrix_shape[:-1])
        an = self.geometry.atomic_numbers.view(-1)
        numbers_l_uniq = an.repeat_interleave(self.basis._shells_per_species[an])

        counts = self.basis.shell_ls * 2 + 1
        counts = counts[counts != -1]
        basis_list = self.basis.shell_ls[self.basis.shell_ls != -1].\
            repeat_interleave(counts.view(-1))
        numbers_list = numbers_l_uniq.repeat_interleave(counts.view(-1))
        orbs_per_shell = self.basis.orbs_per_shell[self.basis.orbs_per_shell != 0].\
            repeat_interleave(counts.view(-1))
        l_shell = pack(split_by_size(basis_list, self.basis.n_orbitals), value=-1)
        numbers_orbs = pack(split_by_size(numbers_list, self.basis.n_orbitals), value=-1)
        orbs_per_shell = pack(split_by_size(orbs_per_shell, self.basis.n_orbitals), value=-1)

        for inum in self.geometry.unique_atomic_numbers():
            for il in self.shell_dict[inum.tolist()]:
                mask_num = numbers_orbs == inum
                mask_l = l_shell == il
                mask = mask_l * mask_num
                qzero_orbs[mask] = self.skparams.sktable_dict[(inum.tolist(), 'occupations')][il]
                qzero_orbs[mask] = qzero_orbs[mask] / orbs_per_shell[mask]

        # mask [n_batch, n_atom, n_orbs]
        tmp = torch.ones(self.basis.orbs_per_shell.sum())
        tmp2 = tmp.split(tuple(self.basis.orbs_per_atom.sum(-1)))
        _mask = pack([pack(icha.split(tuple(jj)))
                        for icha, jj in zip(tmp2, self.basis.orbs_per_atom)])
        _mask_orbital_resolved = _mask != 0

        return numbers_orbs, l_shell, orbs_per_shell, qzero_orbs, _mask_orbital_resolved

    def _qzero_shell(self):
        """Return l resolved charge."""
        qzero_shell = pack(self.qzero_orbs.flatten().split(
            tuple(self.basis.orbs_per_shell.flatten())))
        n_shell = (self.basis.orbs_per_shell != 0).sum(-1)
        qzero_shell = qzero_shell.sum(-1).split(tuple(n_shell))

        return pack(qzero_shell)

    def _gather_plus_u(self):
        u_orbs = torch.zeros(self.basis.orbital_matrix_shape[:-1])
        u_shell = torch.zeros(self.basis.shell_matrix_shape[:-1])
        _mask = self.geometry.atomic_numbers != 0
        _numbers_split = pack(self.numbers_orbs.flatten().split(
            tuple(self.basis.orbs_per_shell.flatten())))
        number_shell = pack(_numbers_split[..., 0].split(tuple((self.basis.shell_ns != -1).sum(-1))))

        for inum in self.geometry.unique_atomic_numbers():
            for il in self.shell_dict[inum.tolist()]:
                mask_num = self.numbers_orbs == inum
                mask_num_shell = number_shell == inum
                mask_orbs = self.l_shell == il
                mask_l = self.basis.shell_ls == il
                mask = mask_orbs * mask_num
                mask_shell = mask_num_shell * mask_l

                if (inum.tolist(), il) in self.plusu_params.keys():
                    u_orbs[mask] = self.plusu_params[(inum.tolist(), il)]
                    u_shell[mask_shell] = self.plusu_params[(inum.tolist(), il)]

        return u_orbs, u_shell

    def _gather_w(self):
        """Spin parameters W matrix for SDFTB."""
        w_mat = torch.zeros(self.basis.shell_matrix_shape)
        atomic_number_matrix = self.basis.atomic_number_matrix('shell')
        azimuthal_matrix = self.basis.azimuthal_matrix('shell')

        for inum in self.geometry.unique_atomic_numbers():
            mask_n = atomic_number_matrix == torch.tensor([inum, inum])
            for il in self.shell_dict[int(inum)]:
                for jl in self.shell_dict[int(inum)]:
                    mask_l = azimuthal_matrix == torch.tensor([il, jl])
                    _mask = mask_l * mask_n
                    mask = _mask[..., 0] * _mask[..., 1]
                    w_mat[mask] = self.spin_params[int(inum)][il, jl]

        mask_shell_block = (self.basis.on_atoms_shell.unsqueeze(-1) -
                            self.basis.on_atoms_shell.unsqueeze(-2)) == 0
        w_mat[~mask_shell_block] = 0

        return w_mat

    def __call__(self, hamiltonian, overlap, iiter, spin=None):
        # calculate the eigen-values & vectors via a Cholesky decomposition
        epsilon, eigvec = maths.eighb(hamiltonian, overlap)

        if not self.geometry.is_periodic:
            # calculate the occupation of electrons via the fermi method
            self.occ, nocc = fermi(epsilon, self.nelectron[self.mask], spin=spin)

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
            return mulliken(overlap, self.rho, self.basis.orbs_per_atom[self.mask],
                            orbital_resolved=self.orbital_resolved)
        else:
            iocc, inocc = fermi(epsilon, self.nelectron[self.mask], spin=spin)
            iden = torch.sqrt(iocc).unsqueeze(1).expand_as(eigvec) * eigvec
            irho = (torch.conj(iden) @ iden.transpose(1, 2))  # -> density
            iq_orbs, iq = mulliken(overlap, irho, self.atom_orbitals[self.mask])
            return iocc, inocc, iden, irho, iq_orbs, iq


    def __hs__(self, hamiltonian, overlap, **kwargs):
        """Hamiltonian or overlap feed."""
        multi_varible = kwargs.get('multi_varible', None)

        if self.geometry.is_periodic:
            hs_obj = self.periodic
        else:
            hs_obj = self.geometry
        if hamiltonian is None:
            self.ham = hs_matrix(
                hs_obj, self.basis, self.h_feed,
                multi_varible=multi_varible, cutoff=self.skparams.cutoff+1.0,
                spin_params=self.spin_params
            )
        else:
            self.ham = hamiltonian
        if overlap is None:
            self.over = hs_matrix(
                hs_obj, self.basis, self.s_feed, multi_varible=multi_varible,
                cutoff=self.skparams.cutoff+1.0)
        else:
            self.over = overlap

    def _next_geometry(self, geometry, **kwargs):
        """Update geometry for DFTB calculations."""
        if (self.geometry.atomic_numbers != geometry.atomic_numbers).any():
            raise ValueError('Atomic numbers in new geometry have changed.')

        self.geometry = geometry

        if self.geometry.is_periodic:
            self.periodic = Periodic(self.geometry, self.geometry.cell,
                                     cutoff=self.skparams.cutoff, **kwargs)
            self.coulomb = Coulomb(self.geometry, self.periodic, method='search')
            self.max_nk = torch.max(self.periodic.n_kpoints)
        # calculate short gamma
        if self.method in ('dftb2', 'dftb3'):
            mask = self.periodic.mask_central_cell if self.geometry.is_periodic else None
            gamma = ShortGamma(
                self.u, self.geometry.atomic_numbers,
                self.distances, self.periodic,
                orbital_resolved=self.orbital_resolved,
                u_orbs=self.u_orbs, orbs_numbers=self.orbs_numbers,
                distance_orbs=self.distance_orbs,
                mask_central_cell=mask)
            self.short_gamma = gamma.short_gamma
            self.short_gamma_pe = gamma.short_gamma_pe
            self.short_gamma_pe_on = gamma.short_gamma_pe_on

        self.atom_orbitals = self.basis.orbs_per_atom
        self.inv_dist = self._inv_distance(self.geometry.distances)

    def _get_shift(self, h: Tensor = None):
        """Return shift term for periodic and non-periodic."""
        if not self.geometry.is_periodic:
            if h is None:
                return self.inv_dist - self.short_gamma
            else:
                return self.inv_dist - self.short_gamma * h
        else:
            if h is None:
                return self.coulomb.invrmat - self.short_gamma
            else:
                return self.coulomb.invrmat - (self.short_gamma_pe * h).sum(1)

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

    def _get_force(self):
        return DftbGradient(
            self.geometry, self.basis, self.h_feed, self.s_feed,
            self.shell_dict, self.skparams, dftb_type=self.dftb_type)

    def _expand_u(self, u):
        """Expand Hubbert U for periodic system."""
        shape_cell = self.distances.shape[1]
        return u.repeat(shape_cell, 1, 1).transpose(0, 1)

    def _expand_u_orbs(self, u):
        """Expand Hubbert U for periodic system."""
        return pack([iu.repeat_interleave(iorb) for iorb, iu in
                     zip(self.atom_orbitals, u)])

    def _expand_uq(self, uq):
        """Expand  dU/dq for periodic system."""
        # TO BE revised
        shape_cell = self.distances.shape[1]
        return uq.repeat(shape_cell, 1, 1).transpose(0, 1)

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
    def charge_orbs(self):
        """Eq 11 in J. Phys. Chem. A 2007, 111, 5671-5677."""
        return self._charge_orbs

    @property
    def total_energy(self):
        return self.electronic_energy + self.repulsive_energy + self.plusu_energy

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
        if self.method == 'Dftb1':
            return self.H0_energy
        if self.method == 'Dftb2':
            return self.H0_energy + self.coulomb_energy
        elif self.method == 'Dftb3':
            return self.H0_energy + self.coulomb_energy + self.dftb3_energy
        else:
            raise ValueError('invalid method')

    @property
    def dftb3_energy(self):
        """Calculate Coulomb energy (atom resolved charge)."""
        d_q = self.charge - self.qzero
        e = d_q * (
                d_q.unsqueeze(-1) *
                d_q.unsqueeze(-2) * self.shift3_off).sum(-1)
        return e.sum(-1) / 3

    @property
    def repulsive_energy(self):
        """Return repulsive energy."""
        return self.cal_repulsive().repulsive_energy

    @property
    def plusu_energy(self):
        if self.plusu_params is None:
            return 0
        else:
            n = 0.5 * (self.over @ self.rho + self.rho @ self.over)
            if self.plusu_params['type'] == 'fll':
                return -0.25 * (self.plusu_orbs * n**2)[self.mask_onsite_block].sum(-1)\
                    + 0.5 * (torch.diag_embed(self.plusu_orbs)*n).sum(-1).sum(-1)
            elif self.plusu_params['type'] == 'psic':
                return -0.25 * (self.plusu_orbs * n**2)[self.mask_onsite_block].sum(-1)

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
        geometry = self.periodic if self.geometry.is_periodic else self.geometry
        return Repulsive(geometry, self.skparams, self.basis)

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

    def _kt(self) -> Union[Tensor, float]:
        return self.temperature * energy_units["k"] / energy_units['ev']

    @property
    def E_fermi(self):
        basis = self.basis  # if self.basis_virtual is None else self.basis_virtual
        return fermi_search(
            eigenvalues=self._epsilon.transpose(0, 1) / energy_units['ev'],
            n_electrons=self.nelectron,
            e_mask=basis,
            kT=self._kt(),
            k_weights=self.k_weights,
        )


class Dftb1(Dftb):
    """Density-functional tight-binding method with 0th correction."""

    def __init__(self,
                 geometry: Geometry,
                 shell_dict: dict = None,
                 path_to_skf: str = './',
                 repulsive=True,
                 skf_type: str = 'h5',
                 spin_params: Dict = None,
                 plusu_params: Dict = None,
                 **kwargs):
        self.method = 'Dftb1'
        self.scc_step = kwargs.get('scc_step', 1)
        super().__init__(
            geometry, shell_dict, path_to_skf, repulsive, skf_type,
            spin_params=spin_params, plusu_params=plusu_params, **kwargs)

        if self.spin_params is not None:
            ValueError('No spin-polarized calculation for DFTB0.')

        super().init_dftb(**kwargs)

    def __call__(self,
                 charge: Tensor = None,  # -> Initial charge
                 geometry: Geometry = None,  # Update Geometry
                 hamiltonian: Tensor = None,
                 overlap: Tensor = None,
                 mask: Tensor = None,
                 **kwargs):
        if geometry is not None:
            super()._next_geometry(geometry, **kwargs)

        self.mask = torch.tensor([True]).repeat(self._n_batch) if mask is None else mask
        super().__hs__(hamiltonian, overlap, **kwargs)

        # self.gamma = self.inv_dist - self.short_gamma

        if charge is not None:
            d_q = charge - self.qzero
            self.shift = self._get_shift()
            self._shift = torch.bmm(d_q.unsqueeze(1), self.shift)
            self.shift_orb = pack([
                ishif.repeat_interleave(iorb) for iorb, ishif in
                zip(self.atom_orbitals, self._shift)])
            self._shift_mat = torch.stack([torch.unsqueeze(ishift, 1) + ishift
                                           for ishift in self.shift_orb])
            if self.geometry.is_periodic:
                self._shift_mat = self._shift_mat.repeat(torch.max(
                    self.periodic.n_kpoints), 1, 1, 1).permute(1, 2, 3, 0)

            self.hamiltonian = self.ham + 0.5 * self.over * self._shift_mat
            ham = self.hamiltonian
            over = self.over
        else:
            ham = self.ham
            over = self.over

        if self.geometry.is_periodic:
            epsilon, eigvec, rho, density, q_new = [], [], [], [], []
            for ik in range(self.max_nk):

                # calculate the eigen-values & vectors
                iep, ieig = maths.eighb(ham[..., ik], over[..., ik])
                epsilon.append(iep), eigvec.append(ieig)

                iocc, inocc = fermi(iep, self.nelectron)
                # nocc.append(inocc)
                iden = torch.sqrt(iocc).unsqueeze(1).expand_as(ieig) * ieig
                irho = (torch.conj(iden) @ iden.transpose(1, 2))  # -> density
                density.append(irho)

                # calculate mulliken charges for each system in batch
                _, iq = mulliken(self.over[..., ik], irho, self.atom_orbitals)

                _q = iq.real
                q_new.append(_q)

            # nocc = pack(nocc).T
            self.rho = pack(density).permute(1, 2, 3, 0)
            self.qm = (pack(q_new).permute(2, 1, 0) * self.periodic.k_weights).sum(-1).T

            self._epsilon = pack(epsilon).permute(1, 0, 2)

        else:
            self.qm_orbs, self.qm = super().__call__(ham, over, iiter=0)

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
                 spin_params: Dict = None,
                 plusu_params: Dict = None,
                 **kwargs):
        self.method = 'Dftb2'
        self.scc_step = kwargs.get('scc_step', 60)
        super().__init__(geometry, shell_dict, path_to_skf,
                         repulsive, skf_type, basis_type, periodic, mixer,
                         spin_params=spin_params,
                         plusu_params=plusu_params, **kwargs)
        super().init_dftb(**kwargs)
        self.converge_number = []

    def __call__(self,
                 charge: Tensor = None,  # -> Initial charge
                 geometry: Geometry = None,  # Update Geometry
                 hamiltonian: Tensor = None,
                 overlap: Tensor = None,
                 charge_orbs: Tensor = None,
                 mask: bool = None,
                 **kwargs):
        """Perform SCC-DFTB calculation."""
        self._charge = self.qzero.clone() if charge is None else charge
        if self.plusu_params is None and self.spin_params is None:
            self.mixer = globals()[self.mixer_type](
                self.qzero, return_convergence=True)
        else:
            self.mixer = None

        if self.spin_params is None:
            if self.plusu_params is not None:
                self._charge_orbs = self.qzero_orbs.clone()
        else:
            self._charge_u = torch.zeros(self.qzero.shape)
            self._charge_d = torch.zeros(self.qzero.shape)
            self._charge_u_orbs = (self.qzero_orbs.clone() + self.q_potential_orbs) / 2.0
            self._charge_d_orbs = (self.qzero_orbs.clone() - self.q_potential_orbs) / 2.0

        self.shift = self._get_shift()

        if geometry is not None:
            super()._next_geometry(geometry)

        self.mask = torch.tensor([True]).repeat(self._n_batch) if mask is None else mask
        super().__hs__(hamiltonian, overlap, **kwargs)

        # Pre-SCC-DFTB for SDFTB
        if self.spin_params is not None:
            shift_pw = torch.einsum('...i,...ij->...j',
                                    self.q_potential[self.mask],
                                    self.w_params[self.mask])
            _repeat_shell = self.basis.shell_ls * 2 + 1
            shift_pw_orb = pack([
                ishif.repeat_interleave(iorb) for iorb, ishif in
                zip(_repeat_shell[self.mask], shift_pw)])
            shift_pw_mat = 0.5 * torch.stack(
                [torch.unsqueeze(ishift, 1) + ishift for ishift in shift_pw_orb])
            ham_spin = shift_pw_mat * self.over

            self.hamiltonian_u = self.ham + ham_spin
            self.hamiltonian_d = self.ham - ham_spin

            # With spin, without +U
            if self.plusu_params is None:
                self.qm_orbs_u, self.qm_u = super().__call__(self.hamiltonian_u, self.over, 0, spin=0)
                self.qm_orbs_d, self.qm_d = super().__call__(self.hamiltonian_d, self.over, 0, spin=1)
                self.qm = self.qm_u + self.qm_d
                self.qm_orbs = self.qm_orbs_u + self.qm_orbs_d

                self._charge_u[self.mask] = self.qm_u
                self._charge_d[self.mask] = self.qm_d
                self._charge[self.mask] = self.qm_u + self.qm_d
                self._charge_orbs[self.mask] = self.qm_orbs_u + self.qm_orbs_d

                # Avoid mix spin-unpolarised and spin-polarised charge
                self.mixer = globals()[self.mixer_type](
                    self.qm, return_convergence=True)
                self._charge[self.mask] = self.qm
                self._charge_orbs[self.mask] = self.qm_orbs

        # Pre-SCC-DFTB for DFTB+U
        if self.plusu_params is not None:
            if self.spin_params is None:
                _shift_plusu1 = 0.5 * self.plusu_orbs
                _shift_plusu2 = 0.5 * self._charge_orbs * self.plusu_orbs
                if self.plusu_params['type'] == 'fll':
                    _shift_mat_u = torch.stack([torch.unsqueeze(ishift, 1) + ishift
                                                for ishift in _shift_plusu1 - _shift_plusu2])
                elif self.plusu_params['type'] == 'psic':
                    _shift_mat_u = torch.stack([torch.unsqueeze(ishift, 1) + ishift
                                                for ishift in -_shift_plusu2])
                ham_u = 0.5 * self.over * _shift_mat_u
                self.hamiltonian = self.ham + ham_u
            else:  # With spin, with +U
                _shift_plusu_u = 0.5 * self._charge_u_orbs * self.plusu_orbs
                _shift_plusu_d = 0.5 * self._charge_d_orbs * self.plusu_orbs
                if self.plusu_params['type'] == 'fll':
                    _shift_plusu1 = 0.25 * self.plusu_orbs
                    _shift_mat_u_u = torch.stack([
                        torch.unsqueeze(ishift, 1) + ishift
                        for ishift in _shift_plusu1 - _shift_plusu_u])
                    _shift_mat_u_d = torch.stack([
                        torch.unsqueeze(ishift, 1) + ishift
                        for ishift in _shift_plusu1 - _shift_plusu_d])
                elif self.plusu_params['type'] == 'psic':
                    _shift_mat_u_u = torch.stack(
                        [torch.unsqueeze(ishift, 1) + ishift
                         for ishift in -_shift_plusu_u])
                    _shift_mat_u_d = torch.stack(
                        [torch.unsqueeze(ishift, 1) + ishift
                         for ishift in -_shift_plusu_d])
                ham_u_u = self.over * _shift_mat_u_u
                ham_u_d = self.over * _shift_mat_u_d

            if self.spin_params is None:
                if not self.geometry.is_periodic:
                    self.qm_orbs, self.qm = super().__call__(self.hamiltonian, self.over, 0)
                else:
                    self.ie, eigvec, nocc, density, q_new, q_new_orbs = [], [], [], [], [], []
                    for ik in range(self.max_nk):
                        iocc, inocc, iden, irho, iq_orbs, iq = super().__call__(self.hamiltonian[..., ik], self.over[..., ik], 0)
                        q_new.append(iq.real)
                        q_new_orbs.append(iq_orbs.real)
                        density.append(irho)
                        nocc.append(inocc)

                    self.rho = pack(density).permute(1, 2, 3, 0)
                    self.qm = (pack(q_new).permute(2, 1, 0) * self.periodic.k_weights[self.mask]).sum(-1).T
                    self.qm_orbs = (pack(q_new_orbs).permute(2, 1, 0) * self.periodic.k_weights[self.mask]).sum(-1).T
                    self._density = self.rho.clone()
                    self._occ = pack(nocc).T

                self._charge[self.mask] = self.qm
                self._charge_orbs[self.mask] = self.qm_orbs
            else:
                self.hamiltonian_u = self.hamiltonian_u + ham_u_u
                self.hamiltonian_d = self.hamiltonian_d + ham_u_d

                self.qm_orbs_u, self.qm_u = super().__call__(self.hamiltonian_u, self.over, 0, spin=0)
                self.rho_u = self.rho
                self.qm_orbs_d, self.qm_d = super().__call__(self.hamiltonian_d, self.over, 0, spin=1)
                self.rho_d = self.rho
                self.qm = self.qm_u + self.qm_d
                self.qm_orbs = self.qm_orbs_u + self.qm_orbs_d

                self._charge_u[self.mask] = self.qm_u
                self._charge_u_orbs[self.mask] = self.qm_orbs_u
                self._charge_d[self.mask] = self.qm_d
                self._charge_d_orbs[self.mask] = self.qm_orbs_d
                self._charge[self.mask] = self.qm
                self._charge_orbs[self.mask] = self.qm_orbs_u + self.qm_orbs_d

            if not self.geometry.is_periodic:
                self._density[self.mask, :self.rho.shape[1], :self.rho.shape[2]] = self.rho
                self._occ[self.mask, :self.occ.shape[-1]] = self.occ

        # Loop for DFTB2
        for iiter in range(self.scc_step):
            if self.mask.any():
                self._single_loop(iiter)
            else:
                break

    def _single_loop(self, iiter):
        """Perform each single SCC-DFTB loop."""
        shift_ = self._update_shift()

        shiftorb_ = pack([ishif.repeat_interleave(iorb) for iorb, ishif in
                          zip(self.atom_orbitals[self.mask], shift_)])
        _shift_mat = torch.stack([torch.unsqueeze(ishift, 1) + ishift
                                 for ishift in shiftorb_])

        if self.geometry.is_periodic:
            _shift_mat = _shift_mat.repeat(torch.max(
                self.periodic.n_kpoints), 1, 1, 1).permute(1, 2, 3, 0)

        # For molecule, the shift_mat will be [n_batch, n_size, n_size]
        # For solid, the size is [n_batch, n_size, n_size, n_kpt]
        this_size = _shift_mat.shape[-2]
        fock = self.ham[self.mask, :this_size, :this_size] + \
            0.5 * self.over[self.mask, :this_size, :this_size] * _shift_mat

        if not self.geometry.is_periodic:
            overlap = self.over[self.mask, :this_size, :this_size]

            # STD SCC-DFTB Hamiltonian
            self.hamiltonian = self.ham[self.mask, :this_size, :this_size] + \
                0.5 * overlap * _shift_mat

            # ********************************************************* #
            # ********* Build Hamiltonian for various methods ********* #
            # ********************************************************* #
            if self.spin_params is not None:
                q_potential_u = pack(self.qm_orbs_u.flatten().split(
                    tuple(self.basis.orbs_per_shell.flatten()))).sum(-1)
                q_potential_d = pack(self.qm_orbs_d.flatten().split(
                    tuple(self.basis.orbs_per_shell.flatten()))).sum(-1)
                q_potential = q_potential_u - q_potential_d
                self.q_potential = pack(q_potential.split(
                    tuple((self.basis.orbs_per_shell != 0).sum(-1))))

                shift_pw = torch.einsum('...i,...ij->...j',
                                        self.q_potential[self.mask],
                                        self.w_params[self.mask])
                _repeat_shell = self.basis.shell_ls * 2 + 1
                shift_pw_orb = pack([
                    ishif.repeat_interleave(iorb) for iorb, ishif in
                    zip(_repeat_shell[self.mask], shift_pw)])
                shift_pw_mat = 0.5 * torch.stack(
                    [torch.unsqueeze(ishift, 1) + ishift for ishift in shift_pw_orb])
                ham_spin = shift_pw_mat * self.over

                self.hamiltonian_u = self.hamiltonian + ham_spin
                self.hamiltonian_d = self.hamiltonian - ham_spin

            if self.plusu_params is not None:
                if self.plusu_params['type'] == 'fll':
                    _shift_plusu1 = 0.5 * self.plusu_orbs

                # With +U, without spin
                if self.spin_params is None:
                    n = 0.5 * (self.over @ self.rho + self.rho @ self.over)
                    shift_mat_u2 = torch.zeros(self.ham.shape)
                    shift_mat_u2[self.mask_onsite_block] = 0.5 * (
                            n * self.plusu_orbs)[self.mask_onsite_block]

                    if self.plusu_params['type'] == 'fll':
                        _shift_mat_u1 = torch.diag_embed(_shift_plusu1)
                        shift_u = _shift_mat_u1 - shift_mat_u2
                    elif self.plusu_params['type'] == 'psic':
                        shift_u = -shift_mat_u2

                    ham_u = 0.5 * (shift_u @ self.over + self.over @ shift_u)
                    self.hamiltonian = self.hamiltonian + ham_u

                else:  # With +U, with spin
                    n_u = self.over @ self.rho_u + self.rho_u @ self.over
                    shift_mat_u2_u = torch.zeros(self.ham.shape)
                    shift_mat_u2_u[self.mask_onsite_block] = 0.5 * (
                            n_u * self.plusu_orbs)[self.mask_onsite_block]
                    n_d = self.over @ self.rho_d + self.rho_d @ self.over
                    shift_mat_u2_d = torch.zeros(self.ham.shape)
                    shift_mat_u2_d[self.mask_onsite_block] = 0.5 * (
                            n_d * self.plusu_orbs)[self.mask_onsite_block]

                    if self.plusu_params['type'] == 'fll':
                        _shift_mat_u1 = torch.diag_embed(_shift_plusu1)
                        shift_u_u = _shift_mat_u1 - shift_mat_u2_u
                        shift_u_d = _shift_mat_u1 - shift_mat_u2_d
                    elif self.plusu_params['type'] == 'psic':
                        shift_u_u = -shift_mat_u2_u
                        shift_u_d = -shift_mat_u2_d

                    ham_u_u = 0.5 * (shift_u_u @ self.over + self.over @ shift_u_u)
                    ham_u_d = 0.5 * (shift_u_d @ self.over + self.over @ shift_u_d)
                    self.hamiltonian_u = self.hamiltonian_u + ham_u_u
                    self.hamiltonian_d = self.hamiltonian_d + ham_u_d

            # ********************************************************* #
            # ************ single loop SCC-DFTB calculation *********** #
            # ********************************************************* #
            if self.spin_params is None:
                self.qm_orbs, self.qm = super().__call__(self.hamiltonian, overlap, iiter)
            else:
                self.qm_orbs_u, self.qm_u = super().__call__(self.hamiltonian_u, overlap, iiter, spin=0)
                self.rho_u = self.rho
                self._charge_u_orbs[self.mask] = self.qm_orbs_u
                self._charge_u[self.mask] = self.qm_u
                self.qm_orbs_d, self.qm_d = super().__call__(self.hamiltonian_d, overlap, iiter, spin=1)
                self.rho_d = self.rho
                self._charge_d_orbs[self.mask] = self.qm_orbs_d
                self._charge_d[self.mask] = self.qm_d
                self.qm = self.qm_u + self.qm_d

            if self.mixer is None and iiter == 0:
                self.mixer = globals()[self.mixer_type](
                    self.qm, return_convergence=True)
                self._charge[self.mask] = self.qm
                self._charge_orbs[self.mask] = self.qm_orbs
                _mask = ~self.mask
            else:
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

                # Beyond standard DFTB
                if self.plusu_params is not None:
                    if self.plusu_params['type'] == 'fll':
                        _shift_plusu1 = 0.5 * self.plusu_orbs

                        # With +U, without spin
                        if self.spin_params is None:
                            n = 0.5 * (self.over[..., 0] @ self.rho[..., 0] +
                                       self.rho[..., 0] @ self.over[..., 0]).real
                            shift_mat_u2 = torch.zeros(self.ham.shape[:-1])
                            shift_mat_u2[self.mask_onsite_block] = 0.5 * (
                                n * self.plusu_orbs)[self.mask_onsite_block]
                            if self.plusu_params['type'] == 'fll':
                                _shift_mat_u1 = torch.diag_embed(_shift_plusu1)
                                shift_u = _shift_mat_u1 - shift_mat_u2
                            elif self.plusu_params['type'] == 'psic':
                                shift_u = -shift_mat_u2
                            ham_u = 0.5 * (shift_u @ self.over.real + self.over.real[..., 0] @ shift_u)
                            self.hamiltonian = self.hamiltonian + ham_u

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
                _, iq = mulliken(self.over[self.mask, :this_size, :this_size, ik],
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
            epsilon = pack(self.ie)
            if self.mixer is None and iiter == 0:
                self.mixer = globals()[self.mixer_type](
                    self.qm, return_convergence=True)
                self._charge[self.mask] = self.qm
                self._charge_orbs[self.mask] = self.qm_orbs
                self._epsilon = epsilon
                self._nocc = nocc
                _mask = ~self.mask
            else:
                self.qmix, _mask = self.mixer(q_new)
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
        if self.geometry.is_periodic:
            return pdos(self.eigenvector.permute(-1, 0, 1, 2),
                        self.over.permute(-1, 0, 1, 2), self._epsilon, energy,
                        is_periodic=self.geometry.is_periodic)
        else:
            return pdos(self.eigenvector, self.over, self._epsilon, energy)


    @property
    def dos(self):
        """Return energy distribution and DOS with fermi energy correction."""
        energy = self.dos_energy
        # energy = energy.repeat(self.system.size_batch, 1)  # -> to batch

        # make sure the 1st dimension is batch
        if self.unit in ('eV', 'EV', 'ev'):
            return dos((self._epsilon - self.E_fermi.unsqueeze(1)),
                       energy, self.geometry.is_periodic)
        elif self.unit in ('hartree', 'Hartree'):
            return dos(self._epsilon - self.E_fermi.unsqueeze(1) * _Hartree__eV,
                       energy, self.geometry.is_periodic)

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
