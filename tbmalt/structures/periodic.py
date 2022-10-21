"""Deal with periodic conditions."""
import torch
import numpy as np
from typing import Union, List, Optional

from tbmalt import Geometry
from tbmalt.common.batch import pack
Tensor = torch.Tensor


class Periodic:
    """Calculate the translation vectors for cells for 3D periodic boundary condition.

    Arguments:
        latvec: Lattice vector describing the geometry of periodic geometry,
            with Bohr as unit.
        cutoff: Interaction cutoff distance for reading SK table.

    Keyword Args:
        distance_extention: Extention of cutoff in SK tables to smooth the tail.
        positive_extention: Extension for the positive lattice vectors.
        negative_extention: Extension for the negative lattice vectors.

    Return:
        cutoff: Global cutoff for the diatomic interactions, unit is Bohr.
        cellvol: Volume of the unit cell.
        reccellvol: Volume of the reciprocal lattice unit cell.
        cellvec: Cell translation vectors in relative coordinates.
        rcellvec: Cell translation vectors in absolute units.
        ncell: Number of lattice cells.

    Examples:
        >>> from tbmalt import Periodic
        >>> import torch
    """

    def __init__(self, geometry: object, latvec: Tensor,
                 cutoff: Union[Tensor, float], **kwargs):
        self.geometry = geometry
        self.is_periodic = geometry.is_periodic
        self._n_batch = geometry._n_batch if geometry._n_batch is not None else 1
        self.n_atoms = geometry.n_atoms

        # mask for periodic and non-periodic systems
        self.mask_pe = self.geometry.periodic_list
        self.latvec, self.cutoff = self._check(latvec, cutoff, **kwargs)

        if self.geometry.frac_list.any():
            self._positions_check()
        dim = self.geometry.atomic_numbers.dim()
        self.atomic_numbers = self.geometry.atomic_numbers.unsqueeze(0) if \
            dim == 1 else self.geometry.atomic_numbers
        self.positions = self.geometry.positions.unsqueeze(0) if \
            dim == 1 else self.geometry.positions

        dist_ext = kwargs.get('distance_extention', 1.0)
        return_distance = kwargs.get('return_distance', True)

        # Global cutoff for the diatomic interactions
        self.cutoff = self.cutoff + dist_ext

        self.invlatvec, self.mask_zero = self._inverse_lattice()

        self.recvec = self._reciprocal_lattice()

        # Unit cell volume
        self.cellvol = abs(torch.det(self.latvec))

        self.cellvec, self.rcellvec, self.ncell = self.get_cell_translations(**kwargs)

        # K-sampling
        self.kpoints, self.n_kpoints, self.k_weights = self._kpoints(**kwargs)

        if return_distance is True:
            self.mask_central_cell, self.positions_pe, self.positions_vec,\
                self.periodic_distances = self._periodic_distance()
            self.neighbour_pos, self.neighbour_vec, self.neighbour_dis = self._neighbourlist()

    def _check(self, latvec, cutoff, **kwargs):
        """Check dimension, type of lattice vector and cutoff."""
        # Default lattice vector is from geometry, therefore default unit is bohr
        unit = kwargs.get('unit', 'bohr')

        # Molecule will be padding with zeros, here select latvec for solid
        if isinstance(latvec, list):
            latvec = pack(latvec)
        elif not isinstance(latvec, Tensor):
            raise TypeError('Lattice vector is tensor or list of tensor.')

        if latvec.dim() == 2:
            latvec = latvec.unsqueeze(0)
        elif latvec.dim() != 3:
            raise ValueError('lattice vector dimension should be 2 or 3')

        if isinstance(cutoff, float):
            cutoff = torch.tensor([cutoff])
        elif not isinstance(cutoff, Tensor):
            raise TypeError(
                f'cutoff should be float or Tensor, but get {type(cutoff)}')
        if cutoff.dim() == 0:
            cutoff = cutoff.unsqueeze(0)
        elif cutoff.dim() >= 2:
            raise ValueError(
                'cutoff should be 0, 1 dimension tensor or float')

        if latvec.size(0) != 1 and cutoff.size(0) == 1:
            cutoff = cutoff.repeat_interleave(latvec.size(0))

        return latvec, cutoff

    def _positions_check(self):
        """Check positions type (fraction or not) and unit."""
        is_frac = self.geometry.frac_list
        dim = self.positions.dim()

        # transfer periodic positions to bohr
        position_pe = self.positions[self.mask_pe]
        _mask = is_frac[self.mask_pe]

        # whether fraction coordinates in the range [0, 1)
        if torch.any(position_pe[_mask] >= 1) or torch.any(position_pe[_mask] < 0):
            position_pe[_mask] = torch.abs(position_pe[_mask]) - \
                            torch.floor(torch.abs(position_pe[_mask]))

        position_pe[_mask] = torch.matmul(
            position_pe[_mask], self.latvec[is_frac])
        self.positions[self.mask_pe] = position_pe


    def get_cell_translations(self, **kwargs):
        """Get cell translation vectors."""
        pos_ext = kwargs.get('positive_extention', 1)
        neg_ext = kwargs.get('negative_extention', 1)

        _tmp = torch.floor(self.cutoff * torch.norm(self.invlatvec, dim=-1).T).T
        ranges = torch.stack([-(neg_ext + _tmp), pos_ext + _tmp])

        # 1D/ 2D cell translation
        ranges[torch.stack([self.mask_zero, self.mask_zero])] = 0

        # Length of the first, second and third column in ranges
        leng = ranges[1, :].long() - ranges[0, :].long() + 1

        # Number of cells
        ncell = leng[..., 0] * leng[..., 1] * leng[..., 2]

        # Cell translation vectors in relative coordinates
        # Large values are padded at the end of short cell vectors to exceed cutoff distance
        cellvec = pack([torch.stack([
            torch.linspace(iran[0, 0], iran[1, 0],
                           ile[0]).repeat_interleave(ile[2] * ile[1]),
            torch.linspace(iran[0, 1], iran[1, 1],
                           ile[1]).repeat(ile[0]).repeat_interleave(ile[2]),
            torch.linspace(iran[0, 2], iran[1, 2],
                           ile[2]).repeat(ile[0] * ile[1])])
                        for ile, iran in zip(leng, ranges.transpose(1, 0))], value=1e3)
        rcellvec = pack([torch.matmul(ilv.transpose(0, 1), icv.T.unsqueeze(-1)).squeeze(-1)
                         for ilv, icv in zip(self.latvec, cellvec)], value=1e3)

        return cellvec, rcellvec, ncell

    def _periodic_distance(self):
        """Get distances between central cell and neighbour cells."""
        mask_central_cell = (self.rcellvec != 0).sum(-1) == 0
        positions = self.rcellvec.unsqueeze(2) + self.positions.unsqueeze(1)
        size_system = self.atomic_numbers.ne(0).sum(-1)
        positions_vec = (-positions.unsqueeze(-3) + self.positions.unsqueeze(1).unsqueeze(-2))
        distance = pack([torch.sqrt(((ipos[:, :inat].repeat(1, inat, 1) - torch.repeat_interleave(
                        icp[:inat], inat, 0)) ** 2).sum(-1)).reshape(-1, inat, inat)
                            for ipos, icp, inat in zip(
                                positions, self.positions, size_system)], value=1e3)

        return mask_central_cell, positions, positions_vec, distance

    @property
    def n_atoms_pe(self):
        """Periodic number of atoms, include all images."""
        mask = self.neighbour.any(-1).any(-1)
        return pack(torch.split(torch.repeat_interleave(
            self.geometry.n_atoms, mask.sum(-1)), tuple(mask.sum(-1))))

    def supercell(self, idx: Tensor):

        # convert single to batch
        idx = idx if idx.dim() == 2 else idx.unsqueeze(0)

        ranges = pack([torch.zeros(*idx.shape), idx - 1])

        cellvec = pack([torch.stack([
            torch.linspace(iran[0, 0], iran[1, 0],
                           ile[0]).repeat_interleave(ile[2] * ile[1]),
            torch.linspace(iran[0, 1], iran[1, 1],
                           ile[1]).repeat(ile[0]).repeat_interleave(ile[2]),
            torch.linspace(iran[0, 2], iran[1, 2],
                           ile[2]).repeat(ile[0] * ile[1])])
                        for ile, iran in zip(idx, ranges.transpose(1, 0))], value=1e3)
        rcellvec = pack([torch.matmul(ilv.transpose(0, 1), icv.T.unsqueeze(-1)).squeeze(-1)
                         for ilv, icv in zip(self.latvec, cellvec)], value=1e3)

        positions = rcellvec.unsqueeze(2) + self.positions.unsqueeze(1)
        atomic_numbers = self.atomic_numbers.repeat(1, positions.shape[-3])
        positions = positions.flatten(-3, -2)
        cell = self.geometry.cell * idx

        return Geometry(atomic_numbers, positions, cell=cell)

    @property
    def atomic_numbers_pe(self):
        """Periodic number of atoms, include all images."""
        return pack([number.repeat(nc, 1) for number, nc in zip(self.atomic_numbers, self.ncell)])

    def _neighbourlist(self):
        """Get distance matrix of neighbour list according to periodic boundary condition."""
        _mask = self.neighbour.any(-1).any(-1)
        neighbour_pos = pack([self.positions_pe[ibatch][_mask[ibatch]]
                              for ibatch in range(self.cutoff.size(0))], value=1e3)
        neighbour_vec = pack([self.positions_vec[ibatch][_mask[ibatch]]
                              for ibatch in range(self.cutoff.size(0))], value=1e3)
        neighbour_dis = pack([self.periodic_distances[ibatch][_mask[ibatch]]
                              for ibatch in range(self.cutoff.size(0))], value=1e3)

        return neighbour_pos, neighbour_vec, neighbour_dis

    def _inverse_lattice(self):
        """Get inverse lattice vectors."""
        # build a mask for zero vectors in 1D/ 2D lattice vectors
        mask_zero = self.latvec.eq(0).all(-1)
        _latvec = self.latvec + torch.diag_embed(mask_zero.type(self.latvec.dtype))

        # inverse lattice vectors
        _invlat = torch.transpose(torch.linalg.solve(
            _latvec, torch.eye(_latvec.shape[-1]).repeat(_latvec.shape[0], 1, 1)), -1, -2)
        _invlat[mask_zero] = 0

        return _invlat, mask_zero

    def _reciprocal_lattice(self):
        """Get reciprocal lattice vectors"""
        return 2 * np.pi * self.invlatvec

    def _kpoints(self, **kwargs):
        """Calculate K-points."""
        _kpoints = kwargs.get('kpoints', None)
        _klines = kwargs.get('klines', None)

        if _kpoints is not None:
            assert _klines is None, 'One of kpoints and klines should be None'
            assert isinstance(_kpoints, Tensor), 'kpoints should be' + \
                f'torch.Tensor, but get {type(_kpoints)}'
            _kpoints = _kpoints if _kpoints.dim() == 2 else _kpoints.unsqueeze(0)
            # all atomic_numbers transfer to batch
            assert len(_kpoints) == len(self.atomic_numbers), \
                f'length of kpoints do not equal to {len(self.atomic_numbers)}'
            assert _kpoints.shape[1] == 3, 'column of _kpoints si not 3'

            return self._super_sampling(_kpoints)
        elif _klines is not None:
            assert isinstance(_klines, Tensor), 'klines should be' + \
                f'torch.Tensor, but get {type(_klines)}'
            _klines = _klines if _klines.dim() == 3 else _klines.unsqueeze(0)

            return self._klines(_klines)

        else:
            _kpoints = torch.ones(self._n_batch, 3, dtype=torch.int32)

            return self._super_sampling(_kpoints)

    def _super_sampling(self, _kpoints):
        """Super sampling."""
        _n_kpoints = _kpoints[..., 0] * _kpoints[..., 1] * _kpoints[..., 2]
        _kpoints_inv = 0.5 / _kpoints
        _kpoints_inv2 = 1.0 / _kpoints
        _nkxyz = _kpoints[..., 0] * _kpoints[..., 1] * _kpoints[..., 2]
        n_ind = tuple(_nkxyz)
        _nkx, _nkyz = _kpoints[..., 0], _kpoints[..., 1] * _kpoints[..., 2]
        _nky, _nkxz = _kpoints[..., 1], _kpoints[..., 0] * _kpoints[..., 2]
        _nkz, _nkxy = _kpoints[..., 2], _kpoints[..., 0] * _kpoints[..., 1]

        # create baseline of kpoints, if n_kpoint in x direction is N,
        # the value will be [0.5 / N] * n_kpoint_x * n_kpoint_y * n_kpoint_z
        _x_base = torch.repeat_interleave(_kpoints_inv[..., 0], _nkxyz)
        _y_base = torch.repeat_interleave(_kpoints_inv[..., 1], _nkxyz)
        _z_base = torch.repeat_interleave(_kpoints_inv[..., 2], _nkxyz)

        # create K-mesh increase in each direction range from 0~1
        _x_incr = torch.cat([torch.repeat_interleave(torch.arange(ii) * iv, yz)
                    for ii, yz, iv in zip(_nkx, _nkyz, _kpoints_inv2[..., 0])])
        _y_incr = torch.cat([torch.repeat_interleave(
            torch.arange(iy) * iv, iz).repeat(ix) for ix, iy, iz, xz, iv in zip(
                _nkx, _nky, _nkz, _nkxz, _kpoints_inv2[..., 1])])
        _z_incr = torch.cat([(torch.arange(iz) * iv).repeat(xy)
                    for iz, xy, iv in zip(_nkz, _nkxy, _kpoints_inv2[..., 2])])

        all_kpoints = torch.stack([
            pack(torch.split((_x_base + _x_incr).unsqueeze(1), n_ind)),
            pack(torch.split((_y_base + _y_incr).unsqueeze(1), n_ind)),
            pack(torch.split((_z_base + _z_incr).unsqueeze(1), n_ind))])

        k_weights = pack(torch.split(torch.ones(_n_kpoints.sum()), tuple(_n_kpoints)))
        k_weights = k_weights / _n_kpoints.unsqueeze(-1)

        return all_kpoints.squeeze(-1).permute(1, 2, 0), _n_kpoints, k_weights

    def _klines(self, klines: Tensor):
        """K-lines for band structure calculations."""
        if self._n_batch == 1:
            if klines.dim() == 2:
                klines = klines.unsqueeze(0)
            elif klines.dim() > 3 or klines.dim() == 1:
                raise ValueError(f"klines dims should be 2 or 3, get {klines.dim()}")
        else:
            assert self._n_batch == len(klines), f"klines size {len(klines)} is not consistent with batch size {self._n_batch}"
            assert klines.dim() == 3, f"klines dims should be 3, get {klines.dim()}"
        assert klines.shape[-1] == 7, (
            "Shape error, for each K-line path, "
            + "last dimension shold include:[kx1, ky1, kz1, kx2, ky2, kz2, N]"
            + f"but get {klines.shape[-1]} numbers in the last dimension"
        )

        # Faltten all K-Lines so that we can deal with K-Lines only once
        _n_kpoints = klines[..., -1].sum(-1).long()
        _klines = klines.flatten(0, 1)
        _mask = _klines[..., -1].ge(1)

        # Extend and get interval K-Points
        delta_k = _klines[..., 3:6] - _klines[..., :3]

        delta_k[_mask] = delta_k[_mask] / (_klines[..., -1][_mask].unsqueeze(-1) - 1)
        delta_k = torch.repeat_interleave(delta_k, _klines[..., -1].long(), 0)

        repeat_nums = torch.cat(
            [torch.arange(ik) for ik in _klines[..., -1].long()]).unsqueeze(-1)
        klines_ext = (
            torch.repeat_interleave(_klines[..., :3], _klines[..., -1].long(), 0)
            + delta_k * repeat_nums
        )
        klines_ext = pack(klines_ext.split(tuple(_n_kpoints.tolist())))

        # Create averaged weight for each K-Lines
        k_weights = pack(
            torch.split(
                torch.repeat_interleave(1.0 / _n_kpoints, _n_kpoints), tuple(_n_kpoints)
            )
        )

        return klines_ext, _n_kpoints, k_weights

    def get_reciprocal_volume(self):
        """Get reciprocal lattice unit cell volume."""
        return abs(torch.det(2 * np.pi * (self.invlatvec.transpose(0, 1))))

    @property
    def neighbour(self):
        """Get neighbour list according to periodic boundary condition."""
        return torch.stack([self.periodic_distances[ibatch].le(self.cutoff[ibatch])
                           for ibatch in range(self.cutoff.size(0))])

    @property
    def distances(self) -> Tensor:
        """Distance matrix between atoms in the system."""
        return self.neighbour_dis.permute(0, 2, 3, 1)

    @property
    def distance_vectors(self) -> Tensor:
        """Distance vector matrix between atoms in the system."""
        return self.neighbour_vec

    @property
    def cellvec_neighbour(self):
        """Return cell vector which distances between all atoms in return cell
        and center cell are smaller than cutoff."""
        _mask = self.neighbour.any(-1).any(-1)
        _cellvec = self.cellvec.permute(0, -1, -2)
        _neighbour_vec = pack([_cellvec[ibatch][_mask[ibatch]]
                              for ibatch in range(self.cutoff.size(0))])

        return _neighbour_vec.permute(0, -1, -2)

    @property
    def phase(self):
        """Select kpoint for each interactions."""
        kpoint = 2.0 * np.pi * self.kpoints

        # shape: [n_batch, n_cell, 3]
        cell_vec = self.cellvec_neighbour

        return pack([torch.exp((0. + 1.0j) * torch.einsum(
            'ij, ijk-> ik', ik, cell_vec)) for ik in kpoint.permute(1, 0, -1)])

    def unique_atomic_numbers(self) -> Tensor:
        """Identifies and returns a tensor of unique atomic numbers.

        This method offers a means to identify the types of elements present
        in the system(s) represented by a `Geometry` object.

        Returns:
            unique_atomic_numbers: A tensor specifying the unique atomic
                numbers present.
        """
        return self.geometry.unique_atomic_numbers()
