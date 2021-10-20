"""Deal with periodic conditions."""
import torch
import numpy as np
from typing import Union, List, Optional
from tbmalt.common.batch import pack
Tensor = torch.Tensor
# _bohr = 0.529177249


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
        >>> from periodic import Periodic
        >>> import torch
    """

    def __init__(self, geometry: object, latvec: Tensor,
                 cutoff: Union[Tensor, float], **kwargs):
        self.geometry = geometry
        self.isperiodic = geometry.isperiodic
        self._n_batch = geometry._n_batch if geometry._n_batch is not None else 1
        self.n_atoms = geometry.n_atoms
        # self.cell = geometry.cell
        # self.periodic_list = geometry.periodic_list

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

        if return_distance is True:
            self.positions_vec, self.periodic_distances = self._periodic_distance()
            self.neighbour_vec, self.neighbour_dis = self._neighbourlist()

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

        # transfer from fraction to Bohr unit positions
        # if dim == 2:
        #     position_pe = torch.matmul(position_pe, self.latvec)
        #     self.geometry.positions = position_pe
        # else:
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
        positions = self.rcellvec.unsqueeze(2) + self.positions.unsqueeze(1)
        size_system = self.atomic_numbers.ne(0).sum(-1)
        positions_vec = (-positions.unsqueeze(-3) + self.positions.unsqueeze(1).unsqueeze(-2))
        distance = pack([torch.sqrt(((ipos[:, :inat].repeat(1, inat, 1) - torch.repeat_interleave(
                        icp[:inat], inat, 0)) ** 2).sum(-1)).reshape(-1, inat, inat)
                            for ipos, icp, inat in zip(
                                positions, self.positions, size_system)], value=1e3)

        return positions_vec, distance

    def _neighbourlist(self):
        """Get distance matrix of neighbour list according to periodic boundary condition."""
        _mask = self.neighbour.any(-1).any(-1)
        neighbour_vec = pack([self.positions_vec[ibatch][_mask[ibatch]]
                              for ibatch in range(self.cutoff.size(0))], value=1e3)
        neighbour_dis = pack([self.periodic_distances[ibatch][_mask[ibatch]]
                              for ibatch in range(self.cutoff.size(0))], value=1e3)
        return neighbour_vec, neighbour_dis

    def _inverse_lattice(self):
        """Get inverse lattice vectors."""
        # build a mask for zero vectors in 1D/ 2D lattice vectors
        mask_zero = self.latvec.eq(0).all(-1)
        _latvec = self.latvec + torch.diag_embed(mask_zero.type(self.latvec.dtype))

        # inverse lattice vectors
        _invlat = torch.transpose(torch.solve(torch.eye(
            _latvec.shape[-1]), _latvec)[0], -1, -2)
        _invlat[mask_zero] = 0
        return _invlat, mask_zero

    def _reciprocal_lattice(self):
        """Get reciprocal lattice vectors"""
        return 2 * np.pi * self.invlatvec

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
        _mask = self.neighbour.any(-1).any(-1)
        _cellvec = self.cellvec.permute(0, -1, -2)
        _neighbour_vec = pack([_cellvec[ibatch][_mask[ibatch]]
                              for ibatch in range(self.cutoff.size(0))])
        return _neighbour_vec.permute(0, -1, -2)

    def unique_atomic_numbers(self) -> Tensor:
        """Identifies and returns a tensor of unique atomic numbers.

        This method offers a means to identify the types of elements present
        in the system(s) represented by a `Geometry` object.

        Returns:
            unique_atomic_numbers: A tensor specifying the unique atomic
                numbers present.
        """
        return self.geometry.unique_atomic_numbers()
