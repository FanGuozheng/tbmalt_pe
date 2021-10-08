"""Deal with periodic conditions."""
import torch
import numpy as np
from typing import Union, List
from tbmalt.common.structures.system import System
from tbmalt.common.batch import pack
Tensor = torch.Tensor
_bohr = 0.529177249


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

        # mask for periodic and non-periodic systems
        self.mask_pe = self.geometry.is_periodic
        self.latvec, self.cutoff = self._check(latvec, cutoff, **kwargs)
        self._positions_check(**kwargs)

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
            self.positions_vec, self.periodic_distances = self._get_periodic_distance()
            self.neighbour_vec, self.neighbour_dis = self._neighbourlist()

    def _check(self, latvec, cutoff, **kwargs):
        """Check dimension, type of lattice vector and cutoff."""
        # Default lattice vector is from geometry, therefore default unit is bohr
        unit = kwargs.get('unit', 'bohr')

        # Molecule will be padding with zeros, here select latvec for solid
        if type(latvec) is list:
            latvec = pack(latvec)
        elif type(latvec) is not Tensor:
            raise TypeError('Lattice vector is tensor or list of tensor.')

        if latvec.dim() == 2:
            latvec = latvec.unsqueeze(0)
        elif latvec.dim() == 3:
            latvec = latvec
        else:
            raise ValueError('lattice vector dimension should be 2 or 3')

        if type(cutoff) is float:
            cutoff = torch.tensor([cutoff])
            if cutoff.dim() == 0:
                cutoff = cutoff.unsqueeze(0)
            elif cutoff.dim() >= 2:
                raise ValueError(
                    'cutoff should be 0, 1 dimension tensor or float')
        elif type(cutoff) is Tensor:
            if cutoff.dim() == 0:
                cutoff = cutoff.unsqueeze(0)
            elif cutoff.dim() >= 2:
                raise ValueError(
                    'cutoff should be 0, 1 dimension tensor or float')
        elif type(cutoff) is not float:
            raise TypeError('cutoff should be tensor or float')

        if latvec.size(0) != 1 and cutoff.size(0) == 1:
            cutoff = cutoff.repeat_interleave(latvec.size(0))

        if unit in ('angstrom', 'Angstrom'):
            latvec = latvec / _bohr
            cutoff = cutoff / _bohr
        elif unit not in ('bohr', 'Bohr'):
            raise ValueError('unit is either angstrom or bohr')

        return latvec, cutoff

    def _positions_check(self, **kwargs):
        """Check positions type (fraction or not) and unit."""
        unit = kwargs.get('unit', 'angstrom')
        is_frac = self.geometry.is_frac

        # transfer periodic positions to bohr
        position_pe = self.geometry.positions[self.mask_pe]
        _mask = is_frac[self.mask_pe]
        if unit in ('angstrom', 'Angstrom'):
            position_pe[~_mask] = position_pe[~_mask] / _bohr
        elif unit not in ('bohr', 'Bohr'):
            raise ValueError('Please select either angstrom or bohr')

        # whether fraction coordinates in the range [0, 1)
        if torch.any(position_pe[_mask] >= 1) or torch.any(position_pe[_mask] < 0):
            position_pe[_mask] = torch.abs(position_pe[_mask]) - \
                            torch.floor(torch.abs(position_pe[_mask]))

        # transfer from fraction to Bohr unit positions
        position_pe[_mask] = torch.matmul(
            position_pe[_mask], self.latvec[is_frac])
        self.geometry.positions[self.mask_pe] = position_pe

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

    def _get_periodic_distance(self):
        """Get distances between central cell and neighbour cells."""
        positions = self.rcellvec.unsqueeze(2) + self.geometry.positions.unsqueeze(1)
        size_system = self.geometry.size_system
        positions_vec = (positions.unsqueeze(-3) - self.geometry.positions.unsqueeze(1).unsqueeze(-2))
        distance = pack([torch.sqrt(((ipos[:, :inat].repeat(1, inat, 1) - torch.repeat_interleave(
                        icp[:inat], inat, 0)) ** 2).sum(-1)).reshape(-1, inat, inat)
                            for ipos, icp, inat in zip(
                                positions, self.geometry.positions, size_system)], value=1e3)

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
