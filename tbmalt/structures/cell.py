from typing import Union, List, Optional
import torch
import numpy as np

from tbmalt.data.units import length_units
from tbmalt.common.batch import pack
Tensor = torch.Tensor

_pbc = ['cluster', '1d', '2d', '3d', 'mix']


class Pbc:
    """Cell class to deal with periodic boundary conditions.

    Arguments:
        cell: Atomic numbers of the atoms.
        frac :
        units: Unit in which ``positions`` were specified. For a list of
            available units see :mod:`.units`. [DEFAULT='bohr']

    Attributes:
        cell: Atomic numbers of the atoms.
        frac : Coordinates of the atoms.
        n_atoms: Number of atoms in the system.


    """

    def __init__(self, cell: Union[Tensor, List[Tensor]], frac=None,
                 units: Optional[str] = 'bohr', **kwargs):
        """Check cell type and dimension, transfer to batch tensor."""

        if isinstance(cell, list):
            cell = pack(cell)
        elif isinstance(cell, Tensor):
            if cell.dim() == 2:
                cell = cell.unsqueeze(0)
            elif cell.dim() < 2 or cell.dim() > 3:
                raise ValueError('input cell dimension is not 2 or 3')

        if cell.size(dim=-2) != 3:
            raise ValueError('input cell should be defined by three lattice vectors')

        # non-periodic systems in cell will be zero
        self.periodic_list = torch.tensor([ic.ne(0).any() for ic in cell])

        # some systems in batch is fraction coordinate
        if frac is not None:
            self.frac_list = torch.stack([ii.ne(0).any() for ii in frac]) & self.periodic_list
        else:
            self.frac_list = torch.zeros(cell.size(0), dtype=bool)

        # transfer positions from angstrom to bohr
        if units != 'bohr':
            cell: Tensor = cell * length_units[units]

        # Sum of the dimensions of periodic boundary condition
        sum_dim = cell.ne(0).any(-1).sum(dim=-1)

        if not torch.all(torch.tensor([isd == sum_dim[0] for isd in sum_dim])):
            self.pbc = [_pbc[isd] for isd in sum_dim]
        else:
            self.pbc = _pbc[sum_dim[0]]

        self.cell = cell

    @property
    def get_cell_lengths(self):
        """Get the length of each lattice vector."""
        return torch.linalg.norm(self.cell, dim=-1)

    @property
    def get_cell_angles(self):
        """Get the angles alpha, beta and gamma of lattice vectors."""
        _cos = torch.nn.CosineSimilarity(dim=0)
        cosine = torch.stack([
            torch.tensor([_cos(self.cell[ibatch, 1], self.cell[ibatch, 2]),
                          _cos(self.cell[ibatch, 0], self.cell[ibatch, 2]),
                          _cos(self.cell[ibatch, 0], self.cell[ibatch, 1])])
                              for ibatch in range(self.cell.size(0))])
        return torch.acos(cosine) * 180 / np.pi
