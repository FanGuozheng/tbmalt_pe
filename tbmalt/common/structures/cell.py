import torch
from typing import Union, List
from tbmalt.common.batch import pack
import numpy as np
Tensor = torch.Tensor
_bohr = 0.529177249
_pbc = ['cluster', '1d', '2d', '3d', 'mix']


class Pbc:
    """Cell class."""

    def __init__(self, cell: Union[Tensor, List[Tensor]], frac=None, **kwargs):
        """Check cell type and dimension, transfer to batch tensor."""
        unit = kwargs['unit'] if 'unit' in kwargs else 'angstrom'
        self.cell = cell
        if type(self.cell) is list:
            self.cell = pack(self.cell)
        elif type(self.cell) is Tensor:
            if self.cell.dim() == 2:
                self.cell = self.cell.unsqueeze(0)
            elif self.cell.dim() < 2 or cell.dim() > 3:
                raise ValueError('input cell dimension is not 2 or 3')

        if self.cell.size(dim=-2) != 3:
            raise ValueError('input cell should be defined by three lattice vectors')

        # non-periodic systems in cell will be zero
        self.is_periodic = torch.stack([ic.ne(0).any() for ic in self.cell])

        # some systems in batch is fraction coordinate
        if frac is not None:
            self.is_frac = torch.stack([ii.ne(0).any() for ii in frac]) & self.is_periodic
        else:
            self.is_frac = torch.zeros(self.cell.size(0), dtype=bool)

        # transfer positions from angstrom to bohr
        if unit in ('angstrom', 'Angstrom'):
            self.cell = self.cell / _bohr
        elif unit not in ('bohr', 'Bohr'):
            raise ValueError('Please select either angstrom or bohr')

        # Sum of the dimensions of periodic boundary condition
        self.sum_dim = self.cell.ne(0).any(-1).sum(dim=-1)

        if not torch.all(torch.tensor([isd == self.sum_dim[0] for isd in self.sum_dim])):
            self.pbc = [_pbc[isd] for isd in self.sum_dim]
        else:
            self.pbc = _pbc[self.sum_dim[0]]

    @property
    def get_cell_lengths(self):
        """Get the length of each lattice vector."""
        return torch.linalg.norm(self.cell, dim=-1)

    @property
    def get_cell_angles(self):
        """Get the angles alpha, beta and gamma of lattice vectors."""
        _cos = torch.nn.CosineSimilarity(dim=0)
        cosine = torch.stack([torch.tensor([_cos(self.cell[ibatch, 1], self.cell[ibatch, 2]),
                                            _cos(self.cell[ibatch, 0], self.cell[ibatch, 2]),
                                            _cos(self.cell[ibatch, 0], self.cell[ibatch, 1])])
                              for ibatch in range(self.cell.size(0))])
        return torch.acos(cosine) * 180 / np.pi
