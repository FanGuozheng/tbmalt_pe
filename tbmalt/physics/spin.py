#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Dict
import torch
from torch import Tensor
from tbmalt import Geometry, Basis


def _gather_w(geometry: Geometry, basis: Basis, w_params: Dict,
              **kwargs) -> Tensor:
    """Gather spin parameters W matrix."""
    an = geometry.atomic_numbers
    a_shape = basis.atomic_matrix_shape[:-1]
    o_shape = basis.orbital_matrix_shape[:-1]

    # Get the onsite values for all non-padding elements & pass on the indices
    # of the atoms just in case they are needed by the SkFeed
    mask = an.nonzero(as_tuple=True)

    if 'atom_indices' not in kwargs:
        kwargs['atom_indices'] = torch.arange(geometry.n_atoms.max()
                                              ).expand(a_shape)

    os_flat = [w_params[(ian.tolist(), il)] for ian in an
               for il in range(max(shell_dict[int(ian)]) + 1)]

