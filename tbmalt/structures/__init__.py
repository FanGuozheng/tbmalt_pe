# -*- coding: utf-8 -*-
"""A module for holding data structures and any associated code.

The `tbmalt.structures` module contains all generic data structure classes,
i.e. those python classes which act primarily as data containers. The TBMaLT
project uses classes for data storage rather than dictionaries as they offer
greater functionality and their contents are more consistent.

All data structure classes are directly accessible from the top level TBMaLT
namespace, e.g.

.. code-block:: python

    # Use this
    from tbmalt import Geometry
    # Rather than this
    from tbmalt.structures.geometry import Geometry

"""
import torch
from torch import Tensor
from tbmalt.common.batch import pack


def orbs_to_atom(intensor: Tensor, orbs_per_atom: Tensor) -> Tensor:
    """Transform orbital resolved tensor to atom resolved tensor."""
    # get a list of accumulative orbital indices
    ind_cumsum = [torch.cat((
        torch.zeros(1), torch.cumsum(ii[ii.ne(0)], -1))).long()
        for ii in orbs_per_atom]

    # return charge of each atom for batch system
    return pack([torch.stack([(it[..., ii: jj]).sum(-1) for ii, jj in zip(
        ic[:-1], ic[1:])]) for ic, it in zip(ind_cumsum, intensor)])
