#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Implementation of many-body tensor representation (MBTR)."""
import warnings
from typing import Literal, Dict
import torch
from torch import Tensor
import numpy as np
from tbmalt import Basis, Geometry
from tbmalt.structures.geometry import unique_atom_pairs
from tbmalt.common.batch import pack

# Define parameters
pi2 = np.sqrt(2 * np.pi)


class Mbtr:
    """Implementation of many-body tensor representation (MBTR).

    Arguments:
        geometry: Geometry from TBMaLT.
        shell: Shell object from TBMaLT.

    Reference:

    """

    def __init__(self,
                 geometry: Geometry,
                 shell: Basis,
                 g1: Dict = None,
                 g2: Dict = None,
                 g3: Dict = None,
                 form: Literal['atom', 'geometry', 'distance'] = 'geometry',
                 logger=None,
                 **kwargs):
        self.geometry = geometry
        self.isperiodic = self.geometry.isperiodic
        self.batch = True if geometry.atomic_numbers.dim() == 2 else False
        if logger is None:
            from tbmalt.common.logger import get_logger
            self.logger = get_logger(__name__)
        if not self.batch:
            self.logger.error('Mbtr do not support single system, transfer to batch')
        if not self.isperiodic:
            self.atomic_numbers = self.geometry.atomic_numbers
            self.distances = self.geometry.distances
            self.d_vect = self.geometry.distance_vectors
            self.atomic_pairs = shell.atomic_number_matrix('atomic')
        else:
            self.atomic_numbers = self.geometry.atomic_numbers
            self.distances = self.geometry.distances_pe
            self.d_vect = self.geometry.distance_vectors_pe
            self.atomic_pairs = shell.atomic_number_matrix('atomic')

        self.fc = torch.exp(2 - self.distances * 0.5)
        self.n_batch = len(self.atomic_numbers)

        self.unique_atomic_numbers = self.geometry.unique_atomic_numbers
        self.unique_atom_pairs = unique_atom_pairs(
            self.geometry, repeat_pairs=False)
        self.shell = shell
        self.form = form

        if self.isperiodic:
            self.periodic = self.geometry.periodic
            self.fc_pe = torch.exp(2-self.periodic.distance_vectors * 0.5)

        # weight of g1 set as 1
        self.g = None
        if g1 is not None:
            self.g1 = self._get_g1(g1) * 1.0
            self.g = self.g1.clone()
        if g2 is not None:
            self.g2 = self._get_g2(g2)
            self.g = torch.cat([self.g, self.g2 / torch.max(self.g2)], 1) \
                if self.g is not None else self.g2
        if g3 is not None:
            self.g3 = self._get_g3(g3)
            self.g = torch.cat([self.g, self.g3 / torch.max(self.g3)], 1) \
                if self.g is not None else self.g3

    def _get_g1(self, g1: Dict):
        r"""Check $\mathcal{D}_1$ function parameters."""
        # Check for the input, where g1 should include range start, end,
        # length and sigma in Gaussian
        assert len(g1) == 4, f'len(g1) should be 4, but get {len(g1)}'

        # To get erf function difference, we need extra one length
        dx = (g1['max'] - g1['min']) / g1['length']
        _space = torch.linspace(g1['min'] - dx, g1['max'] + dx, g1['length'] + 1)

        # Use erf function difference to calculate delta Gaussian distribution
        _map = []
        if self.form in ('atom', 'geometry'):
            for iua in self.unique_atomic_numbers:
                _mask = self.atomic_numbers == iua
                imap = 0.5 * (1.0 + torch.erf(
                    (_space - self.atomic_numbers[_mask].unsqueeze(-1)) / (2.0 * g1['sigma'])))
                _map.append(
                    pack(torch.split(imap, tuple(_mask.sum(-1)))).sum(1))

            g1 = torch.stack([(im[..., 1:] - im[..., :-1]) for im in _map], 1)
            if self.form == 'atom':
                g1 = g1[self.atomic_numbers.ne(0)]

        elif self.form == 'distances':
            imap = 0.5 * (1.0 + torch.erf(
                (_space - self.atomic_pairs.unsqueeze(-1)) / (2.0 * g1['sigma'])))
            g1 = (imap[..., 1:] - imap[..., :-1]).sum(-2)

        else:
            raise ValueError(f'Unknown form {self.form}')

        return g1  # / torch.max(_g1)

    def _get_g2(self, g2: Dict):
        """Get two-body tensor with distances."""
        assert len(g2) == 4, f'len(g2) should be 4, but get {len(g2)}'
        # _min, _max, _len, sigma = g2
        if g2['min'] != 0:
            warnings.warn('min of g2 is not zero, reset it as 0')
            _min = 0
        assert g2['max'] > g2['min'], 'max in g2 is smaller than 0'

        # Reset self.distances so that it ranges from 0 to _max
        _dist = self.distances / (torch.max(self.distances) / g2['max'])

        # To get erf function difference, we need extra one length
        dx = (g2['max'] - g2['min']) / g2['length']
        _space = torch.linspace(g2['min'] - dx, g2['max'] + dx, g2['length'] + 1)

        if self.form in ('atom', 'geometry'):
            _map = []
            for iuap in self.unique_atom_pairs:
                _mask = ((self.atomic_pairs == iuap).sum(-1) == 2) * _dist.ne(0)
                imap = 1.0 / (pi2 * g2['sigma']) * 0.5 * (1.0 + torch.erf(
                    (_space - _dist[_mask].unsqueeze(1)) / (2.0 * g2['sigma'])))

                # Split the atomic pair distances in each geometries, return
                # shape from [n_pair, n_map] -> [n_batch, n_map]
                _map.append(pack(torch.split(imap, tuple(
                    _mask.sum(-1).sum(-1)))).sum(1))
                _g2 = torch.stack([(im[..., 1:] - im[..., :-1])
                                   for im in _map]).transpose(1, 0)

        elif self.form == 'distances':
            imap = 0.5 * (1.0 + torch.erf(
                (_space - self.distances.unsqueeze(-1)) / (2.0 * g2['sigma'])))
            _g2 = (imap[..., 1:] - imap[..., :-1])
        else:
            raise ValueError(f'Unknown form {self.form}')

        return _g2

    def _get_g3(self, g, g3: Dict = None, smear=10.0):
        """Get three-body tensor with angular parameters."""
        assert len(g3) == 4, f'len(g3) should be 4, but get {len(g3)}'
        assert -1 <= g3['min'] < 1, 'min out of range (0, 1)'
        assert g3['max'] > g3['min'], 'max is smaller than min'

        # To get erf function difference, we need extra one length
        dx = (g3['max'] - g3['min']) / g3['length']
        _space = torch.linspace(g3['min'] - dx, g3['max'] + dx, g3['length'] + 1)
        pad_value = (g3['max'] + dx) * smear + 1

        # For convenience, transfer single to batch
        assert self.distances.dim() == 3
        _dist = self.distances
        _atomic_pairs = self.atomic_pairs
        d_vect_ijk = (self.d_vect.unsqueeze(-2) *
                      self.d_vect.unsqueeze(-3)).sum(-1)

        # the dimension of d_ij * d_ik is [n_batch, n_atom_ij, n_atom_jk]
        dist_ijk = _dist.unsqueeze(-1) * _dist.unsqueeze(-2)
        # dist2_ijk = _dist.unsqueeze(-1) ** 2 + _dist.unsqueeze(-2) ** 2

        # create the terms in G4 or G5
        # Set initial values as 2 is to exclude Atom1-Atom1 like angle
        cos = torch.ones(dist_ijk.shape) * pad_value
        mask = dist_ijk.ne(0)
        cos[mask] = d_vect_ijk[mask] / dist_ijk[mask]

        # Set only lower diagonal is not zero to avoid repeat calculations
        ut = torch.unbind(torch.triu_indices(cos.shape[-1], cos.shape[-1], 0))
        cos.permute(2, 3, 0, 1)[ut] = pad_value

        # THIS could be improved by parallel or cpython
        _map, uniq_atom_pairs = [], []
        for i, ian in enumerate(self.unique_atomic_numbers):
            for j, jan in enumerate(self.unique_atomic_numbers[i:]):
                uniq_atom_pairs.append(torch.tensor([ian, jan]))
        uniq_atom_pairs = pack(uniq_atom_pairs)

        for u_atom_pair in uniq_atom_pairs:
            # if not self.isperiodic:
            #     ig = torch.ones(*self.atomic_numbers.shape) * 2
            # else:
            #     ig = torch.ones(self.pe_atomic_numbers.shape) * 2

            # Select ALL the interactions with u_atom_pair
            _im = torch.nonzero((self.atomic_pairs == u_atom_pair).all(dim=-1))

            # If atom pair is not homo, we have to consider inverse u_atom_pair
            if u_atom_pair[0] != u_atom_pair[1]:
                _im = torch.cat(
                    [_im, torch.nonzero((self.atomic_pairs == u_atom_pair.flip(0)).all(dim=-1))])
                _im = _im[_im[..., 0].sort()[1]]

            # Select last two dims which equals to atom-pairs in _im
            g_im = cos[_im[..., 0], :, _im[..., 1], _im[..., 2]]
            _imask, count = torch.unique_consecutive(
                _im[..., 0], return_counts=True)

            # If there is such atom pairs
            _g3 = []
            if count.shape[0] > 0:
                for jj in self.unique_atomic_numbers:
                    _g3.append(pack([ii[ii.le(1) * ia == jj] for ii, ia in zip(
                        g_im.split(tuple(count)), self.atomic_numbers)], value=pad_value))

            _g3 = pack(_g3, value=pad_value)
            # For each geometry, add the same atom pair in the last 2 dimension
            _map.append(1.0 / (pi2 * g3['sigma']) * 0.5 *
                        (1.0 + torch.erf((_space - _g3.unsqueeze(-1)) * smear) /
                         (2.0 * g3['sigma'])).sum(-2).transpose(1, 0))

        g3_tensor = torch.cat([(im[..., 1:] - im[..., :-1]) for im in _map], -2)

        return torch.cat([g, g3_tensor], -2).squeeze()

    @staticmethod
    def _select_distance_onecell(mbtr: Tensor,
                                 geometry_one: object,
                                 cutoff: float = 10.0,
                                 min_distance: float = 1.0):
        """Select.

        Return:
            Mbtr_tensor: Selected tensor from Mbtr method.
            indices: Indices to select Mbtr tensor.

        """
        assert mbtr.dim() == 4, 'dimension of cell_ind wrong'

        # Get central cell indices
        # ind_cent = (geometry_one.cell2d_mat[..., :3] == torch.tensor([0, 0, 0]))\
        #                .sum(-1) == 3
        # ind_cent = (geometry_one.cell_mat == torch.tensor([0, 0, 0]))\
        #                .sum(-1) == 3
        # ind_cut1 = geometry_one.distances.lt(cutoff)
        # ind_cut2 = geometry_one.distances.gt(min_distance)
        decay = torch.exp(0.5 - geometry_one.distances * 0.5).unsqueeze(-1)
        # decay =  0.5 * (torch.cos(np.pi * geometry_one.distances / cutoff) + 1.0).unsqueeze(-1)

        # import matplotlib.pyplot as plt
        # plt.plot(np.linspace(0, 100, 100), (mbtr * decay).sum(1)[0, 10])
        # plt.show()
        # ind = ind_cut1 * ind_cut2
        return mbtr * decay  # [ind], ind

    @staticmethod
    def to_pbc_mbtr(mbtr_tensor: Tensor,
                    GeometryPbcOneCell) -> Tensor:
        """Transfer quasi PBC MBTR tensor to PBC MBTR tensor.

        For convenience, we transfer PBC cell to a molecule like positions. This method
        works to make the molecule like MBTR tensor to PBC tensor, which means an extra
        dimension for cell vector is added to MBTR tensor, looks like:
        (n_atom * n_cell, n_atom * n_cell) -> (n_atom, n_atom, n_cell).
        """
        print('mbtr_tensor', mbtr_tensor.shape)
