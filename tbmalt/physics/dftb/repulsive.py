#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 12:04:56 2021

@author: gz_fan
"""
import torch
from tbmalt.structures.geometry import unique_atom_pairs


class Repulsive:
    """Calculate repulsive for DFTB."""

    def __init__(self, geometry, rep_feed, basis):
        """Initialize parameters."""
        # self.skf = skf
        self.geometry = geometry
        self.rep_feed = rep_feed
        self.basis = basis
        self.repulsive_energy = self.rep_energy()

    def rep_energy(self):
        """Calculate repulsive energy."""
        self.rep_energy = torch.zeros(self.geometry.n_atoms.shape)
        uap = unique_atom_pairs(self.geometry)
        atom_pairs = self.basis.atomic_number_matrix('atomic')
        energy = torch.zeros(self.geometry.distances.shape)

        if self.geometry.distances.dim() == 4:  # PBC, not efficient here
            atom_pairs = atom_pairs.repeat(
                self.geometry.distances.shape[-1], 1, 1, 1, 1).permute(1, 2, 3, 0, -1)

        for iap in uap:
            # get rid of the same atom interaction
            mask_dist = self.geometry.distances.ne(0)

            r_cutoff = self.rep_feed.sktable_dict[(*iap.tolist(), 'rep_cut')]

            # get mask for different atom pairs
            mask_cut = self.geometry.distances.lt(r_cutoff)
            mask = ((iap == atom_pairs).sum(-1) == 2) * mask_dist * mask_cut

            d_mask = self.geometry.distances[mask]

            # 1. exponential repulsive
            r_a123 = self.rep_feed.sktable_dict[(*iap.tolist(), 'exp_coef')]
            energy[mask] = energy[mask] + torch.exp(
                -r_a123[0] * d_mask + r_a123[1]) + r_a123[2]

            # 2. spline repulsive
            r_table = self.rep_feed.sktable_dict[(*iap.tolist(), 'spline_coef')]
            grid = self.rep_feed.sktable_dict[(*iap.tolist(), 'grid')]

            mask2 = (self.geometry.distances.le(grid[-1]) *
                     self.geometry.distances.ge(grid[0]) * mask) * mask_dist
            ind1 = (torch.searchsorted(
                grid, self.geometry.distances) - 1)[mask2]

            r_pol = r_table[ind1]
            deltar = self.geometry.distances[mask2] - grid[ind1]
            energy[mask2] = r_pol[..., 0] + r_pol[..., 1] * deltar + \
                r_pol[..., 2] * deltar ** 2 + r_pol[..., 3] * deltar ** 3

            # 3. bounds distances spline repulsive
            r_table_l = self.rep_feed.sktable_dict[(*iap.tolist(), 'tail_coef')]
            grid_l = self.rep_feed.sktable_dict[(*iap.tolist(), 'long_grid')]

            mask_l = (self.geometry.distances.le(grid_l[1]) *
                      self.geometry.distances.ge(grid_l[0]) * mask) * mask_dist
            ind_l = (torch.searchsorted(
                grid_l, self.geometry.distances) - 1)[mask_l]

            # r_pol_l = r_table_l[ind_l]
            deltar_l = self.geometry.distances[mask_l] - grid_l[ind_l]

            if mask_l.any():
                energy[mask_l] = r_table_l[0] + r_table_l[1] * deltar_l + \
                    r_table_l[2] * deltar_l ** 2 +  r_table_l[3] * deltar_l ** 3 + \
                        r_table_l[4] * deltar_l ** 4 + r_table_l[5] * deltar_l ** 5

        if not self.geometry.distances.dim() == 4:
            return 0.5 * energy.sum(-1).sum(-1)
        else:
            return 0.5 * energy.sum(-1).sum(-1).sum(-1)
