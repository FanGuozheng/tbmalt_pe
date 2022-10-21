#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate forces in DFTB.
"""
import torch
from torch import Tensor
from tbmalt import Basis, Geometry, hs_matrix
from tbmalt.structures.geometry import unique_atom_pairs
from tbmalt.structures import orbs_to_atom
from tbmalt.physics.dftb.shortgamma import ShortGamma


class DftbGradient:
    """Calculate density functional tight binding gradients.

    Arguments:
        geometry: Geometry object in TBMaLT.
        basis: Basis object in TBMaLT.
        h_feed: Hamiltonian feed.
        s_feed: Overlap feed.

   """

    def __init__(self, geometry: Geometry,
                 basis: Basis,
                 h_feed: object,
                 s_feed: object,
                 shell_dict,
                 skparams: object,
                 dftb_type: str = 'dftb2',
                 h: Tensor = None,
                 exponent: float = None):
        self.geometry = geometry
        self.basis = basis
        self.h_feed = h_feed
        self.s_feed = s_feed
        self.shell_dict = shell_dict
        self.skparams = skparams
        self.dftb_type = dftb_type
        self.exponent = exponent

        _pos = self.geometry.positions if self.geometry.positions.dim() == 3 \
            else self.geometry.positions.unsqueeze(0)
        self.n_batch, self.max_atom = _pos.shape[0], _pos.shape[1]

    def __call__(self, density, density_e, epsilon, deltaq, shift, U, **kwargs):
        """Calculate gradients of DFTB.

        Arguments:
            density: Density matrix (energy wighted) in DFTB.
            epsilon: Eigenvalue from DFTB calculations.
            deltaq: Mulliken charge difference from DFTB calculations.

        Returns:
            gradient: DFTB gradients.

        """
        self.density = density
        self.density_e = density_e
        self.epsilon = epsilon
        self.deltaq = deltaq
        self.shift = shift
        self.U = U
        self.max_atm = self.geometry.distances.shape[-1]

        self.dist = self.geometry.distances
        self.mask = self.dist.ne(0.0)
        self.mask_vec = self.mask.repeat(3, 1, 1, 1)
        self.vec = self.geometry.distance_vectors.clone()
        self.vec_dist = torch.zeros(self.vec.shape).permute(-1, 0, 1, 2)
        self.vec_dist[self.mask_vec] = (self.vec.permute(-1, 0, 1, 2) /
                                        self.dist)[self.mask_vec]

        self.vect = self.geometry.distance_vectors.clone().permute(-1, 0, 1, 2)

        # calculate gradients over H0 and S0
        self._dftb1_grad(**kwargs)
        inv, grad2, grad3 = self.dftb2_grad(**kwargs)

        return self.H0_grad() + inv - grad2 - grad3 + self.repulsive_grad()


    def _dftb1_grad(self, **kwargs):
        """Get first derivative of Hamiltonian and overlap SKF integrals."""
        grid = kwargs.pop('gird', 1E-2)

        # repeat three times to generate gradients over x, y, z directions
        _basis = Basis(torch.repeat_interleave(
            self.geometry.atomic_numbers, 3, 0), shell_dict=self.shell_dict)
        self._geometry = Geometry(
            torch.repeat_interleave(self.geometry.atomic_numbers, 3, 0),
            torch.repeat_interleave(self.geometry.positions, 3, 0))
        _mask_u = self.geometry.distances.ne(0) * self.geometry.distances.triu().ne(0)
        _mask_l = self.geometry.distances.ne(0) * self.geometry.distances.tril().ne(0)
        _grid = _mask_l * grid - _mask_u * grid
        dist_v = self._geometry.distance_vectors
        n = torch.arange(self.n_batch) * 3

        # calculate gradients in x, y and z directions
        # 1. get Hamiltonian and overlap with x, y, z distance vectors move
        # one step towards negative distance_ij
        dist_v[n, :, :, 0] = dist_v[n, :, :, 0] + _grid
        dist_v[n+1, :, :, 1] = dist_v[n+1, :, :, 1] + _grid
        dist_v[n+2, :, :, 2] = dist_v[n+2, :, :, 2] + _grid

        self._geometry.updated_dist_vec = dist_v.clone()
        ham0 = hs_matrix(self._geometry, _basis, self.h_feed)
        over0 = hs_matrix(self._geometry, _basis, self.s_feed)

        # 2. get Hamiltonian and overlap with larger distances
        dist_v[n, :, :, 0] = dist_v[n, :, :, 0] - 2.0 * _grid
        dist_v[n+1, :, :, 1] = dist_v[n+1, :, :, 1] - 2.0 * _grid
        dist_v[n+2, :, :, 2] = dist_v[n+2, :, :, 2] - 2.0 * _grid

        self._geometry.updated_dist_vec = dist_v.clone()
        ham1 = hs_matrix(self._geometry, _basis, self.h_feed)
        over1 = hs_matrix(self._geometry, _basis, self.s_feed)
        max_norb = ham1.shape[-1]

        # first derivative: \partial H0/\partial R and \partial S0/\partial S
        self.ham0_grad = ((ham1 - ham0) / (2.0 * grid)).reshape(
            self.n_batch, 3, max_norb, max_norb).transpose(0, 1)
        self.over0_grad = ((over1 - over0) / (2.0 * grid)).reshape(
            self.n_batch, 3, max_norb, max_norb).transpose(0, 1)
        mask = torch.unbind(torch.triu_indices(
            self.ham0_grad.shape[-1], self.ham0_grad.shape[-1], 1))
        self.ham0_grad[..., mask[1], mask[0]] = -self.ham0_grad[..., mask[1], mask[0]]
        self.over0_grad[..., mask[1], mask[0]] = -self.over0_grad[..., mask[1], mask[0]]

    def _dftb1_grad2(self, **kwargs):
        """Get first derivative of Hamiltonian and overlap SKF integrals."""
        grid = kwargs.pop('gird', 1E-2)

        # repeat three times to generate gradients over x, y, z directions
        _basis = Basis(self.geometry.atomic_numbers.repeat(3, 1),
                        shell_dict=self.shell_dict)
        _geometry = Geometry(self.geometry.atomic_numbers.repeat(3, 1),
                        self.geometry.positions.repeat(3, 1, 1))
        _mask_u = self.geometry.distances.ne(0) * self.geometry.distances.triu().ne(0)
        _mask_l = self.geometry.distances.ne(0) * self.geometry.distances.tril().ne(0)

        dist_v = _geometry.distance_vectors
        n1, n2 = self.n_batch, int(self.n_batch * 2)

        # calculate gradients in x, y and z directions
        # 1. get Hamiltonian and overlap with x, y, z distance vectors move
        # one step towards negative distance_ij
        dist_v[: n1, :, :, 0][_mask_u] = dist_v[: n1, :, :, 0][
            _mask_u] - grid
        dist_v[n1: n2, :, :, 1][_mask_u] = dist_v[n1: n2, :, :, 1][
            _mask_u] - grid
        dist_v[n2:, :, :, 2][_mask_u] = dist_v[n2:, :, :, 2][
            _mask_u] - grid
        dist_v[: n1, :, :, 0][_mask_l] = dist_v[: n1, :, :, 0][
            _mask_l] + grid
        dist_v[n1: n2, :, :, 1][_mask_l] = dist_v[n1: n2, :, :, 1][
            _mask_l] + grid
        dist_v[n2:, :, :, 2][_mask_l] = dist_v[n2:, :, :, 2][
            _mask_l] + grid
        _geometry.updated_dist_vec = dist_v.clone()

        ham0 = hs_matrix(_geometry, _basis, self.h_feed)
        over0 = hs_matrix(_geometry, _basis, self.s_feed)

        # 2. get Hamiltonian and overlap with larger distances
        dist_v[: n1, :, :, 0][_mask_u] = dist_v[: n1, :, :, 0][_mask_u] \
            + 2.0 * grid
        dist_v[n1: n2, :, :, 1][_mask_u] = dist_v[n1: n2, :, :, 1][
            _mask_u] + 2.0 * grid
        dist_v[n2:, :, :, 2][_mask_u] = dist_v[n2:, :, :, 2][
            _mask_u] + 2.0 *  grid
        dist_v[: n1, :, :, 0][_mask_l] = dist_v[: n1, :, :, 0][_mask_l] \
            - 2.0 * grid
        dist_v[n1: n2, :, :, 1][_mask_l] = dist_v[n1: n2, :, :, 1][
            _mask_l] - 2.0 * grid
        dist_v[n2: , :, :, 2][_mask_l] = dist_v[n2: , :, :, 2][
            _mask_l] - 2.0 *  grid

        _geometry.updated_dist_vec = dist_v.clone()
        ham1 = hs_matrix(_geometry, _basis, self.h_feed)
        over1 = hs_matrix(_geometry, _basis, self.s_feed)
        max_norb = ham1.shape[-1]

        # first derivative: \partial H0/\partial R and \partial S0/\partial S
        self.ham0_grad = ((ham1 - ham0) / (2.0 * grid)).reshape(
            3, self.n_batch, max_norb, max_norb)
        self.over0_grad = ((over1 - over0) / (2.0 * grid)).reshape(
            3, self.n_batch, max_norb, max_norb)
        mask = torch.unbind(torch.triu_indices(
            self.ham0_grad.shape[-1], self.ham0_grad.shape[-1], 1))
        self.ham0_grad[..., mask[1], mask[0]] = -self.ham0_grad[..., mask[1], mask[0]]
        self.over0_grad[..., mask[1], mask[0]] = -self.over0_grad[..., mask[1], mask[0]]

    def H0_grad(self):
        _h0_grad = 2.0 * (self.ham0_grad * self.density
                         - self.over0_grad * self.density_e)
        n_orb = _h0_grad.shape[-1]
        max_orb = self.over0_grad.shape[-1]
        _grad = (self.over0_grad.reshape(3, -1, max_orb, max_orb) * self.shift
                 * self.density)

        return orbs_to_atom((_h0_grad + _grad).sum(-1).reshape(-1, n_orb),
                            self.basis.orbs_per_atom.repeat(3, 1)).reshape(
                                3, -1, self.max_atm).permute(1, 2, 0)

    def dftb2_grad(self, **kwargs):
        _mask = self.geometry.distances.eq(0.0)
        d_q = self.deltaq.unsqueeze(-1) * self.deltaq.unsqueeze(-2)

        # Calculate 1/R gradients
        invr_grad = torch.zeros(self.dist.shape)
        invr_grad[self.mask] = -d_q[self.mask] / (self.dist ** 3)[self.mask]
        invr_grad = (invr_grad * self.vect).sum(-1).permute(1, 2, 0)

        s_r = ShortGamma.gamma_grad_r(
            self.U, self.geometry.atomic_numbers,
            self.geometry.distances, self.geometry.is_periodic)

        # gradient for corrected DFTB2
        if self.dftb_type == 'dftb3':
            h = kwargs.get('h')
            damp_exp = kwargs.get('damp_exp')
            s = kwargs.get('s')
            u_derivative = kwargs.get('u_derivative')
            h_u = kwargs.get('h_u')
            h_r = self.get_h_grad_r(damp_exp, h)
            sh_r = s_r * h + h_r * s
            tmp2 = sh_r * d_q

            damp_exp = kwargs.get('damp_exp')
            s_u = kwargs.get('s_u')
            tmp3 = self.dftb3_grad(damp_exp, h=h, h_r=h_r, h_u=h_u,
                                    s=s, s_r=s_r, s_u=s_u, u_derivative=u_derivative)
        else:
            tmp2 = s_r * d_q

        tmp2 = tmp2 * self.geometry.distance_vectors.permute(
            -1, 0, 1, 2) / self.geometry.distances
        tmp2.permute(1, 2, 3, 0)[_mask] = 0
        grad2 = tmp2.sum(-1).permute(1, 2, 0)

        if self.dftb_type == 'dftb3':
            tmp3 = tmp3 * self.geometry.distance_vectors.permute(
                -1, 0, 1, 2) / self.geometry.distances
            tmp3.permute(1, 2, 3, 0)[_mask] = 0
            grad3 = tmp3.sum(-1).permute(1, 2, 0)

            return invr_grad, grad2, grad3
        else:
            return invr_grad, grad2, torch.zeros(grad2.shape)

    def dftb3_grad(self, damp_exp, h, h_u, h_r, s, s_r, s_u, u_derivative):
        _mask = self.geometry.distances.eq(0.0)
        Gamma_r = u_derivative.unsqueeze(-1) * ShortGamma.gamma_grad_UR(
            self.U, self.geometry.atomic_numbers,
            self.geometry.distances, s=s, s_r=s_r, s_u=s_u,
            damp_exp=damp_exp, h=h, h_u=h_u, h_r=h_r,)

        d_q = self.deltaq.unsqueeze(-1) * self.deltaq.unsqueeze(-2)
        d_q2 = self.deltaq.unsqueeze(-1)
        tmp = ((d_q2 * Gamma_r).permute(0, -1, -2) + d_q2 * Gamma_r) * d_q

        return -1/3 * tmp

    def get_h_grad_r(self, damp_exp, h):
        uab = (self.U.unsqueeze(-1) + self.U.unsqueeze(-2)) / 2.0
        return -2 * self.geometry.distances * h * uab ** damp_exp

    # def coulomb_grad(self):
    #     """Calculate coulomb gradients."""
    #     grad = torch.zeros(self.dist.shape)
    #     # vect = self.geometry.distance_vectors.clone().permute(-1, 0, 1, 2)
    #     grad[self.mask] = -(self.deltaq.unsqueeze(1) *
    #                   self.deltaq.unsqueeze(2))[self.mask] / \
    #         (self.dist ** 3)[self.mask]
    #
    #     return (grad * self.vect).sum(-1).permute(1, 2, 0)

    def repulsive_grad(self):
        grad = torch.zeros(self.geometry.distances.shape)
        # vec = torch.zeros(self.vect.shape)
        uap = unique_atom_pairs(self.geometry)

        # vec[self.mask] = self.vect / self.dist
        atom_pairs = self.basis.atomic_number_matrix('atomic')
        # energy = torch.zeros(self.geometry.distances.shape)

        for iap in uap:
            # get rid of the same atom interaction
            mask_dist = self.geometry.distances.ne(0)
            # r_cutoff = self.h_feed.other_params[(*iap.tolist(), 'cutoff')]
            r_cutoff = self.skparams.sktable_dict[(*iap.tolist(), 'rep_cut')]

            # get mask for different atom pairs
            mask_cut =  self.geometry.distances.lt(r_cutoff)
            mask = ((iap == atom_pairs).sum(-1) == 2) * mask_dist * mask_cut

            d_mask = self.geometry.distances[mask]

            # 1. exponential repulsive
            r_a123 = self.skparams.sktable_dict[(*iap.tolist(), 'exp_coef')]
            grad[mask] = grad[mask] - r_a123[0] * torch.exp(
                -r_a123[0] * d_mask + r_a123[1])

            # 2. spline repulsive
            r_table = self.skparams.sktable_dict[(*iap.tolist(), 'spline_coef')]
            grid = self.skparams.sktable_dict[(*iap.tolist(), 'grid')]

            mask2 = (self.geometry.distances.le(grid[-1]) *
                self.geometry.distances.ge(grid[0]) * mask) * mask_dist
            ind1 = (torch.searchsorted(
                grid, self.geometry.distances) - 1)[mask2]

            r_pol = r_table[ind1]
            deltar = self.geometry.distances[mask2] - grid[ind1]
            grad[mask2] = r_pol[..., 1] + 2.0 * r_pol[..., 2] * deltar + \
                3.0 * r_pol[..., 3] * deltar ** 2

            # 3. bounds distances spline repulsive
            r_table_l = self.skparams.sktable_dict[(*iap.tolist(), 'tail_coef')]
            grid_l = self.skparams.sktable_dict[(*iap.tolist(), 'long_grid')]

            mask_l = (self.geometry.distances.le(grid_l[1]) *
                self.geometry.distances.ge(grid_l[0]) * mask) * mask_dist
            ind_l = (torch.searchsorted(
                grid_l, self.geometry.distances) - 1)[mask_l]

            # r_pol_l = r_table_l[ind_l]
            deltar_l = self.geometry.distances[mask_l] - grid_l[ind_l]

            if mask_l.any():
                grad[mask_l] = r_table_l[1] + 2.0 * r_table_l[2] * deltar_l +  \
                    3.0 * r_table_l[3] * deltar_l ** 2 + \
                        4.0 * r_table_l[4] * deltar_l ** 3 + \
                            5.0 * r_table_l[5] * deltar_l ** 4

        return (grad * self.vec_dist).sum(-1).permute(1, 2, 0)
