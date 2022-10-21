# -*- coding: utf-8 -*-
"""Short gamma calculations."""
from typing import Union
import torch
from torch import Tensor

_expcutoff = {(1, 1, 'cutoff'): torch.tensor([20.024999999999999]),
              (1, 6, 'cutoff'): torch.tensor([22.037500000000001]),
              (1, 7, 'cutoff'): torch.tensor([19.521874999999998]),
              (1, 8, 'cutoff'): torch.tensor([18.515625000000000]),
              (6, 1, 'cutoff'): torch.tensor([22.037500000000001]),
              (6, 6, 'cutoff'): torch.tensor([22.540625000000002]),
              (6, 7, 'cutoff'): torch.tensor([22.037500000000001]),
              (6, 8, 'cutoff'): torch.tensor([20.528124999999999]),
              (6, 14, 'cutoff'): torch.tensor([20.528124999999999]),
              (7, 1, 'cutoff'): torch.tensor([19.521874999999998]),
              (7, 6, 'cutoff'): torch.tensor([22.037500000000001]),
              (7, 7, 'cutoff'): torch.tensor([20.024999999999999]),
              (7, 8, 'cutoff'): torch.tensor([19.018749999999997]),
              (8, 1, 'cutoff'): torch.tensor([18.515625000000000]),
              (8, 6, 'cutoff'): torch.tensor([20.528124999999999]),
              (8, 7, 'cutoff'): torch.tensor([19.018749999999997]),
              (8, 8, 'cutoff'): torch.tensor([17.006250000000001]),
              (8, 14, 'cutoff'): torch.tensor([17.006250000000001]),
              (8, 31, 'cutoff'): torch.tensor([33.003124999999997]),
              (14, 6, 'cutoff'): torch.tensor([33.003124999999997]),
              (14, 8, 'cutoff'): torch.tensor([33.003124999999997]),
              (14, 14, 'cutoff'): torch.tensor([33.003124999999997]),
              (22, 8, 'cutoff'): torch.tensor([33.003124999999997]),
              (31, 31, 'cutoff'): torch.tensor([33.003124999999997]),
              (31, 33, 'cutoff'): torch.tensor([33.003124999999997]),
              (33, 31, 'cutoff'): torch.tensor([33.003124999999997]),
              (33, 33, 'cutoff'): torch.tensor([33.003124999999997]),
              (0, 0, 'cutoff'): torch.tensor([1.1]), (0, 1, 'cutoff'): torch.tensor([1.1]),
              (0, 6, 'cutoff'): torch.tensor([1.1]), (0, 7, 'cutoff'): torch.tensor([1.1]),
              (0, 8, 'cutoff'): torch.tensor([1.1]), (0, 14, 'cutoff'): torch.tensor([1.1]),
              (1, 0, 'cutoff'): torch.tensor([1.1]), (6, 0, 'cutoff'): torch.tensor([1.1]),
              (7, 0, 'cutoff'): torch.tensor([1.1]), (8, 0, 'cutoff'): torch.tensor([1.1]),
              (14, 0, 'cutoff'): torch.tensor([1.1])}

_expcutoff2 = {(1, 1, 'cutoff'): 20.024999999999999,
               (1, 6, 'cutoff'): 22.037500000000001,
               (1, 7, 'cutoff'): 19.521874999999998,
               (1, 8, 'cutoff'): 18.515625000000000,
               (6, 1, 'cutoff'): 22.037500000000001,
               (6, 6, 'cutoff'): 22.540625000000002,
               (6, 7, 'cutoff'): 22.037500000000001,
               (6, 8, 'cutoff'): 20.528124999999999,
               (6, 14, 'cutoff'): 20.528124999999999,
               (7, 1, 'cutoff'): 19.521874999999998,
               (7, 6, 'cutoff'): 22.037500000000001,
               (7, 7, 'cutoff'): 20.024999999999999,
               (7, 8, 'cutoff'): 19.018749999999997,
               (8, 1, 'cutoff'): 18.515625000000000,
               (8, 6, 'cutoff'): 20.528124999999999,
               (8, 7, 'cutoff'): 19.018749999999997,
               (8, 8, 'cutoff'): 17.006250000000001,
               (8, 14, 'cutoff'): 17.006250000000001,
               (8, 31, 'cutoff'): 33.003124999999997,
               (14, 6, 'cutoff'): 33.003124999999997,
               (14, 8, 'cutoff'): 33.003124999999997,
               (14, 14, 'cutoff'): 33.003124999999997,
               (22, 8, 'cutoff'): 33.003124999999997,
               (31, 31, 'cutoff'): 33.003124999999997,
               (31, 33, 'cutoff'): 33.003124999999997,
               (33, 31, 'cutoff'): 33.003124999999997,
               (33, 33, 'cutoff'): 33.003124999999997,
               (8, 31, 'cutoff'): 33.003124999999997,
               (0, 0, 'cutoff'): 1.1, (0, 1, 'cutoff'): 1.1,
               (0, 6, 'cutoff'): 1.1, (0, 7, 'cutoff'): 1.1,
               (0, 8, 'cutoff'): 1.1, (0, 14, 'cutoff'): 1.1,
               (1, 0, 'cutoff'): 1.1, (6, 0, 'cutoff'): 1.1,
               (7, 0, 'cutoff'): 1.1, (8, 0, 'cutoff'): 1.1,
               (14, 0, 'cutoff'): 1.1}


class ShortGamma:
    """Calculate the short gamma in second-order term of DFTB.

    Arguments:
        U: Non-orbital resolved Hubbert U.
        distances: Distance of single or batch systems.
        gamma_type: Short gamma calculation method.

    Attributes:
        gamma: Calculated short gamma for single or batch systems.

    TODO:
        Add periodic conditions and gaussian method.
    """

    def __init__(self, u: Tensor, number: Tensor, distances: Tensor,
                 periodic: bool, gamma_type: str = 'exponential', **kwargs):
        self.u = u
        self.number = number
        self.distances = distances
        self.periodic = periodic
        self.method = kwargs.get('method', 'read')
        self.orbital_resolved = kwargs.get('orbital_resolved', False)
        self.mask_central_cell = kwargs.get('mask_central_cell', None)

        self.short_gamma, self.short_gamma_pe, self.short_gamma_pe_on = \
            getattr(ShortGamma, gamma_type)(self, **kwargs)

    def exponential(self, **kwargs) -> Union[Tensor, Tensor]:
        """Build the Slater type gamma in second-order term."""
        if self.orbital_resolved:
            U = kwargs.get('u_orbs')
            distances = kwargs.get('distance_orbs')
            number = kwargs.get('orbs_numbers')

        # Construct index list for upper triangle gather operation
        ut = torch.unbind(torch.triu_indices(self.u.shape[-1], self.u.shape[-1], 1))

        # deal with single and batch problem
        U = self.u.unsqueeze(0) if self.u.dim() == 1 else self.u

        # make sure the unfied dim for both periodic and non-periodic
        U = U.unsqueeze(1) if U.dim() == 2 else U
        dist = self.distances.unsqueeze(1) if self.distances.dim() == 3 else self.distances

        # # build the whole gamma, shortgamma (without 1/R) and triangular gamma
        # dist_half = dist[..., ut[0], ut[1]]
        # gamma = torch.zeros(*U.shape, U.shape[-1])
        # gamma_tr = torch.zeros(U.shape[0], U.shape[1], len(ut[0]))
        # # add (0th row in cell dim) so called chemical hardness Hubbert
        # if not self.periodic:
        #     gamma.diagonal(0, -1, -2)[:, 0] = -U[:, 0]
        # else:
        #     gamma.diagonal(0, -1, -2)[self.mask_central_cell] = -U[:, 0]

        # alpha, beta = U[..., ut[0]] * 3.2, U[..., ut[1]] * 3.2
        # aa, bb = self.number[..., ut[0]], self.number[..., ut[1]]
        # # mask of homo or hetero Hubbert in triangular gamma
        # mask_homo, mask_hetero = alpha == beta, alpha != beta
        # mask_homo[dist_half.eq(0)], mask_hetero[dist_half.eq(0)] = False, False

        # # expcutoff for different atom pairs
        # if self.method == 'read':
        #     expcutoff = torch.stack([torch.cat(
        #         [_expcutoff[(*[ii.tolist(), jj.tolist()], 'cutoff')]
        #           for ii, jj in zip(aa[ibatch], bb[ibatch])])
        #           for ibatch in range(aa.size(0))]
        #         ).unsqueeze(-2).repeat_interleave(alpha.size(-2), dim=-2)
        # else:
        #     expcutoff = ShortGamma._expgamma_cutoff(
        #         alpha, beta, torch.clone(gamma_tr))
        # # new masks of homo or hetero Hubbert
        # mask_cutoff = dist_half < expcutoff
        # mask_homo = mask_homo & mask_cutoff
        # mask_hetero = mask_hetero & mask_cutoff
        # # triangular gamma values
        # gamma_tr = ShortGamma._expgamma(
        #     dist_half, alpha, beta, mask_homo, mask_hetero, gamma_tr)
        # # symmetric gamma values
        # gamma[..., ut[0], ut[1]] = gamma_tr
        # gamma[..., ut[1], ut[0]] = gamma[..., ut[0], ut[1]]
        # gamma2 = gamma.sum(1)  # sum gamma of all images


        # mask of homo or hetero Hubbert in triangular gamma
        ut = torch.unbind(torch.triu_indices(U.shape[-1], U.shape[-1], 1))
        lt = torch.unbind(torch.tril_indices(U.shape[-1], U.shape[-1], -1))
        idx = (torch.cat([ut[0], lt[0]]), torch.cat([ut[1], lt[1]]))
        gamma = torch.zeros(*U.shape, U.shape[-1])
        alpha, beta = U[..., idx[0]] * 3.2, U[..., idx[1]] * 3.2
        gamma_idx = torch.zeros(U.shape[0], U.shape[1], len(idx[0]))
        aa, bb = self.number[..., idx[0]], self.number[..., idx[1]]
        dist_flat = dist[..., idx[0], idx[1]]
        mask_homo, mask_hetero = alpha == beta, alpha != beta
        mask_homo[dist_flat.eq(0)], mask_hetero[dist_flat.eq(0)] = False, False
        # expcutoff for different atom pairs
        if self.method == 'read':
            expcutoff = torch.stack([torch.cat(
                [_expcutoff[(*[ii.tolist(), jj.tolist()], 'cutoff')]
                  for ii, jj in zip(aa[ibatch], bb[ibatch])])
                  for ibatch in range(aa.size(0))]
                ).unsqueeze(-2).repeat_interleave(alpha.size(-2), dim=-2)
        else:
            expcutoff = ShortGamma._expgamma_cutoff(
                alpha, beta, torch.clone(gamma_idx))
        # new masks of homo or hetero Hubbert
        mask_cutoff = dist_flat < expcutoff
        mask_homo = mask_homo & mask_cutoff
        mask_hetero = mask_hetero & mask_cutoff
        # triangular gamma values
        gamma_idx = ShortGamma._expgamma2(
            dist_flat, alpha, beta, mask_homo, mask_hetero, gamma_idx)
        # symmetric gamma values
        gamma[..., idx[0], idx[1]] = gamma_idx
        if not self.periodic:
            gamma.diagonal(0, -1, -2)[:, 0] = -U[:, 0]
        else:
            gamma.diagonal(0, -1, -2)[self.mask_central_cell] = -U[:, 0]
        gamma2 = gamma.sum(1)  # sum gamma of all images



        # onsite part for PBC, in molecule, the onsite distance is zero
        # in PBC, the onsite not in the central image, should be considered
        if self.periodic:
            gamma_tem = torch.zeros(U.shape)
            dist_on = dist.diagonal(0, -1, -2)
            alpha_o, beta_o = U * 3.2, U * 3.2
            mask_homo2, mask_hetero2 = alpha_o == beta_o, alpha_o != beta_o
            mask_homo2[dist_on.eq(0)], mask_hetero2[dist_on.eq(0)] = False, False
            if self.method == 'read':
                expcutoff2 = torch.stack([torch.cat(
                    [_expcutoff[(*[ii.tolist(), ii.tolist()], 'cutoff')]
                     for ii in self.number[ibatch]])
                    for ibatch in range(aa.size(0))]).unsqueeze(
                    -2).repeat_interleave(U.size(-2), dim=-2)
            else:
                expcutoff2 = ShortGamma._expgamma_cutoff(
                    alpha_o, beta_o, torch.clone(gamma_tem))

            mask_cutoff2 = dist_on < expcutoff2
            mask_homo2 = mask_homo2 & mask_cutoff2
            mask_hetero2 = mask_hetero2 & mask_cutoff2  # this is not used here
            gamma_tem = ShortGamma._expgamma(
                dist_on, alpha_o, beta_o, mask_homo2, mask_hetero2, gamma_tem)
            # The gamma will be not used here, but will be used for DFTB3 for S*h
            gamma.diagonal(0, -1, -2)[:] = gamma.diagonal(0, -1, -2) + gamma_tem
            gamma_on = gamma_tem.sum(1)

            # add periodic onsite part to the whole gamma
            gamma2.diagonal(0, -1, -2)[:] = gamma2.diagonal(0, -1, -2) + gamma_on
        else:
            gamma_tem = None

        return gamma2, gamma, gamma_tem

    def gaussian(self):
        """Build the Gaussian type gamma in second-order term."""
        raise NotImplementedError('Not implement gaussian yet.')

    @staticmethod
    def _expgamma_cutoff(alpha, beta, gamma_tem,
                         minshortgamma=1.0e-10, tolshortgamma=1.0e-10):
        """Cutoff distance for short range part."""
        # initial distance
        rab = torch.ones_like(alpha)

        # mask of homo or hetero Hubbert in triangular gamma
        mask_homo, mask_hetero = alpha == beta, alpha != beta
        mask_homo[alpha.eq(0)], mask_hetero[alpha.eq(0)] = False, False
        mask_homo[beta.eq(0)], mask_hetero[beta.eq(0)] = False, False

        # mask for batch calculation
        gamma_init = ShortGamma._expgamma(rab, alpha, beta, mask_homo,
                                          mask_hetero, torch.clone(gamma_tem))
        mask = gamma_init > minshortgamma

        # determine rab
        while True:
            rab[mask] = 2.0 * rab[mask]
            gamma_init[mask] = ShortGamma._expgamma(
                rab[mask], alpha[mask], beta[mask], mask_homo[mask],
                mask_hetero[mask], torch.clone(gamma_tem)[mask])
            mask = gamma_init > minshortgamma
            if (~mask).all() is True:
                break

        # bisection search for expcutoff
        mincutoff = rab + 0.1
        maxcutoff = 0.5 * rab - 0.1
        cutoff = maxcutoff + 0.1
        maxgamma = ShortGamma._expgamma(maxcutoff, alpha, beta, mask_homo,
                                        mask_hetero, torch.clone(gamma_tem))
        mingamma = ShortGamma._expgamma(mincutoff, alpha, beta, mask_homo,
                                        mask_hetero, torch.clone(gamma_tem))
        lowergamma = torch.clone(mingamma)
        gamma = ShortGamma._expgamma(cutoff, alpha, beta, mask_homo,
                                     mask_hetero, torch.clone(gamma_tem))

        # mask for batch calculation
        mask2 = (gamma - lowergamma) > tolshortgamma
        while True:
            maxcutoff = 0.5 * (cutoff + mincutoff)
            mask_search = (maxgamma >= mingamma) == (
                    minshortgamma >= ShortGamma._expgamma(
                        maxcutoff, alpha, beta, mask_homo, mask_hetero, torch.clone(gamma_tem)))
            mask_a = mask2 & mask_search
            mask_b = mask2 & (~mask_search)
            mincutoff[mask_a] = maxcutoff[mask_a]
            lowergamma[mask_a] = ShortGamma._expgamma(
                mincutoff[mask_a], alpha[mask_a], beta[mask_a],
                mask_homo[mask_a], mask_hetero[mask_a], torch.clone(gamma_tem)[mask_a])
            cutoff[mask_b] = maxcutoff[mask_b]
            gamma[mask_b] = ShortGamma._expgamma(
                cutoff[mask_b], alpha[mask_b], beta[mask_b], mask_homo[mask_b],
                mask_hetero[mask_b], torch.clone(gamma_tem)[mask_b])
            mask2 = (gamma - lowergamma) > tolshortgamma
            if (~mask2).all() is True:
                break

        return mincutoff

    @staticmethod
    def _expgamma(distance, alpha, beta, mask_homo, mask_hetero, gamma_tem):
        """Calculate the value of short range gamma."""
        # distance of site a and b
        r_homo, r_hetero = 1. / distance[mask_homo], 1. / distance[mask_hetero]

        # homo Hubbert
        aa, dd_homo = alpha[mask_homo], distance[mask_homo]
        taur = aa * dd_homo
        efac = torch.exp(-taur) / 48. * r_homo
        gamma_tem[mask_homo] = \
            (48. + 33. * taur + 9. * taur ** 2 + taur ** 3) * efac

        # hetero Hubbert
        aa, bb = alpha[mask_hetero], beta[mask_hetero]
        dd_hetero = distance[mask_hetero]
        aa2, aa4, aa6 = aa ** 2, aa ** 4, aa ** 6
        bb2, bb4, bb6 = bb ** 2, bb ** 4, bb ** 6
        rab, rba = 1 / (aa2 - bb2), 1 / (bb2 - aa2)
        exp_a, exp_b = torch.exp(-aa * dd_hetero), torch.exp(-bb * dd_hetero)
        val_ab = exp_a * (0.5 * aa * bb4 * rab ** 2 -
                          (bb6 - 3. * aa2 * bb4) * rab ** 3 * r_hetero)
        val_ba = exp_b * (0.5 * bb * aa4 * rba ** 2 -
                          (aa6 - 3. * bb2 * aa4) * rba ** 3 * r_hetero)
        gamma_tem[mask_hetero] = val_ab + val_ba

        return gamma_tem

    @staticmethod
    def _expgamma2(distance, alpha, beta, mask_homo, mask_hetero, gamma_tem):
        """Calculate the value of short range gamma."""
        # distance of site a and b
        r_homo, r_hetero = 1. / distance[mask_homo], 1. / distance[mask_hetero]

        # homo Hubbert
        aa, dd_homo = alpha[mask_homo], distance[mask_homo]
        taur = aa * dd_homo
        efac = torch.exp(-taur) / 48. * r_homo
        gamma_tem[mask_homo] = \
            (48. + 33. * taur + 9. * taur ** 2 + taur ** 3) * efac

        # hetero Hubbert
        aa, bb = alpha[mask_hetero], beta[mask_hetero]
        dd_hetero = distance[mask_hetero]
        aa2, aa4, aa6 = aa ** 2, aa ** 4, aa ** 6
        bb2, bb4, bb6 = bb ** 2, bb ** 4, bb ** 6
        rab, rba = 1 / (aa2 - bb2), 1 / (bb2 - aa2)
        exp_a, exp_b = torch.exp(-aa * dd_hetero), torch.exp(-bb * dd_hetero)
        val_ab = exp_a * (0.5 * aa * bb4 * rab ** 2 -
                          (bb6 - 3. * aa2 * bb4) * rab ** 3 * r_hetero)
        val_ba = exp_b * (0.5 * bb * aa4 * rba ** 2 -
                          (aa6 - 3. * bb2 * aa4) * rba ** 3 * r_hetero)
        gamma_tem[mask_hetero] = val_ab + val_ba

        return gamma_tem


    def gaussian(self):
        """Build the Gaussian type gamma in second-order term."""
        raise NotImplementedError('Not implement gaussian yet.')

    @staticmethod
    def gamma_grad_r(U: Tensor, number: Tensor, distances: Tensor, periodic: bool) -> Tensor:
        """Build the Slater type gamma in second-order term."""
        # to minimum the if & else conditions raise from single & batch,
        # increase dimensions U and distances in single system
        U = U.unsqueeze(1) if U.dim() == 2 else U
        distances = distances.unsqueeze(1) if distances.dim() == 3 else distances

        # Construct index list for upper triangle gather operation
        ut = torch.unbind(torch.triu_indices(U.shape[-1], U.shape[-1], 1))
        dist_half = distances[..., ut[0], ut[1]]

        # build the whole gamma, shortgamma (without 1/R) and triangular gamma
        gamma = torch.zeros(*U.shape, U.shape[-1])
        # gamma_tr = torch.zeros(U.shape[0], len(ut[0]))
        gamma_tr = torch.zeros(U.shape[0], U.shape[1], len(ut[0]))

        # diagonal values is so called chemical hardness Hubbert
        # gamma.diagonal(0, -(U.dim() - 1))[:] = -U
        gamma.diagonal(0, -1, -2)[:, 0] = -U[:, 0]

        alpha, beta = U[..., ut[0]] * 3.2, U[..., ut[1]] * 3.2
        aa, bb = number[..., ut[0]], number[..., ut[1]]

        # mask of homo or hetero Hubbert in triangular gamma
        mask_homo, mask_hetero = alpha == beta, alpha != beta
        mask_homo[dist_half.eq(0)], mask_hetero[dist_half.eq(0)] = False, False

        expcutoff = torch.stack([torch.cat([_expcutoff[(*[ii.tolist(), jj.tolist()], 'cutoff')]
                                            for ii, jj in zip(aa[ibatch], bb[ibatch])])
                                 for ibatch in range(aa.size(0))]).unsqueeze(
            -2).repeat_interleave(alpha.size(-2), dim=-2)

        # new masks of homo or hetero Hubbert
        mask_cutoff = dist_half < expcutoff
        mask_homo = mask_homo & mask_cutoff
        mask_hetero = mask_hetero & mask_cutoff

        # triangular gamma values
        gamma_tr = ShortGamma._expgamma_r(dist_half, alpha, beta, mask_homo, mask_hetero, gamma_tr)

        # to make sure gamma values symmetric
        gamma[..., ut[0], ut[1]] = gamma_tr
        gamma[..., ut[1], ut[0]] = gamma[..., ut[0], ut[1]]
        gamma2 = gamma.sum(1)  # sum gamma of all images

        # onsite part for periodic condition, in molecule, the onsite distance is zero
        # in PBC, the onsite term, which is not in the central image, should be considered
        if periodic:
            gamma_tem = torch.zeros(U.shape[0], U.shape[1], U.shape[2])
            dist_on = distances.diagonal(0, -1, -2)
            alpha_o, beta_o = U * 3.2, U * 3.2
            mask_homo2, mask_hetero2 = alpha_o == beta_o, alpha_o != beta_o
            mask_homo2[dist_on.eq(0)], mask_hetero2[dist_on.eq(0)] = False, False

            expcutoff2 = torch.stack([torch.cat(
                [_expcutoff[(*[ii.tolist(), ii.tolist()], 'cutoff')]
                 for ii in number[ibatch]])
                for ibatch in range(aa.size(0))]).unsqueeze(
                    -2).repeat_interleave(U.size(-2), dim=-2)

            mask_cutoff2 = dist_on < expcutoff2
            mask_homo2 = mask_homo2 & mask_cutoff2
            mask_hetero2 = mask_hetero2 & mask_cutoff2
            gamma_tem = ShortGamma._expgamma_r(dist_on, alpha_o, beta_o, mask_homo2, mask_hetero2, gamma_tem)
            gamma_on = gamma_tem.sum(1)

            # add periodic onsite part to the whole gamma
            _tem = gamma2.diagonal(0, -1, -2) + gamma_on
            gamma2.diagonal(0, -1, -2)[:] = _tem[:]

        return gamma2

    @staticmethod
    def _expgamma_r(dist_half, alpha, beta, mask_homo, mask_hetero, gamma_tr):
        """d_{shortgamma} / d_R."""
        r_homo, r_hetero = 1.0 / dist_half[mask_homo], 1.0 / dist_half[mask_hetero]

        # homo Hubbert
        tauab = 0.5 * (alpha[mask_homo] + beta[mask_homo])
        dd_homo = dist_half[mask_homo]

        gamma_tr[mask_homo] = -tauab * torch.exp(-tauab * dd_homo) * (
                r_homo + 0.6875 * tauab + 0.1875 * dd_homo * (tauab ** 2)
                + 0.02083333333333333333 * (dd_homo ** 2) * (tauab ** 3)) + torch.exp(
            -tauab * dd_homo) * (-r_homo ** 2 + 0.1875 * (tauab ** 2) +
                                 2.0 * 0.02083333333333333333 * dd_homo * (tauab ** 3))

        # hetero Hubbert
        aa, bb = alpha[mask_hetero], beta[mask_hetero]
        dd_homo = dist_half[mask_hetero]

        val_ab = -aa * torch.exp(- aa * dd_homo) * (
                (0.5 * bb ** 4 * aa / (aa ** 2 - bb ** 2) ** 2) - (
                 bb ** 6 - 3.0 * bb ** 4 * aa ** 2) / (
                 dd_homo * (aa ** 2 - bb ** 2) ** 3)) + torch.exp(
            - aa * dd_homo) * (bb ** 6 - 3.0 * bb ** 4 * aa ** 2) /\
                 (dd_homo ** 2 * (aa ** 2 - bb ** 2) ** 3)
        val_ba = -bb * torch.exp(- bb * dd_homo) * (
                (0.5 * aa ** 4 * bb / (bb ** 2 - aa ** 2) ** 2) - (
                 aa ** 6 - 3.0 * aa ** 4 * bb ** 2) / (
                 dd_homo * (bb ** 2 - aa ** 2) ** 3)) + torch.exp(
            - bb * dd_homo) * (aa ** 6 - 3.0 * aa ** 4 * bb ** 2) /\
                 (dd_homo ** 2 * (bb ** 2 - aa ** 2) ** 3)

        gamma_tr[mask_hetero] = val_ab + val_ba

        return gamma_tr

    @staticmethod
    def shortgamma_grad_U(U: Tensor, number: Tensor, distances: Tensor, damp: bool,
                          periodic: bool, h: Tensor, grad_h: Tensor,
                          short_gamma_pe: Tensor, short_gamma_pe_on: Tensor = None,
                          mask_central_cell=None) -> Tensor:
        """d_{shortgamma} / d_U."""
        # make sure the unfied dim for both periodic and non-periodic
        U = U.unsqueeze(0) if U.dim() == 1 else U
        U = U.unsqueeze(1) if U.dim() == 2 else U
        dist = distances.unsqueeze(1) if distances.dim() == 3 else distances

        # Construct index list for upper triangle gather operation
        ut = torch.unbind(torch.triu_indices(U.shape[-1], U.shape[-1], 1))

        # build the whole gamma, shortgamma (without 1/R) and triangular gamma
        # this will include the on-site gamma3, and onsite has pre-factor 0.5
        s_u = torch.zeros(*U.shape, U.shape[-1]) * 0.5
        sh_u = torch.zeros(*U.shape, U.shape[-1]) * 0.5

        # The central image equals to molecules, diagonal should be 0.5
        s_u.diagonal(0, -1, -2)[mask_central_cell] = 0.5
        sh_u.diagonal(0, -1, -2)[mask_central_cell] = 0.5


        # This is wrong for PBC, because the symmetry from PBC: such as [0, 0, -1] and
        # [0, 0, 1] in the distance, or s_u are antisymmetric, should use full matrix
        # dist_half = dist[..., ut[0], ut[1]]
        # alpha, beta = U[..., ut[0]] * 3.2, U[..., ut[1]] * 3.2
        # gamma_tr = torch.zeros(U.shape[0], U.shape[1], len(ut[0]))
        # gamma_tr2 = torch.zeros(U.shape[0], U.shape[1], len(ut[0]))
        # aa, bb = number[..., ut[0]], number[..., ut[1]]
        # # mask of homo or hetero Hubbert in triangular gamma
        # mask_homo, mask_hetero = alpha == beta, alpha != beta
        # mask_homo[dist_half.eq(0)], mask_hetero[dist_half.eq(0)] = False, False
        # expcutoff = torch.stack([torch.cat([_expcutoff[(*[ii.tolist(), jj.tolist()], 'cutoff')]
        #                                     for ii, jj in zip(aa[ibatch], bb[ibatch])])
        #                           for ibatch in range(aa.size(0))]).unsqueeze(
        #     -2).repeat_interleave(alpha.size(-2), dim=-2)
        # # new masks of homo or hetero Hubbert
        # mask_cutoff = dist_half < expcutoff
        # mask_homo = mask_homo & mask_cutoff
        # mask_hetero = mask_hetero & mask_cutoff
        # s_u, sh_u = ShortGamma._expgamma_u2(
        #     dist_half, alpha, beta, mask_homo, mask_hetero, mask_central_cell, ut, damp,
        #     h, grad_h, short_gamma_pe, gamma_tr, gamma_tr2, s_u, sh_u)

        # mask of homo or hetero Hubbert in triangular gamma
        idx = (torch.repeat_interleave(torch.arange(U.shape[-1]), U.shape[-1]),
               torch.arange(U.shape[-1]).repeat(U.shape[-1]))
        mask_diag = idx[0] == idx[1]
        alpha, beta = U[..., idx[0]] * 3.2, U[..., idx[1]] * 3.2
        gamma_idx = torch.zeros(U.shape[0], U.shape[1], len(idx[0]))
        aa, bb = number[..., idx[0]], number[..., idx[1]]
        dist_flat = dist[..., idx[0], idx[1]]
        # mask of homo or hetero Hubbert in triangular gamma
        mask_homo, mask_hetero = alpha == beta, alpha != beta
        mask_homo[dist_flat.eq(0)], mask_hetero[dist_flat.eq(0)] = False, False
        expcutoff = torch.stack([torch.cat([_expcutoff[(*[ii.tolist(), jj.tolist()], 'cutoff')]
                                            for ii, jj in zip(aa[ibatch], bb[ibatch])])
                                 for ibatch in range(aa.size(0))]).unsqueeze(
            -2).repeat_interleave(alpha.size(-2), dim=-2)
        # new masks of homo or hetero Hubbert
        mask_cutoff = dist_flat < expcutoff
        mask_homo = mask_homo & mask_cutoff
        mask_hetero = mask_hetero & mask_cutoff
        gamma_idx[mask_central_cell][..., mask_diag] = 0.5
        s_u, sh_u = ShortGamma._expgamma_u(
            dist_flat, alpha, beta, mask_homo, mask_hetero, mask_central_cell, idx, damp,
            h, grad_h, short_gamma_pe, gamma_idx, s_u, sh_u)



        # onsite part for periodic condition, in molecule, the onsite distance
        # is zero in PBC, the onsite term, which is not in the central image,
        # should be considered
        if periodic:
            # print('sh_u 37', sh_u[:, 37])
            # print('s_u 37', s_u[:, 37])
            # print('gamma_pe 37', short_gamma_pe[:, 37])
            # print('\n sh_u 87', sh_u[:, 87])
            # print('s_u 87', s_u[:, 87])
            # print('gamma_pe 87', short_gamma_pe[:, 87])
            gamma_tr = torch.zeros(U.shape[0], U.shape[1], len(ut[0]))
            shu_tem = torch.zeros(*U.shape)
            dist_on = dist.diagonal(0, -1, -2)
            alpha_o, beta_o = U * 3.2, U * 3.2
            mask_homo2, mask_hetero2 = alpha_o == beta_o, alpha_o != beta_o
            mask_homo2[dist_on.eq(0)], mask_hetero2[dist_on.eq(0)] = False, False
            expcutoff2 = torch.stack([torch.cat([_expcutoff[
                (*[ii.tolist(), ii.tolist()], 'cutoff')]
                for ii in number[ibatch]])
                for ibatch in range(aa.size(0))]
                ).unsqueeze(-2).repeat_interleave(U.size(-2), dim=-2)
            mask_cutoff2 = dist_on < expcutoff2
            mask_homo2 = mask_homo2 & mask_cutoff2
            su_tem, shu_tem = ShortGamma._expgamma_u_homo(
                dist_on, alpha_o, beta_o, mask_homo2, damp,
                h, grad_h, short_gamma_pe_on, gamma_tr, shu_tem)

            # add periodic onsite part to the whole gamma
            s_u.diagonal(0, -1, -2)[:] = s_u.diagonal(0, -1, -2) + su_tem
            sh_u.diagonal(0, -1, -2)[:] = sh_u.diagonal(0, -1, -2) + shu_tem

        return s_u.sum(1), sh_u.sum(1)

    @staticmethod
    def _expgamma_u_homo(dist_half, alpha, beta, mask_homo,
                         damp, h, grad_h, short_gamma_pe, gamma_tr, sh_u):
        """Only works for periodic onsite gradient d_{shortgamma} / d_U."""
        tauab = 0.5 * (alpha[mask_homo] + beta[mask_homo])
        dd_homo = dist_half[mask_homo]

        # when A and B is the same element specie, the dgamma/dU is symmetric
        print(gamma_tr.shape, mask_homo.shape)
        gamma_tr[mask_homo] = torch.exp(-tauab * dd_homo) * (
                1 + tauab * dd_homo + 0.4 * tauab ** 2 * dd_homo ** 2
                + 0.06666666666666667 * tauab ** 3 * dd_homo ** 3)

        if damp:
            sh_u[mask_homo] = gamma_tr[mask_homo] * h.diagonal(0, -1, -2)[:][mask_homo] - \
                                      grad_h.diagonal(0, -1, -2)[:][mask_homo] * short_gamma_pe[mask_homo]

        return gamma_tr, sh_u


    @staticmethod
    def _expgamma_u(dist, alpha, beta, mask_homo, mask_hetero, mask_central_cell,
                    ut, damp, h, grad_h, short_gamma_pe, gamma_tr, s_u, sh_u):

        # 1. homo Hubbert
        tauab = 0.5 * (alpha[mask_homo] + beta[mask_homo])
        dd_homo = dist[mask_homo]

        # when A and B is the same element specie, the dgamma/dU is symmetric
        gamma_tr[mask_homo] = torch.exp(-tauab * dd_homo) * (
                1 + tauab * dd_homo + 0.4 * tauab ** 2 * dd_homo ** 2
                + 0.06666666666666667 * tauab ** 3 * dd_homo ** 3)

        # 2. hetero Hubbert
        aa, bb = alpha[mask_hetero], beta[mask_hetero]
        dd_hetero = dist[mask_hetero]

        val_ab = -3.2 * torch.exp(-aa * dd_hetero) * (
                -(bb ** 6 + 3.0 * bb ** 4 * aa ** 2) / (2.0 * (aa ** 2 - bb ** 2) ** 3) -
                (12.0 * aa ** 3 * bb ** 4) / (dd_hetero * (aa ** 2 - bb ** 2) ** 4)) \
            + 3.2 * torch.exp(-aa * dd_hetero) * dd_hetero * (
                         (aa * bb ** 4) / (2.0 * (aa ** 2 - bb ** 2) ** 2) -
                         (bb ** 6 - 3.0 * aa ** 2 * bb ** 4) / (dd_hetero * (aa ** 2 - bb ** 2) ** 3))
        val_ba = -3.2 * torch.exp(-bb * dd_hetero) * (
                (2.0 * aa ** 3 * bb ** 3) / (bb ** 2 - aa ** 2) ** 3 +
                (12.0 * bb ** 4 * aa ** 3) / (dd_hetero * (bb ** 2 - aa ** 2) ** 4))

        # unlike gamma, Gamma is not symmetric
        val_ab2 = -3.2 * torch.exp(-bb * dd_hetero) * (
                -(aa ** 6 + 3.0 * aa ** 4 * bb ** 2) / (2.0 * (bb ** 2 - aa ** 2) ** 3) -
                (12.0 * bb ** 3 * aa ** 4) / (dd_hetero * (bb ** 2 - aa ** 2) ** 4)) \
                  + 3.2 * torch.exp(-bb * dd_hetero) * dd_hetero * (
                          (bb * aa ** 4) / (2.0 * (bb ** 2 - aa ** 2) ** 2) -
                          (aa ** 6 - 3.0 * bb ** 2 * aa ** 4) / (dd_hetero * (bb ** 2 - aa ** 2) ** 3))
        val_ba2 = -3.2 * torch.exp(-aa * dd_hetero) * (
                (2.0 * bb ** 3 * aa ** 3) / (aa ** 2 - bb ** 2) ** 3 +
                (12.0 * aa ** 4 * bb ** 3) / (dd_hetero * (aa ** 2 - bb ** 2) ** 4))

        gamma_tr[mask_hetero] = val_ab + val_ba

        # to make sure gamma values symmetric
        s_u[..., ut[0], ut[1]] = gamma_tr
        # s_u[..., ut[1], ut[0]] = gamma_tr2

        if damp:
            sh_u[..., ut[0], ut[1]] = s_u[..., ut[0], ut[1]] * h[..., ut[0], ut[1]] - \
                                      grad_h[..., ut[0], ut[1]] * short_gamma_pe[..., ut[0], ut[1]]
            # sh_u[..., ut[1], ut[0]] = s_u[..., ut[1], ut[0]] * h[..., ut[1], ut[0]] - \
            #                           grad_h[..., ut[1], ut[0]] * short_gamma_pe[..., ut[1], ut[0]]

        return s_u, sh_u

    @staticmethod
    def _expgamma_u2(dist_half, alpha, beta, mask_homo, mask_hetero, mask_central_cell,
                    ut, damp, h, grad_h, short_gamma_pe, gamma_tr, gamma_tr2, s_u, sh_u):

        # 1. homo Hubbert
        tauab = 0.5 * (alpha[mask_homo] + beta[mask_homo])
        dd_homo = dist_half[mask_homo]

        # when A and B is the same element specie, the dgamma/dU is the still symmetric
        gamma_tr[mask_homo] = torch.exp(-tauab * dd_homo) * (
                1 + tauab * dd_homo + 0.4 * tauab ** 2 * dd_homo ** 2
                + 0.06666666666666667 * tauab ** 3 * dd_homo ** 3)
        gamma_tr2[mask_homo] = gamma_tr[mask_homo]

        # 2. hetero Hubbert
        aa, bb = alpha[mask_hetero], beta[mask_hetero]
        dd_hetero = dist_half[mask_hetero]

        val_ab = -3.2 * torch.exp(-aa * dd_hetero) * (
                -(bb ** 6 + 3.0 * bb ** 4 * aa ** 2) / (2.0 * (aa ** 2 - bb ** 2) ** 3) -
                (12.0 * aa ** 3 * bb ** 4) / (dd_hetero * (aa ** 2 - bb ** 2) ** 4)) \
                 + 3.2 * torch.exp(-aa * dd_hetero) * dd_hetero * (
                         (aa * bb ** 4) / (2.0 * (aa ** 2 - bb ** 2) ** 2) -
                         (bb ** 6 - 3.0 * aa ** 2 * bb ** 4) / (dd_hetero * (aa ** 2 - bb ** 2) ** 3))
        val_ba = -3.2 * torch.exp(-bb * dd_hetero) * (
                (2.0 * aa ** 3 * bb ** 3) / (bb ** 2 - aa ** 2) ** 3 +
                (12.0 * bb ** 4 * aa ** 3) / (dd_hetero * (bb ** 2 - aa ** 2) ** 4))

        # unlike gamma, Gamma is not symmetric
        val_ab2 = -3.2 * torch.exp(-bb * dd_hetero) * (
                -(aa ** 6 + 3.0 * aa ** 4 * bb ** 2) / (2.0 * (bb ** 2 - aa ** 2) ** 3) -
                (12.0 * bb ** 3 * aa ** 4) / (dd_hetero * (bb ** 2 - aa ** 2) ** 4)) \
                  + 3.2 * torch.exp(-bb * dd_hetero) * dd_hetero * (
                          (bb * aa ** 4) / (2.0 * (bb ** 2 - aa ** 2) ** 2) -
                          (aa ** 6 - 3.0 * bb ** 2 * aa ** 4) / (dd_hetero * (bb ** 2 - aa ** 2) ** 3))
        val_ba2 = -3.2 * torch.exp(-aa * dd_hetero) * (
                (2.0 * bb ** 3 * aa ** 3) / (aa ** 2 - bb ** 2) ** 3 +
                (12.0 * aa ** 4 * bb ** 3) / (dd_hetero * (aa ** 2 - bb ** 2) ** 4))

        gamma_tr[mask_hetero] = val_ab + val_ba
        gamma_tr2[mask_hetero] = val_ab2 + val_ba2

        # to make sure gamma values symmetric
        s_u[..., ut[0], ut[1]] = gamma_tr
        s_u[..., ut[1], ut[0]] = gamma_tr2
        # print('s_u', s_u.sum(1))
        # print('h', h.sum(1))
        # print('grad_h', grad_h.sum(1))
        # print('short_gamma_pe', short_gamma_pe.sum(1))
        # print(s_u[mask_central_cell] * h[mask_central_cell] - grad_h[mask_central_cell] * short_gamma_pe[mask_central_cell])
        # if mask_central_cell is not None:
        #     for i in range(s_u.shape[1]):
        #         print(i, '\n', s_u[:, i] * h[:, i] - grad_h[:, i] * short_gamma_pe[:, i],'\n',  s_u[:, i], '\n', h[:, i], '\n',grad_h[:, i],'\n', short_gamma_pe[:, i])

        if damp:
            sh_u[..., ut[0], ut[1]] = s_u[..., ut[0], ut[1]] * h[..., ut[0], ut[1]] - \
                                      grad_h[..., ut[0], ut[1]] * short_gamma_pe[..., ut[0], ut[1]]
            sh_u[..., ut[1], ut[0]] = s_u[..., ut[1], ut[0]] * h[..., ut[1], ut[0]] - \
                                      grad_h[..., ut[1], ut[0]] * short_gamma_pe[..., ut[1], ut[0]]
        # print('sh_u 1', sh_u.sum(1))

        return s_u, sh_u


    @staticmethod
    def gamma_grad_UR(U: Tensor, number: Tensor, distances: Tensor,
                      s: Tensor, s_r: Tensor, s_u: Tensor,
                      damp_exp: float = None,
                      h: Tensor = None, h_r: Tensor = None, h_u: Tensor = None,
                      ):
        assert U.dim() in (1, 2), 'dimension of U is 1 or 2, get %d' % U.dim()
        assert distances.dim() in (2, 3), 'dimension of distances is 2 ' + \
                                          'or 3, get %d' % distances.dim()

        # to minimum the if & else conditions raise from single & batch,
        # increase dimensions U and distances in single system
        U = U.unsqueeze(0) if U.dim() == 1 else U
        distances = distances.unsqueeze(0) if distances.dim() == 2 else distances

        # Construct index list for upper triangle gather operation
        ut = torch.unbind(torch.triu_indices(U.shape[-1], U.shape[-1], 1))
        distance = distances[..., ut[0], ut[1]]

        # 1. build the whole gamma, shortgamma (without 1/R) and triangular gamma
        # this will include the on-site gamma3, and onsite has pre-factor 0.5
        s_grad = torch.zeros(*U.shape, U.shape[-1])
        grad_ur = torch.zeros(*U.shape, U.shape[-1])
        s_tr_grad = torch.zeros(U.shape[0], len(ut[0]))
        s_tr_grad2 = torch.zeros(U.shape[0], len(ut[0]))

        alpha, beta = U[..., ut[0]] * 3.2, U[..., ut[1]] * 3.2
        aa, bb = number[..., ut[0]], number[..., ut[1]]

        # mask of homo or hetero Hubbert in triangular gamma
        mask_homo, mask_hetero = alpha == beta, alpha != beta
        mask_homo[distance.eq(0)], mask_hetero[distance.eq(0)] = False, False

        # 2. homo Hubbert
        tauab = 0.5 * (alpha[mask_homo] + beta[mask_homo])
        dd_homo = distance[mask_homo]

        g_tr_homo = \
            (48.0 + 33.0 * (tauab * dd_homo) + 9.0 * (tauab * dd_homo) ** 2 +
             (tauab * dd_homo) ** 3) / (48.0 * dd_homo)
        g_grad_tr_u_homo = (11.0 + 6.0 * tauab*dd_homo + (tauab*dd_homo)**2) / 16.0
        g_grad_r_homo = -1.0 / (dd_homo**2) + 0.1875 * (tauab**2) +\
                        0.041666666666666664 * dd_homo * tauab**3
        g_tr_grad_ur_homo = 3.0 / 8.0 * tauab + 1.0 / 8.0 * tauab ** 2 * dd_homo
        s_tr_grad[mask_homo] = 3.2 * torch.exp(-tauab * dd_homo) * (
            (tauab * dd_homo - 1.0) * g_tr_homo -
            tauab * g_grad_tr_u_homo + g_tr_grad_ur_homo - dd_homo * g_grad_r_homo)
        s_tr_grad2[mask_homo] = s_tr_grad[mask_homo]

        # 3. hetero Hubbert
        aa, bb = alpha[mask_hetero], beta[mask_hetero]
        dd_hetero = distance[mask_hetero]
        aa2, aa4, aa6 = aa ** 2, aa ** 4, aa ** 6
        bb2, bb4, bb6 = bb ** 2, bb ** 4, bb ** 6
        rab, rba = aa2 - bb2, bb2 - aa2
        exp_a, exp_b = torch.exp(-aa * dd_hetero), torch.exp(-bb * dd_hetero)
        f_ab = (0.5 * aa * bb4) / (rab ** 2) -\
               (bb6 - 3. * aa2 * bb4) / (rab ** 3 * dd_hetero)
        f_ab2 = (0.5 * bb * aa4) / (rba ** 2) -\
                (aa6 - 3. * bb2 * aa4) / (rba ** 3 * dd_hetero)
        f_ab_ua = -(bb6 + 3.0 * bb4 * aa2) / (2.0 * rab**3) -\
                   (12.0 * aa**3 * bb4) / (dd_hetero * rab**4)
        f_ba_ub = -(aa6 + 3.0 * aa4 * bb2) / (2.0 * rba**3) -\
                   (12.0 * bb**3 * aa4) / (dd_hetero * rba**4)
        f_ba_ua = (2.0 * bb ** 3 * aa ** 3) / (rba ** 3) +\
                  (12.0 * bb4 * aa**3) / (dd_hetero * rba**4)
        f_ab_ub = (2.0 * aa ** 3 * bb ** 3) / (rab ** 3) +\
                  (12.0 * aa4 * bb**3) / (dd_hetero * rab**4)
        f_ab_r = (bb6 - 3.0 * aa2 * bb4) / (rab**3 * dd_hetero**2)
        f_ba_r = (aa6 - 3.0 * bb2 * aa4) / (rba**3 * dd_hetero**2)
        f_ab_ur = (12.0 * aa**3 * bb4) / (rab**4 * dd_hetero**2)
        f_ba_ur = -(12.0 * aa**3 * bb4) / (rba**4 * dd_hetero**2)
        f_ab_ur2 = (12.0 * bb**3 * aa4) / (rba**4 * dd_hetero**2)
        f_ba_ur2 = -(12.0 * bb**3 * aa4) / (rab**4 * dd_hetero**2)

        val_ab = 3.2 * exp_a * (
                (aa * dd_hetero - 1.0) * f_ab
                - aa * f_ab_ua + f_ab_ur - dd_hetero * f_ab_r)
        val_ba = 3.2 * exp_b * (f_ba_ur - bb * f_ba_ua)

        # unlike gamma, Gamma is not symmetric
        val_ab2 = 3.2 * exp_b * (
                (bb * dd_hetero - 1.0) * f_ab2
                - bb * f_ba_ub + f_ab_ur2 - dd_hetero * f_ba_r)
        val_ba2 = 3.2 * exp_a * (f_ba_ur2 - aa * f_ab_ub)

        s_tr_grad[mask_hetero] = val_ab + val_ba
        s_tr_grad2[mask_hetero] = val_ab2 + val_ba2

        # to make sure gamma values symmetric
        s_grad[..., ut[0], ut[1]] = s_tr_grad
        s_grad[..., ut[1], ut[0]] = s_tr_grad2

        if damp_exp is not None:
            uab = (U.unsqueeze(-1) + U.unsqueeze(-2)) / 2.0
            h_ur = damp_exp * distances * uab ** (damp_exp - 1) * (
                distances ** 2 * uab ** damp_exp - 1) * h
            grad_ur[..., ut[0], ut[1]] = -s_grad[..., ut[0], ut[1]] * h[..., ut[0], ut[1]] -\
                s_u[..., ut[0], ut[1]] * h_r[..., ut[0], ut[1]] -\
                s_r[..., ut[0], ut[1]] * h_u[..., ut[0], ut[1]] -\
                s[..., ut[0], ut[1]] * h_ur[..., ut[0], ut[1]]
            grad_ur[..., ut[1], ut[0]] = -s_grad[..., ut[1], ut[0]] * h[..., ut[1], ut[0]] -\
                s_u[..., ut[1], ut[0]] * h_r[..., ut[1], ut[0]] -\
                s_r[..., ut[1], ut[0]] * h_u[..., ut[1], ut[0]] -\
                s[..., ut[1], ut[0]] * h_ur[..., ut[1], ut[0]]

        return grad_ur


def gamma_grad_rU(U: Tensor, distances: Tensor) -> Tensor:
    """Build the Slater type gamma in second-order term."""
