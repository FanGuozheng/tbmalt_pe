# -*- coding: utf-8 -*-
"""Short gamma calculations."""
from typing import Literal
import torch
from torch import Tensor

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

    def __init__(self, U: Tensor, distances: Tensor,
                 gamma_type: Literal['exponential', 'gaussian']):
        assert U.dim() in (1, 2), 'dimension of U is 1 or 2, get %d' % U.dim()
        assert distances.dim() in (2, 3), 'dimension of distances is 2 ' + \
            'or 3, get %d' % distances.dim()

        self.short_gamma = getattr(ShortGamma, gamma_type)(self, U, distances)

    def exponential(self, U: Tensor, distances: Tensor) -> Tensor:
        """Build the Slater type gamma in second-order term."""
        # to minimum the if & else conditions raise from single & batch,
        # increase dimensions U and distances in single system
        U = U.unsqueeze(0) if U.dim() == 1 else U
        distances = distances.unsqueeze(0) if distances.dim() == 2 else distances

        # Construct index list for upper triangle gather operation
        ut = torch.unbind(torch.triu_indices(U.shape[-1], U.shape[-1], 1))
        distance = distances[..., ut[0], ut[1]]

        # build the whole gamma, shortgamma (without 1/R) and triangular gamma
        gamma = torch.zeros(*U.shape, U.shape[-1])
        gamma_tr = torch.zeros(U.shape[0], len(ut[0]))

        # diagonal values is so called chemical hardness Hubbert
        gamma.diagonal(0, -(U.dim() - 1))[:] = -U

        alpha, beta = U[..., ut[0]] * 3.2, U[..., ut[1]] * 3.2

        # mask of homo or hetero Hubbert in triangular gamma
        mask_homo, mask_hetero = alpha == beta, alpha != beta
        mask_homo[distance.eq(0)], mask_hetero[distance.eq(0)] = False, False
        r_homo, r_hetero = 1.0 / distance[mask_homo], 1.0 / distance[mask_hetero]

        # homo Hubbert
        aa, dd_homo = alpha[mask_homo], distance[mask_homo]
        taur = aa * dd_homo
        efac = torch.exp(-taur) / 48.0 * r_homo
        gamma_tr[mask_homo] = \
            (48.0 + 33.0 * taur + 9.0 * taur ** 2 + taur ** 3) * efac

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
        gamma_tr[mask_hetero] = val_ab + val_ba

        # to make sure gamma values symmetric
        gamma[..., ut[0], ut[1]] = gamma_tr
        gamma[..., ut[1], ut[0]] = gamma[..., ut[0], ut[1]]

        return gamma

    def gaussian(self):
        """Build the Gaussian type gamma in second-order term."""
        raise NotImplementedError('Not implement gaussian yet.')
