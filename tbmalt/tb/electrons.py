"""Electronic calculations."""
import torch
import torch.nn.functional as F
from tbmalt.common.batch import pack
Tensor = torch.Tensor
_expcutoff = {(1, 1, 'cutoff'): torch.tensor([20.024999999999999]),
              (1, 6, 'cutoff'): torch.tensor([22.037500000000001]),
              (1, 7, 'cutoff'): torch.tensor([19.521874999999998]),
              (1, 8, 'cutoff'): torch.tensor([18.515625000000000]),
              (6, 1, 'cutoff'): torch.tensor([22.037500000000001]),
              (6, 6, 'cutoff'): torch.tensor([22.540625000000002]),
              (6, 7, 'cutoff'): torch.tensor([22.037500000000001]),
              (6, 8, 'cutoff'): torch.tensor([20.528124999999999]),
              (7, 1, 'cutoff'): torch.tensor([19.521874999999998]),
              (7, 6, 'cutoff'): torch.tensor([22.037500000000001]),
              (7, 7, 'cutoff'): torch.tensor([20.024999999999999]),
              (7, 8, 'cutoff'): torch.tensor([19.018749999999997]),
              (8, 1, 'cutoff'): torch.tensor([18.515625000000000]),
              (8, 6, 'cutoff'): torch.tensor([20.528124999999999]),
              (8, 7, 'cutoff'): torch.tensor([19.018749999999997]),
              (8, 8, 'cutoff'): torch.tensor([17.006250000000001]),
              (14, 14, 'cutoff'): torch.tensor([33.003124999999997]),
              (0, 0, 'cutoff'): torch.tensor([1.1]), (0, 1, 'cutoff'): torch.tensor([1.1]),
              (0, 6, 'cutoff'): torch.tensor([1.1]), (0, 7, 'cutoff'): torch.tensor([1.1]),
              (0, 8, 'cutoff'): torch.tensor([1.1]), (0, 14, 'cutoff'): torch.tensor([1.1]),
              (1, 0, 'cutoff'): torch.tensor([1.1]), (6, 0, 'cutoff'): torch.tensor([1.1]),
              (7, 0, 'cutoff'): torch.tensor([1.1]), (8, 0, 'cutoff'): torch.tensor([1.1]),
              (14, 0, 'cutoff'): torch.tensor([1.1])}


class Gamma:
    """Build the gamma in second-order term of DFTB.

    Arguments:
        U: non orbital resolved Hubbert
        distance: distance of single system or batch, unit is bohr

    Keyword Args:
        gamma_type: slater or gaussian type, the option with short, e.g.
            slater_short will not include 1 / distance term.
    """

    def __init__(self, U: Tensor, distance: Tensor,
                 number: Tensor, periodic=None, **kwargs) -> Tensor:
        self.U = U
        self.distance = distance
        self.periodic = periodic
        self.number = number
        self.gamma_type = kwargs.get('gamma_type', 'slater')
        self.method = kwargs.get('method', 'read')

        # call gamma funcitons
        if self.gamma_type == 'slater':
            self.gamma = self.gamma_slater()
        elif self.gamma_type == 'gaussian':
            self.gamma = self.gamma_gaussian()

    def gamma_slater(self):
        """Build the Slater type gamma in second-order term."""
        # Construct index list for upper triangle gather operation
        ut = torch.unbind(torch.triu_indices(
            self.U.shape[-1], self.U.shape[-1], 1))

        # deal with single and batch problem
        U = self.U.unsqueeze(0) if self.U.dim() == 1 else self.U

        # make sure the unfied dim for both periodic and non-periodic
        U = self.U.unsqueeze(1) if self.U.dim() == 2 else self.U
        dist = self.distance.unsqueeze(1) if self.distance.dim() == 3 else self.distance

        distance = dist[..., ut[0], ut[1]]

        # build the whole gamma, shortgamma (without 1/R) and triangular gamma
        gamma = torch.zeros(*U.shape, U.shape[-1])
        gamma_tr = torch.zeros(U.shape[0], U.shape[1], len(ut[0]))

        # add (0th row in cell dim) so called chemical hardness Hubbert
        gamma.diagonal(0, -1, -2)[:, 0] = -U[:, 0]

        alpha, beta = U[..., ut[0]] * 3.2, U[..., ut[1]] * 3.2
        aa, bb = self.number[..., ut[0]], self.number[..., ut[1]]

        # mask of homo or hetero Hubbert in triangular gamma
        mask_homo, mask_hetero = alpha == beta, alpha != beta
        mask_homo[distance.eq(0)], mask_hetero[distance.eq(0)] = False, False

        # expcutoff for different atom pairs
        if self.method == 'read':
            expcutoff = torch.stack([torch.cat([_expcutoff[(*[ii.tolist(), jj.tolist()], 'cutoff')]
                                                for ii, jj in zip(aa[ibatch], bb[ibatch])])
                                     for ibatch in range(aa.size(0))]).unsqueeze(
                                             -2).repeat_interleave(alpha.size(-2), dim=-2)
        else:
            expcutoff = self._expgamma_cutoff(alpha, beta, torch.clone(gamma_tr))

        # new masks of homo or hetero Hubbert
        mask_cutoff = distance < expcutoff
        mask_homo = mask_homo & mask_cutoff
        mask_hetero = mask_hetero & mask_cutoff

        # triangular gamma values
        gamma_tr = self._expgamma(distance, alpha, beta, mask_homo, mask_hetero, gamma_tr)

        # symmetric gamma values
        gamma[..., ut[0], ut[1]] = gamma_tr
        gamma[..., ut[1], ut[0]] = gamma[..., ut[0], ut[1]]
        gamma2 = gamma.sum(1)

        # onsite part for periodic condition
        if self.periodic:
            gamma_tem = torch.zeros(U.shape[0], U.shape[1], U.shape[2])
            dist_on = dist.diagonal(0, -1, -2)
            alpha_o, beta_o = U * 3.2, U * 3.2
            mask_homo2, mask_hetero2 = alpha_o == beta_o, alpha_o != beta_o
            mask_homo2[dist_on.eq(0)], mask_hetero2[dist_on.eq(0)] = False, False
            if self.method == 'read':
                expcutoff2 = torch.stack([torch.cat([_expcutoff[(*[ii.tolist(), ii.tolist()], 'cutoff')]
                                                     for ii in self.number[ibatch]])
                                          for ibatch in range(aa.size(0))]).unsqueeze(
                                                 -2).repeat_interleave(U.size(-2), dim=-2)
            else:
                expcutoff2 = self._expgamma_cutoff(alpha_o, beta_o, torch.clone(gamma_tem))

            mask_cutoff2 = dist_on < expcutoff2
            mask_homo2 = mask_homo2 & mask_cutoff2
            mask_hetero2 = mask_hetero2 & mask_cutoff2
            gamma_tem = self._expgamma(dist_on, alpha_o, beta_o, mask_homo2, mask_hetero2, gamma_tem)
            gamma_on = gamma_tem.sum(1)

            # add periodic onsite part to the whole gamma
            _tem = gamma2.diagonal(0, -1, -2) + gamma_on
            gamma2.diagonal(0, -1, -2)[:] = _tem[:]

        return gamma2

    def gaussian(self):
        """Build the Gaussian type gamma in second-order term."""
        raise NotImplementedError('Not implement gaussian yet.')

    def _expgamma_cutoff(self, alpha, beta, gamma_tem,
                         minshortgamma=1.0e-10, tolshortgamma=1.0e-10):
        """The cutoff distance for short range part."""
        # initial distance
        rab = torch.ones_like(alpha)

        # mask of homo or hetero Hubbert in triangular gamma
        mask_homo, mask_hetero = alpha == beta, alpha != beta
        mask_homo[alpha.eq(0)], mask_hetero[alpha.eq(0)] = False, False
        mask_homo[beta.eq(0)], mask_hetero[beta.eq(0)] = False, False

        # mask for batch calculation
        gamma_init = self._expgamma(rab, alpha, beta, mask_homo,
                                    mask_hetero, torch.clone(gamma_tem))
        mask = gamma_init > minshortgamma

        # determine rab
        while True:
            rab[mask] = 2.0 * rab[mask]
            gamma_init[mask] = self._expgamma(rab[mask], alpha[mask], beta[mask], mask_homo[mask],
                                              mask_hetero[mask], torch.clone(gamma_tem)[mask])
            mask = gamma_init > minshortgamma
            if (~mask).all() == True:
                break

        # bisection search for expcutoff
        mincutoff = rab + 0.1
        maxcutoff = 0.5 * rab - 0.1
        cutoff = maxcutoff + 0.1
        maxgamma = self._expgamma(maxcutoff, alpha, beta, mask_homo,
                                  mask_hetero, torch.clone(gamma_tem))
        mingamma = self._expgamma(mincutoff, alpha, beta, mask_homo,
                                  mask_hetero, torch.clone(gamma_tem))
        lowergamma = torch.clone(mingamma)
        gamma = self._expgamma(cutoff, alpha, beta, mask_homo,
                               mask_hetero, torch.clone(gamma_tem))

        # mask for batch calculation
        mask2 = (gamma - lowergamma) > tolshortgamma
        while True:
            maxcutoff = 0.5 * (cutoff + mincutoff)
            mask_search = (maxgamma >= mingamma) == (
                minshortgamma >= self._expgamma(
                    maxcutoff, alpha, beta, mask_homo, mask_hetero, torch.clone(gamma_tem)))
            mask_a = mask2 & mask_search
            mask_b = mask2 & (~mask_search)
            mincutoff[mask_a] = maxcutoff[mask_a]
            lowergamma[mask_a] = self._expgamma(mincutoff[mask_a], alpha[mask_a], beta[mask_a],
                                                mask_homo[mask_a], mask_hetero[mask_a],
                                                torch.clone(gamma_tem)[mask_a])
            cutoff[mask_b] = maxcutoff[mask_b]
            gamma[mask_b] = self._expgamma(cutoff[mask_b], alpha[mask_b], beta[mask_b], mask_homo[mask_b],
                                           mask_hetero[mask_b], torch.clone(gamma_tem)[mask_b])
            mask2 = (gamma - lowergamma) > tolshortgamma
            if (~mask2).all() == True:
                break

        return mincutoff

    def _expgamma(self, distance, alpha, beta, mask_homo, mask_hetero, gamma_tem):
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


def fermi(eigenvalue: Tensor, nelectron: Tensor, kT=0.):
    """Fermi-Dirac distributions without smearing.

    Arguments:
        eigenvalue: Eigen-energies.
        nelectron: number of electrons.
        kT: temperature.

    Returns
        occ: occupancies of electrons
    """
    # make sure each system has at least one electron
    assert False not in torch.ge(nelectron, 1)
    # the number of full occupied state
    electron_pair = torch.true_divide(nelectron.clone().detach(), 2).int()
    # the left single electron
    electron_single = (nelectron.clone().detach() % 2).unsqueeze(1)

    # zero temperature
    if kT != 0:
        raise NotImplementedError('not implement smearing method.')

    # occupied state for batch, if full occupied, occupied will be 2
    # with unpaired electron, return 1
    occ_ = pack([
        torch.cat((torch.ones(electron_pair[i]) * 2, electron_single[i]), 0)
        for i in range(nelectron.shape[0])])

    # pad the rest unoccupied states with 0
    occ = F.pad(input=occ_, pad=(
        0, eigenvalue.shape[-1] - occ_.shape[-1]), value=0)

    # all occupied states (include full and not full occupied)
    nocc = (nelectron.clone().detach() / 2.).ceil()

    return occ, nocc
