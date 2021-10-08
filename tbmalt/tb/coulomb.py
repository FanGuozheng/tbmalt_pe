# -*- coding: utf-8 -*-
"""Code associated with coulombic interactions.

This module calculate the Ewald summation for periodic
boundary conditions.
"""
import torch
import numpy as np
from scipy import special
from abc import ABC
from tbmalt.common.structures.periodic import Periodic
from tbmalt.common.structures.system import System
from tbmalt.common.batch import pack
Tensor = torch.Tensor
_euler = 0.5772156649


class Coulomb:
    """Class to assist the calculation of coulomb interaction by ABC 'Ewald'.

    The 'Coulomb' class checks the type of periodic boundary condition and
    decides which subclass of ewald summation to use. This class also splits
    mini batches of different peridoci boundary conditions for mix batch
    calculation.

    Arguments:
        geometry: Object for calculation, storing the data of input geometry.
        periodic: Object for calculation, storing the data of translation vectors
            and neighbour list for periodic boundary conditoin.

    Keyword Args:
        tol_ewald: EWald tolerance.
        method: Method to obtain parameters of alpha and cutoff.
        nsearchiter: Maximum of iteration for searching alpha, maxg and maxr.

    Return:
        invrmat: 1/R matrix for the periodic geometry.

    Warning:
        The result of 1D ewald summation is sensitive to the selection of
        splitting parameter alpha. Using the default parameter can achieve the
        convergence.

    """

    def __init__(self, geometry: object, periodic: object, **kwargs):
        self.geometry = geometry
        self.periodic = periodic

        # Whether mix batch
        if not isinstance(self.geometry.pbc, list):
            if self.geometry.pbc == '1d':
                _coulomb = Ewald1d(self.geometry, self.periodic, **kwargs)
            elif self.geometry.pbc == '2d':
                _coulomb = Ewald2d(self.geometry, self.periodic, **kwargs)
            elif self.geometry.pbc == '3d':
                _coulomb = Ewald3d(self.geometry, self.periodic, **kwargs)
            self.invrmat = _coulomb.invrmat

        else:  # -> Mix batch
            raise ValueError('Mix pbc not implemented.')


class Ewald(ABC):
    """ABC for calculating the coulombic interaction in periodic geometry.

    Arguments:
        geometry: Object for calculation, storing the data of input geometry.
        periodic: Object for calculation, storing the data of translation vectors
            and neighbour list for periodic boundary conditoin.

    Keyword Args:
        tol_ewald: EWald tolerance.
        method: Method to obtain parameters of alpha and cutoff.
        nsearchiter: Maximum of iteration for searching alpha, maxg and maxr.

    Return:
        invrmat: 1/R matrix for the periodic geometry.
        alpha: Optimal alpha for the Ewald summation
        maxr: The longest real space vector that gives a bigger
              contribution to the EWald sum than tolerance.
        maxg: The longest reciprocal vector that gives a bigger
              contribution to the EWald sum than tolerance.

    References:
        [1]: Journal of Computational Physics 285 (2015): 280-315.
        [2]: Advances in Computational Mathematics 42.1 (2016): 227-248.

    """

    def __init__(self, geometry: object, periodic: object, param, **kwargs):
        self.geometry = geometry
        self.periodic = periodic
        self.latvec = self.geometry.cell
        self.natom = torch.tensor(self.geometry.size_system)
        self.coord = self.geometry.positions

        # Classify single or batch calculation
        self.param = param
        self.recvec = self.periodic.recvec
        self.tol_ewald = kwargs.get('tol_ewald', torch.tensor(1e-9))
        self.method = kwargs.get('method', 'default')
        self.nsearchiter = kwargs.get('nsearchiter', 30)

        # Number of batches if in batch mode (for internal use only)
        self._n_batch = len(self.geometry.size_system)

        # Maximum atom number
        self._max_natoms = torch.max(self.natom)

        # Optimal parameters for the Ewald summation
        if self.method == 'default':

            # Splitting parameter
            self.alpha = self._default_alpha()
            ff = torch.sqrt(-torch.log(self.tol_ewald))

            # The longest real space vector
            self.maxr = ff / self.alpha

            # The longest reciprocal vector
            self.maxg = 2.0 * self.alpha * ff

        else:
            self.alpha = self._get_alpha()
            self.maxr = self._get_maxr()
            self.maxg = self._get_maxg()

        # The updated lattice points
        self.rcellvec_ud, self.ncell_ud = self._update_latvec()

        # The updated neighbour lists
        self.distmat, self.neighbour = self._update_neighbour()

        # Real part of the Ewald summation
        self.ewald_r, self.mask_g = self._invr_periodic_real()

        # Reciprocal part of the Ewald summation
        self.ewald_g = self._invr_periodic_reciprocal()

        # 1/R matrix for the periodic geometry
        self.invrmat = self._invr_periodic()

    def _update_latvec(self):
        """Update the lattice points for reciprocal Ewald summation."""
        update = Periodic(self.geometry, self.recvec, cutoff=self.maxg,
                          distance_extention=0, positive_extention=0,
                          negative_extention=0, unit='Bohr', return_distance='False')
        return update.rcellvec, update.ncell

    def _update_neighbour(self):
        """Update the neighbour lists for real Ewald summation."""
        update = Periodic(self.geometry, self.latvec, cutoff=self.maxr,
                          distance_extention=0, unit='Bohr')
        return update.periodic_distances, update.neighbour

    def _invr_periodic(self):
        """Calculate the 1/R matrix for the periodic geometry."""
        # Extra contribution for self interaction
        extra = torch.eye(self._max_natoms).unsqueeze(0).repeat_interleave(
            self._n_batch, dim=0) * (2.0 * self.alpha / np.sqrt(np.pi)).unsqueeze(-1).unsqueeze(-1)
        invr = self.ewald_r + self.ewald_g - extra
        invr[self.mask_g] = 0
        return invr

    def _invr_periodic_reciprocal(self):
        """Calculate the reciprocal part of 1/R matrix for the periodic geometry."""
        # Lattice points for the reciprocal sum
        n_low = torch.ceil(torch.clone(self.ncell_ud / 2.0))

        # Batch
        # Large values are padded in the end of short vectors
        gvec_tem = pack([torch.unsqueeze(self.rcellvec_ud[
            ibatch, int(n_low[ibatch]): int(2 * n_low[ibatch] - 1)], 0)
            for ibatch in range(self._n_batch)], value=1e3)
        dd2 = torch.sum(torch.clone(gvec_tem) ** 2, -1)
        mask = dd2 < self.maxg.unsqueeze(-1).unsqueeze(-1) ** 2
        gvec = pack([gvec_tem[ibatch, mask[ibatch]]
                     for ibatch in range(self._n_batch)], value=1e3)

        # Vectors for calculating the reciprocal Ewald sum
        rr = self.coord.repeat(1, self._max_natoms, 1) - self.coord.repeat(
            1, 1, self._max_natoms).view(self._n_batch, -1, 3)

        # The reciprocal Ewald sum
        recsum = self._ewald_reciprocal(rr, gvec, self.alpha, self.param)
        ewald_g = torch.reshape(recsum, (self._n_batch, self._max_natoms,
                                         self._max_natoms))
        ewald_g[self.mask_g] = 0

        return ewald_g

    def _invr_periodic_real(self):
        """Calculate the real part of 1/R matrix for the periodic geometry."""
        ewaldr_tmp = self._ewald_real()

        # Mask for summation
        mask = ewaldr_tmp < float('inf')
        mask_real = self.neighbour & mask
        ewaldr_tmp[~mask_real] = 0
        ewald_r = torch.sum(ewaldr_tmp, dim=-3)

        # Mask used for calculation of reciprocal part
        mask_g = ewald_r == 0

        return ewald_r, mask_g

    def _get_alpha(self):
        """Get optimal alpha for the Ewald sum."""
        # Mask for zero vector
        maskg = self.recvec.ne(0).any(-1)
        maskr = self.latvec.ne(0).any(-1)

        # Ewald parameter
        alphainit = torch.tensor([1.0e-8]).repeat(self._n_batch)
        # Length of the shortest vector in reciprocal space
        min_g = torch.sqrt(torch.min(torch.sum(self.recvec[maskg].unsqueeze(
            0).view(self._n_batch, -1, 3) ** 2, -1), 1).values)

        # Length of the shortest vector in real space
        min_r = torch.sqrt(torch.min(torch.sum(self.latvec[maskr].unsqueeze(
            0).view(self._n_batch, -1, 3) ** 2, -1), 1).values)

        alpha = torch.clone(alphainit)

        # Difference between reciprocal and real parts of the decrease of Ewald sum
        diff = self._diff_rec_real(alpha, min_g, min_r, self.param)
        ierror = 0

        # Mask for batch calculation
        mask = diff < - self.tol_ewald

        # Loop to find the alpha
        while ((alpha[mask] < float('inf')).all()):
            alpha[mask] = 2.0 * alpha[mask]
            diff[mask] = self._diff_rec_real(alpha[mask], min_g[mask],
                                             min_r[mask], self.param[mask])
            mask = diff < - self.tol_ewald
            if (~mask).all() == True:
                break
        if torch.max(alpha >= float('inf')):
            ierror = 1
        elif torch.max(alpha == alphainit):
            ierror = 2

        if ierror == 0:
            alphaleft = 0.5 * alpha
            mask = diff < self.tol_ewald
            while((alpha[mask] < float('inf')).all()):
                alpha[mask] = 2.0 * alpha[mask]
                diff[mask] = self._diff_rec_real(alpha[mask], min_g[mask],
                                                 min_r[mask], self.param[mask])
                mask = diff < self.tol_ewald
                if (~mask).all() == True:
                    break

        if torch.max(alpha >= float('inf')):
            ierror = 3

        if ierror == 0:
            alpharight = alpha
            alpha = (alphaleft + alpharight) / 2.0
            iiter = 0
            diff = self._diff_rec_real(alpha, min_g, min_r, self.param)
            mask = torch.abs(diff) > self.tol_ewald
            while(iiter <= self.nsearchiter):
                mask_neg = diff < 0
                alphaleft[mask_neg] = alpha[mask_neg]
                alpharight[~mask_neg] = alpha[~mask_neg]
                alpha[mask] = (alphaleft[mask] + alpharight[mask]) / 2.0
                diff[mask] = self._diff_rec_real(alpha[mask], min_g[mask],
                                                 min_r[mask], self.param[mask])
                mask = torch.abs(diff) > self.tol_ewald
                iiter += 1
                if (~mask).all() == True:
                    break
            if iiter > self.nsearchiter:
                ierror = 4

        if ierror != 0:
            raise ValueError('Fail to get optimal alpha for Ewald sum.')
        return alpha

    def _get_maxg(self):
        """Get the longest reciprocal vector that gives a bigger
           contribution to the EWald sum than tolerance."""
        ginit = torch.tensor([1.0e-8]).repeat(self.alpha.size(0))
        ierror = 0
        xx = torch.clone(ginit)
        yy = self._gterm(xx, self.alpha, self.param)

        # Mask for batch
        mask = yy > self.tol_ewald

        # Loop
        while ((xx[mask] < float('inf')).all()):
            xx[mask] = xx[mask] * 2.0
            yy[mask] = self._gterm(xx[mask], self.alpha[mask], self.param[mask])
            mask = yy > self.tol_ewald
            if (~mask).all() == True:
                break
        if torch.max(xx >= float('inf')):
            ierror = 1
        elif torch.max(xx == ginit):
            ierror = 2

        if ierror == 0:
            xleft = xx * 0.5
            xright = torch.clone(xx)
            yleft = self._gterm(xleft, self.alpha, self.param)
            yright = torch.clone(yy)
            iiter = 0
            mask = (yleft - yright) > self.tol_ewald

            while(iiter <= self.nsearchiter):
                xx[mask] = 0.5 * (xleft[mask] + xright[mask])
                yy[mask] = self._gterm(xx[mask], self.alpha[mask], self.param[mask])
                mask_yy = yy >= self.tol_ewald
                xleft[mask_yy] = xx[mask_yy]
                yleft[mask_yy] = yy[mask_yy]
                xright[~mask_yy] = xx[~mask_yy]
                yright[~mask_yy] = yy[~mask_yy]
                mask = (yleft - yright) > self.tol_ewald
                iiter += 1
                if (~mask).all() == True:
                    break
            if iiter > self.nsearchiter:
                ierror = 3
        if ierror != 0:
            raise ValueError('Fail to get maxg for Ewald sum.')
        return xx

    def _get_maxr(self):
        """Get the longest real space vector that gives a bigger
           contribution to the EWald sum than tolerance."""
        rinit = torch.tensor([1.0e-8]).repeat(self.alpha.size(0))
        ierror = 0
        xx = torch.clone(rinit)
        yy = self._rterm(xx, self.alpha)

        # Mask for batch
        mask = yy > self.tol_ewald

        # Loop
        while ((xx[mask] < float('inf')).all()):
            xx[mask] = xx[mask] * 2.0
            yy[mask] = self._rterm(xx[mask], self.alpha[mask])
            mask = yy > self.tol_ewald
            if (~mask).all() == True:
                break
        if torch.max(xx >= float('inf')):
            ierror = 1
        elif torch.max(xx == rinit):
            ierror = 2

        if ierror == 0:
            xleft = xx * 0.5
            xright = torch.clone(xx)
            yleft = self._rterm(xleft, self.alpha)
            yright = torch.clone(yy)
            iiter = 0
            mask = (yleft - yright) > self.tol_ewald

            while(iiter <= self.nsearchiter):
                xx[mask] = 0.5 * (xleft[mask] + xright[mask])
                yy[mask] = self._rterm(xx[mask], self.alpha[mask])
                mask_yy = yy >= self.tol_ewald
                xleft[mask_yy] = xx[mask_yy]
                yleft[mask_yy] = yy[mask_yy]
                xright[~mask_yy] = xx[~mask_yy]
                yright[~mask_yy] = yy[~mask_yy]
                mask = (yleft - yright) > self.tol_ewald
                iiter += 1
                if (~mask).all() == True:
                    break
            if iiter > self.nsearchiter:
                ierror = 3
        if ierror != 0:
            raise ValueError('Fail to get maxg for Ewald sum.')
        return xx

    def _ewald_real(self):
        """Batch calculation of the Ewald sum in the real part for a certain
        vector length."""
        return torch.erfc(self.alpha.unsqueeze(-1).unsqueeze(
            -1).unsqueeze(-1) * self.distmat) / self.distmat

    def _diff_rec_real(self, alpha, min_g, min_r, param):
        """Returns the difference between reciprocal and real parts of the
        decrease of Ewald sum."""
        return (self._gterm(4.0 * min_g, alpha, param) - self._gterm(
            5.0 * min_g, alpha, param)) - (self._rterm(2.0 * min_r, alpha) -
                                           self._rterm(3.0 * min_r, alpha))


class Ewald3d(Ewald):
    """Implement of ewald summation for 3D boundary condition."""

    def __init__(self, geometry: object, periodic: object, **kwargs):
        param = periodic.cellvol
        super().__init__(geometry, periodic, param, **kwargs)

    def _default_alpha(self):
        """Returns the default value of alpha."""
        return (self.natom / self.param ** 2) ** (1/6) * np.pi ** 0.5

    def _ewald_reciprocal(self, rr, gvec, alpha, vol):
        """Calculate the reciprocal part of the Ewald sum."""
        g2 = torch.sum(gvec ** 2, -1)
        dot = torch.matmul(rr, gvec.transpose(-1, -2))

        recsum = torch.sum((torch.exp(- g2 / (4.0 * alpha.unsqueeze(-1) ** 2)
                                      ) / g2).unsqueeze(-2) * torch.cos(dot), -1)
        tem = 2.0 * recsum * 4.0 * np.pi / vol.unsqueeze(-1)
        return (tem - (np.pi / (self.param * self.alpha ** 2)).unsqueeze(-1)).unsqueeze(-2)

    def _gterm(self, len_g, alpha, cellvol):
        """Returns the maximum value of the Ewald sum in the
           reciprocal part for a certain vector length."""
        return 4.0 * np.pi * (torch.exp((-0.25 * len_g ** 2) / (alpha ** 2))
                              / (cellvol * len_g ** 2))

    def _rterm(self, len_r, alpha):
        """Returns the maximum value of the Ewald sum in the
           real part for a certain vector length."""
        return torch.erfc(alpha * len_r) / len_r


class Ewald2d(Ewald):
    """Implement of ewald summation for 2D boundary condition."""

    def __init__(self, geometry: object, periodic: object, **kwargs):
        self.length = geometry._cell.get_cell_lengths
        _tem = torch.clone(self.length)
        _tem[_tem.eq(0)] = 1e5
        param = torch.min(_tem, -1).values
        super().__init__(geometry, periodic, param, **kwargs)

    def _default_alpha(self):
        """Returns the default value of alpha."""
        return self.natom ** (1/6) / self.param * np.pi ** 0.5

    def _ewald_reciprocal(self, rr, gvec, alpha, length):
        """Calculate the reciprocal part of the Ewald sum."""
        # Mask of the periodic direction
        _mask_pd = self.periodic.latvec.ne(0).any(-1)
        _index_pd = torch.tensor([0, 1, 2]).repeat(self._n_batch, 1)[~_mask_pd]
        _length = self.length[_mask_pd].reshape(self._n_batch, 2)

        g2 = torch.sum(gvec ** 2, -1)
        gg = torch.sqrt(g2)
        dot = torch.matmul(rr, gvec.transpose(-1, -2))

        # Reciprocal, L
        tem = gg.unsqueeze(-2).repeat_interleave(rr.size(-2), -2) * rr[
            torch.arange(self._n_batch), :, _index_pd].unsqueeze(-2).transpose(-1, -2)

        aa = torch.exp(tem)
        tem2 = gg / (alpha.unsqueeze(-1) * 2.0)

        bb = bb = tem2.unsqueeze(-2).repeat_interleave(rr.size(-2), -2) + (
            alpha.unsqueeze(-1) * rr[torch.arange(self._n_batch), :, _index_pd]
            ).unsqueeze(-1).repeat_interleave(tem2.size(-1), -1)

        cc = torch.exp(- tem)
        dd = tem2.unsqueeze(-2).repeat_interleave(rr.size(-2), -2) - (
            alpha.unsqueeze(-1) * rr[torch.arange(self._n_batch), :, _index_pd]
            ).unsqueeze(-1).repeat_interleave(tem2.size(-1), -1)

        yyt = aa * torch.erfc(bb) + cc * torch.erfc(dd)
        yy = yyt / gg.unsqueeze(-2)
        yy[yy != yy] = 0
        recl = torch.sum(torch.cos(dot) * yy, -1) * 2.0 * np.pi / (
            _length[:, 0] * _length[:, 1]).unsqueeze(-1)

        # Reciprocal, 0
        tem3 = torch.exp(- alpha.unsqueeze(-1) ** 2 * rr[torch.arange(
            self._n_batch), :, _index_pd] ** 2) / alpha.unsqueeze(-1) + (
                np.pi) ** 0.5 * rr[torch.arange(self._n_batch), :, _index_pd] * torch.erf(
            alpha.unsqueeze(-1) * rr[torch.arange(self._n_batch), :, _index_pd])
        rec0 = tem3 * (- 2.0 * np.pi ** 0.5 / ((
            _length[:, 0] * _length[:, 1]).unsqueeze(-1)))

        return recl + rec0

    def _gterm(self, len_g, alpha, length):
        """Returns the maximum value of the Ewald sum in the
           reciprocal part for a certain vector length."""

        return (torch.erfc(len_g / (alpha * 2)) * 2) / len_g * np.pi / length ** 2

    def _rterm(self, len_r, alpha):
        """Returns the maximum value of the Ewald sum in the
           real part for a certain vector length."""
        return torch.erfc(alpha * len_r) / len_r


class Ewald1d(Ewald):
    """Implement of ewald summation for 1D boundary condition."""

    def __init__(self, geometry: object, periodic: object, **kwargs):
        self.length = geometry._cell.get_cell_lengths
        _tem = torch.clone(self.length)
        _tem[_tem.eq(0)] = 1e5
        param = torch.min(_tem, -1).values
        super().__init__(geometry, periodic, param, **kwargs)

    def _default_alpha(self):
        """Returns the default value of alpha."""
        return self.natom ** (1/6) / self.param * np.pi ** 0.5

    def _ewald_reciprocal(self, rr, gvec, alpha, length):
        """Calculate the reciprocal part of the Ewald sum."""
        # Mask of the periodic direction
        _mask_pd = self.periodic.latvec.ne(0).any(-1)
        _index_pd = torch.tensor([0, 1, 2]).repeat(
            self._n_batch, 1)[~_mask_pd].reshape(self._n_batch, 2)

        g2 = torch.sum(gvec ** 2, -1)
        dot = torch.matmul(rr, gvec.transpose(-1, -2))

        # Reciprocal, L
        aa = g2 / (4 * alpha.unsqueeze(-1) ** 2)

        bb = (rr[torch.arange(self._n_batch), :, _index_pd[:, 0]] ** 2 + rr[
            torch.arange(self._n_batch), :, _index_pd[:, 1]] ** 2) * alpha.unsqueeze(-1) ** 2

        xx = torch.linspace(10.0 ** -20, 1.0, 5000)
        kk0 = torch.tensor([[[torch.trapz(1.0 / xx * torch.exp(-iaa / xx - ibb * xx), xx)
                              for iaa in aa[ibatch]]
                             for ibb in bb[ibatch]]
                            for ibatch in range(self._n_batch)])

        recl = torch.sum(torch.cos(dot) * kk0, -1) * 2.0 / length.unsqueeze(-1)

        # Reciprocal, 0
        rec0 = torch.zeros_like(bb)
        mask = bb != 0

        rec0[mask] = (- _euler - torch.log(bb[mask])
                      - special.exp1(bb[mask])) / length.unsqueeze(
                      -1).repeat_interleave(bb.size(-1), -1)[mask]

        return recl + rec0

    def _gterm(self, len_g, alpha, length):
        """Returns the maximum value of the Ewald sum in the
           reciprocal part for a certain vector length."""
        return special.exp1(len_g ** 2 / (4 * alpha ** 2)) / length

    def _rterm(self, len_r, alpha):
        """Returns the maximum value of the Ewald sum in the
           real part for a certain vector length."""
        return torch.erfc(alpha * len_r) / len_r
