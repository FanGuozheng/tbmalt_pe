import bisect
import torch
from tbmalt.common.batch import pack
import numpy as np
Tensor = torch.Tensor


class BicubInterp:
    """Vectorized bicubic interpolation method designed for molecule.

    The bicubic interpolation is designed to interpolate the integrals of
    whole molecule. The xmesh, ymesh are the grid points and they are the same,
    therefore only xmesh is needed here.
    The zmesh is a 4D or 5D Tensor. The 1st, 2nd dimensions are corresponding
    to the pairwise atoms in molecule. The 3rd and 4th are corresponding to
    the xmesh and ymesh. The 5th dimension is optional. For bicubic
    interpolation of single integral such as ss0 orbital, it is 4D Tensor.
    For bicubic interpolation of all the orbital integrals, zmesh is 5D Tensor.

    Arguments:
        xmesh: 1D Tensor.
        zmesh: 2D or 3D Tensor, 2D is for single integral with vrious
            compression radii, 3D is for multi integrals.

    References:
        .. [wiki] https://en.wikipedia.org/wiki/Bicubic_interpolation
    """

    def __init__(self, xmesh: Tensor, zmesh: Tensor, hs_grid=None):
        """Get interpolation with two variables."""
        if zmesh.dim() < 2:
            raise ValueError('zmesh dim should >= 2')
        if zmesh.dim() == 2:
            zmesh = zmesh.unsqueeze(0)  # -> single to batch

        self.xmesh = xmesh
        self.zmesh = zmesh
        self.hs_grid = hs_grid

    def __call__(self, xnew: Tensor, distance=None):
        """Calculate bicubic interpolation.

        Arguments:
            xnew: the interpolation points of atom-pairs in molecule, it can be
                single atom-pair or multi atom-pairs.
        """
        assert xnew.dim() <= 2
        self.xi = xnew if xnew.dim() == 2 else xnew.unsqueeze(0)  # -> to batch
        self.atp = self.xi.shape[0]  # number of atom pairs
        self.arange_atp = torch.arange(self.atp)

        if distance is not None:  # with DFTB+ distance interpolation
            assert self.hs_grid is not None  # -> grid mesh for distance
            pdim = [2, 0, 1] if self.zmesh.dim() == 3 else [2, 3, 0, 1]
            zmesh = self.zmesh.permute(pdim)
            ski = SKInterpolation(self.hs_grid, zmesh)
            zmesh = ski(distance)
        else:
            zmesh = self.zmesh

        coeff = torch.tensor([[1., 0., 0., 0.], [0., 0., 1., 0.],
                              [-3., 3., -2., -1.], [2., -2., 1., 1.]])
        coeff_ = torch.tensor([[1., 0., -3., 2.], [0., 0., 3., -2.],
                               [0., 1., -2., 1.], [0., 0., -1., 1.]])

        # get the nearest grid points, 1st and second neighbour indices of xi
        self._get_indices()

        # this is to transfer x to fraction and its square, cube
        x_fra = (self.xi - self.xmesh[self.nx0]) / (
            self.xmesh[self.nx1] - self.xmesh[self.nx0])
        xmat = torch.stack([x_fra ** 0, x_fra ** 1, x_fra ** 2, x_fra ** 3])

        # get four nearest grid points values, each will be: [natom, natom, 20]
        f00, f10, f01, f11 = self._fmat0th(zmesh)

        # get four nearest grid points derivative over x, y, xy
        f02, f03, f12, f13, f20, f21, f30, f31, f22, f23, f32, f33 = \
            self._fmat1th(zmesh, f00, f10, f01, f11)
        fmat = torch.stack([torch.stack([f00, f01, f02, f03]),
                            torch.stack([f10, f11, f12, f13]),
                            torch.stack([f20, f21, f22, f23]),
                            torch.stack([f30, f31, f32, f33])])

        pdim = [2, 0, 1] if fmat.dim() == 3 else [2, 3, 0, 1]
        a_mat = torch.matmul(torch.matmul(coeff, fmat.permute(pdim)), coeff_)

        return torch.stack([torch.matmul(torch.matmul(
            xmat[:, i, 0], a_mat[i]), xmat[:, i, 1]) for i in range(self.atp)])

    def _get_indices(self):
        """Get indices and repeat indices."""
        self.nx0 = torch.searchsorted(self.xmesh, self.xi.detach()) - 1

        # get all surrounding 4 grid points indices and repeat indices
        self.nind = torch.tensor([ii for ii in range(self.atp)])
        self.nx1 = torch.clamp(torch.stack([ii + 1 for ii in self.nx0]), 0,
                               len(self.xmesh) - 1)
        self.nx_1 = torch.clamp(torch.stack([ii - 1 for ii in self.nx0]), 0)
        self.nx2 = torch.clamp(torch.stack([ii + 2 for ii in self.nx0]), 0,
                               len(self.xmesh) - 1)

    def _fmat0th(self, zmesh: Tensor):
        """Construct f(0/1, 0/1) in fmat."""
        f00 = zmesh[self.arange_atp, self.nx0[..., 0], self.nx0[..., 1]]
        f10 = zmesh[self.arange_atp, self.nx1[..., 0], self.nx0[..., 1]]
        f01 = zmesh[self.arange_atp, self.nx0[..., 0], self.nx1[..., 1]]
        f11 = zmesh[self.arange_atp, self.nx1[..., 0], self.nx1[..., 1]]
        return f00, f10, f01, f11

    def _fmat1th(self, zmesh: Tensor, f00: Tensor, f10: Tensor, f01: Tensor,
                 f11: Tensor):
        """Get the 1st derivative of four grid points over x, y and xy."""
        f_10 = zmesh[self.arange_atp, self.nx_1[..., 0], self.nx0[..., 1]]
        f_11 = zmesh[self.arange_atp, self.nx_1[..., 0], self.nx1[..., 1]]
        f0_1 = zmesh[self.arange_atp, self.nx0[..., 0], self.nx_1[..., 1]]
        f02 = zmesh[self.arange_atp, self.nx0[..., 0], self.nx2[..., 1]]
        f1_1 = zmesh[self.arange_atp, self.nx1[..., 0], self.nx_1[..., 1]]
        f12 = zmesh[self.arange_atp, self.nx1[..., 0], self.nx2[..., 1]]
        f20 = zmesh[self.arange_atp, self.nx2[..., 0], self.nx0[..., 1]]
        f21 = zmesh[self.arange_atp, self.nx2[..., 0], self.nx1[..., 1]]

        # calculate the derivative: (F(1) - F(-1) / (2 * grid)
        fy00 = (f01 - f0_1) / (self.nx1[..., 1] - self.nx_1[..., 1])
        fy01 = (f02 - f00) / (self.nx2[..., 1] - self.nx0[..., 1])
        fy10 = (f11 - f1_1) / (self.nx1[..., 1] - self.nx_1[..., 1])
        fy11 = (f12 - f10) / (self.nx2[..., 1] - self.nx0[..., 1])
        fx00 = (f10 - f_10) / (self.nx1[..., 0] - self.nx_1[..., 0])
        fx01 = (f20 - f00) / (self.nx2[..., 0] - self.nx0[..., 0])
        fx10 = (f11 - f_11) / (self.nx1[..., 0] - self.nx_1[..., 0])
        fx11 = (f21 - f01) / (self.nx2[..., 0] - self.nx0[..., 0])
        fxy00, fxy11 = fy00 * fx00, fx11 * fy11
        fxy01, fxy10 = fx01 * fy01, fx10 * fy10
        return fy00, fy01, fy10, fy11, fx00, fx01, fx10, fx11, fxy00, fxy01, fxy10, fxy11


class Spline1d:
    """Polynomial natural (linear, cubic) non-periodic spline.

    Arguments:
        x: 1D Tensor variable.
        y: 1D (single) or 2D (batch) Tensor variable.

    Keyword Args:
        kind: Define spline method, 'cubic' or 'linear'.
        abcd: 0th, 1st, 2nd and 3rd order parameters in cubic spline.

    References:
        .. [wiki] https://en.wikipedia.org/wiki/Spline_(mathematics)

    Examples:
        >>> import tbmalt.common.maths.interpolator as interp
        >>> import torch
        >>> x = torch.linspace(1, 10, 10)
        >>> y = torch.sin(x)
        >>> fit = interp.Spline1d(x, y)
        >>> fit(torch.tensor([3.5]))
        >>> tensor([-0.3526])
        >>> torch.sin(torch.tensor([3.5]))
        >>> tensor([-0.3508])
    """

    def __init__(self, xx: Tensor, yy: Tensor, **kwargs):
        self.xp, self.yp = xx, yy
        self.kind = kwargs.get('kind', 'cubic')
        self.incr = self.xp[1] - self.xp[0]

        # Add points in the range (9.98, 10.98]
        xtail = torch.from_numpy(np.linspace(9.98, 10.98, 51))
        ninterp = 8
        delta_r = 1E-5
        l_tail = 1
        dr = xtail - self.xp[-2] - l_tail
        ilast = len(self.xp)

        # get grid points and grid point values
        xa = (ilast - ninterp + torch.arange(ninterp)) * self.incr
        yb = self.yp[ilast - ninterp - 1: ilast - 1]
        xa = xa.repeat(dr.shape[0]).reshape(dr.shape[0], -1)
        yb = yb.repeat(dr.shape[0]).reshape(dr.shape[0], -1)

        # get derivative
        y0 = poly_interp_2d(xa, yb, xa[:, ninterp - 1] - delta_r)
        y2 = poly_interp_2d(xa, yb, xa[:, ninterp - 1] + delta_r)
        y1 = self.yp[ilast - 2]
        y1p = (y2 - y0) / (2.0 * delta_r)
        y1pp = (y2 + y0 - 2.0 * y1) / (delta_r * delta_r)
        tail_ = poly5_zero(y1, y1p, y1pp, dr, -1.0 * l_tail)
        self.xp = torch.cat((self.xp[:-2], xtail))
        self.yp = torch.cat((self.yp[:-2], tail_))

        if self.kind == 'cubic':
            if kwargs.get('abcd') is not None:
                self.aa, self.bb, self.cc, self.dd = kwargs.get('abcd')
            else:
                self.aa, self.bb, self.cc, self.dd = self._get_abcd()
            self.abcd = pack([self.aa, self.bb, self.cc, self.dd])
        elif self.kind != 'linear':
            raise NotImplementedError('%s not implemented' % self.kind)

    def __call__(self, xnew: Tensor, abcd=None):
        """Evaluate the polynomial linear or cubic spline.

        Arguments:
            xnew: 0D Tensor.

        Returns:
            ynew: 0D Tensor.
        """
        # according to the order to choose spline method
        self.xnew = xnew if xnew.dim() == 1 else xnew.unsqueeze(0)
        self.knot = [ii in self.xp for ii in self.xnew]

        # boundary condition of xnew,  xp[0] < xnew < xp[-1], now xp[-1] = 10.98
        assert self.xnew.ge(self.xp[0]).all()
        self.mask = self.xnew.lt(self.xp[-1])

        # get the nearest grid point index of d in x
        self.dind = [
            bisect.bisect(self.xp.detach().numpy(), ii.detach().numpy()) - 1
            for ii in self.xnew[self.mask]]

        # generate new index for 2D yp batch
        self.ind = [torch.arange(len(self.mask)), self.dind] if \
            self.yp.dim() == 2 else self.dind

        if self.kind == 'cubic':
            return self._cubic(abcd)
        elif self.kind == 'linear':
            return self._linear()

    def _linear(self):
        """Calculate linear interpolation."""
        return self.yp[self.ind] + (self.xnew - self.xp[self.dind]) / (
            self.xp[1:] - self.xp[:-1])[self.dind] * (
            self.yp[..., 1:] - self.yp[..., :-1])[self.ind]

    def _cubic(self, abcd):
        """Calculate cubic spline interpolation."""
        if abcd is None:
            aa, bb, cc, dd = self.aa, self.bb, self.cc, self.dd
        else:
            aa, bb, cc, dd = abcd[0], abcd[1], abcd[2], abcd[3]

        # calculate a, b, c, d parameters, need input x and y
        dx = self.xnew[self.mask] - self.xp[self.dind]
        intergal_ = aa[self.ind] + bb[self.ind] * dx + \
            cc[self.ind] * dx ** 2 + dd[self.ind] * dx ** 3
        integral = torch.zeros(self.xnew.shape[0], *intergal_.shape[1:])
        integral[self.mask] = intergal_

        return integral

    def _get_abcd(self):
        """Get parameter aa, bb, cc, dd for cubic spline interpolation."""
        # get the first dim of x
        nx = self.xp.shape[0]
        assert nx > 3  # the length of x variable must > 3

        # get the differnce between grid points
        dxp = self.xp[1:] - self.xp[:-1]
        dyp = self.yp[..., 1:] - self.yp[..., :-1]

        # get b, c, d from reference website: step 3~9, first calculate c
        A = torch.zeros(nx, nx)
        A.diagonal()[1:-1] = 2 * (dxp[:-1] + dxp[1:])  # diag
        A[torch.arange(nx - 1), torch.arange(nx - 1) + 1] = dxp  # off-diag
        A[torch.arange(nx - 1) + 1, torch.arange(nx - 1)] = dxp
        A[0, 0], A[-1, -1] = 1., 1.
        A[0, 1], A[1, 0] = 0., 0.   # natural condition
        A[-1, -2], A[-2, -1] = 0., 0.

        B = torch.zeros(*self.yp.shape)
        B[..., 1:-1] = 3 * (dyp[..., 1:] / dxp[1:] - dyp[..., :-1] / dxp[:-1])
        B = B.permute(1, 0) if B.dim() == 2 else B

        cc, _ = torch.lstsq(B, A)
        cc = cc.permute(1, 0) if cc.squeeze().dim() == 2 else cc.squeeze()
        bb = dyp / dxp - dxp * (cc[..., 1:] + 2 * cc[..., :-1]) / 3
        dd = (cc[..., 1:] - cc[..., :-1]) / (3 * dxp)
        return self.yp, bb, cc, dd


class SKInterpolation:
    """Interpolation method for SK tables.

    Arguments:
        xx: Grid points of distances.
        yy: Integral tables.
    """

    def __init__(self, xx: Tensor, yy: Tensor):
        self.yy = yy
        self.incr = xx[1] - xx[0]
        self.ngridpoint = len(xx)

    def __call__(self, rr: Tensor, ninterp=8, delta_r=1E-5, tail=1) -> Tensor:
        """Interpolation SKF according to distance from integral tables.

        Arguments:
            rr: interpolation points for batch.
            ninterp: Number of total interpolation grid points.
            delta_r: Delta distance for 1st, 2nd derivative.
            tail: Distance to smooth the tail, unit is bohr.
        """
        ntail = int(tail / self.incr)
        rmax = (self.ngridpoint - 1) * self.incr + tail
        ind = (rr / self.incr).int()
        result = torch.zeros(rr.shape) if self.yy.dim() == 1 else torch.zeros(
            rr.shape[0], *self.yy.shape[1:])

        # thye input SKF must have more than 8 grid points
        if self.ngridpoint < ninterp + 1:
            raise ValueError("Not enough grid points for interpolation!")

        # => polynomial fit
        if (ind <= self.ngridpoint).any():
            _mask = ind <= self.ngridpoint

            # get the index of rr in grid points
            ind_last = (ind[_mask] + ninterp / 2 + 1).int()
            ind_last[ind_last > self.ngridpoint] = self.ngridpoint
            ind_last[ind_last < ninterp] = ninterp

            xa = (ind_last.unsqueeze(1) - ninterp + torch.arange(ninterp)
                  ) * self.incr  # get the interpolation gird points
            yb = torch.stack([self.yy[ii - ninterp - 1: ii - 1]
                              for ii in ind_last])  # grid point values
            result[_mask] = poly_interp_2d(xa, yb, rr[_mask])

        # Beyond the grid => extrapolation with polynomial of 5th order
        max_ind = self.ngridpoint - 1 + int(tail / self.incr)
        is_tail = ind.masked_fill(ind.ge(self.ngridpoint) * ind.le(max_ind), -1).eq(-1)
        if is_tail.any():
            dr = rr[is_tail] - rmax
            ilast = self.ngridpoint

            # get grid points and grid point values
            xa = (ilast - ninterp + torch.arange(ninterp)) * self.incr
            yb = self.yy[ilast - ninterp - 1: ilast - 1]
            xa = xa.repeat(dr.shape[0]).reshape(dr.shape[0], -1)
            yb = yb.unsqueeze(0).repeat_interleave(dr.shape[0], dim=0)

            # get derivative
            y0 = poly_interp_2d(xa, yb, xa[:, ninterp - 1] - delta_r)
            y2 = poly_interp_2d(xa, yb, xa[:, ninterp - 1] + delta_r)
            y1 = self.yy[ilast - 2]
            y1p = (y2 - y0) / (2.0 * delta_r)
            y1pp = (y2 + y0 - 2.0 * y1) / (delta_r * delta_r)

            if y1pp.dim() == 3:  # -> compression radii, not good
                dr = dr.repeat(y1pp.shape[1], y1pp.shape[2], 1).transpose(-1, 0)
            result[is_tail] = poly5_zero(y1, y1p, y1pp, dr, -1.0 * tail)

        return result


def smooth_tail_batch(xx: Tensor, yy: Tensor, n_grid: Tensor, ninterp=8,
                      delta_r=1E-5, tail=1.0):
    """Smooth the tail with input xx and yy."""
    assert xx.dim() == 2
    assert yy.dim() == 3
    assert n_grid.shape[0] == xx.shape[0]

    ilast = n_grid.clone()
    incr = xx[0, 1] - xx[0, 0]

    # get grid points and grid point values
    xa = (ilast.unsqueeze(1) - ninterp + torch.arange(ninterp)) * incr
    yb = torch.stack([yy[ii, il - ninterp - 1: il - 1]
                      for ii, il in enumerate(ilast)])

    # return smooth grid points in the tail for each SKF
    dr = -torch.linspace(4, 0, 5) * incr

    # get derivative
    y0 = poly_interp_2d(xa, yb, xa[:, ninterp - 1] - delta_r)
    y2 = poly_interp_2d(xa, yb, xa[:, ninterp - 1] + delta_r)
    y1 = yy[torch.arange(yy.shape[0]), ilast - 2]
    y1p = (y2 - y0) / (2.0 * delta_r)
    y1pp = (y2 + y0 - 2.0 * y1) / (delta_r * delta_r)
    # dr = dr.unsqueeze(1).unsqueeze(2).repeat(1, y0.shape[0], y0.shape[1])
    integral_tail = poly5_zero(
        y1.unsqueeze(-1), y1p.unsqueeze(-1), y1pp.unsqueeze(-1), dr, -tail).permute(0, 2, 1)

    for ii, iy in enumerate(yy):  # -> add tail
        yy[ii, n_grid[ii]: n_grid[ii] + 5] = integral_tail[ii]

    return yy


def poly5_zero(y0: Tensor, y0p: Tensor, y0pp: Tensor, xx: Tensor,
               dx: Tensor) -> Tensor:
    """Get integrals if beyond the grid range with 5th polynomial."""
    dx1 = y0p * dx
    dx2 = y0pp * dx * dx
    dd = 10.0 * y0 - 4.0 * dx1 + 0.5 * dx2
    ee = -15.0 * y0 + 7.0 * dx1 - 1.0 * dx2
    ff = 6.0 * y0 - 3.0 * dx1 + 0.5 * dx2
    xr = xx / dx
    yy = ((ff * xr + ee) * xr + dd) * xr * xr * xr
    return yy


def poly_interp_2d(xp: Tensor, yp: Tensor, rr: Tensor) -> Tensor:
    """Interpolate from DFTB+ (lib_math) with uniform grid.

    Arguments:
        xp: 2D tensor, 1st dimension if batch size, 2nd is grid points.
        yp: 2D tensor of integrals.
        rr: interpolation points.
    """
    nn0, nn1 = xp.shape[0], xp.shape[1]
    index_nn0 = torch.arange(nn0)
    icl = torch.zeros(nn0).long()
    cc, dd = yp.clone(), yp.clone()
    dxp = abs(rr - xp[index_nn0, icl])

    # find the most close point to rr (single atom pair or multi pairs)
    _mask, ii = torch.zeros(len(rr)) == 0, 0
    dxNew = abs(rr - xp[index_nn0, 0])
    while (dxNew < dxp).any():
        ii += 1
        assert ii < nn1 - 1  # index ii range from 0 to nn1 - 1
        _mask = dxNew < dxp
        icl[_mask] = ii
        dxp[_mask] = abs(rr - xp[index_nn0, ii])[_mask]

    yy = yp[index_nn0, icl]

    for mm in range(nn1 - 1):
        for ii in range(nn1 - mm - 1):
            rtmp0 = xp[index_nn0, ii] - xp[index_nn0, ii + mm + 1]

            # use transpose to realize div: (N, M, K) / (N)
            rtmp1 = ((cc[index_nn0, ii + 1] - dd[index_nn0, ii]).transpose(
                0, -1) / rtmp0).transpose(0, -1)
            cc[index_nn0, ii] = ((xp[index_nn0, ii] - rr) *
                                 rtmp1.transpose(0, -1)).transpose(0, -1)
            dd[index_nn0, ii] = ((xp[index_nn0, ii + mm + 1] - rr) *
                                 rtmp1.transpose(0, -1)).transpose(0, -1)
        if (2 * icl < nn1 - mm - 1).any():
            _mask = 2 * icl < nn1 - mm - 1
            yy[_mask] = (yy + cc[index_nn0, icl])[_mask]
        else:
            _mask = 2 * icl >= nn1 - mm - 1
            yy[_mask] = (yy + dd[index_nn0, icl - 1])[_mask]
            icl[_mask] = icl[_mask] - 1
    return yy
