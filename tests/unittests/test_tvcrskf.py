#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""""""
import os
import torch
import h5py
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import interpn
import matplotlib.pyplot as plt
from tbmalt.io.skf import TvcrSkf, Skf
from tbmalt.common.maths.interpolation import MultiVarInterp


def test_load():
    path = '/home/gz_fan/Downloads/test/work/twoparams/H_H_denwav/H_H_denwav'
    file_list = [os.path.join(path, isk) for isk in os.listdir(path)]
    path_homo = ['./data/slko/mio/H-H.skf']
    vcrskf = TvcrSkf.read(file_list, path_homo, dtype=torch.float64,
                         smooth_to_zero=False, write=True, overwrite=True)

def test_interp():
    grid = torch.tensor([2., 2.5, 3., 4., 5., 7., 10.])
    path = '/home/gz_fan/Downloads/test/work/twoparams/test/HH/H-H.skf'
    sk = Skf.read(path)
    with h5py.File('./tmp.db', 'r') as f:
        H = f['H-H']['integrals']['H'][()][0]
        H2 = torch.from_numpy(f['H-H']['integrals']['H'][()][0])

        interp = MultiVarInterp(grid.repeat(4, 1), torch.from_numpy(H), n_dims=4)
        y = interp(torch.tensor([3.5, 3.5, 3.5, 3.5]))
        ref = sk.hamiltonian[(0, 0)][0]

        N, bpred = 20, []
        _g = grid.numpy()
        linear = interpn((_g, _g, _g, _g), H, np.array([3.5, 3.5, 3.5, 3.5]))

        # bspline
        for i in range(1, N):
            bspline = TensorSpline(torch.from_numpy(H).squeeze()[..., i],
                                   grid, grid, grid, grid)
            bpred.append(bspline(torch.tensor([3.5]), torch.tensor([3.5]),
                            torch.tensor([3.5]), torch.tensor([3.5])))

        x = torch.linspace(1, N-1, N-1) * 0.2
        # plt.plot(ref[1:N], ref[1:N], label='skgen reference')
        # plt.plot(ref[1:N], y.squeeze()[1:N])
        plt.plot(x, torch.zeros(N-1), 'k')
        plt.plot(x, ref[1:N]-torch.from_numpy(linear).squeeze()[1:N], label='linear')
        plt.plot(x, ref[1:N]-torch.tensor(bpred), label='bspline')
        # plt.plot(x, ref[1:N]-H2[3, 3, 3, 3, 0, 1:N], label='neighbouring 1')
        # plt.plot(x, ref[1:N]-H2[4, 4, 4, 4, 0, 1:N], label='neighbouring 2')
        plt.legend()
        plt.ylabel('error (reference from skgen)')
        plt.xlabel('distances (Bohr)')
        plt.show()


class TensorSpline:
    """Providing routines for multivariate interpolation with tensor product
    splines.

    The number of nodes has to be at least 4 in every dimension. Otherwise a
    singularity error will occur during  calculation.

    Arguments:
        f_nodes (Tensor): Values at the given sites.
        x_nodes (Tensor): x-nodes of the interpolation data.
        *args (Tensor): y-, z-, ... nodes. Every set of nodes is a
            tensor.

    Notes:
        The theory of the univariate b-splines were taken from [3]_. The
        mathematical background about the calculations of the coefficients
        during a multivariate interpolation were taken from [4]_.

    References:
        .. [3] Boor, C.d. 2001, "A Practical Guide to Splines". Rev. ed.
           Springer-Verlag
        .. [4] Floater, M.S. (2007) "Tensor Product Spline Surfaces"[Online].
           Available at: https://www.uio.no/studier/emner/matnat/ifi/nedlagte-emner/INF-MAT5340/v07/undervisningsmateriale/kap7.pdf
           (Accessed: 01 December 2020)
    """

    def __init__(self, f_nodes: torch.Tensor, x_nodes: torch.Tensor,
                 *args: torch.Tensor):
        # Dimensions will be called the following: x, y, z, a, b, c, d, ...
        # f_nodes has shape (x, y, z, a, b, ...)
        self.f_nodes = f_nodes

        self.nodes = [x_nodes, *args]

        self.device = self.f_nodes.device

        self.num = len(self.nodes)

        # Permute f_nodes: (z, a, b, ... , x, y)
        # print(f_nodes.shape, tuple(torch.arange(self.num).roll(-2)))
        self.f_nodes = f_nodes.permute(tuple(torch.arange(self.num).roll(-2)))

        # Calculate a list containing the knot-vectors of every dimension
        self.knot_vec = [self.get_knot_vector(nodes) for nodes in self.nodes]

        # Calculate a list containing tensors of b-splines for every
        # dimension.
        self.bsplines = [self.get_bsplines(self.knot_vec[ii], self.nodes[ii],
                                           self.nodes[ii], 4)
                         for ii in range(self.num)]

        # Create cyclic permutation for the coefficients.
        roll_permutation = torch.arange(self.num).roll(-1)

        _dd = self.f_nodes
        for ii in self.bsplines:
            _dd = torch.solve(_dd, ii.T).solution.permute(*roll_permutation)

        # c has now the following shape: (z, a, b, ... , x, y)
        self.cc = _dd

    def get_knot_vector(self, nodes: torch.Tensor) -> torch.Tensor:
        """Calculates the corresponding knot vector for the given nodes.

        Arguments:
            nodes (Tensor): Nodes of which the knot vector should be
                calculated.

        Returns:
            tt (Tensor): Knot vector for the given nodes.
        """

        tt = torch.zeros((len(nodes) + 4,), device=self.device)
        tt[0:4] = nodes[0]
        tt[4:-4] = nodes[2:-2]
        tt[-4:] = nodes[-1]

        return tt

    def get_bsplines(self, tt: torch.Tensor, xx: torch.Tensor,
                     nodes: torch.Tensor, kk: int) -> torch.Tensor:
        r"""Calculates a tensor containing the values of the b-splines at the
        given sites.

        Assume that: :math:`\text{len}(x) = m, \text{len}(x_{\text{nodes}}) =
        n`. The tensor has dimensions: :math:`(m, n)`. The b-splines are
        row-vectors. Hence the tensor has the following structure:

        .. math::

           \begin{matrix}
           B_1(x_1) & ... & B_1(x_m)\\
           \vdots &  & \vdots\\
           B_n(x_1) & ... & B_n(x_m)
           \end{matrix}

        Arguments:
            tt (Tensor): Knot vector of the corresponding nodes.
            xx (Tensor): Values where the b-splines should be evaluated.
            nodes (Tensor): Interpolation nodes.
            kk (int): Order of the b-splines. Order of `k` means that the
                b-splines have degree `k`-1.

        Returns:
            b_tensor (Tensor): Tensor containing the values of the
                b-splines at the corresponding x-values.
        """

        j_num = torch.arange(0, len(tt) - kk, device=self.device)

        # Calculate a tensor containing the b-splines for every dimension
        b_tensor = [self._b_spline(tt, xx, nodes, jj, kk) for jj
                    in j_num]

        b_tensor = torch.stack(b_tensor)

        return b_tensor

    def _b_spline(self, tt: torch.Tensor, xx: torch.Tensor,
                  nodes: torch.Tensor, jj: int, kk: int) -> torch.Tensor:
        r"""Calculates the b-spline for a given knot vector.

        It calculates the y-values of the `j`th b-spline of order `k`.
        The b-spline will be calculated for the knot vector `t`. The
        calculation follows the recurrence relation:

        .. math::

           B_{j, 1} = 1 if t_j \leq x < t_{jj + 1}, 0 \text{otherwise}
           B_{j, k} = \frac{x - t_j}{t_{j + k - 1} - t_j} B_{j, k-1}
           + (1 - \frac{x - t_{j + 1}}{t_{j + k} -
           t_{j + 1}} B_{j + 1, k-1}

        Arguments:
            tt (Tensor): Knot vector. Tensor containing the knots. You
                need at least `k`+1 knots.
            xx (Tensor): Tensor containing the x-values where the
                b-spline should be evaluated.
            jj (int): Specifies which b-spline should be calculated.
            kk (int): Specifies the order of the b-spline. Order of `k` means
                that the calculated polynomial has degree `k-1`.

        Returns:
            yy (Tensor): Tensor containing the y-values of the `j`th
                b-spline of order `k` for the corresponding x-values.
        """

        t1 = tt[jj]
        t2 = tt[jj + 1]
        t3 = tt[jj + kk - 1]
        t4 = tt[jj + kk]

        if kk == 1:
            yy = torch.where((tt[jj] <= xx) & (xx < tt[jj + 1]), 1, 0)
            if jj == len(tt) - 4:
                yy = torch.where((tt[jj] <= xx) & (xx <= tt[jj + 1]), 1, 0)
            if len(nodes) == 2 and jj == 1:
                yy = torch.where((tt[jj] <= xx) & (xx <= tt[jj + 1]), 1, 0)
        else:
            # Here the recursion formula will be executed. The 'if' and 'else'
            # blocks ensure that one avoid the division by zero.
            if tt[jj + kk - 1] == tt[jj] and tt[jj + kk] == tt[jj + 1]:
                yy = self._b_spline(tt, xx, nodes, jj + 1, kk - 1)
            elif tt[jj + kk - 1] == tt[jj] and tt[jj + kk] != tt[jj + 1]:
                yy = (1 - (xx - t2) / (t4 - t2)) * \
                     self._b_spline(tt, xx, nodes, jj + 1, kk - 1)
            elif tt[jj + kk - 1] != tt[jj] and tt[jj + kk] == tt[jj + 1]:
                yy = (xx - t1) / (t3 - t1) * \
                     self._b_spline(tt, xx, nodes, jj, kk - 1) \
                     + self._b_spline(tt, xx, nodes, jj + 1, kk - 1)
            else:
                yy = (xx - t1) / (t3 - t1) * \
                     self._b_spline(tt, xx, nodes, jj, kk - 1) \
                     + (1 - (xx - t2) / (t4 - t2)) * \
                     self._b_spline(tt, xx, nodes, jj + 1, kk - 1)
        return yy

    def __call__(self, x_new: torch.Tensor, *args: torch.Tensor, grid=True)\
            -> torch.Tensor:
        """Evaluates the spline function at the desired sites.

        Arguments:
            x_new (Tensor): x-values, where you want to evaluate the spline
                function.
            *args (Tensor): New values for the other dimensions.
            grid (bool): You can decide whether you want to evaluate the
                results on a grid or not. If `grid=True` (default) the grid is
                spanned by the input tensors. If `grid=False` the spline
                function will be evaluated at single points specified by the
                rows of one single input tensor with dimension of 2 or by the
                values of multiple tensors.

        Returns:
            ff (Tensor): Tensor containing the values at the given sites.
        """

        if len(x_new.size()) == 1:
            new_vals = [x_new, *args]
        else:
            new_vals = [x_new[:, ii] for ii in range(x_new.shape[1])]

        inverted_num = torch.arange(0, self.num).flip(0)

        # matrices contains the b-spline tensors but in inverted order.
        matrices = [self.get_bsplines(self.knot_vec[ii], new_vals[ii],
                                      self.nodes[ii], 4).T
                    for ii in inverted_num]

        # permutation1: cyclic permutation.
        # Permute dd so it has shape: (y, z, a, b, ..., x)
        dd = self.cc.permute(*torch.arange(self.num).roll(1))

        permutation = [-2, *range(self.num - 2), -1]

        for ii in range(self.num):
            if ii == self.num - 1:
                # For the last multiplication permute dd so the shapes matches
                # properly
                dd = dd.transpose(-1, -2)
            dd = torch.matmul(matrices[ii], dd)
            dd = dd.permute(permutation)

        # Final permutation of f so it has the same shape as f_nodes in the
        # input: (x, y, z, a, b, ...).
        ff = dd

        if not grid:
            for ii in range(self.num - 1):
                ff = torch.diagonal(ff)

        return ff


test_interp()
