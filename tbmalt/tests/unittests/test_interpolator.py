"""Performs tests on functions in the tbmalt.common.maths.interpolator."""
import torch
from torch.autograd import gradcheck
import pytest
from tbmalt.tests.test_utils import *
from scipy.interpolate import interp1d, interp2d
from tbmalt.common.maths.interpolator import SKInterpolation, Spline1d, BicubInterp
torch.set_default_dtype(torch.float64)
torch.set_printoptions(15)

##################################
# TBMaLT.common.maths.Spline1d #
##################################
@fix_seed
def test_polyspline_cubic_single(device):
    x = torch.linspace(1, 10, 10)
    y = torch.rand(10, device=device)
    fit = Spline1d(x, y)
    pred = [fit(ii) for ii in torch.tensor([1.5, 5, 8.5])]
    ref = [0.8632885619144542, 0.645024120122765, 0.5224871031995648]
    error = max([abs(ii - jj) for ii, jj in zip(ref, pred)])
    assert error < 1E-12

@fix_seed
def test_polyspline_cubic_batch(device):
    x = torch.linspace(1, 10, 10)
    y = torch.rand(3, 10, device=device)
    fit = Spline1d(x, y)
    pred = fit(torch.tensor([1.5, 5, 8.5]))
    ref = [0.8632885619144542, 0.481848400781709, 0.7634905376056228]
    error = max([abs(ii - jj) for ii, jj in zip(ref, pred)])
    assert error < 1E-12

@fix_seed
def test_polyspline_linear_single(device):
    x = torch.linspace(1, 10, 10)
    y = torch.rand(10, device=device)
    fit = Spline1d(x, y, kind='linear')
    pred = [fit(ii) for ii in torch.tensor([1.5, 5, 8.5])]
    ref = [0.8389364331031705, 0.645024120122765, 0.46622427210075146]
    error = max([abs(ii - jj) for ii, jj in zip(ref, pred)])
    assert error < 1E-12

@fix_seed
def test_polyspline_linear_batch(device):
    x = torch.linspace(1, 10, 10)
    y = torch.rand(3, 10, device=device)
    fit = Spline1d(x, y, kind='linear')
    pred = fit(torch.tensor([1.5, 5, 8.5]))
    ref = [0.8389364331031705, 0.481848400781709, 0.7136609231288971]
    error = max([abs(ii - jj) for ii, jj in zip(ref, pred)])
    assert error < 1E-12

@pytest.mark.grad
@fix_seed
def test_polyspline_grad(device):
    """Gradient evaluation of maths.sym function."""
    x = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.])
    y = torch.rand(10, device=device)
    fit = Spline1d(x, y)
    xi = torch.tensor([0.6], requires_grad=True)
    print("gradcheck(PolySpline, (x, y), raise_exception=False)",
          gradcheck(fit, xi, raise_exception=False))
    grad_is_safe = gradcheck(fit, xi, raise_exception=False)
    assert grad_is_safe, 'Gradient stability test'


##################################
# TBMaLT.common.maths.BicubInterp #
##################################
def test_bicubic(device):
    """Test bicub interpolation, reference data is from skgen."""
    mesh = torch.tensor([2., 2.5, 3., 3.5, 4., 4.5, 5.])
    # C-C 7th column, 2.4 angstrom with compression radii in mesh
    cc = torch.tensor([
        [-.09372742397236, -.1029017477537, -.1106843505678, -.11721676770750,
         -.1224776690475, -.1266236843412, -.12987648184730],
        [-.10290174775370, -.1143789924960, -.1237463331506, -.13136245657760,
         -.1373963019333, -.1421211304664, -.14582423450440],
        [-.11068435056780, -.1237463331506, -.1341703696824, -.14248641933590,
         -.1490047084563, -.1540834266064, -.15805724441820],
        [-.11721676770750, -.1313624565776, -.1424864193359, -.15124911396030,
         -.1580644273014, -.1633521720510, -.16748135080120],
        [-.12247766904750, -.1373963019333, -.1490047084563, -.15806442730140,
         -.1650671904758, -.1704791849583, -.17469560402620],
        [-.12662368434120, -.1421211304664, -.1540834266064, -.16335217205100,
         -.1704791849583, -.1759667364147, -.18023105518010],
        [-.12987648184730, -.1458242345044, -.1580572444182, -.16748135080120,
         -.1746956040262, -.1802310551801, -.18452129647520]])
    zmesh = cc  # torch.stack([torch.stack([cc, cc]), torch.stack([cc, cc])])
    interp = BicubInterp(mesh.squeeze(), zmesh)

    # interpolation point, choose the middle between grid points
    xi = torch.tensor([[2.25, 2.25], [3.25, 3.25], [4.25, 4.25]])
    ref = torch.tensor([-1.040054565948E-01, -1.431125379623E-01, -1.708458454207E-01])
    # interp(xi)
    error = max([abs(interp(ix) - ref[ii]) for ii, ix in enumerate(xi)])
    interp_b = BicubInterp(mesh, torch.stack([zmesh, zmesh, zmesh]))
    # assert error < 1E-3

@pytest.mark.grad
@fix_seed
def test_polyspline_grad(device):
    """Gradient evaluation of maths.sym function."""
    x = torch.tensor([[1., 2., 3., 4., 5., 6., 7., 8., 9., 10.],
                      [1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]])
    z = torch.rand(2, 2, 10, 10, device=device)
    fit = BicubInterp(x, z)
    xi = torch.tensor([5.], requires_grad=True)
    # grad_is_safe = gradcheck(fit, xi, raise_exception=False)
    # assert grad_is_safe, 'Gradient stability test'


def test_sk_interp(device):
    x = torch.tensor([[1., 2., 3., 4., 5., 6., 7., 8., 9., 10.],
                      [1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]])
    y = torch.rand(2, 2, 10, 10, device=device)
    SKInterpolation(x, y)