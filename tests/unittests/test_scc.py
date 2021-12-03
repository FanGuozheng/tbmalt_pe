"""Test SCC DFTB."""
import torch
import pytest
from torch.autograd import gradcheck
from ase.build import molecule
from tbmalt import Geometry, Dftb1, Dftb2
torch.set_printoptions(15)
torch.set_default_dtype(torch.float64)

shell_dict = {1: [0], 6: [0, 1], 7: [0, 1], 8: [0, 1]}


def test_h2o(device):
    """Test H2O DFTB from ase input."""
    # Standard SCC-DFTB calculations
    geometry = Geometry.from_ase_atoms([molecule('H2O')], device=device)
    path_to_skf = './tests/unittests/data/slko/mio'

    dftb1 = Dftb1(geometry, shell_dict=shell_dict,
                  path_to_skf=path_to_skf, skf_type='skf')
    dftb2 = Dftb2(geometry, shell_dict=shell_dict,
                  path_to_skf=path_to_skf, skf_type='skf')

    assert torch.max(abs(dftb1.charge - torch.tensor([
        6.760316843429, 0.619841578285, 0.619841578285]))) < 1E-10
    assert torch.max(abs(dftb2.charge - torch.tensor([
        6.587580500853424, 0.706209749573288, 0.706209749573288]))) < 1E-10

    # Standard SCC-DFTB calculations with binary SKF
    path_to_skf = './tests/unittests/data/slko/mio.hdf'

    dftb2 = Dftb2(geometry, shell_dict=shell_dict,
                  path_to_skf=path_to_skf, skf_type='h5')
    assert torch.max(abs(dftb2.charge - torch.tensor([
        6.587580500853424, 0.706209749573288, 0.706209749573288]))) < 1E-10


@pytest.mark.skip(reason="Test SKF input too huge.")
def test_h2o_var(device):
    geometry = Geometry.from_ase_atoms([molecule('H2O')], device=device)

    # 1.1 Test basis with one variable
    # all wavefunction compression radii set as 3.5
    path_to_skf = './tests/unittests/data/slko/vcr.h5'
    grids = torch.tensor([1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5., 6., 8., 10.])
    compr = torch.ones(*geometry.atomic_numbers.shape) * 3.5
    dftb2 = Dftb2(geometry, shell_dict=shell_dict,
                  path_to_skf=path_to_skf, skf_type='h5', basis_type='vcr',
                  interpolation='BicubInterp', grids=grids, multi_varible=compr)
    assert torch.max(abs(dftb2.charge - torch.tensor([
        6.622131504505886, 0.688934247747058, 0.688934247747058]))) < 1E-5

    # 2.2 set all wavefunction compression radii as 2.75
    compr = torch.ones(*geometry.atomic_numbers.shape) * 2.75
    dftb2 = Dftb2(geometry, shell_dict=shell_dict,
                  path_to_skf=path_to_skf, skf_type='h5', basis_type='vcr',
                  interpolation='BicubInterp', grids=grids, multi_varible=compr)
    assert torch.max(abs(dftb2.charge - torch.tensor([
        6.587862806962196, 0.706068596518902, 0.706068596518902]))) < 1E-3

    # 2.3 set wavefunction compression radii for O and H different
    compr = torch.tensor([[2.3, 3.0, 3.0]])
    dftb2 = Dftb2(geometry, shell_dict=shell_dict,
                  path_to_skf=path_to_skf, skf_type='h5', basis_type='vcr',
                  interpolation='BicubInterp', grids=grids, multi_varible=compr)
    assert torch.max(abs(dftb2.charge - torch.tensor([
        6.588870121994031, 0.705564939002985, 0.705564939002985]))) < 5E-3

    # 2.1 Test basis with two variable: density compression radii and
    # wavefunction compression radii, both are set the same here
    path_to_skf = './tests/unittests/data/slko/tvcr.h5'
    grids = torch.tensor([2., 2.5, 3., 4., 5., 7., 10.])
    compr = torch.ones(*geometry.atomic_numbers.shape, 2) * 3.5
    dftb2 = Dftb2(geometry, shell_dict=shell_dict,
                  path_to_skf=path_to_skf, skf_type='h5', basis_type='tvcr',
                  interpolation='BSpline', grids=grids, multi_varible=compr)
    assert torch.max(abs(dftb2.charge - torch.tensor([
        6.662066383899750, 0.668966808050125, 0.668966808050125]))) < 1E-4
    # 2.2 set all compression radii as 2.75
    compr = torch.ones(*geometry.atomic_numbers.shape, 2) * 2.75
    dftb2 = Dftb2(geometry, shell_dict=shell_dict,
                  path_to_skf=path_to_skf, skf_type='h5', basis_type='tvcr',
                  interpolation='BSpline', grids=grids, multi_varible=compr)
    assert torch.max(abs(dftb2.charge - torch.tensor([
        6.659845691910212, 0.670077154044895, 0.670077154044895]))) < 1E-4

    # 2.3 set density, compression radii differently
    compr = torch.tensor([[[8.0, 2.3], [2.5, 3.0], [2.5, 3.0]]])
    dftb2 = Dftb2(geometry, shell_dict=shell_dict,
                  path_to_skf=path_to_skf, skf_type='h5', basis_type='tvcr',
                  interpolation='BSpline', grids=grids, multi_varible=compr)
    assert(abs(dftb2.charge - torch.tensor([
        6.588870121994031, 0.705564939002985, 0.705564939002985]))) < 5E-3


def test_batch(device):
    """Test batch DFTB from ase input."""
    ref1 = torch.tensor(
        [[4.305343486193386, 0.923664128451654, 0.923664128451654,
          0.923664128451654, 0.923664128451654, 0., 0., 0.],
         [6.587580500853424, 0.706209749573288, 0.706209749573288,
          0., 0., 0., 0., 0.],
         [4.191613720237371, 4.191613720237375, 0.936128660536281,
          0.936128809613175, 0.936128809613175, 0.936128660536280,
          0.936128809613174, 0.936128809613174]])
    geometry = Geometry.from_ase_atoms([
        molecule('CH4'), molecule('H2O'), molecule('C2H6')])

    # 1. Perfrom DFTB calculations with standard SKF input
    path_to_skf = './tests/unittests/data/slko/mio'
    dftb2 = Dftb2(geometry, shell_dict=shell_dict,
                  path_to_skf=path_to_skf, skf_type='skf')
    assert torch.max(abs(dftb2.charge - ref1)) < 1E-10

    geometry = Geometry.from_ase_atoms([
        molecule('CH4'), molecule('H2O'), molecule('C2H6')])

    # 1.1 Perfrom DFTB calculations with standard SKF input
    path_to_skf = './tests/unittests/data/slko/mio'
    dftb2 = Dftb2(geometry, shell_dict=shell_dict,
                  path_to_skf=path_to_skf, skf_type='skf')

    # 2. Perfrom DFTB calculations with h5 SK input
    path_to_skf = './tests/unittests/data/slko/mio.hdf'
    dftb2 = Dftb2(geometry, shell_dict=shell_dict,
                  path_to_skf=path_to_skf, skf_type='h5')
    assert torch.max(abs(dftb2.charge - ref1)) < 1E-10


@pytest.mark.skip(reason="Test SKF input too huge.")
def test_batch_vcr(device):
    geometry = Geometry.from_ase_atoms([
        molecule('CH4'), molecule('H2O'), molecule('C2H6')])

    # 1. Test basis with two variable: density compression radii and
    # wavefunction compression radii, both are set the same here
    path_to_skf = '/tests/unittests/data/slko/tvcr.h5'
    grids = torch.tensor([2., 2.5, 3., 4., 5., 7., 10.])
    compr = torch.ones(*geometry.atomic_numbers.shape, 2) * 3.5
    dftb2 = Dftb2(geometry, shell_dict=shell_dict,
                  path_to_skf=path_to_skf, skf_type='h5', basis_type='tvcr',
                  interpolation='BSpline', grids=grids, multi_varible=compr)
    assert torch.max(abs(dftb2.charge - torch.tensor([
        [4.411561510172980, 0.897109622456755, 0.897109622456755,
         0.897109622456755, 0.897109622456755, 0., 0., 0.],
        [6.662066383899750, 0.668966808050125, 0.668966808050125,
         0., 0., 0., 0., 0.],
        [4.251468958474244, 4.251468958474247, 0.916176893075964,
         0.916177074224895, 0.916177074224895, 0.916176893075966,
         0.916177074224895, 0.916177074224895]]))) < 1E-4

    # 2. set each element specie density and wavefunction radii different
    init_dict = {1: torch.tensor([2.5, 3.0]),
                 6: torch.tensor([7.0, 2.7]),
                 7: torch.tensor([8.0, 2.2]),
                 8: torch.tensor([8.0, 2.3])}
    unique_atomic_numbers = geometry.unique_atomic_numbers()
    compr[:] = 0
    for ii, iu in enumerate(unique_atomic_numbers):
        mask = geometry.atomic_numbers == iu
        compr[mask] = init_dict[iu.tolist()]
    dftb2 = Dftb2(geometry, shell_dict=shell_dict,
                  path_to_skf=path_to_skf, skf_type='h5', basis_type='tvcr',
                  interpolation='BSpline', grids=grids, multi_varible=compr)
    assert torch.max(abs(dftb2.charge - torch.tensor([
            [4.305378934388628, 0.923655266402844, 0.923655266402843,
             0.923655266402843, 0.923655266402844, 0., 0., 0.],
            [6.588870121994031, 0.705564939002985, 0.705564939002986,
             0., 0., 0., 0., 0.],
            [4.191632595649794, 4.191632595649796, 0.936122368740296,
             0.936122517804954, 0.936122517804955, 0.936122368740296,
             0.936122517804955, 0.936122517804955]]))) < 1E-3
