"""Test SCC DFTB."""
import torch
import os
import pytest
from torch.autograd import gradcheck
from ase.build import molecule
from tbmalt import Geometry, Basis, SkfFeed, Dftb1, Dftb2
from tbmalt.common.parameter import params
torch.set_printoptions(15)
torch.set_default_dtype(torch.float64)
shell_dict = {1: [0], 6: [0, 1], 7: [0, 1], 8: [0, 1]}


def test_ase_h2o(device):
    """Test H2O DFTB from ase input."""
    geometry = Geometry.from_ase_atoms([molecule('H2O')], device=device)
    basis = Basis(geometry.atomic_numbers, shell_dict)
    path_to_skf = './data/slko/mio'

    # dftb1 = Dftb1(params, geometry, shell_dict=shell_dict, skf_type='skf')
    # dftb1()
    # assert torch.max(abs(dftb1.charge - torch.tensor([
    #     6.760316843429209, 0.619841578285396, 0.619841578285396]))) < 1E-14

    dftb2 = Dftb2(params, geometry, shell_dict=shell_dict,
                  path_to_skf=path_to_skf, skf_type='skf')
    # dftb2()
    assert torch.max(abs(dftb2.charge - torch.tensor([
        6.587580500853424, 0.706209749573288, 0.706209749573288]))) < 1E-10


def test_ase_c2h6():
    """Test C2H6 DFTB from ase input."""
    c2h6 = Geometry.from_ase_atoms([molecule('C2H6')])
    path_to_skf = './data/slko/mio'

    dftb2 = Dftb2(params, c2h6, shell_dict=shell_dict,
                  path_to_skf=path_to_skf, skf_type='skf')
    assert torch.max(abs(dftb2.charge - torch.tensor([
        4.1916322572147706, 4.1916322572147635, 0.9361224815474558,
        0.9361226306188899, 0.9361226306188895, 0.9361224815474540,
        0.9361226306188882, 0.9361226306188880]))) < 1E-14


def test_batch_ase(device):
    """Test batch DFTB from ase input."""
    geo = Geometry.from_ase_atoms([molecule('CH4'),
                                   molecule('H2O'),
                                   molecule('C2H6')])
    path_to_skf = './data/slko/mio'

    # Perfrom DFTB calculations
    dftb2 = Dftb2(params, geo, shell_dict=shell_dict,
                  path_to_skf=path_to_skf, skf_type='skf')
    assert torch.max(abs(dftb2.charge - torch.tensor([
        [4.305343486193386, 0.923664128451654, 0.923664128451654,
         0.923664128451654, 0.923664128451654, 0., 0., 0.],
        [6.587580500853424, 0.706209749573288, 0.706209749573288,
         0., 0., 0., 0., 0.],
        [4.191613720237371, 4.191613720237375, 0.936128660536281,
         0.936128809613175, 0.936128809613175, 0.936128660536280,
         0.936128809613174, 0.936128809613174]]))) < 1E-10

test_batch_ase(torch.device('cpu'))

def test_scc_spline():
    """Test non-SCC DFTB from ase input."""
    molecule = System.from_ase_atoms([molecule_database('CH4')])
    sktable = IntegralGenerator.from_dir(
        './slko/auorg-1-1', molecule, interpolation='spline')
    skt = SKT(molecule, sktable)
    parameter = Parameter()
    properties = ['dipole']
    scc = Dftb2(molecule, skt, parameter, properties)
    sktable = IntegralGenerator.from_dir(
        './slko/auorg-1-1', molecule, repulsive=False, interpolation='spline', with_variable=True)
    skt = SKT(molecule, sktable, with_variable=True)
    parameter = Parameter()
    properties = ['dipole']
    scc = Dftb2(molecule, skt, parameter, properties)
    print('charge', scc.charge, 'dipole', scc.properties.dipole)


def test_read_compr_single(device):
    """Test SKF data with various compression radii."""
    molecule = System.from_ase_atoms([molecule_database('CH4')])
    compression_radii_grid = torch.tensor([
        01.00, 01.50, 02.00, 02.50, 03.00, 03.50, 04.00,
        04.50, 05.00, 05.50, 06.00, 07.00, 08.00, 09.00, 10.00])
    compression_r = torch.tensor([[3., 3.1, 3.2, 3.3, 3.4, 0, 0, 0]])
    sk = IntegralGenerator.from_dir(
        './slko/compr', molecule, repulsive=True,
        sk_type='compression_radii', homo=False, interpolation='bicubic_interpolation',
        compression_radii_grid=compression_radii_grid)

    skt = SKT(molecule, sk, compression_radii=compression_r, fix_onsite=True,
              fix_U=True)
    parameter = Parameter()
    properties = ['dipole']
    scc = Dftb2(molecule, skt, parameter, properties)


def test_read_compr_batch(device):
    """Test SKF data with various compression radii."""
    molecule = System.from_ase_atoms([molecule_database('CH4'),
                                      molecule_database('NH3'),
                                      molecule_database('C2H6')])
    compression_radii_grid = torch.tensor([
        01.00, 01.50, 02.00, 02.50, 03.00, 03.50, 04.00,
        04.50, 05.00, 05.50, 06.00, 07.00, 08.00, 09.00, 10.00])
    compression_r = torch.tensor([[3., 3.1, 3.2, 3.3, 3.4, 0, 0, 0],
                                  [3., 3., 3., 3., 0., 0., 0., 0],
                                  [3., 3., 3., 3., 3., 3.5, 3.5, 3.5]])
    sk = IntegralGenerator.from_dir(
        './slko/compr', molecule, repulsive=True,
        sk_type='compression_radii', homo=False, interpolation='bicubic_interpolation',
        compression_radii_grid=compression_radii_grid)

    skt = SKT(molecule, sk, compression_radii=compression_r, fix_onsite=True,
              fix_U=True)
    parameter = Parameter()
    properties = ['dipole']
    scc = Dftb2(molecule, skt, parameter, properties)
