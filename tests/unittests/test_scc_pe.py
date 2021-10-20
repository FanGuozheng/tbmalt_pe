"""Test SCC DFTB."""
import torch
import os
import pytest
from ase.build import molecule

from tbmalt import Geometry, Basis, SkfFeed, SkfParamFeed, Dftb1, Dftb2
from tbmalt.structures.periodic import Periodic
from tbmalt.common.parameter import params
torch.set_printoptions(15)
torch.set_default_dtype(torch.float64)
shell_dict = {1: [0], 6: [0, 1], 7: [0, 1], 8: [0, 1]}


def test_ase_h2o_pe(device):
    """Test H2O DFTB from ase input."""
    h2o = molecule('H2O')
    h2o.cell = [6.0, 6.0, 6.0]
    geometry = Geometry.from_ase_atoms([h2o])
    basis = Basis(geometry.atomic_numbers, shell_dict)
    path_to_skf = './data/slko/mio'
    dftb2 = Dftb2(params, geometry, shell_dict=shell_dict,
                  path_to_skf=path_to_skf, skf_type='skf')
    assert torch.max(abs(dftb2.charge - torch.tensor([[
        6.591468709378842, 0.704265645310579, 0.704265645310579]]))) < 1E-4

    path_to_skf = '../train/vcr.h5'
    grids = torch.tensor([1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5., 6., 8., 10.])
    multi_varible = torch.tensor([[3.0, 3.0, 3.0]])
    dftb2 = Dftb2(params, geometry, shell_dict=shell_dict,
                  path_to_skf=path_to_skf, skf_type='h5', basis_type='vcr',
                  interpolation='BicubInterp',
                  grids=grids, multi_varible=multi_varible)
    assert torch.max(abs(dftb2.charge - torch.tensor([[
        6.591468709378842, 0.704265645310579, 0.704265645310579]]))) < 1E-4


def test_scc_ch4_pe(device):
    """Test SCC DFTB for ch4 with periodic boundary condition."""
    ch4 = molecule('CH4')
    ch4.cell = [6.0, 6.0, 6.0]
    geometry = Geometry.from_ase_atoms([ch4])
    basis = Basis(geometry.atomic_numbers, shell_dict)
    path_to_skf = './data/slko/mio'
    dftb2 = Dftb2(params, geometry, shell_dict=shell_dict,
                  path_to_skf=path_to_skf, skf_type='skf')
    print(dftb2.charge)
    assert torch.max(abs(dftb2.charge - torch.tensor([[
        4.305475062065351, 0.923631234483662, 0.923631234483662,
         0.923631234483662, 0.923631234483662]]))) < 1E-4


def test_batch_pe(device):
    """Test SCC DFTB for c2h6 with periodic boundary condition."""
    mol = [molecule('H2O'), molecule('CH4')]
    for im in mol:
        im.cell = [6.0, 6.0, 6.0]
    geometry = Geometry.from_ase_atoms(mol)
    basis = Basis(geometry.atomic_numbers, shell_dict)
    path_to_skf = './data/slko/mio'
    dftb2 = Dftb2(params, geometry, shell_dict=shell_dict,
                  path_to_skf=path_to_skf, skf_type='skf')
    assert torch.max(abs(dftb2.charge - torch.tensor([[
        6.591468709378842, 0.704265645310579, 0.704265645310579, 0., 0.],
        [4.305475062065351, 0.923631234483662, 0.923631234483662,
         0.923631234483662, 0.923631234483662]]
        ))) < 1E-8, 'Tolerance check'


test_ase_h2o_pe(torch.device('cpu'))

def test_batch_pe_2():
    """Test scc batch calculation."""
    latvec = [torch.tensor([[4., 4., 0.], [5., 0., 5.], [0., 6., 6.]]),
              torch.tensor([[4., 0., 0.], [0., 4., 0.], [0., 0., 4.]]),
              torch.tensor([[5., 0., 0.], [0., 5., 0.], [0., 0., 5.]]),
              torch.tensor([[99., 0., 0.], [0., 99., 0.], [0., 0., 99.]])]
    cutoff = torch.tensor([9.98])
    positions = [torch.tensor([
        [3., 3., 3.], [3.6, 3.6, 3.6], [2.4, 3.6, 3.6], [3.6, 2.4, 3.6], [3.6, 3.6, 2.4]]),
         torch.tensor([[0., 0., 0.], [0., 2., 0.]]),
         torch.tensor([[0.965, 0.075, 0.088], [1.954, 0.047, 0.056], [2.244, 0.660, 0.778]]),
         torch.tensor([[0.965, 0.075, 0.088], [1.954, 0.047, 0.056], [2.244, 0.660, 0.778]])]
    numbers = [torch.tensor([6, 1, 1, 1, 1]), torch.tensor([1, 1]),
               torch.tensor([1, 8, 1]), torch.tensor([1, 8, 1])]
    molecule = System(numbers, positions, latvec)
    sktable = IntegralGenerator.from_dir('./slko/mio-1-1', molecule)
    periodic = Periodic(molecule, molecule.cell, cutoff=cutoff)
    skt = SKT(molecule, sktable, periodic)
    coulomb = Coulomb(molecule, periodic, method='search')
    parameter = Parameter()
    scc = Scc(molecule, skt, parameter, coulomb, periodic)
    assert torch.max(abs(scc.charge - torch.tensor([[
            4.6122976812946259, 0.83320615382097674, 0.85273810385371818,
            0.85182728982744738, 0.84993077120323290],
            [1.000000000000000, 1.000000000000000, 0.000000000000000,
             0.000000000000000, 0.000000000000000],
            [0.70282850018606047, 6.5936446382800851, 0.70352686153385458,
             0.000000000000000, 0.000000000000000],
            [0.70794447853157250, 6.5839848726758881, 0.70807064879254611,
             0.000000000000000, 0.000000000000000]]))) < 1E-8, 'Tolerance check'
