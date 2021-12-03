"""Test SCC DFTB."""
import torch
import pytest
import numpy as np
from ase.build import molecule

from tbmalt import Geometry, Dftb1, Dftb2
from tbmalt.io import read_band
torch.set_printoptions(15)
torch.set_default_dtype(torch.float64)

shell_dict = {1: [0], 6: [0, 1], 7: [0, 1], 8: [0, 1],
              14: [0, 1], 22: [0, 1, 2]}


def test_h2_pe(device):
    """Test SCC DFTB for ch4 with periodic boundary condition."""
    h2 = molecule('H2')
    h2.cell = [6.0, 6.0, 6.0]
    kpoints = torch.tensor([1, 1, 1])
    geometry = Geometry.from_ase_atoms([h2])
    path_to_skf = './tests/unittests/data/slko/mio'

    dftb2 = Dftb2(geometry, shell_dict=shell_dict,
                  path_to_skf=path_to_skf, skf_type='skf', kpoints=kpoints)

    check_q = torch.max(abs(dftb2.charge - torch.tensor([[1.0, 1.0]]))) < 1E-14

    kpoints2 = torch.tensor([2, 2, 2])
    dftb2 = Dftb2(geometry, shell_dict=shell_dict,
                  path_to_skf=path_to_skf, skf_type='skf', kpoints=kpoints2)

    assert check_q, 'charge tolerance check'


def test_h2o_pe(device):
    """Test H2O DFTB from ase input."""
    h2o = molecule('H2O')
    h2o.cell = [6.0, 6.0, 6.0]
    kpoints = torch.tensor([1, 1, 1])
    kpoints2 = torch.tensor([2, 2, 2])
    geometry = Geometry.from_ase_atoms([h2o])
    path_to_skf = './tests/unittests/data/slko/mio'

    dftb1 = Dftb1(geometry, shell_dict=shell_dict,
                  path_to_skf=path_to_skf, skf_type='skf', kpoints=kpoints)
    dftb2 = Dftb2(geometry, shell_dict=shell_dict,
                  path_to_skf=path_to_skf, skf_type='skf', kpoints=kpoints)
    dftb22 = Dftb2(geometry, shell_dict=shell_dict,
                  path_to_skf=path_to_skf, skf_type='skf', kpoints=kpoints2)

    check_q1 = torch.max(abs(dftb1.charge - torch.tensor(
        [[6.760326589760, 0.619836705120, 0.619836705120]]))) < 1E-11
    check_q2 = torch.max(abs(dftb2.charge - torch.tensor(
        [[6.591506603167, 0.704246698416, 0.704246698416]]))) < 1E-10
    check_q22 = torch.max(abs(dftb22.charge - torch.tensor(
        [[6.591487673531, 0.704256163234, 0.704256163234]]))) < 1E-10
    assert check_q1, 'charge tolerance check'
    assert check_q2, 'charge tolerance check'
    assert check_q22, 'charge tolerance check'


@pytest.mark.skip(reason="Test SKF input too huge.")
def test_h2o_pe_vcr(device):
    """Test H2O with various compression radii."""
    path_to_skf = './tests/unittests/data/slko/vcr.h5'
    h2o = molecule('H2O')
    h2o.cell = [6.0, 6.0, 6.0]
    geometry = Geometry.from_ase_atoms([h2o])
    grids = torch.tensor([1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5., 6., 8., 10.])
    multi_varible = torch.tensor([[3.0, 3.0, 3.0]])
    dftb2 = Dftb2(geometry, shell_dict=shell_dict,
                  path_to_skf=path_to_skf, skf_type='h5', basis_type='vcr',
                  interpolation='BicubInterp',
                  grids=grids, multi_varible=multi_varible)

    check_q = torch.max(abs(dftb2.charge - torch.tensor([[
        6.591468709378842, 0.704265645310579, 0.704265645310579]]))) < 1E-3
    assert check_q, 'charge tolerance check'


def test_ch4_pe(device):
    """Test SCC DFTB for ch4 with periodic boundary condition."""
    ch4 = molecule('CH4')
    ch4.cell = [6.0, 6.0, 6.0]
    kpoints = torch.tensor([1, 1, 1])
    geometry = Geometry.from_ase_atoms([ch4])
    path_to_skf = './tests/unittests/data/slko/mio'
    dftb2 = Dftb2(geometry, shell_dict=shell_dict, path_to_skf=path_to_skf,
                  skf_type='skf')

    dftb2 = Dftb2(geometry, shell_dict=shell_dict, path_to_skf=path_to_skf,
                  skf_type='skf', kpoints=kpoints)

    check_q = torch.max(abs(dftb2.charge - torch.tensor([[
        4.305475062065351, 0.923631234483662, 0.923631234483662,
        0.923631234483662, 0.923631234483662]]))) < 1E-9
    assert check_q, 'charge tolerance check'


def test_c2h6_pe(device):
    """Test SCC DFTB for ch4 with periodic boundary condition."""
    ch4 = molecule('C2H6')
    ch4.cell = [6.0, 6.0, 6.0]
    kpoints = torch.tensor([1, 1, 1])
    geometry = Geometry.from_ase_atoms([ch4])
    path_to_skf = './tests/unittests/data/slko/mio'
    dftb2 = Dftb2(geometry, shell_dict=shell_dict,
                  path_to_skf=path_to_skf, skf_type='skf')

    dftb2 = Dftb2(geometry, shell_dict=shell_dict,
                  path_to_skf=path_to_skf, skf_type='skf', kpoints=kpoints)


def test_si_pe(device):
    """Test SCC DFTB for c2h6 with periodic boundary condition."""
    band = read_band('./tests/unittests/data/sk/si/band.out')
    bandd = read_band('./tests/unittests/data/sk/si/band.out.d')

    geometry = Geometry(
        torch.tensor([[14, 14]]),
        torch.tensor([[[0., 0.,  0.], [1.356773, 1.356773, 1.356773]]]),
        cell=torch.tensor([[
            [2.713546, 2.713546, 0.0], [0.0, 2.713546, 2.713546],
            [2.713546, 0.0, 2.713546]]]),
        units='angstrom')
    klines = torch.tensor([[0.5, 0.5, -0.5, 0], [0, 0, 0, 11],
                           [0, 0, 0.5, 11], [0.25, 0.25, 0.25, 11]])
    path_to_skf = './tests/unittests/data/slko'
    dftb2 = Dftb2(geometry, shell_dict=shell_dict,
                  path_to_skf=path_to_skf, skf_type='skf', klines=klines)

    shell_dict.update({14: [0, 1, 2]})
    dftb2d = Dftb2(geometry, shell_dict=shell_dict,
                   path_to_skf=path_to_skf, skf_type='skf', klines=klines)

    check_band = torch.max(abs(dftb2.eigenvalue.squeeze() - band.T)) < 1E-8
    check_bandd = torch.max(abs(dftb2d.eigenvalue.squeeze() - bandd.T)) < 1E-7

    assert check_band, 'eigenvalue tolerance check'
    assert check_bandd, 'eigenvalue tolerance check'


def test_tio2(device):
    """Test TiO2."""
    band = read_band('./tests/unittests/data/sk/tio2/band.out')
    band2 = read_band('./tests/unittests/data/sk/tio2/band.out.scc1')
    geometry = Geometry(
        torch.tensor([[22, 22, 8, 8, 8, 8]]),
        torch.tensor([[
            [2.783207184819172, -0.000000000099763, 1.344461013714925],
            [0.000000000000000, 0.000000000000000, 0.000000000000000],
            [4.542807941001271, 0.970773411850806, 1.344461013714925],
            [6.775397066812810, 3.737993010494626, 0.000000000000000],
            [1.023606428637073, -0.970773412050332, 1.344461013714925],
            [1.759600756182100, 0.970773411950569, 0.000000000000000]]]),
        cell=torch.tensor([[
            [5.566414370000000, 0.000000000000000, 0.000000000000000],
            [2.968583452994909, 4.708766422445195, 0.000000000000000],
            [-4.267498911859111, -2.354383211422124, 2.688922027429850]]]),
        units='angstrom')
    path_to_skf = './tests/unittests/data/slko/tiorg'
    kpoints1 = torch.tensor([[1, 1, 1]])
    kpoints3 = torch.tensor([[3, 3, 3]])
    klines = torch.tensor([[0.5, 0.5, -0.5, 0], [0.0, 0.0, 0.0, 10],
                           [0.0, 0.0, 0.5, 10], [0.25, 0.25, 0.25, 10]])
    charge111 = torch.tensor([[
        2.763957905122, 2.763957905122, 6.618021047439,
        6.618021047439, 6.618021047439, 6.618021047439]])

    dftb11 = Dftb1(geometry, shell_dict=shell_dict,
                   path_to_skf=path_to_skf, skf_type='skf', kpoints=kpoints1)

    assert torch.max(abs(charge111 - dftb11.charge)) < 1E-9
    assert torch.max(abs(band - dftb11.eigenvalue.squeeze())) < 1E-7

    dftb21 = Dftb2(geometry, shell_dict=shell_dict,
                   path_to_skf=path_to_skf, skf_type='skf', kpoints=kpoints1)
    dftb23 = Dftb2(geometry, shell_dict=shell_dict,
                   path_to_skf=path_to_skf, skf_type='skf', kpoints=kpoints3)
    dftb_band = Dftb1(geometry, shell_dict=shell_dict, path_to_skf=path_to_skf,
                      skf_type='skf', klines=klines, charge=dftb23.charge)

    assert torch.max(abs(band2 - dftb21.eigenvalue.squeeze())) < 1E-4
    assert torch.max(abs(dftb21.charge - torch.tensor([[
        3.058571734175, 3.058571734175, 6.470714132913, 6.470714132913,
        6.470714132913, 6.470714132913]]))) < 1E-5


def _get_matrix(filename, device):
    """Read DFTB+ hamsqr1.dat and oversqr.dat."""
    return torch.from_numpy(np.loadtxt(filename)).to(device)
