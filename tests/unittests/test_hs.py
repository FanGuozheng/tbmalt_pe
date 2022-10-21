# -*- coding: utf-8 -*-
"""Perform tests on functions which generate Hamiltonian and overlap matrices.

The tests on reading SKF is in test_sk_read.py, here the tests will test
Hamiltonian and overlap matrices compare with DFTB+ results. The tests cover
the different orbitals, interpolation methods and effect of `max_l`.
"""
import re
import torch
import pytest
import numpy as np
from ase.build import molecule
from torch.autograd import gradcheck
from tbmalt import Geometry, Basis, SkfFeed, SkfParamFeed
from tbmalt.physics.dftb.slaterkoster import hs_matrix
from tbmalt.structures.periodic import Periodic
from tbmalt.common.batch import pack
torch.set_default_dtype(torch.float64)
torch.set_printoptions(6)

shell_dict = {1: [0], 6: [0, 1], 7: [0, 1], 8: [0, 1],
              14: [0, 1], 22: [0, 1, 2]}


def _get_matrix(filename, device, in_type='txt'):
    """Read DFTB+ hamsqr1.dat and oversqr.dat."""
    if in_type == 'txt':
        return torch.from_numpy(np.loadtxt(filename)).to(device)
    else:
        text = ''.join(open(filename, 'r').readlines())
        string = re.search('(?<=MATRIX\n).+(?=\n)', text, flags=re.DOTALL).group(0)
        return torch.tensor([[float(i) for i in row.split()]
                             for row in string.split('\n')]).to(device)


def test_hs_single_npe(device):
    """Test single Hamiltonian and overlap after SK transformations."""
    h_ch4 = _get_matrix('./tests/unittests/data/sk/ch4/hamsqr1.dat', device)
    s_ch4 = _get_matrix('./tests/unittests/data/sk/ch4/oversqr.dat', device)
    path_sk = './tests/unittests/data/slko/mio'
    mol = molecule('CH4')
    geometry = Geometry.from_ase_atoms(mol, device=device)
    basis = Basis(geometry.atomic_numbers, shell_dict)

    # build single Hamiltonian and overlap feeds
    h_feed = SkfFeed.from_dir(
        path_sk, shell_dict, skf_type='skf',
        geometry=geometry, interpolation='PolyInterpU', integral_type='H')
    s_feed = SkfFeed.from_dir(
        path_sk, shell_dict, skf_type='skf',
        geometry=geometry, interpolation='PolyInterpU', integral_type='S')

    ham = hs_matrix(geometry, basis, h_feed)
    over = hs_matrix(geometry, basis, s_feed)
    check_h = torch.max(abs(ham - h_ch4)) < 1E-14
    check_s = torch.max(abs(over - s_ch4)) < 1E-14
    check_persistence_h = ham.device == device
    check_persistence_s = over.device == device

    assert check_h, 'Hamiltonian are outside of permitted tolerance thresholds'
    assert check_s, 'Overlap are outside of permitted tolerance thresholds'
    assert check_persistence_h, 'Device persistence check failed'
    assert check_persistence_s, 'Device persistence check failed'

    # build Hamiltonian and overlap feeds with python interpolation
    geometry2 = Geometry.from_ase_atoms(mol, device=torch.device('cpu'))
    h_feed2 = SkfFeed.from_dir(
        path_sk, shell_dict, skf_type='skf',
        geometry=geometry2, interpolation='CubicSpline', integral_type='H')
    s_feed2 = SkfFeed.from_dir(
        path_sk, shell_dict, skf_type='skf',
        geometry=geometry2, interpolation='CubicSpline', integral_type='S')

    ham2 = hs_matrix(geometry2, basis, h_feed2)
    over2 = hs_matrix(geometry2, basis, s_feed2)

    check_h2 = torch.max(abs(ham2 - h_ch4.to('cpu'))) < 5E-9
    check_s2 = torch.max(abs(over2 - s_ch4.to('cpu'))) < 1E-10
    check_persistence_h2 = ham2.device == torch.device('cpu')
    check_persistence_s2 = over2.device == torch.device('cpu')

    assert check_h2, 'Hamiltonian tolerance failed'
    assert check_s2, 'Overlap tolerance failed'
    assert check_persistence_h2, 'Device persistence check failed'
    assert check_persistence_s2, 'Device persistence check failed'

    # build only Hamiltonian feed
    h_feed3 = SkfFeed.from_dir(
        path_sk, shell_dict, skf_type='skf',
        geometry=geometry, interpolation='PolyInterpU', integral_type='H')

    ham3 = hs_matrix(geometry, basis, h_feed3)
    check_h3 = torch.max(abs(ham3 - h_ch4)) < 1E-14
    check_persistence_h3 = ham3.device == device

    assert check_h3, 'Hamiltonian are outside of permitted tolerance thresholds'
    assert check_persistence_h3, 'Device persistence check failed'

    # build only Overlap feed
    s_feed4 = SkfFeed.from_dir(
        path_sk, shell_dict, skf_type='skf',
        geometry=geometry, interpolation='PolyInterpU', integral_type='S')

    over4 = hs_matrix(geometry, basis, s_feed4)
    check_s4 = torch.max(abs(over4 - s_ch4)) < 1E-14
    check_persistence_s4 = over4.device == device

    assert check_s4, 'Hamiltonian are outside of permitted tolerance thresholds'
    assert check_persistence_s4, 'Device persistence check failed'


def test_h2_pe(device):
    """Test single Hamiltonian and overlap after SK transformations."""
    h_h2 = _get_matrix('./tests/unittests/data/sk/h2/hamsqr1.dat.pe', device)
    s_h2 = _get_matrix('./tests/unittests/data/sk/h2/oversqr.dat.pe', device)
    h_h22 = _get_matrix('./tests/unittests/data/sk/h2/hamsqr1.dat.pe2', device)
    s_h22 = _get_matrix('./tests/unittests/data/sk/h2/oversqr.dat.pe2', device)
    path_sk = './tests/unittests/data/slko/mio'
    kpoints = torch.tensor([1, 1, 1])
    kpoints2 = torch.tensor([1, 2, 2])

    # Define positions all positive to make sure all atoms in the same cell
    geometry = Geometry(torch.tensor([[1, 1]]), torch.tensor([[
        [0, 0, 0], [0, 0, 0.6]]]), cell=torch.eye(3).unsqueeze(0) * 6.0,
        units='angstrom')
    basis = Basis(geometry.atomic_numbers, shell_dict)
    geometry2 = Geometry(
        torch.tensor([[1, 1, 1]]), torch.tensor([[
            [0, 0, 0], [0, 0, 0.6], [0, 0, 1.2]]]),
        cell=torch.eye(3).unsqueeze(0) * 6.0,
        units='angstrom')
    basis2 = Basis(geometry2.atomic_numbers, shell_dict)

    # build single Hamiltonian and overlap feeds
    h_feed = SkfFeed.from_dir(
        path_sk, shell_dict, skf_type='skf',
        geometry=geometry, interpolation='PolyInterpU', integral_type='H')
    s_feed = SkfFeed.from_dir(
        path_sk, shell_dict, skf_type='skf',
        geometry=geometry, interpolation='PolyInterpU', integral_type='S')
    skparams = SkfParamFeed.from_dir(path_sk, geometry, skf_type='skf')
    periodic = Periodic(geometry, geometry.cell, cutoff=skparams.cutoff,
                        kpoints=kpoints)
    periodic2 = Periodic(geometry2, geometry2.cell, cutoff=skparams.cutoff,
                         kpoints=kpoints2)

    ham = hs_matrix(periodic, basis, h_feed, n_kpoints=1, cutoff=skparams.cutoff + 1.0)
    over = hs_matrix(periodic, basis, s_feed, n_kpoints=1, cutoff=skparams.cutoff + 1.0)
    ham2 = hs_matrix(periodic2, basis2, h_feed, n_kpoints=4, cutoff=skparams.cutoff + 1.0)
    over2 = hs_matrix(periodic2, basis2, s_feed, n_kpoints=4, cutoff=skparams.cutoff + 1.0)

    ind_r = torch.arange(0, h_h2.shape[-1], 2)
    ind_r2 = torch.arange(0, h_h22.shape[-1], 2)
    check_h_r = torch.max(abs(ham.real.squeeze() - h_h2[..., ind_r])) < 1E-14
    check_s_r = torch.max(abs(over.real.squeeze() - s_h2[..., ind_r])) < 1E-11
    check_ih_r = torch.max(
        abs(ham.imag.squeeze() - h_h2[..., ind_r + 1])) < 1E-14
    check_is_r = torch.max(abs(
        over.imag.squeeze() - s_h2[..., ind_r + 1])) < 1E-11
    check_h_r2 = torch.max(
        abs(ham2.real.squeeze()[..., 0] - h_h22[:3, ind_r2])) < 1E-14
    check_ih_r2 = torch.max(
        abs(ham2.imag.squeeze()[..., 0] - h_h22[:3, ind_r2 + 1])) < 1E-14
    check_s_r2 = torch.max(
        abs(over2.real.squeeze()[..., 0] - s_h22[:3, ind_r2])) < 1E-11
    # print(ham2.imag.squeeze()[..., 0], '\n', h_h22[:3, ind_r2 + 1])
    check_persistence_h = ham.device == device
    check_persistence_s = over.device == device

    assert check_h_r, 'tolerance check'
    assert check_s_r, 'tolerance check'
    assert check_ih_r, 'tolerance check'
    assert check_is_r, 'tolerance check'
    assert check_h_r2, 'tolerance check'
    assert check_ih_r2, 'tolerance check'
    assert check_s_r2, 'tolerance check'
    assert check_persistence_h, 'device check'
    assert check_persistence_s, 'device check'


def test_h2o_pe(device):
    """Test single Hamiltonian and overlap after SK transformations."""
    h_h2o = _get_matrix('./tests/unittests/data/sk/h2o/hamsqr1.dat.pe', device)
    s_h2o = _get_matrix('./tests/unittests/data/sk/h2o/oversqr.dat.pe', device)
    h_h2 = _get_matrix('./tests/unittests/data/sk/h2/hamsqr1.dat.pe', device)
    s_h2 = _get_matrix('./tests/unittests/data/sk/h2/oversqr.dat.pe', device)
    path_sk = './tests/unittests/data/slko/mio'
    kpoints = torch.tensor([[1, 1, 1], [1, 1, 1]])
    geometry = Geometry(torch.tensor([[8, 1, 1], [1, 1, 0]]), torch.tensor([[
        [0, 0, 0], [0, 0.6, 0], [0, 0., 0.6]], [[0, 0, 0], [0, 0, 0.6], [0, 0, 0]]]),
        cell=torch.eye(3).unsqueeze(0).repeat(2, 1, 1) * 6.0, units='angstrom')
    basis = Basis(geometry.atomic_numbers, shell_dict)

    # build single Hamiltonian and overlap feeds
    h_feed = SkfFeed.from_dir(
        path_sk, shell_dict, skf_type='skf',
        geometry=geometry, interpolation='PolyInterpU', integral_type='H')
    s_feed = SkfFeed.from_dir(
        path_sk, shell_dict, skf_type='skf',
        geometry=geometry, interpolation='PolyInterpU', integral_type='S')
    skparams = SkfParamFeed.from_dir(path_sk, geometry, skf_type='skf')
    periodic = Periodic(geometry, geometry.cell, cutoff=skparams.cutoff,
                        kpoints=kpoints)

    ham = hs_matrix(periodic, basis, h_feed, n_kpoints=1, cutoff=skparams.cutoff + 1.0)
    over = hs_matrix(periodic, basis, s_feed, n_kpoints=1, cutoff=skparams.cutoff + 1.0)

    ind_h2o = torch.arange(0, h_h2o.shape[-1], 2)
    ind_h2 = torch.arange(0, h_h2.shape[-1], 2)
    check_h_h2o = torch.max(abs(
        ham.real.squeeze()[0] - h_h2o[..., ind_h2o])) < 1E-12
    check_s_h2o = torch.max(abs(
        over.real.squeeze()[0] - s_h2o[..., ind_h2o])) < 1E-11
    check_ih_h2o = torch.max(abs(
        ham.imag.squeeze()[0] - h_h2o[..., ind_h2o + 1])) < 1E-12
    check_is_h2o = torch.max(abs(
        over.imag.squeeze()[0] - s_h2o[..., ind_h2o + 1])) < 1E-11
    check_is_h2o = torch.max(abs(
        over.imag.squeeze()[0] - s_h2o[..., ind_h2o + 1])) < 1E-11
    check_h_h2 = torch.max(abs(
        ham.real.squeeze()[1, :2, :2] - h_h2[..., ind_h2][:2, :2])) < 1E-14
    check_s_h2 = torch.max(abs(
        over.real.squeeze()[1, :2, :2] - s_h2[..., ind_h2][:2, :2])) < 1E-11
    check_persistence_h = ham.device == device
    check_persistence_s = over.device == device

    assert check_h_h2o, 'real H tolerance check'
    assert check_s_h2o, 'real S tolerance check'
    assert check_ih_h2o, 'imaginary H tolerance check'
    assert check_is_h2o, 'imaginary S tolerance check'
    assert check_h_h2, 'real H tolerance check'
    assert check_s_h2, 'real S tolerance check'
    assert check_persistence_h, 'device persistence_ check'
    assert check_persistence_s, 'device persistence_ check'


def test_si_pe(device):
    """Test single Hamiltonian and overlap after SK transformations."""
    hr = _get_matrix('./tests/unittests/data/sk/si/hamsqr1.dat', device)
    sr = _get_matrix('./tests/unittests/data/sk/si/oversqr.dat', device)
    hr_pe = _get_matrix('./tests/unittests/data/sk/si/hamsqr1.dat.pe', device)
    sr_pe = _get_matrix('./tests/unittests/data/sk/si/oversqr.dat.pe', device)
    hr_pe2 = _get_matrix('./tests/unittests/data/sk/si/hamsqr1.dat.pe2', device)
    sr_pe2 = _get_matrix('./tests/unittests/data/sk/si/oversqr.dat.pe2', device)
    hr_pe3 = _get_matrix('./tests/unittests/data/sk/si/hamsqr1.dat.pe3', device)
    sr_pe3 = _get_matrix('./tests/unittests/data/sk/si/oversqr.dat.pe3', device)
    hr_pe5 = _get_matrix('./tests/unittests/data/sk/si/hamsqr1.dat.pe5', device)
    sr_pe5 = _get_matrix('./tests/unittests/data/sk/si/oversqr.dat.pe5', device)

    path_sk = './tests/unittests/data/slko'
    kpoints = torch.tensor([[1, 1, 1]])
    kpoints3 = torch.tensor([[3, 3, 3]])
    kpoints4 = torch.tensor([[3, 3, 3], [1, 2, 2]])
    klines = torch.tensor([[0.5, 0.5, -0.5, 0, 0, 0, 11]])

    # Non-periodic geometry
    geometry = Geometry(
        torch.tensor([[14, 14]]),
        torch.tensor([[
            [0., 0.,  0.], [0.1356773E+01, 0.13567730000E+01, 0.13567730000E+01]]]),
        units='angstrom')
    basis = Basis(geometry.atomic_numbers, shell_dict)

    # Periodic geometry
    geometry_pe = Geometry(
        torch.tensor([[14]]),
        torch.tensor([[
            [0., 0.,  0.]]]),
        cell=torch.tensor([[
            [6.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 6.0]]]),
        units='angstrom')
    basis_pe = Basis(geometry_pe.atomic_numbers, shell_dict, geometry_pe)

    # Periodic geometry
    geometry_pe2 = Geometry(
        torch.tensor([[14, 14]]),
        torch.tensor([[[0., 0.,  0.], [1.356773, 1.356773, 1.356773]]]),
        cell=torch.tensor([[
            [2.713546, 2.713546, 0.0], [0.0, 2.713546, 2.713546],
            [2.713546, 0.0, 2.713546]]]),
        units='angstrom')
    basis_pe2 = Basis(geometry_pe2.atomic_numbers, shell_dict)

    # Batch periodic
    geometry_pe4 = Geometry(
        torch.tensor([[14, 14], [14, 0]]),
        torch.tensor([[[0., 0.,  0.], [1.356773, 1.356773, 1.356773]],
                      [[0., 0.,  0.], [0, 0, 0]]]),
        cell=torch.tensor([[
            [2.713546, 2.713546, 0.0], [0.0, 2.713546, 2.713546],
            [2.713546, 0.0, 2.713546]],
            [[6.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 6.0]]]),
        units='angstrom')
    basis_pe4 = Basis(geometry_pe4.atomic_numbers, shell_dict)

    # Klines
    geometry_pe5 = Geometry(
        torch.tensor([[14, 14]]),
        torch.tensor([[[0., 0.,  0.], [1.356773, 1.356773, 1.356773]]]),
        cell=torch.tensor([[
            [2.713546, 2.713546, 0.0], [0.0, 2.713546, 2.713546],
            [2.713546, 0.0, 2.713546]]]),
        units='angstrom')
    basis_pe5 = Basis(geometry_pe5.atomic_numbers, shell_dict)

    # build single Hamiltonian and overlap feeds
    h_feed = SkfFeed.from_dir(
        path_sk, shell_dict, skf_type='skf',
        geometry=geometry, interpolation='PolyInterpU', integral_type='H')
    s_feed = SkfFeed.from_dir(
        path_sk, shell_dict, skf_type='skf',
        geometry=geometry, interpolation='PolyInterpU', integral_type='S')
    skparams = SkfParamFeed.from_dir(path_sk, geometry, skf_type='skf')
    periodic = Periodic(geometry_pe, geometry_pe.cell, cutoff=skparams.cutoff,
                        kpoints=kpoints)
    periodic2 = Periodic(geometry_pe2, geometry_pe2.cell,
                         cutoff=skparams.cutoff, kpoints=kpoints)
    periodic3 = Periodic(geometry_pe2, geometry_pe2.cell,
                         cutoff=skparams.cutoff, kpoints=kpoints3)
    periodic4 = Periodic(geometry_pe4, geometry_pe4.cell,
                         cutoff=skparams.cutoff, kpoints=kpoints4)
    periodic5 = Periodic(geometry_pe5, geometry_pe5.cell,
                         cutoff=skparams.cutoff, klines=klines)

    ham = hs_matrix(geometry, basis, h_feed)
    over = hs_matrix(geometry, basis, s_feed)

    h_pe = hs_matrix(periodic, basis_pe, h_feed, n_kpoints=1, cutoff=skparams.cutoff)

    s_pe = hs_matrix(periodic, basis_pe, s_feed, n_kpoints=1, cutoff=skparams.cutoff + 1.0)
    h_pe2 = hs_matrix(periodic2, basis_pe2, h_feed, n_kpoints=1, cutoff=skparams.cutoff + 1.0)
    s_pe2 = hs_matrix(periodic2, basis_pe2, s_feed, n_kpoints=1, cutoff=skparams.cutoff + 1.0)
    h_pe3 = hs_matrix(periodic3, basis_pe2, h_feed, n_kpoints=27, cutoff=skparams.cutoff + 1.0)
    s_pe3 = hs_matrix(periodic3, basis_pe2, s_feed, n_kpoints=27, cutoff=skparams.cutoff + 1.0)
    h_pe4 = hs_matrix(periodic4, basis_pe4, h_feed, n_kpoints=torch.tensor([27, 4]), cutoff=skparams.cutoff + 1.0)
    s_pe4 = hs_matrix(periodic4, basis_pe4, s_feed, n_kpoints=torch.tensor([27, 4]), cutoff=skparams.cutoff + 1.0)
    h_pe5 = hs_matrix(periodic5, basis_pe5, h_feed, n_kpoints=11, cutoff=skparams.cutoff + 1.0)
    s_pe5 = hs_matrix(periodic5, basis_pe5, s_feed, n_kpoints=11, cutoff=skparams.cutoff + 1.0)

    ind_r = torch.arange(0, hr.shape[-1], 2)
    ind_r2 = torch.arange(0, hr_pe2.shape[-1], 2)
    check_h = torch.max(abs(ham.squeeze() - hr)) < 1E-14
    check_s = torch.max(abs(over.squeeze() - sr)) < 1E-14
    check_h_pe = torch.max(abs(
        h_pe.real.squeeze() - hr_pe[..., ind_r])) < 1E-14
    check_h_ipe = torch.max(abs(
        h_pe.imag.squeeze() - hr_pe[..., ind_r + 1])) < 1E-14
    check_s_pe = torch.max(abs(
        s_pe.real.squeeze() - sr_pe[..., ind_r])) < 1E-14
    check_s_ipe = torch.max(abs(
        s_pe.imag.squeeze() - sr_pe[..., ind_r + 1])) < 1E-14
    check_h_pe2 = torch.max(abs(
        h_pe2.real.squeeze() - hr_pe2[..., ind_r2])) < 1E-12
    check_h_ipe2 = torch.max(abs(
        h_pe2.imag.squeeze() - hr_pe2[..., ind_r2 + 1])) < 1E-12
    check_s_pe2 = torch.max(abs(
        s_pe2.real.squeeze() - sr_pe2[..., ind_r2])) < 1E-11
    check_s_ipe2 = torch.max(abs(
        s_pe2.imag.squeeze() - sr_pe2[..., ind_r2 + 1])) < 1E-11
    check_h_pe31 = torch.max(abs(
        h_pe3.real.squeeze()[..., 0] - hr_pe3[:8, ind_r2])) < 1E-12
    check_s_pe31 = torch.max(abs(
        s_pe3.real.squeeze()[..., 0] - sr_pe3[:8, ind_r2])) < 1E-11
    check_h_ipe31 = torch.max(abs(
        h_pe3.imag.squeeze()[..., 0] - hr_pe3[:8, ind_r2 + 1])) < 1E-12
    check_h_pe32 = torch.max(abs(
        h_pe3.real.squeeze()[..., 1] - hr_pe3[8:16, ind_r2])) < 1E-12
    check_s_pe32 = torch.max(abs(
        s_pe3.real.squeeze()[..., 1] - sr_pe3[8:16, ind_r2])) < 1E-11
    check_ih_pe32 = torch.max(abs(
        h_pe3.imag.squeeze()[..., 1] - hr_pe3[8:16, ind_r2 + 1])) < 1E-12
    check_is_pe32 = torch.max(abs(
        s_pe3.imag.squeeze()[..., 0] - sr_pe3[:8, ind_r2 + 1])) < 1E-11
    check_h_pe33 = torch.max(abs(
        h_pe3.real.squeeze()[..., 2] - hr_pe3[16:24, ind_r2])) < 1E-12
    check_s_pe33 = torch.max(abs(
        s_pe3.real.squeeze()[..., 2] - sr_pe3[16:24, ind_r2])) < 1E-11
    check_h_pe41 = torch.max(abs(
        h_pe4.real.squeeze()[0, :, :, 0] - hr_pe3[:8, ind_r2])) < 1E-12
    check_h_pe42 = torch.max(abs(
        h_pe4.real.squeeze()[0, :, :, 1] - hr_pe3[8:16, ind_r2])) < 1E-12
    check_s_pe41 = torch.max(abs(
        s_pe4.real.squeeze()[0, :, :, 0] - sr_pe3[:8, ind_r2])) < 1E-11
    check_s_pe42 = torch.max(abs(
        s_pe4.real.squeeze()[0, :, :, 1] - sr_pe3[8:16, ind_r2])) < 1E-11
    check_ih_pe41 = torch.max(abs(
        h_pe4.imag.squeeze()[0, :, :, 0] - hr_pe3[:8, ind_r2 + 1])) < 1E-12
    check_ih_pe42 = torch.max(abs(
        h_pe4.imag.squeeze()[0, :, :, 1] - hr_pe3[8:16, ind_r2 + 1])) < 1E-12
    check_is_pe41 = torch.max(abs(
        s_pe4.imag.squeeze()[0, :, :, 0] - sr_pe3[:8, ind_r2 + 1])) < 1E-11
    check_is_pe42 = torch.max(abs(
        s_pe4.imag.squeeze()[0, :, :, 1] - sr_pe3[8:16, ind_r2 + 1])) < 1E-11

    check_h_pe51 = torch.max(abs(
        h_pe5.real.squeeze()[..., 0] - hr_pe5[:8, ind_r2])) < 1E-12
    check_s_pe51 = torch.max(abs(
        s_pe5.real.squeeze()[..., 0] - sr_pe5[:8, ind_r2])) < 1E-11
    check_h_ipe51 = torch.max(abs(
        h_pe5.imag.squeeze()[..., 0] - hr_pe5[:8, ind_r2 + 1])) < 1E-12
    check_s_ipe51 = torch.max(abs(
        s_pe5.imag.squeeze()[..., 0] - hr_pe5[:8, ind_r2 + 1])) < 1E-12
    check_h_pe52 = torch.max(abs(
        h_pe5.real.squeeze()[..., 1] - hr_pe5[8:16, ind_r2])) < 1E-12
    check_s_pe52 = torch.max(abs(
        s_pe5.real.squeeze()[..., 1] - sr_pe5[8:16, ind_r2])) < 1E-11
    check_h_ipe52 = torch.max(abs(
        h_pe5.imag.squeeze()[..., 1] - hr_pe5[8:16, ind_r2 + 1])) < 1E-12
    check_s_ipe52 = torch.max(abs(
        s_pe5.imag.squeeze()[..., 1] - sr_pe5[8:16, ind_r2 + 1])) < 1E-11
    check_h_pe53 = torch.max(abs(
        h_pe5.real.squeeze()[..., 2] - hr_pe5[16:24, ind_r2])) < 1E-12
    check_s_pe53 = torch.max(abs(
        s_pe5.real.squeeze()[..., 2] - sr_pe5[16:24, ind_r2])) < 1E-11
    check_h_ipe53 = torch.max(abs(
        h_pe5.imag.squeeze()[..., 2] - hr_pe5[16:24, ind_r2 + 1])) < 1E-12
    check_s_ipe53 = torch.max(abs(
        s_pe5.imag.squeeze()[..., 2] - sr_pe5[16:24, ind_r2 + 1])) < 1E-11
    check_h_pe56 = torch.max(abs(
        h_pe5.real.squeeze()[..., 5] - hr_pe5[40:48, ind_r2])) < 1E-12
    check_s_pe56 = torch.max(abs(
        s_pe5.real.squeeze()[..., 5] - sr_pe5[40:48, ind_r2])) < 1E-11
    check_h_ipe56 = torch.max(abs(
        h_pe5.imag.squeeze()[..., 5] - hr_pe5[40:48, ind_r2 + 1])) < 1E-12
    check_s_ipe56 = torch.max(abs(
        s_pe5.imag.squeeze()[..., 5] - sr_pe5[40:48, ind_r2 + 1])) < 1E-11

    check_h_pe5_1 = torch.max(abs(
        h_pe5.real.squeeze()[..., -1] - hr_pe5[-8:, ind_r2])) < 1E-11
    check_s_pe5_1 = torch.max(abs(
        s_pe5.real.squeeze()[..., -1] - sr_pe5[-8:, ind_r2])) < 5E-10
    check_h_ipe5_1 = torch.max(abs(
        h_pe5.imag.squeeze()[..., -1] - hr_pe5[-8:, ind_r2 + 1])) < 1E-11
    check_s_ipe5_1 = torch.max(abs(
        s_pe5.imag.squeeze()[..., -1] - hr_pe5[-8:, ind_r2 + 1])) < 5E-10

    check_persistence_h = ham.device == device
    check_persistence_s = over.device == device
    check_persistence_h_pe = h_pe.device == device
    check_persistence_s_pe = s_pe.device == device

    assert check_h, 'real H tolerance check'
    assert check_s, 'real S tolerance check'
    assert check_h_pe, 'real H tolerance check'
    assert check_h_ipe, 'real H tolerance check'
    assert check_s_pe, 'real S tolerance check'
    assert check_s_ipe, 'real S tolerance check'
    assert check_h_pe2, 'real H tolerance check'
    assert check_h_ipe2, 'real H tolerance check'
    assert check_s_pe2, 'real S tolerance check'
    assert check_s_ipe2, 'real S tolerance check'
    assert check_h_pe31, 'real H tolerance check'
    assert check_h_ipe31, 'real H tolerance check'
    assert check_s_pe31, 'real S tolerance check'
    assert check_h_pe32, 'real H tolerance check'
    assert check_ih_pe32, 'imaginary H tolerance check'
    assert check_is_pe32, 'imaginary S tolerance check'
    assert check_s_pe32, 'real S tolerance check'
    assert check_h_pe33, 'real H tolerance check'
    assert check_s_pe33, 'real S tolerance check'
    assert check_h_pe41, 'real H tolerance check'
    assert check_h_pe42, 'real H tolerance check'
    assert check_s_pe41, 'real S tolerance check'
    assert check_s_pe42, 'real S tolerance check'
    assert check_ih_pe41, 'imaginary H tolerance check'
    assert check_ih_pe42, 'imaginary H tolerance check'
    assert check_is_pe41, 'imaginary S tolerance check'
    assert check_is_pe42, 'imaginary S tolerance check'

    assert check_h_pe51
    assert check_s_pe51
    assert check_h_ipe51
    assert check_s_ipe51
    assert check_h_pe52
    assert check_s_pe52
    assert check_h_ipe52
    assert check_s_ipe52
    assert check_h_pe53
    assert check_s_pe53
    assert check_h_ipe53
    assert check_s_ipe53
    assert check_h_pe56
    assert check_s_pe56
    assert check_h_ipe56
    assert check_s_ipe56
    assert check_h_pe5_1
    assert check_s_pe5_1
    assert check_h_ipe5_1
    assert check_s_ipe5_1

    assert check_persistence_h, 'device check'
    assert check_persistence_s, 'device check'
    assert check_persistence_h_pe, 'device check'
    assert check_persistence_s_pe, 'device check'


test_si_pe(torch.device('cpu'))

def test_ch3cho(device):
    """Test TiO2."""
    h_tio2 = _get_matrix('./tests/unittests/data/sk/ch3cho/hamsqr1.dat', device)
    s_tio2 = _get_matrix('./tests/unittests/data/sk/ch3cho/oversqr.dat', device)
    geometry = Geometry.from_ase_atoms([molecule('CH3CHO')])

    path_sk = './tests/unittests/data/slko/mio'
    basis = Basis(geometry.atomic_numbers, shell_dict)

    # build single Hamiltonian and overlap feeds
    h_feed = SkfFeed.from_dir(
        path_sk, shell_dict, skf_type='skf',
        geometry=geometry, interpolation='PolyInterpU', integral_type='H')
    s_feed = SkfFeed.from_dir(
        path_sk, shell_dict, skf_type='skf',
        geometry=geometry, interpolation='PolyInterpU', integral_type='S')

    ham = hs_matrix(geometry, basis, h_feed)
    over = hs_matrix(geometry, basis, s_feed)

    check_h_tio2 = torch.max(abs(ham.squeeze() - h_tio2)) < 1E-14
    check_s_tio2 = torch.max(abs(over.squeeze() - s_tio2)) < 1E-14

    check_persistence_h = ham.device == device
    check_persistence_s = over.device == device

    assert check_h_tio2, 'real H tolerance check'
    assert check_s_tio2, 'real S tolerance check'
    check_persistence_h, 'device check'
    check_persistence_s, 'device check'


def test_tio2(device):
    """Test TiO2."""
    h_tio2 = _get_matrix('./tests/unittests/data/sk/tio2/hamsqr1.dat', device)
    s_tio2 = _get_matrix('./tests/unittests/data/sk/tio2/oversqr.dat', device)
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

    path_sk = './tests/unittests/data/slko/tiorg'
    kpoints = torch.tensor([[1, 1, 1]])
    basis = Basis(geometry.atomic_numbers, shell_dict)

    # build single Hamiltonian and overlap feeds
    h_feed = SkfFeed.from_dir(
        path_sk, shell_dict, skf_type='skf',
        geometry=geometry, interpolation='PolyInterpU', integral_type='H')
    s_feed = SkfFeed.from_dir(
        path_sk, shell_dict, skf_type='skf',
        geometry=geometry, interpolation='PolyInterpU', integral_type='S')
    skparams = SkfParamFeed.from_dir(path_sk, geometry, skf_type='skf')
    periodic = Periodic(geometry, geometry.cell, cutoff=skparams.cutoff,
                        kpoints=kpoints)

    ham = hs_matrix(periodic, basis, h_feed, n_kpoints=1, cutoff=skparams.cutoff + 1.0)
    over = hs_matrix(periodic, basis, s_feed, n_kpoints=1, cutoff=skparams.cutoff + 1.0)

    ind_tio2 = torch.arange(0, h_tio2.shape[-1], 2)
    check_h_tio2 = torch.max(abs(
        ham.real.squeeze() - h_tio2[..., ind_tio2])) < 3E-9
    check_s_tio2 = torch.max(abs(
        over.real.squeeze() - s_tio2[..., ind_tio2])) < 5E-9
    check_ih_tio2 = torch.max(abs(
        ham.imag.squeeze() - h_tio2[..., ind_tio2 + 1])) < 1E-9
    check_is_tio2 = torch.max(abs(
        over.imag.squeeze() - s_tio2[..., ind_tio2 + 1])) < 1E-9

    check_persistence_h = ham.device == device
    check_persistence_s = over.device == device

    assert check_h_tio2, 'real H tolerance check'
    assert check_s_tio2, 'real S tolerance check'
    assert check_ih_tio2, 'imaginary H tolerance check'
    assert check_is_tio2, 'imaginary S tolerance check'
    check_persistence_h, 'device check'
    check_persistence_s, 'device check'


def test_hs_matrix_hdf_npe(device):
    """Test single Hamiltonian and overlap after SK transformations."""
    h_ch4 = _get_matrix('./tests/unittests/data/sk/ch4/hamsqr1.dat', device)
    s_ch4 = _get_matrix('./tests/unittests/data/sk/ch4/oversqr.dat', device)
    path_skf = './tests/unittests/data/slko/mio.hdf'
    mol = molecule('CH4')
    geometry = Geometry.from_ase_atoms(mol, device=device)
    basis = Basis(geometry.atomic_numbers, shell_dict)

    # build single Hamiltonian and overlap feeds
    h_feed = SkfFeed.from_dir(
        path_skf, shell_dict, skf_type='h5',
        geometry=geometry, interpolation='PolyInterpU', integral_type='H')
    s_feed = SkfFeed.from_dir(
        path_skf, shell_dict, skf_type='h5',
        geometry=geometry, interpolation='PolyInterpU', integral_type='S')

    ham = hs_matrix(geometry, basis, h_feed)
    over = hs_matrix(geometry, basis, s_feed)

    check_h = torch.max(abs(ham - h_ch4)) < 1E-14
    check_s = torch.max(abs(over - s_ch4)) < 1E-14
    check_persistence_h = ham.device == device
    check_persistence_s = over.device == device

    assert check_h, 'Hamiltonian are outside of permitted tolerance thresholds'
    assert check_s, 'Overlap are outside of permitted tolerance thresholds'
    assert check_persistence_h, 'Device persistence check failed'
    assert check_persistence_s, 'Device persistence check failed'


def test_hs_matrix_batch_npe(device):
    """Test batch Hamiltonian and overlap after SK transformations."""
    h_ch4 = _get_matrix('./tests/unittests/data/sk/ch4/hamsqr1.dat', device)
    s_ch4 = _get_matrix('./tests/unittests/data/sk/ch4/oversqr.dat', device)
    h_c2h4 = _get_matrix('./tests/unittests/data/sk/ch3cho/hamsqr1.dat', device)
    s_c2h4 = _get_matrix('./tests/unittests/data/sk/ch3cho/oversqr.dat', device)
    path_skf = './tests/unittests/data/slko/mio.hdf'
    geometry = Geometry.from_ase_atoms([
        molecule('CH4'), molecule('CH3CHO')], device=device)
    basis = Basis(geometry.atomic_numbers, shell_dict)

    # build Hamiltonian and overlap tables feed from original SKF files
    h_feed = SkfFeed.from_dir(
        path_skf, shell_dict, skf_type='h5',
        geometry=geometry, interpolation='PolyInterpU', integral_type='H')
    s_feed = SkfFeed.from_dir(
        path_skf, shell_dict, skf_type='h5',
        geometry=geometry, interpolation='PolyInterpU', integral_type='S')

    ham = hs_matrix(geometry, basis, h_feed)
    over = hs_matrix(geometry, basis, s_feed)
    h_ref = pack([h_ch4, h_c2h4])
    s_ref = pack([s_ch4, s_c2h4])

    # Tolerance threshold tests are not implemented, so just fail here
    check_h = torch.max(abs(ham - h_ref)) < 1E-14
    check_s = torch.max(abs(over - s_ref)) < 1E-14
    check_persistence = ham.device == device
    assert check_h, 'Hamiltonian are outside of permitted tolerance thresholds'
    assert check_s, 'Overlap are outside of permitted tolerance thresholds'
    assert check_persistence, 'Device persistence check failed'


# def test_hs_d(device):
#     """Test SK transformation values of d orbitals."""
#     h_au_p = _get_matrix('./tests/unittests/data/sk/au/hamsqr1.dat.p', device)
#     s_au_p = _get_matrix('./tests/unittests/data/sk/au/oversqr.dat.p', device)
#     h_au_d = _get_matrix('./tests/unittests/data/sk/au/hamsqr1.dat', device)
#     s_au_d = _get_matrix('./tests/unittests/data/sk/au/oversqr.dat', device)
#     numbers = torch.tensor([79, 79]).to(device)
#     max_l_p, max_l_d = {79: 1}, {79: 2}
#     positions = torch.tensor([[0., 0., 0.], [1., 1., 0.]]).to(device)
#     geometry = Geometry(numbers, positions, 'angstrom')
#     basis_p = Basis(geometry.atomic_numbers, max_l_p)
#     basis_d = Basis(geometry.atomic_numbers, max_l_d)

#     # build Hamiltonian and overlap tables feed from original SKF files
#     h_feed_p, s_feed_p = SkfFeed.from_dir(
#         './tests/unittests/data/slko/auorg', max_l_p, geometry,
#         interpolation='PolyInterpU', h_feed=True, s_feed=True)

#     # build Hamiltonian and overlap tables feed with d orbitals
#     h_feed_d, s_feed_d = SkfFeed.from_dir(
#         './tests/unittests/data/slko/auorg', max_l_d, geometry,
#         interpolation='PolyInterpU', h_feed=True, s_feed=True)

#     ham_p = hs_matrix(geometry, basis_p, h_feed_p,  max_ls=max_l_p)
#     over_p = hs_matrix(geometry, basis_p, s_feed_p)

#     check_h_p = torch.max(abs(ham_p - h_au_p)) < 1E-14
#     check_s_p = torch.max(abs(over_p - s_au_p)) < 1E-14
#     check_persistence_p = ham_p.device == device

#     ham_d = hs_matrix(geometry, basis_d, h_feed_d,  max_ls=max_l_d)
#     over_d = hs_matrix(geometry, basis_d, s_feed_d)

#     check_h_d = torch.max(abs(ham_d - h_au_d)) < 1E-14
#     check_s_d = torch.max(abs(over_d - s_au_d)) < 1E-14
#     check_persistence_d = ham_d.device == device

#     assert check_h_p, 'Hamiltonian are outside of permitted tolerance thresholds'
#     assert check_s_p, 'Overlap are outside of permitted tolerance thresholds'
#     assert check_persistence_p, 'Device persistence check failed'

#     assert check_h_d, 'Hamiltonian are outside of permitted tolerance thresholds'
#     assert check_s_d, 'Overlap are outside of permitted tolerance thresholds'
#     assert check_persistence_d, 'Device persistence check failed'



# @pytest.mark.grad
# def test_hs_matrix_grad(device):
#     """

#     Warnings:
#         This gradient check can take a **VERY, VERY LONG TIME** if great care
#         is not taken to limit the number of input variables. Therefore, tests
#         are only performed on H2 and CH4, change at your own peril!
#     """

#     def proxy(geometry_in, basis_in, sk_feed_in, *args):
#         """Proxy function is needed to enable gradcheck to operate properly"""
#         return hs_matrix(geometry_in, basis_in, sk_feed_in)

#     mol = molecule('CH4')
#     path_to_skf = './tests/unittests/data/slko/mio'
#     geometry = Geometry.from_ase_atoms(mol, device=device)
#     basis = Basis(geometry.atomic_numbers, shell_dict)
#     h_feed = SkfFeed.from_dir(
#         path_to_skf, shell_dict, skf_type='skf', geometry=geometry,
#         interpolation='PolyInterpU', integral_type='H', h_grad=True)
#     s_feed = SkfFeed.from_dir(
#         path_to_skf, shell_dict, skf_type='skf', geometry=geometry,
#         interpolation='PolyInterpU', integral_type='S', s_grad=True)

#     # Identify what variables the gradient will be calculated with respect to.
#     argh = (*h_feed.off_site_dict[(6, 6, 0, 0)].yy,
#             *h_feed.off_site_dict[(6, 6, 1, 1)].yy,
#             *h_feed.off_site_dict[(1, 1, 0, 0)].yy)

#     args = (*s_feed.off_site_dict[(6, 6, 0, 0)].yy,
#             *s_feed.off_site_dict[(6, 6, 1, 1)].yy,
#             *s_feed.off_site_dict[(1, 1, 0, 0)].yy)

#     grad_h = gradcheck(proxy, (geometry, basis, h_feed, *argh),
#                        raise_exception=False)
#     grad_s = gradcheck(proxy, (geometry, basis, s_feed, *args),
#                        raise_exception=False)

#     assert grad_h, 'Hamiltonian gradient stability test failed.'
#     assert grad_s, 'Overlap gradient stability test failed.'


# @pytest.mark.grad
# def test_hs_matrix_batch_grad(device):
#     """

#     Warnings:
#         This gradient check can take a **VERY, VERY LONG TIME** if great care
#         is not taken to limit the number of input variables. Therefore, tests
#         are only performed on H2 and CH4, change at your own peril!
#     """

#     def proxy(geometry_in, basis_in, sk_feed_in, *args):
#         """Proxy function is needed to enable gradcheck to operate properly"""
#         return hs_matrix(geometry_in, basis_in, sk_feed_in)

#     mol = [molecule('CH4'), molecule('H2')]
#     path_to_skf = './tests/unittests/data/slko/mio'
#     geometry = Geometry.from_ase_atoms(mol, device=device)
#     basis = Basis(geometry.atomic_numbers, shell_dict)
#     h_feed = SkfFeed.from_dir(
#         path_to_skf, shell_dict, skf_type='skf', geometry=geometry,
#         interpolation='PolyInterpU', integral_type='H', h_grad=True)
#     s_feed = SkfFeed.from_dir(
#         path_to_skf, shell_dict, skf_type='skf', geometry=geometry,
#         interpolation='PolyInterpU', integral_type='S', s_grad=True)

#     # Identify what variables the gradient will be calculated with respect to.
#     argh = (*h_feed.off_site_dict[(6, 6, 0, 0)].yy,
#             *h_feed.off_site_dict[(6, 6, 1, 1)].yy,
#             *h_feed.off_site_dict[(1, 1, 0, 0)].yy)

#     args = (*s_feed.off_site_dict[(6, 6, 0, 0)].yy,
#             *s_feed.off_site_dict[(6, 6, 1, 1)].yy,
#             *s_feed.off_site_dict[(1, 1, 0, 0)].yy)

#     grad_h = gradcheck(proxy, (geometry, basis, h_feed, *argh),
#                        raise_exception=False)
#     grad_s = gradcheck(proxy, (geometry, basis, s_feed, *args),
#                        raise_exception=False)

#     assert grad_h, 'Hamiltonian gradient stability test failed.'
#     assert grad_s, 'Overlap gradient stability test failed.'
