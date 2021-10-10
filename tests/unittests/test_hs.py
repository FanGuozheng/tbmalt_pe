# -*- coding: utf-8 -*-
"""Perform tests on functions which generate Hamiltonian and overlap matrices.

The tests on reading SKF is in test_sk_read.py, here the tests will test
Hamiltonian and overlap matrices compare with DFTB+ results. The tests cover
the different orbitals, interpolation methods and effect of `max_l`.
"""
import re
import torch
import pytest
from ase.build import molecule
from torch.autograd import gradcheck
from tbmalt import Geometry, Basis, SkfFeed
from tbmalt.physics.dftb.slaterkoster import hs_matrix
from tbmalt.common.batch import pack
torch.set_default_dtype(torch.float64)
torch.set_printoptions(15)
shell_dict = {1: [0], 6: [0, 1], 7: [0, 1], 8: [0, 1]}


def test_hs_matrix_single(device):
    """Test single Hamiltonian and overlap after SK transformations."""
    h_ch4 = _get_matrix('./tests/unittests/data/sk/ch4/hamsqr1.dat', device)
    s_ch4 = _get_matrix('./tests/unittests/data/sk/ch4/oversqr.dat', device)
    mol = molecule('CH4')
    shell_dict = {1: [0], 6: [0, 1]}
    geometry = Geometry.from_ase_atoms(mol, device=device)
    basis = Basis(geometry.atomic_numbers, shell_dict)

    # build single Hamiltonian and overlap feeds
    h_feed = SkfFeed.from_dir(
        './tests/unittests/data/slko/mio', shell_dict, skf_type='skf',
        geometry=geometry, interpolation='PolyInterpU', integral_type='H')
    s_feed = SkfFeed.from_dir(
        './tests/unittests/data/slko/mio', shell_dict, skf_type='skf',
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
        './tests/unittests/data/slko/mio', shell_dict, skf_type='skf',
        geometry=geometry2, interpolation='CubicSpline', integral_type='H')
    s_feed2 = SkfFeed.from_dir(
        './tests/unittests/data/slko/mio', shell_dict, skf_type='skf',
        geometry=geometry2, interpolation='CubicSpline', integral_type='S')

    ham2 = hs_matrix(geometry2, basis, h_feed2, shell_dict=shell_dict)
    over2 = hs_matrix(geometry2, basis, s_feed2, shell_dict=shell_dict)

    check_h2 = torch.max(abs(ham2 - h_ch4.to('cpu'))) < 5E-9
    check_s2 = torch.max(abs(over2 - s_ch4.to('cpu'))) < 1E-10
    check_persistence_h2 = ham2.device == torch.device('cpu')
    check_persistence_s2 = over2.device == torch.device('cpu')

    assert check_h2, 'Hamiltonian are outside of permitted tolerance thresholds'
    assert check_s2, 'Overlap are outside of permitted tolerance thresholds'
    assert check_persistence_h2, 'Device persistence check failed'
    assert check_persistence_s2, 'Device persistence check failed'

    # build only Hamiltonian feed
    h_feed3 = SkfFeed.from_dir(
        './tests/unittests/data/slko/mio', shell_dict, skf_type='skf',
        geometry=geometry, interpolation='PolyInterpU', integral_type='H')

    ham3 = hs_matrix(geometry, basis, h_feed3, shell_dict=shell_dict)
    check_h3 = torch.max(abs(ham3 - h_ch4)) < 1E-14
    check_persistence_h3 = ham3.device == device

    assert check_h3, 'Hamiltonian are outside of permitted tolerance thresholds'
    assert check_persistence_h3, 'Device persistence check failed'

    # build only Overlap feed
    s_feed4 = SkfFeed.from_dir(
        './tests/unittests/data/slko/mio', shell_dict, skf_type='skf',
        geometry=geometry, interpolation='PolyInterpU', integral_type='S')

    over4 = hs_matrix(geometry, basis, s_feed4, shell_dict=shell_dict)
    check_s4 = torch.max(abs(over4 - s_ch4)) < 1E-14
    check_persistence_s4 = over4.device == device

    assert check_s4, 'Hamiltonian are outside of permitted tolerance thresholds'
    assert check_persistence_s4, 'Device persistence check failed'


def test_hs_matrix_hdf(device):
    """Test single Hamiltonian and overlap after SK transformations."""
    h_ch4 = _get_matrix('./tests/unittests/data/sk/ch4/hamsqr1.dat', device)
    s_ch4 = _get_matrix('./tests/unittests/data/sk/ch4/oversqr.dat', device)
    mol = molecule('CH4')
    geometry = Geometry.from_ase_atoms(mol, device=device)
    basis = Basis(geometry.atomic_numbers, shell_dict)

    # build single Hamiltonian and overlap feeds
    h_feed = SkfFeed.from_dir(
        './tests/unittests/data/slko/ch.hdf', shell_dict, skf_type='skf',
        geometry=geometry, interpolation='PolyInterpU', integral_type='H')
    s_feed = SkfFeed.from_dir(
        './tests/unittests/data/slko/ch.hdf', shell_dict, skf_type='skf',
        geometry=geometry, interpolation='PolyInterpU', integral_type='S')

    ham = hs_matrix(geometry, basis, h_feed, shell_dict=shell_dict)
    over = hs_matrix(geometry, basis, s_feed, shell_dict=shell_dict)

    check_h = torch.max(abs(ham - h_ch4)) < 1E-14
    check_s = torch.max(abs(over - s_ch4)) < 1E-14
    check_persistence_h = ham.device == device
    check_persistence_s = over.device == device

    assert check_h, 'Hamiltonian are outside of permitted tolerance thresholds'
    assert check_s, 'Overlap are outside of permitted tolerance thresholds'
    assert check_persistence_h, 'Device persistence check failed'
    assert check_persistence_s, 'Device persistence check failed'


def test_hs_matrix_batch(device):
    """Test batch Hamiltonian and overlap after SK transformations."""
    h_ch4 = _get_matrix('./tests/unittests/data/sk/ch4/hamsqr1.dat', device)
    s_ch4 = _get_matrix('./tests/unittests/data/sk/ch4/oversqr.dat', device)
    h_c2h4 = _get_matrix('./tests/unittests/data/sk/c2h4/hamsqr1.dat', device)
    s_c2h4 = _get_matrix('./tests/unittests/data/sk/c2h4/oversqr.dat', device)
    geometry = Geometry.from_ase_atoms([
        molecule('CH4'), molecule('C2H4')], device=device)
    basis = Basis(geometry.atomic_numbers, shell_dict)

    # build Hamiltonian and overlap tables feed from original SKF files
    h_feed = SkfFeed.from_dir(
        './tests/unittests/data/slko/mio', shell_dict, skf_type='skf',
        geometry=geometry, interpolation='PolyInterpU', integral_type='H')
    s_feed = SkfFeed.from_dir(
        './tests/unittests/data/slko/mio', shell_dict, skf_type='skf',
        geometry=geometry, interpolation='PolyInterpU', integral_type='S')

    ham = hs_matrix(geometry, basis, h_feed)
    over = hs_matrix(geometry, basis, s_feed)
    h_ref = pack([h_ch4, h_c2h4])
    s_ref = pack([s_ch4, s_c2h4])

    # Tolerance threshold tests are not implemented, so just fail here
    check_h = torch.max(abs(ham - h_ref)) < 1E-14
    check_s = torch.max(abs(over - s_ref)) < 1E-14
    check_persistence = ham.device == device
    # print(ham, '\n', h_ref[0], '\n', abs(ham - h_ref).ge(1E-14))
    assert check_h, 'Hamiltonian are outside of permitted tolerance thresholds'
    assert check_s, 'Overlap are outside of permitted tolerance thresholds'
    assert check_persistence, 'Device persistence check failed'


def test_hs_d(device):
    """Test SK transformation values of d orbitals."""
    h_au_p = _get_matrix('./tests/unittests/data/sk/au/hamsqr1.dat.p', device)
    s_au_p = _get_matrix('./tests/unittests/data/sk/au/oversqr.dat.p', device)
    h_au_d = _get_matrix('./tests/unittests/data/sk/au/hamsqr1.dat', device)
    s_au_d = _get_matrix('./tests/unittests/data/sk/au/oversqr.dat', device)
    numbers = torch.tensor([79, 79]).to(device)
    max_l_p, max_l_d = {79: 1}, {79: 2}
    positions = torch.tensor([[0., 0., 0.], [1., 1., 0.]]).to(device)
    geometry = Geometry(numbers, positions, 'angstrom')
    basis_p = Basis(geometry.atomic_numbers, max_l_p)
    basis_d = Basis(geometry.atomic_numbers, max_l_d)

    # build Hamiltonian and overlap tables feed from original SKF files
    h_feed_p, s_feed_p = SkfFeed.from_dir(
        './tests/unittests/data/slko/auorg', max_l_p, geometry,
        interpolation='PolyInterpU', h_feed=True, s_feed=True)

    # build Hamiltonian and overlap tables feed with d orbitals
    h_feed_d, s_feed_d = SkfFeed.from_dir(
        './tests/unittests/data/slko/auorg', max_l_d, geometry,
        interpolation='PolyInterpU', h_feed=True, s_feed=True)

    ham_p = hs_matrix(geometry, basis_p, h_feed_p,  max_ls=max_l_p)
    over_p = hs_matrix(geometry, basis_p, s_feed_p)

    check_h_p = torch.max(abs(ham_p - h_au_p)) < 1E-14
    check_s_p = torch.max(abs(over_p - s_au_p)) < 1E-14
    check_persistence_p = ham_p.device == device

    ham_d = hs_matrix(geometry, basis_d, h_feed_d,  max_ls=max_l_d)
    over_d = hs_matrix(geometry, basis_d, s_feed_d)

    check_h_d = torch.max(abs(ham_d - h_au_d)) < 1E-14
    check_s_d = torch.max(abs(over_d - s_au_d)) < 1E-14
    check_persistence_d = ham_d.device == device

    assert check_h_p, 'Hamiltonian are outside of permitted tolerance thresholds'
    assert check_s_p, 'Overlap are outside of permitted tolerance thresholds'
    assert check_persistence_p, 'Device persistence check failed'

    assert check_h_d, 'Hamiltonian are outside of permitted tolerance thresholds'
    assert check_s_d, 'Overlap are outside of permitted tolerance thresholds'
    assert check_persistence_d, 'Device persistence check failed'



@pytest.mark.grad
def test_hs_matrix_grad(device):
    """

    Warnings:
        This gradient check can take a **VERY, VERY LONG TIME** if great care
        is not taken to limit the number of input variables. Therefore, tests
        are only performed on H2 and CH4, change at your own peril!
    """

    def proxy(geometry_in, basis_in, sk_feed_in, *args):
        """Proxy function is needed to enable gradcheck to operate properly"""
        return hs_matrix(geometry_in, basis_in, sk_feed_in)

    max_l = {1: 0, 6: 1}
    mol = molecule('CH4')
    geometry = Geometry.from_ase_atoms(mol, device=device)
    basis = Basis(geometry.atomic_numbers, max_l)
    h_feed, s_feed = SkfFeed.from_dir(
        './tests/unittests/data/slko/mio', max_l, geometry,
        interpolation='PolyInterpU', h_feed=True, s_feed=True,
        h_grad=True, s_grad=True)

    # Identify what variables the gradient will be calculated with respect to.
    argh = (*h_feed.off_site_dict[(6, 6, 0, 0, 0)].yy,
            *h_feed.off_site_dict[(6, 6, 1, 1, 0)].yy,
            *h_feed.off_site_dict[(1, 1, 0, 0, 0)].yy)

    args = (*s_feed.off_site_dict[(6, 6, 0, 0, 0)].yy,
            *s_feed.off_site_dict[(6, 6, 1, 1, 0)].yy,
            *s_feed.off_site_dict[(1, 1, 0, 0, 0)].yy)

    grad_h = gradcheck(proxy, (geometry, basis, h_feed, *argh),
                       raise_exception=False)
    grad_s = gradcheck(proxy, (geometry, basis, s_feed, *args),
                       raise_exception=False)

    assert grad_h, 'Hamiltonian gradient stability test failed.'
    assert grad_s, 'Overlap gradient stability test failed.'


@pytest.mark.grad
def test_hs_matrix_batch_grad(device):
    """

    Warnings:
        This gradient check can take a **VERY, VERY LONG TIME** if great care
        is not taken to limit the number of input variables. Therefore, tests
        are only performed on H2 and CH4, change at your own peril!
    """

    def proxy(geometry_in, basis_in, sk_feed_in, *args):
        """Proxy function is needed to enable gradcheck to operate properly"""
        return hs_matrix(geometry_in, basis_in, sk_feed_in)

    max_l = {1: 0, 6: 1}
    mol = [molecule('CH4'), molecule('H2')]
    geometry = Geometry.from_ase_atoms(mol, device=device)
    basis = Basis(geometry.atomic_numbers, max_l)
    h_feed, s_feed = SkfFeed.from_dir(
        './tests/unittests/data/slko/mio', max_l, geometry,
        interpolation='PolyInterpU', h_feed=True, s_feed=True,
        h_grad=True, s_grad=True)

    # Identify what variables the gradient will be calculated with respect to.
    argh = (*h_feed.off_site_dict[(6, 6, 0, 0, 0)].yy,
            *h_feed.off_site_dict[(6, 6, 1, 1, 0)].yy,
            *h_feed.off_site_dict[(1, 1, 0, 0, 0)].yy)

    args = (*s_feed.off_site_dict[(6, 6, 0, 0, 0)].yy,
            *s_feed.off_site_dict[(6, 6, 1, 1, 0)].yy,
            *s_feed.off_site_dict[(1, 1, 0, 0, 0)].yy)

    grad_h = gradcheck(proxy, (geometry, basis, h_feed, *argh),
                       raise_exception=False)
    grad_s = gradcheck(proxy, (geometry, basis, s_feed, *args),
                       raise_exception=False)

    assert grad_h, 'Hamiltonian gradient stability test failed.'
    assert grad_s, 'Overlap gradient stability test failed.'


def _get_matrix(filename, device):
    """Read DFTB+ hamsqr1.dat and oversqr.dat."""
    text = ''.join(open(filename, 'r').readlines())
    string = re.search('(?<=MATRIX\n).+(?=\n)', text, flags=re.DOTALL).group(0)
    return torch.tensor([[float(i) for i in row.split()]
                         for row in string.split('\n')]).to(device)

test_hs_matrix_single(torch.device('cpu'))
