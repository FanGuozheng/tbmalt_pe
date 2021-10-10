#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from contextlib import contextmanager
import pytest
import numpy as np
import h5py
import torch
from tbmalt.io.skf import VcrSkf


def _ref_interaction(reference, atom_pair, lp, hs, device=None):
    """Random looking data for testing electronic integrals with.

    For a given electronic integral, as defined by its azimuthal pair an its
    integral type (i.e. H/S), this function will return some random looking
    data that is unique to that interaction. This prevents having to hard
    code in the expected results.
    """
    _dict = {(0, 0, 'h'): 9, (0, 1, 'h'): 8, (1, 1, 'h'): [5, 6],
             (0, 0, 's'): 19, (0, 1, 's'): 18, (1, 1, 's'): [15, 16]}
    col = _dict[(*lp, hs)]
    _data = torch.from_numpy(reference[str(atom_pair.tolist()[0])+str(
        atom_pair.tolist()[1])][()][..., col])
    _data = _data.unsqueeze(-1) if _data.dim() == 2 else _data

    nr = int(np.sqrt(_data.shape[0]))  # get number of compression radii

    # reshape reference H or S as [n_compr, n_compr, n_interactions, n_grid]
    _data = _data.reshape(nr, nr, _data.shape[-2], _data.shape[-1])
    return _data.permute(0, 1, -1, -2)


def skf_files():
    """Returns a generator that loops over skf test files and their contents.

    File 1: Homo-atomic system with a repulsive polynomial & spline.
    File 2: Same as file 1 but includes f orbitals.
    File 3: Hetero-atomic system with a repulsive polynomial & spline.
    File 4: Hetero-atomic system without a repulsive polynomial or spline.
    File 5: Same as file 4 but with some commonly encountered errors.

    Returns:
        path: Path to skf file.
        args:
            has_r_poly: True if the file contains a valid repulsive polynomial.
            has_r_spline: True if the file contains a repulsive spline.
    """
    path1 = 'tests/unittests/data/slko/vcrskf'
    files1 = ['C-C.skf']
    reference = h5py.File('tests/unittests/data/slko/vcrskf/reference.h5', 'r')

    path2 = './tests/unittests/data/slko/vcrskf'
    files2 = ['C-H_02.00_02.00.skf', 'C-H_02.00_05.00.skf',
              'C-H_05.00_02.00.skf', 'C-H_05.00_05.00.skf',
              'H-C_02.00_02.00.skf', 'H-C_02.00_05.00.skf',
              'H-C_05.00_02.00.skf', 'H-C_05.00_05.00.skf',
              'C-C_02.00_02.00.skf', 'C-C_02.00_05.00.skf',
              'C-C_05.00_02.00.skf', 'C-C_05.00_05.00.skf',]

    args = (False, True)
    return [os.path.join(path2, iname) for iname in files2], reference, \
        [os.path.join(path1, iname) for iname in files1], args


@contextmanager
def file_cleanup(path_in, skf_homo, path_out):
    """Context manager for cleaning up temporary files once no longer needed."""
    # Loads the file located at `path_in` and saves it to `path_out`
    vcrskf = VcrSkf.read(path_in, path_homo=skf_homo, dtype=torch.float64,
                         smooth_to_zero=False)
    for iskf in vcrskf:
        iskf.write(path_out)


def test_read(device):
    """Ensure the `Skf.read` method operates as anticipated."""
    skf, reference, skf_homo, args = skf_files()

    # Check 1: ensure read method redirects to from_skf/from_hdf5 and that the
    # device information is passed on. Note that no actual check is performed
    # as previous tests would have failed & upcoming test will fail if did/does
    # not work correctly.

    # Check 2: warning issued when passing in ``atom_pair`` for an skf file.
    with pytest.warns(UserWarning, match='"atom_pair" argument is*'):
        _check_skf_contents(VcrSkf.read(
            skf, atom_pair=[0, 0], path_homo=skf_homo, smooth_to_zero=False,
            dtype=torch.float64, device=device), reference, *args, device)

    # Check 3: read should not need the ``atom_pair`` argument to read form an
    # HDF5 database that only has a single system in it.
    file_cleanup(skf, skf_homo, 'skfdb.hdf5')

    _check_skf_contents([VcrSkf.read(
        'skfdb.hdf5', atom_pair=ii, smooth_to_zero=False, device=device)
        for ii in [[1, 6], [6, 1], [6, 6]]], reference, *args, device)
    os.remove('skfdb.hdf5')

    # Check 4: an exception should be raise if multiple entries are present in
    # the source HDF5 database but the ``atom_pair`` argument was not given.
    temp = VcrSkf.read(skf, path_homo=skf_homo, device=device)
    [ii.write('skfdb.hdf5') for ii in temp]
    temp[-1].atom_pair = (ap := torch.tensor([6, 6]))

    with pytest.raises(ValueError):
        VcrSkf.read('skfdb.hdf5')

    # Check 5: correct pair is returned
    check_4 = (VcrSkf.read('skfdb.hdf5', atom_pair=ap).atom_pair == ap).all()
    assert check_4, 'Wrong atom pair returned'

    os.remove('skfdb.hdf5')


def _check_skf_contents(skf, reference, has_r_poly, has_r_spline, device):
    """Helper function to test the contents of an `Skf` instances.

    Arguments:
        has_r_poly: True if the file contains a valid repulsive polynomial.
        has_r_spline: True if the file contains a repulsive spline.
        device: Device on on which the `Skf` object should be created.
    """
    d = {'device': device}

    def check_it(attrs, src):
        for name, ref in attrs.items():
            n = f'{src.__class__.__qualname__}.{name}'
            pred = src.__getattribute__(name)
            # Check is done this way to ensure that the error message is
            # correctly displayed.
            if dev_check := pred.device == device:
                assert torch.allclose(pred, ref), f'`{n}` is in error'
            else:
                assert dev_check, f'`{n}` is on the wrong device'

    # Check integral grid integrity
    for iskf in skf:
        step = iskf.grid[..., -1] - iskf.grid[..., -2]
        check_grid = torch.allclose(
            iskf.grid, (torch.arange(21., **d) + 1.) * step)
        assert check_grid, '`Skf.grid` is in error'

    # Ensure atomic data was parsed
    for iskf in skf:
        if iskf.atomic:
            check_it({
                'on_sites': _on_site[iskf.atom_pair[0].tolist()],
                'hubbard_us': _hubbard_us[iskf.atom_pair[0].tolist()],
                'occupations': _occupations[iskf.atom_pair[0].tolist()],
                'mass': _mass[iskf.atom_pair[0].tolist()]}, iskf)

        elif iskf.atom_pair[0] == iskf.atom_pair[1]:
            pytest.fail(f'Skf is homo, but atomic is {iskf.atomic}')

    # Repulsive polynomial
    for iskf in skf:
        if iskf.r_poly is not None:
            check_it({'coef': torch.arange(8., **d) + 0.1,
                      'cutoff': torch.tensor(5.321, **d)}, iskf.r_poly)
            if not has_r_poly:
                pytest.fail('Unexpectedly found valid repulsive polynomial')
        elif has_r_poly:
            pytest.fail('Failed to locate valid repulsive polynomial')

        # Verify the integrals are read in correctly:
        check_h = all([torch.allclose(v, _ref_interaction(
            reference, iskf.atom_pair, k, 'h', device))
                       for k, v in iskf.hamiltonian.items()])
        check_s = all([torch.allclose(v, _ref_interaction(
            reference, iskf.atom_pair, k, 's', device))
                       for k, v in iskf.overlap.items()])

        check_hs = check_h and check_s
        assert check_hs, 'Electronic integrals are in error'

    # Repulsive spline
    for iskf in skf:
        if iskf.r_spline is not None:
            check_it({
                'exp_coef': torch.tensor([
                    1.1293533468e+02, 2.8013737015e+00, -1.1199948353e-01],
                    dtype=torch.float64).repeat(4, 1),
                'grid': torch.tensor([
                    0.035, 0.0375, 0.04, 0.0425, 0.045, 0.04533, 0.045334,
                    0.046259, 0.047184, 0.0493131, 0.0503195, 0.0513259],
                    dtype=torch.float64).repeat(4, 1),
                'spline_coef': torch.tensor([
                    [2.04206e-01, -3.571077211e+01,  2.016504e+03, 2.4177937621e+04],
                    [1.2791e-01, -2.517491578e+01,  2.1978385322e+03, -1.208896881e+05],
                    [7.68203e-02, -1.6452404771e+01,  1.2911658714e+03, -5.7585585206e+04],
                    [4.28593e-02, -1.1076305137e+01,  8.5927398233e+02,  1.6659228929e+04],
                    [2.07993e-02, -6.4675746826e+00,  9.8421819930e+02, -2.1671735721e+06],
                    [1.86943e-02, -6.5260062770e+00, -1.1612836371e+03, 3.5321322249e+08],
                    [1.86682e-02, -6.5183423114e+00,  3.0772750328e+03, -1.3245595712e+06],
                    [1.42234e-02, -4.2253623501e+00, -5.9837777730e+02, 5.6181111108e+05],
                    [1.02476e-02, -3.8902623423e+00,  9.6064805593e+02, -1.0076352105e+05],
                    [5.34702e-03, -1.1699341094e+00,  3.1704121793e+02, -1.4302691445e+05],
                    [4.34492e-03, -9.6638409795e-01, -1.1478564218e+02, 1.0348588939e+04]],
                    dtype=torch.float64).repeat(4, 1, 1),
                'tail_coef': torch.tensor([
                    3.26664e-03, -1.1659802143e+00, -8.3541182457e+01,
                    -5.7825151694e+03, 2.7636944827e+07, -3.8779595521e+09],
                    dtype=torch.float64).repeat(4, 1),
                'cutoff': torch.tensor([0.0553584993], dtype=torch.float64).repeat(4)},
                iskf.r_spline)
            if not has_r_spline:
                pytest.fail('Unexpectedly found repulsive spline')
        elif has_r_spline:
            pytest.fail('Failed to locate repulsive spline')

# atomic data from mio as reference
_on_site = {6: torch.tensor([-0.5048917654803, -0.1943551799182],
                            dtype=torch.float64)}
_hubbard_us = {6: torch.tensor([0.3646664973641, 0.3646664973641],
                               dtype=torch.float64)}
_occupations = {6: torch.tensor([2., 2., 0.], dtype=torch.float64)}
_mass = {6: torch.tensor([12.01], dtype=torch.float64)}
