#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 23:03:21 2021

@author: gz_fan
"""
import os
from typing import Dict, Optional, Literal
import torch
# from tbmalt.data.sk import hdf_suffix
from tbmalt.structures.geometry import unique_atom_pairs
from tbmalt.io.skf import Skf
from tbmalt.structures.geometry import batch_chemical_symbols


class SkfParamFeed:
    """This is the standardard method to supply Slater-Koster integral feeds.

    The standard suggests that the feeds are similar to the method in DFTB+.
    The `from_dir` function can be used to read normal Slater-Koster files
    and return Hamiltonian and overlap feeds separatedly in default.

    Arguments:
        off_site_dict: Collections of off-site data as off-site feeds of
            Hamiltonian or overlap.
        on_site_dict: Collections of on-site data as on-site feeds of
            Hamiltonian or overlap.

    Attributes:
        off_site_dict: Collections of off-site data as off-site feeds of
            Hamiltonian or overlap.
        on_site_dict: Collections of on-site data as on-site feeds of
            Hamiltonian or overlap.

    """
    def __init__(self, U, qzero, cutoff):
        self.U = U
        self.qzero = qzero
        self.cutoff =  cutoff

    @classmethod
    def from_dir(cls, path: str, geometry: Optional[dict] = None,
                 elements: Optional[list] = None,
                 skf_type: Literal['h5', 'skf'] = 'h5',
                 **kwargs) -> 'SkfParamFeed':
        """Read all skf files like the normal way in DFTB+ and return SkFeed.

        The geometry and elements are optional, which give the information of
        the element species in SKF files to be read. The `h_feed` and `s_feed`
        control if return Hamiltonian or overlap feeds, if False, it will
        return an empty Hamiltonian feed or overlap feed. Besides Hamiltonian,
        overlap and on-site, all other parameters original from SKF files are
        packed to `params_dict`.

        Arguments:
            path: Path to SKF files or joint path to binary SKF file.
            geometry: `Geometry` object, which contains element species.
            elements: All element specie names for reading SKF files.
            max_l: Maximum quantum ℓ for each element.
            h_feed: If build Hamiltonian feed from SKF integral tables.
            s_feed: If build overlap feed from SKF integral tables.

        Keyword Args:
            interpolation: Interpolation method of integrals which are not
                in the grid points.
            orbital_resolve: If each orbital is resolved for U.

        Returns:
            sktable_dict: Dictionary contains SKF integral tables.

        Notes:
            The interactions will rely on the maximum of quantum ℓ in the
            system. Current only support up to d orbital. If you define the
            interactions as `int_d`, which means the maximum of ℓ is 2, the
            code will read all the s, p, d related integral tables.

        """
        # The device will first read from geometry, if geometry is None
        # then from kwargs dictionary, default is cpu
        if geometry is not None:
            device = geometry.positions.device
            dtype = geometry.positions.dtype
        else:
            device = kwargs.get('device', torch.device('cpu'))
            dtype = kwargs.get('device', torch.get_default_dtype())

        if orbital_resolve:= kwargs.get('orbital_resolve', False):
            raise NotImplementedError('Not implement orbital resolved U.')

        # check path existence, type (dir or file) and suffix
        assert os.path.exists(path), '%s do not exist' % path
        if not (is_dir := os.path.isdir(path)):
            hdf_suffix = ('hdf', 'HDF', 'Hdf', 'H5', 'h5')
            assert path.split('.')[-1] in hdf_suffix, 'suffix error, ' + \
                'suffix of %s is not in %s' % (path, hdf_suffix)

        # # do not support reading f orbitals and will raise error
        # assert max(max_l.values()) <= 2, 'do not support f orbitals.'

        # unique atom pairs is from either elements or geometry
        assert elements is not None or geometry is not None, 'both ' + \
            'elements and geometry are None.'

        # get global unique element species pair with geometry object
        assert geometry is not None

        sktable_dict = {}

        # get global unique element species pair
        element_pair = unique_atom_pairs(geometry=geometry)
        hs_cut = 0.0

        # get a list of interpolations for each unique atom pairs
        # interactions = _get_interactions(max_l, element_pair)

        # loop of all unique element pairs
        # if skf_type == 'skf':
        for ii, ielement in enumerate(element_pair):

            element = batch_chemical_symbols(ielement)
            path_to_skf = _path_to_skf(path, element, is_dir)
            skf = Skf.read(path_to_skf, ielement, mask_hs=False,
                           read_hamiltonian=False,
                           read_overlap=False,
                           device=device, **kwargs)

            if ielement[0] == ielement[1]:
                # retutn onsite parameters for Hamiltonian or
                # sktable_dict = _get_homo_dict(
                #     sktable_dict, skf, device=device, **kwargs)

                if not orbital_resolve:  # -> only s orbital
                    sktable_dict[(ielement[0].tolist(),
                                  'U')] = skf.hubbard_us[0].to(dtype)

                sktable_dict[(ielement[0].tolist(),
                              'occupations')] = skf.occupations.to(dtype)

            hs_cut = skf.hs_cutoff if skf.hs_cutoff > hs_cut else hs_cut
        # else:
        #     for ii, ielement in enumerate(element_pair):
        #         if ielement[0] == ielement[1]:
        #             sktable_dict[(ielement[0].tolist(), 'U')] = _U[
        #                 (ielement[0].tolist(), 'U')].to(dtype)
        #             sktable_dict[(ielement[0].tolist(), 'occupations')] = _occ[
        #                 (ielement[0].tolist(), 'occ')].to(dtype)
        #         hs_cut = skf.hs_cutoff if skf.hs_cutoff > hs_cut else hs_cut

        U = _build_U(sktable_dict, geometry)
        qzero = _build_qzero(sktable_dict, geometry)

        return cls(U, qzero, hs_cut)

def _path_to_skf(path, element, is_dir):
    """Return the joint path to the skf file or binary file."""
    if not is_dir:
        return path
    else:
        return os.path.join(path, element[0] + '-' + element[1] + '.skf')


def _get_homo_dict(sktable_dict: dict, skf: object, **kwargs) -> dict:
    """Write onsite, Hubbert U and other homo parameters into dict."""
    if kwargs.get('orbital_resolve', False):
        raise NotImplementedError('Not implement orbital resolved Hubbert U.')

    assert skf.atom_pair[0] == skf.atom_pair[1]

    # return Hubbert U without orbital resolve
    sktable_dict[(skf.atom_pair[0].tolist(), 'U')] = skf.hubbard_us.unsqueeze(1)[-1]
    sktable_dict[(skf.atom_pair[0].tolist(), 'occupations')] = skf.occupations

    return sktable_dict


def _build_U(sktable_dict: dict, geometry: object):
    U = torch.zeros(geometry.atomic_numbers.shape)
    for inum in geometry.unique_atomic_numbers():
        mask = geometry.atomic_numbers == inum
        U[mask] = sktable_dict[(inum.tolist(), 'U')]

    return U

def _build_qzero(sktable_dict: dict, geometry: object):
    qzero = torch.zeros(geometry.atomic_numbers.shape)
    for inum in geometry.unique_atomic_numbers():
        mask = geometry.atomic_numbers == inum
        qzero[mask] = sktable_dict[(inum.tolist(), 'occupations')].sum()
    return qzero


_U = {(1, 'U'): torch.tensor([4.195E-01]),
      (6, 'U'): torch.tensor([3.647E-01]),
      (7, 'U'): torch.tensor([4.309E-01]),
      (8, 'U'): torch.tensor([4.954E-01])}

_occ = {(1, 'occ'): torch.tensor([0, 0, 1.0]),
        (6, 'occ'): torch.tensor([0, 2.0, 2.0]),
        (7, 'occ'): torch.tensor([0, 3.0, 2.0]),
        (8, 'occ'): torch.tensor([0, 4.0, 2.0])}
