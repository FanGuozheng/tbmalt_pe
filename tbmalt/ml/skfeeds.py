# -*- coding: utf-8 -*-
"""Slater-Koster integral feed objects.

This contains all Slater-Koster integral feed objects. These objects are
responsible for generating the Slater-Koster integrals used in constructing
Hamiltonian and overlap matrices. The on-site and off-site terms are yielded
by the `on_site` and `off_site` class methods respectively.
"""
import os
from typing import Union, Tuple, Literal, Optional, List, Dict
from abc import ABC, abstractmethod
from inspect import getfullargspec
from warnings import warn
import numpy as np
from h5py import Group
import torch
from torch import Tensor
from scipy.interpolate import CubicSpline
from torch import Tensor
from tbmalt.common.batch import pack
from tbmalt.structures.geometry import unique_atom_pairs
from tbmalt.common.maths.interpolation import PolyInterpU, BicubInterp, Spline1d
from tbmalt.io.skf import Skf, VcrSkf
from tbmalt.structures.geometry import batch_chemical_symbols


class _SkFeed(ABC):
    """ABC for objects responsible for supplying Slater-Koster integrals.

    Subclasses of the this abstract base class are responsible for supplying
    the Slater-Koster integrals needed to construct the Hamiltonian & overlap
    matrices.

    Arguments:
        device: Device on which the `SkFeed` object and its contents resides.
        dtype: Floating point dtype used by `SkFeed` object.

    Developers Notes:
        This class provides a common fabric upon which all Slater-Koster
        integral feed objects are built. As the `_SkFeed` class is in its
        infancy it is subject to change; e.g. the addition of an `update`
        method which allows relevant model variables to be updated via a
        single call during backpropagation.
    """

    def __init__(self, device: torch.device, dtype: torch.dtype):
        # These are static, private variables and must NEVER be modified!
        self.__device = device
        self.__dtype = dtype

    def __init_subclass__(cls, check_sig: bool = True):
        """Check the signature of subclasses' methods.

        Issues non-fatal warnings if invalid signatures are detected in
        subclasses' `off_site` or `on_site` methods. Both methods must accept
        an arbitrary number of keyword arguments, i.e. `**kwargs`. The
        `off_site` & `on_site` method must take the keyword arguments
        (atom_pair, shell_pair, distances) and (atomic_numbers) respectively.

        This behaviour is enforced to maintain consistency between the various
        subclasses of `_SkFeed`'; which is necessary as the various subclasses
        will likely differ significantly from one another & may become quite
        complex.

        Arguments:
            check_sig: Signature check not performed if ``check_sig = False``.
                This offers a way to override these warnings if needed.
        """

        def check(func, has_args):
            sig = getfullargspec(func)
            name = func.__qualname__
            if check_sig:  # This check can be skipped
                missing = ', '.join(has_args - set(sig.args))
                if len(missing) != 0:
                    warn(f'Signature Warning: keyword argument(s) "{missing}"'
                         f' missing from method "{name}"',
                         stacklevel=4)

            if sig.varkw is None:  # This check cannot be skipped
                warn(f'Signature Warning: method "{name}" must accept an '
                     f'arbitrary keyword arguments, i.e. **kwargs.',
                     stacklevel=4)

        check(cls.off_site, {'atom_pair', 'shell_pair', 'distances'})
        check(cls.on_site, {'atomic_numbers'})

    @property
    def device(self) -> torch.device:
        """The device on which the geometry object resides."""
        return self.__device

    @device.setter
    def device(self, value):
        # Instruct users to use the ".to" method if wanting to change device.
        name = self.__class__.__name__
        raise AttributeError(f'{name} object\'s dtype can only be modified '
                             'via the ".to" method.')

    @property
    def dtype(self) -> torch.dtype:
        """Floating point dtype used by geometry object."""
        return self.__dtype

    # @abstractmethod
    def off_site(self, atom_pair: Tensor, shell_pair: Tensor,
                 distances: Tensor, **kwargs) -> Tensor:
        """Evaluate the selected off-site Slater-Koster integrals.

        This evaluates & returns the off-site Slater-Koster integrals between
        orbitals `l_pair` on atoms `atom_pair` at the distances specified by
        `distances`. Note that only one `atom_pair` & `shell_pair` can be
        evaluated at at time. The dimensionality of the the returned tensor
        depends on the number of distances evaluated & the number of bonding
        integrals associated with the interaction.

        Arguments:
            atom_pair: Atomic numbers of the associated atoms.
            shell_pair: Shell numbers associated with the interaction.
            distances: Distances between the atoms pairs.

        Keyword Arguments:
            atom_indices: Tensor: The indices of the atoms associated with the
                 ``distances`` specified. This is automatically passed in by
                 the Slater-Koster transformation code.

        Return:
            integrals: Off-site integrals between orbitals ``shell_pair`` on
                atoms ``atom_pair`` at the specified distances.

        Developers Notes:
            The Slater-Koster transformation passes "atom_pair", "shell_pair",
            & "distances" as keyword arguments. This avoids having to change
            the Slater-Koster transformation code every time a new feed is
            created. These four arguments were made default as they will be
            required by most Slater-Koster feed implementations. A warning
            will be issued if a `_SkFeed` subclass is found to be missing any
            of these arguments. However, this behaviour can be suppressed by
            adding the class argument `check_sig=False`.

            It is imperative that this method accepts an arbitrary number of
            keyword arguments, i.e. has a `**kwarg` argument. This allows for
            additional data to be passed in. By default the Slater-Koster
            transformation code will add the keyword argument "atom_indices".
            This specifies the indices of the atoms involved, which is useful
            if the feed takes into account environmental dependency.

            Any number of additional arguments can be added to this method.
            However, to get the Slater-Koster transform code to pass this
            information through one must pass the requisite data as keyword
            arguments to the Slater-Koster transform function itself. As it
            will pass through any keyword arguments it encounters.

        """
        pass

    # @abstractmethod
    def on_site(self, atomic_numbers: Tensor, **kwargs) -> Tuple[Tensor, ...]:
        """Returns the specified on-site terms.

        Arguments:
            atomic_numbers: Atomic numbers for which on-site terms should be
                returned.

        Keyword Arguments:
            atom_indices: Tensor: The indices of the atoms associated with the
                 ``distances`` specified. This is automatically passed in by
                 the Slater-Koster transformation code.

        Returns:
            on_sites: Tuple of on-site term tensors, one for each atom in
                ``atomic_numbers``.

        Developers Notes:
            See the documentation for the _SkFeed.off_site method for
            more information.

        """
        pass

    # @abstractmethod
    def to(self, device: torch.device) -> 'SkFeed':
        """Returns a copy of the `SkFeed` instance on the specified device.
        This method creates and returns a new copy of the `SkFeed` instance
        on the specified device "``device``".
        Arguments:
            device: Device on which the clone should be placed.
        Returns:
            sk_feed: A copy of the `SkFeed` instance placed on the specified
                device.
        Notes:
            If the `SkFeed` instance is already on the desired device then
            `self` will be returned.
        """
        pass

    @classmethod
    def load(cls, source: Union[str, Group]) -> 'SkFeed':
        """Load a stored Slater Koster integral feed object.

        This is only for loading preexisting Slater-Koster feed objects, from
        HDF5 databases, not instantiating new ones.

        Arguments:
            source: Name of a file to load the integral feed from or an HDF5
                group from which it can be extracted.

        Returns:
            ski_feed: A Slater Koster integral feed object.

        """
        raise NotImplementedError()

    def save(self, target: Union[str, Group]):
        """Save the Slater Koster integral feed to an HDF5 database.

        Arguments:
            target: Name of a file to save the integral feed to or an HDF5
                group in which it can be saved.

        Notes:
            If `target` is a string then a new HDF5 database will be created
            at the path specified by the string. If an HDF5 entity was given
            then a new HDF5 group will be created and added to it.

            Under no circumstances should this just pickle an object. Doing so
            is unstable, unsafe and inflexible.

            It is good practice to save the name of the class so that the code
            automatically knows how to unpack it.
        """
        if isinstance(target, str):
            # Create a HDF5 database and save the feed to it
            raise NotImplementedError()
        elif isinstance(target, Group):
            # Create a new group, save the feed in it and add it to the Group
            raise NotImplementedError()


class SkfFeed(_SkFeed):
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
    def __init__(self, off_site_dict: dict, on_site_dict: dict, **kwargs):
        self.off_site_dict = off_site_dict
        self.on_site_dict = on_site_dict

    @classmethod
    def from_dir(cls, path: str,
                 shell_dict: Dict[int, List[int]],
                 skf_type: Literal['h5', 'skf'] = 'h5',
                 geometry: Optional[dict] = None,
                 elements: Optional[list] = None,
                 integral_type: Literal['H', 'S'] = 'H',
                 **kwargs) -> 'SkfFeed':
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
            shell_dict: : Dictionary of shell numbers associated with the
                interaction.
            integral_type:

        Keyword Args:
            interpolation: Interpolation method of integrals which are not
                in the grid points.
            orbital_resolve: If each orbital is resolved for U.

        Returnpaths:
            sktable_dict: Dictionary contains SKF integral tables.

        Notes:
            The interactions will rely on the maximum of quantum ℓ in the
            system. Current only support up to d orbital. If you define the
            interactions as `int_d`, which means the maximum of ℓ is 2, the
            code will read all the s, p, d related integral tables.

        """
        interpolation = kwargs.get('interpolation', 'PolyInterpU')

        # The device will first read from geometry, if geometry is None
        # then from kwargs dictionary, default is cpu
        if geometry is not None:
            device = geometry.positions.device
        else:
            device = kwargs.get('device', torch.device('cpu'))

        if kwargs.get('orbital_resolve', False):
            raise NotImplementedError('Not implement orbital resolved U.')

        # Bicubic interpolation will be implemented soon
        if interpolation == 'CubicSpline':
            interpolator = CubicSpline
            assert device == torch.device('cpu'), 'device must be cpu if ' + \
                ' interpolation is CubicSpline, but get %s' % device
        elif interpolation == 'PolyInterpU':
            interpolator = PolyInterpU
        elif interpolation == 'Spline1d':
            interpolator = Spline1d
        else:
            raise NotImplementedError('%s is not implemented.' % interpolation)

        if (is_dir := os.path.isdir(path)):
            warn('"hdf" binary Slater-Koster files are suggested, TBMaLT'
                 ' will generate binary with smoothed integral tails.')

        # unique atom pairs is from either elements or geometry
        assert elements is not None or geometry is not None, 'both ' + \
            'elements and geometry are None.'

        # create a blank dict for integrals
        hs_dict, onsite_hs_dict = {}, {}

        # get global unique element species pair with geometry object
        if geometry is not None:
            element_pair = unique_atom_pairs(geometry)
        elif elements is not None:
            element_pair = unique_atom_pairs(symbols=elements)
        if skf_type == 'skf':
            _element_name = [batch_chemical_symbols(ie) for ie in element_pair]
            _path = [os.path.join(path, ien[0] + '-' + ien[1] + '.skf')
                     for ie, ien in zip(element_pair, _element_name)]

        # check path existence, type (dir or file) and suffix
        if skf_type == 'h5':
            _path = [path] * len(element_pair)
        hs_dict, onsite_hs_dict = cls._read(
            hs_dict, onsite_hs_dict, interpolator, element_pair,
            _path, skf_type, integral_type, shell_dict, device, **kwargs)

        return cls(hs_dict, onsite_hs_dict, **kwargs)

    @classmethod
    def _read(cls, hs_dict: dict, onsite_hs_dict: dict, interpolator:object,
              element_pair: Tensor, _path: List[str], skf_type: str,
              integral_type: str, shell_dict: Dict[int, List[int]],
              device: torch.device, **kwargs) -> [dict, dict]:
        """Read."""
        if kwargs.get('build_abcd', False):
            hs_dict['variable'] = []

        for ielement, ipath in zip(element_pair, _path):

            atom_pair = ielement if skf_type == 'h5' else None
            skf = Skf.read(ipath, atom_pair, device=device, **kwargs)

            # generate H or S in SKF files dict
            hs_dict = _get_hs_dict(
                hs_dict, interpolator, skf, integral_type, **kwargs)

            if ielement[0] == ielement[1]:
                onsite_hs_dict = _get_onsite_dict(
                    onsite_hs_dict, skf, shell_dict, integral_type)

        return hs_dict, onsite_hs_dict

    def off_site(self, atom_pair: Tensor, shell_pair: Tensor,
                 distances: Tensor, **kwargs) -> Tensor:
        """Get integrals for given geometrys.

        Arguments:
            distances: distances of single & multi systems.
            atom_pair: skf files type. Support normal skf, h5py binary skf.
            l_pair: The quantum number ℓ pairs.
            ski_type: Type of integral, H or S.

        Keyword Args:
            orbital_resolve: If each orbital is resolved for U.
            abcd: abcd parameters in cubic spline method.

        Returns:
            integral: Getting integral in SKF tables with given atom pair, ℓ
                number pair, distance, or compression radii pair.
        """
        if kwargs.get('orbital_resolve', False):
            raise NotImplementedError('Not implement orbital resolved U.')
        g_compr = kwargs.get('g_compr', None)

        splines = self.off_site_dict[(*atom_pair.tolist(), *shell_pair.tolist())]

        # call the interpolator
        if g_compr is None:
            integral = splines(distances)
        else:
            integral = splines(g_compr[0], g_compr[0], distances)

        if isinstance(integral[0], np.ndarray):
            integral = torch.from_numpy(integral)

        return integral

    def on_site(self, atomic_numbers: Tensor, **kwargs) -> List[Tensor]:
        """Returns the specified on-site terms.

        In sktable dictionary, s, p, d and f orbitals of original on-site
        from Slater-Koster files have been expanded one, three, five and
        seven times in default. The expansion of on-site could be controlled
        by `orbital_expand` when loading Slater-Koster integral tables. The
        output on-site size relies on the defined `max_ls`.

        Arguments:
            atomic_numbers: Atomic numbers for which on-site terms should be
                returned.
        max_ls: A dictionary specifying the maximum permitted angular momentum
            associated with a each atomic number. keys must be integers not
            torch tensors.

        Returns:
            on_sites: Tuple of on-site term tensors, one for each atom in
                `atomic_numbers`.

        """
        return [self.on_site_dict[(ian.tolist())] for ian in atomic_numbers]

    def to_hdf5(self, target: Group):
        """Saves standard instance into a target HDF5 Group.

        Arguments:
            target: The hdf5 group to which the system's data should be saved.

        Notes:
            This function does not create its own group as it expects that
            `target` is the group into which data should be writen.

        """
        pass


class VcrFeed(_SkFeed):
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
    def __init__(self, off_site_dict: dict, on_site_dict: dict,
                 compression_radii_grid: Tensor, **kwargs):
        self.off_site_dict = off_site_dict
        self.on_site_dict = on_site_dict
        self.compression_radii_grid = compression_radii_grid

    @classmethod
    def from_dir(cls, path: str,
                 shell_dict: dict,
                 vcr: Tensor,
                 skf_type: Literal['h5', 'skf'] = 'h5',
                 geometry: Optional[dict] = None,
                 elements: Optional[list] = None,
                 integral_type: Literal['H', 'S'] = 'H',
                 **kwargs) -> Tuple['SkFeed', 'SkFeed']:
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
            shell_dict: : Dictionary of shell numbers associated with the
                interaction.
            vcr:
            integral_type:

        Keyword Args:
            interpolation: Interpolation method of integrals which are not
                in the grid points.
            orbital_resolve: If each orbital is resolved for U.

        Returnpaths:
            sktable_dict: Dictionary contains SKF integral tables.

        Notes:
            The interactions will rely on the maximum of quantum ℓ in the
            system. Current only support up to d orbital. If you define the
            interactions as `int_d`, which means the maximum of ℓ is 2, the
            code will read all the s, p, d related integral tables.

        """
        interpolation = kwargs.get('interpolation', 'PolyInterpU')

        # The device will first read from geometry, if geometry is None
        # then from kwargs dictionary, default is cpu
        if geometry is not None:
            device = geometry.positions.device
        else:
            device = kwargs.get('device', torch.device('cpu'))

        if kwargs.get('orbital_resolve', False):
            raise NotImplementedError('Not implement orbital resolved U.')

        # Bicubic interpolation will be implemented soon
        if interpolation == 'CubicSpline':
            interpolator = CubicSpline
            assert device == torch.device('cpu'), 'device must be cpu if ' + \
                ' interpolation is CubicSpline, but get %s' % device
        elif interpolation == 'PolyInterpU':
            interpolator = PolyInterpU
        elif interpolation == 'BicubInterp':
            interpolator = BicubInterp
        else:
            raise NotImplementedError('%s is not implemented.' % interpolation)

        if (is_dir := os.path.isdir(path)):
            warn('"hdf" binary Slater-Koster files are suggested, TBMaLT'
                 ' will generate binary with smoothed integral tails.')

        # unique atom pairs is from either elements or geometry
        assert elements is not None or geometry is not None, 'both ' + \
            'elements and geometry are None.'

        # create a blank dict for integrals
        hs_dict, onsite_hs_dict = {}, {}

        # get global unique element species pair with geometry object
        if geometry is not None:
            element_pair = unique_atom_pairs(geometry)
        elif elements is not None:
            element_pair = unique_atom_pairs(symbols=elements)
        if skf_type == 'skf':
            _element_name = [batch_chemical_symbols(ie) for ie in element_pair]
            _path = [os.path.join(path, ien[0] + '-' + ien[1] + '.skf')
                     for ie, ien in zip(element_pair, _element_name)]

        # check path existence, type (dir or file) and suffix
        if skf_type == 'skf':
            _path_vcr = [os.path.join(
                path, ien[0] + '-' + ien[1] + '_' + "{:05.2f}".format(ir)
                + '_' + "{:05.2f}".format(jr) + '.skf')
                for ie, ien in zip(element_pair, _element_name)
                for ir in vcr for jr in vcr]
        else:
            _path_vcr = path

        hs_dict, onsite_hs_dict = cls._read_vcr(
            hs_dict, onsite_hs_dict, interpolator, vcr, element_pair,
            _path_vcr, _path, skf_type, integral_type, shell_dict, device, **kwargs)

        return cls(hs_dict, onsite_hs_dict, vcr, **kwargs)

    @classmethod
    def _read(cls, hs_dict: dict, onsite_hs_dict: dict,
              interpolator, element_pair, _path, skf_type,
              integral_type, shell_dict,
              device, **kwargs):
        """Read."""
        for ielement, ipath in zip(element_pair, _path):

            atom_pair = ielement if skf_type == 'h5' else None
            skf = Skf.read(ipath, atom_pair, device=device, **kwargs)

            # generate H or S in SKF files dict
            hs_dict = _get_hs_dict(
                hs_dict, interpolator, skf, integral_type, **kwargs)

            if ielement[0] == ielement[1]:
                onsite_hs_dict = _get_onsite_dict(
                    onsite_hs_dict, skf, shell_dict, integral_type)

        return hs_dict, onsite_hs_dict

    @classmethod
    def _read_vcr(cls, hs_dict: dict, onsite_hs_dict: dict,
              interpolator, vcr, element_pair, _path_vcr, _path, skf_type,
              integral_type, shell_dict,
              device, **kwargs):
        """Read."""
        vcrskf = VcrSkf.read(_path_vcr, element_pair, path_homo=_path)
        if isinstance(vcr, list):
            vcr = torch.tensor(vcr)

        for skf in vcrskf:

            # generate H or S in SKF files dict
            hs_dict = _get_hs_dict(
                hs_dict, interpolator, skf, integral_type,
                vcr=vcr, **kwargs)

            if skf.atom_pair[0] == skf.atom_pair[1]:
                onsite_hs_dict = _get_onsite_dict(
                    onsite_hs_dict, skf, shell_dict, integral_type)

        return hs_dict, onsite_hs_dict

    def off_site(self, atom_pair: Tensor, shell_pair: Tensor,
                 distances: Tensor, g_compr: tuple, **kwargs) -> Tensor:
        """Get integrals for given geometrys.

        Arguments:
            distances: distances of single & multi systems.
            atom_pair: skf files type. Support normal skf, h5py binary skf.
            l_pair: The quantum number ℓ pairs.
            ski_type: Type of integral, H or S.

        Keyword Args:
            orbital_resolve: If each orbital is resolved for U.
            abcd: abcd parameters in cubic spline method.

        Returns:
            integral: Getting integral in SKF tables with given atom pair, ℓ
                number pair, distance, or compression radii pair.
        """
        if kwargs.get('orbital_resolve', False):
            raise NotImplementedError('Not implement orbital resolved U.')

        splines = self.off_site_dict[(*atom_pair.tolist(), *shell_pair.tolist())]

        # call the interpolator
        if g_compr[0] is None:
            integral = splines(distances)
        else:
            integral = splines(g_compr[0], g_compr[0], distances)

        if isinstance(integral[0], np.ndarray):
            integral = torch.from_numpy(integral)

        return integral
        # return pack(list_integral).T

    def on_site(self, atomic_numbers: Tensor, **kwargs) -> List[Tensor]:
        """Returns the specified on-site terms.

        In sktable dictionary, s, p, d and f orbitals of original on-site
        from Slater-Koster files have been expanded one, three, five and
        seven times in default. The expansion of on-site could be controlled
        by `orbital_expand` when loading Slater-Koster integral tables. The
        output on-site size relies on the defined `max_ls`.

        Arguments:
            atomic_numbers: Atomic numbers for which on-site terms should be
                returned.
        max_ls: A dictionary specifying the maximum permitted angular momentum
            associated with a each atomic number. keys must be integers not
            torch tensors.

        Returns:
            on_sites: Tuple of on-site term tensors, one for each atom in
                `atomic_numbers`.

        """
        return [self.on_site_dict[(ian.tolist())] for ian in atomic_numbers]

    def to_hdf5(self, target: Group):
        """Saves standard instance into a target HDF5 Group.

        Arguments:
            target: The hdf5 group to which the system's data should be saved.

        Notes:
            This function does not create its own group as it expects that
            `target` is the group into which data should be writen.

        """
        pass


def _get_hs_dict(hs_dict: dict, interpolator: object,
                 skf: object, skf_type: str, vcr=None,
                 **kwargs) -> Tuple[dict, dict]:
    """Get Hamiltonian or overlap tables for each orbital interaction.

    Arguments:
        h_dict: Hamiltonian tables dictionary.
        s_dict: Overlap tables dictionary.
        interpolator: Slater-Koster interpolation method.
        interactions: Orbital interactions, e.g. (0, 0, 0) for ss0 orbital.
        skf: Object with original SKF integrals data.

    Returns:
        h_dict: Dictionary with updated Hamiltonian tables.
        s_dict: Dictionary with updated overlap tables.

    """
    build_abcd = kwargs.get('build_abcd', False)
    hs_data = getattr(skf, 'hamiltonian') if skf_type == 'H' else \
        getattr(skf, 'overlap')

    for ii, interaction in enumerate(hs_data.keys()):
        if vcr is None:
            hs_dict[(*skf.atom_pair.tolist(), *interaction)] = interpolator(
                skf.grid, hs_data[interaction].T)

            # write spline parameters into list
            if build_abcd:
                hs_dict['variable'].append(
                    hs_dict[(*skf.atom_pair.tolist(),
                             *interaction)].abcd.requires_grad_(build_abcd))

        else:
            hs_dict[(*skf.atom_pair.tolist(), *interaction)] = interpolator(
                vcr, vcr, skf.grid, hs_data[interaction])

    return hs_dict


def _get_onsite_dict(onsite_hs_dict: dict, skf: object, shell_dict, integral_type,
                     **kwargs) -> Tuple[dict, dict]:
    """Write on-site of Hamiltonian or overlap.

    In default, the on-site has been expanded according to orbital shell.
    If maximum orbital from SKF fikes is d, the original s, p and d on-site
    will repeat one, three and five times.

    Arguments:
        onsite_h_dict: Hamiltonian onsite dictionary.
        onsite_s_dict: Overlap onsite dictionary.
        skf: Object with original SKF data.
        max_l: An integer specifying the maximum permitted angular momentum
            associated with the specific atomic number in this function.
        onsite_h_feed: If True, return Hamiltonian onsite, else return dict
            `onsite_h_dict` with any changes.
        onsite_s_feed: If True, return overlap onsite, else return dict
            `onsite_s_dict` with any changes.

    Returns:
        onsite_h_dict: Hamiltonian onsite dictionary.
        onsite_s_dict: Overlap onsite dictionary.

    """
    if kwargs.get('orbital_resolve', False):
        raise NotImplementedError('Not implement orbital resolved Hubbert U.')

    # # get index to expand homo parameters according to the orbitals
    max_l = max(shell_dict[int(skf.atom_pair[0])])
    if kwargs.get('orbital_expand', True):
        orb_index = [(ii + 1) ** 2 - ii ** 2 for ii in range(max_l + 1)]
    else:
        orb_index = [1] * len(skf.onsite)

    # flip make sure the order is along s, p ...
    if integral_type == 'H':
        onsite_hs_dict[(skf.atom_pair[0].tolist())] = torch.cat([
            isk.repeat(ioi) for ioi, isk in zip(
                orb_index, skf.on_sites[: max_l + 1])])
    # onsite_hs_dict[(skf.atom_pair[0].tolist())] = skf.on_sites
    elif integral_type == 'S':
        onsite_hs_dict[(skf.atom_pair[0].tolist())] = torch.cat([
            torch.ones(ioi) for ioi in orb_index])

    return onsite_hs_dict



# Type alias to improve PEP484 readability
SkFeed = _SkFeed
