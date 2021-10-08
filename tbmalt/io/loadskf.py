"""Load Slater-Koster Tables."""
import os
import numpy as np
import torch
import h5py
from scipy.interpolate import CubicSpline
from tbmalt.common.batch import pack
from tbmalt.common.maths.interpolator import SKInterpolation, BicubInterp, Spline1d, smooth_tail_batch
Tensor = torch.Tensor
_orb = {1: 's', 6: 'p', 7: 'p', 8: 'p', 14:'p', 15: 'p', 16: 'p', 79: 'd'}
_onsite = {(1, 1, 'onsite'): torch.tensor([-2.386005440483E-01]),
           (6, 6, 'onsite'): torch.tensor([
               -5.048917654803E-01, -1.943551799182E-01, -1.943551799182E-01,
               -1.943551799182E-01]),
           (7, 7, 'onsite'): torch.tensor([
               -6.400000000000E-01, -2.607280834222E-01, -2.607280834222E-01,
               -2.607280834222E-01]),
           (8, 8, 'onsite'): torch.tensor([
               -8.788325840767E-01, -3.321317735288E-01, -3.321317735288E-01,
               -3.321317735288E-01]),
           (14, 14, 'onsite'): torch.tensor([
               -0.39572506, -0.15031380, -0.15031380,
               -0.15031380])}
_U = {(1, 1, 'U'): torch.tensor([4.196174261214E-01]),
      (6, 6, 'U'): torch.tensor([3.646664973641E-01]),
      (7, 7, 'U'): torch.tensor([4.308879578818E-01]),
      (8, 8, 'U'): torch.tensor([4.954041702122E-01]),
      (14, 14, 'U'): torch.tensor([0.247609])}


class IntegralGenerator:
    """Read skf files and return integrals.

    IntegralGenerator from_dir will read skf files with path of skf files and
    system object. Then return integrals according to parameters, such as
    distance, l_pair, etc.

    Arguments:
        sktable_dict: This should be a dictionary of scipy splines keyed by a
            tuple of strings of the form: (z_1, z_2, ℓ_1, ℓ_2, b, O), where the
            ℓ's are the azimuthal quantum numbers, the z's are the atomic
            numbers, b is the bond type, & O is the operator which the spline
            represents i.e. S or H.

    Examples:
        >>> from ase.build import molecule as molecule_database
        >>> from tbmalt.common.structures.system import System
        >>> from tbmalt.io.loadskf import IntegralGenerator
        >>> molecule = molecule_database('CH4')
        >>> system = System.from_ase_atoms(molecule)
        >>> sk = IntegralGenerator.from_dir('./slko/mio-1-1', system)
        >>> atom_pair = torch.tensor([6, 6])  # Carbon-Carbon pair
        >>> distance = torch.tensor([2.0])
        >>> sk(distance, atom_pair, torch.tensor([1, 1]), hs_type='H')
        >>> tensor([[ 0.329381666662100, -0.263189996289000]])
    """

    def __init__(self, sktable_dict):
        self.sktable_dict = sktable_dict

    @classmethod
    def from_dir(cls, path: str, system=None, elements=None, **kwargs) -> dict:
        """Read all skf files in a directory & return an SKIG instance.

        Arguments:
            path: path to the directory in which skf files can be found. If
                sk_type is normal, the path represents only the directory. if
                sk_type is h5py, the path is the joint directory and h5py file.
                sk_type is compression_radii, the path is the path to list of
                    SKF files with various compression radii.
            system: Optional, system object.
            element: Optional, global elements for read SKF files.

        Keyword Args:
            sk_type: type of skf files.
            interpolation: interpolation method of integrals which are not
                in the grid points.
            orbital_resolve: If each orbital is resolved for U.
        """
        repulsive = kwargs.get('repulsive', False)
        with_variable = kwargs.get('with_variable', False)

        if kwargs.get('orbital_resolve', False):
            raise NotImplementedError('Not implement orbital resolved U.')

        sk_interp = kwargs.get('interpolation', 'sk_interpolation')

        # The interactions: ddσ, ddπ, ddδ, ...
        interactions = [(2, 2, 0), (2, 2, 1), (2, 2, 2), (1, 2, 0), (1, 2, 1),
                        (1, 1, 0), (1, 1, 1), (0, 2, 0), (0, 1, 0), (0, 0, 0)]

        # create a blank dict for integrals
        sktable_dict = {}
        sktable_dict['variable'] = [] if with_variable else None
        sk_cutoff = []
        assert elements is not None or system is not None

        # get global element species and all corresponding SKF files
        if system is not None:
            element, element_number, element_pair, element_number_pair = \
                system.get_global_species()

        # get global element species and all corresponding SKF files
        elif elements is not None:
            element_pair, element_number_pair = _get_element_info(elements)

        elif system is None and element is None:
            raise ValueError('At least one of system and element is not None')

        if sk_interp == 'sk_interpolation':
            interpolator = SKInterpolation
        elif sk_interp == 'bicubic_interpolation':
            interpolator = BicubInterp
        elif sk_interp == 'numpy_interpolation':
            interpolator = CubicSpline
        elif sk_interp == 'spline':
            interpolator = Spline1d

        # loop of all global element pairs
        for ielement, ielement_number in zip(element_pair, element_number_pair):

            # read skf files
            skf = LoadSKF.read(path, ielement, ielement_number, **kwargs)
            sk_cutoff.append(skf.hs_cutoff)

            # generate skf files dict
            sktable_dict = _get_hs_dict(sktable_dict, interpolator, interactions, skf, **kwargs)
            sktable_dict = _get_other_params_dict(sktable_dict, skf)

            if skf.homo:
                # return onsite
                if (*skf.elements.tolist(), 'onsite') not in sktable_dict:
                    sktable_dict = _get_onsite_dict(sktable_dict, skf)

                # retutn U
                if (*skf.elements.tolist(), 'U') not in sktable_dict:
                    sktable_dict = _get_u_dict(sktable_dict, skf, **kwargs)

            if repulsive:
                sktable_dict = _get_repulsive_dict(sktable_dict, skf)

        sktable_dict['sk_cutoff_element_pair'] = sk_cutoff
        # sktable_dict['sk_cutoff'] = max(sk_cutoff)  # return the max cutoff

        return cls(sktable_dict)

    def __call__(self, input1: Tensor, atom_pair: Tensor,
                 l_pair: Tensor, **kwargs) -> Tensor:
        """Get integrals for given systems.

        Arguments:
            input1: distances or compression radii of single & multi systems.
            atom_pair: skf files type. Support normal skf, h5py binary skf.
            l_pair:

        Keyword Args:
            hs_type: type of skf files.
            input2: if input1 is compression radii, input2 is distances.
            orbital_resolve: If each orbital is resolved for U.
        """
        if kwargs.get('orbital_resolve', False):
            raise NotImplementedError('Not implement orbital resolved U.')
        hs_type = kwargs.get('hs_type', 'H')
        input2 = kwargs.get('input2', None)
        get_abcd = kwargs.get('get_abcd', None)

        # retrieve the appropriate splines
        splines = [self.sktable_dict[(
            *atom_pair.tolist(), *l_pair.tolist(), il, hs_type)]
            for il in range(min(l_pair) + 1)]

        # call the interpolator
        if input2 is None and get_abcd is None:
            list_integral = [spline(input1) for spline in splines]
        elif input2 is not None:
            list_integral = [spline(input1, input2) for spline in splines]
        elif get_abcd is not None:
            abcd = [self.sktable_dict[(
                *atom_pair.tolist(), *l_pair.tolist(), il, hs_type, 'abcd')]
                for il in range(min(l_pair) + 1)]
            list_integral = [spline(input1, iabcd) for spline, iabcd in zip(splines, abcd)]

        if type(list_integral[0]) == np.ndarray:
            list_integral = [torch.from_numpy(ii) for ii in list_integral]

        return pack(list_integral).T

    def get_onsite(self, atom_number: Tensor, **kwargs) -> Tensor:
        """Return onsite with onsite_blocks.

        Arguments:
            atom_number: Atomic numbers.
        """
        fix_onsite = kwargs.get('fix_onsite', False)

        if not fix_onsite:
            return torch.cat([self.sktable_dict[
                (*[ii.tolist(), ii.tolist()], 'onsite')] for ii in atom_number])
        else:
            return torch.cat([_onsite[(*[ii.tolist(), ii.tolist()], 'onsite')]
                              for ii in atom_number])

    def get_U(self, atom_number: Tensor, **kwargs) -> Tensor:
        """Return onsite with onsite_blocks.

        Arguments:
            atom_number: Atomic numbers.
        """
        fix_U = kwargs.get('fix_U', False)

        if not fix_U:
            return torch.cat([self.sktable_dict[
                (*[ii.tolist(), ii.tolist()], 'U')] for ii in atom_number])
        else:
            return torch.cat([_U[
                (*[ii.tolist(), ii.tolist()], 'U')] for ii in atom_number])

    @property
    def cutoff(self):
        """Return cutoff from SK input."""
        return self.sktable_dict['sk_cutoff']

    @property
    def cutoff_element_pair(self):
        """Return cutoff for all atom pairs."""
        return self.sktable_dict['sk_cutoff_element_pair']

def _get_element_info(elements: list):
    """Generate element pair information."""
    _elements_dict = {"H": 1, "Li": 3, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9,
                      "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
                      "S": 16, "Ti": 22, "Au": 79}
    element_pair = [[iel, jel] for iel, jel in zip(
        sorted(elements * len(elements)), elements * len(elements))]
    element_number_pair = [[_elements_dict[ii[0]], _elements_dict[ii[1]]]
                           for ii in element_pair]
    return element_pair, element_number_pair


def _get_hs_dict(sktable_dict: dict, interpolator: object,
                 interactions: list, skf: object, **kwargs) -> dict:
    """Get sk tables for each orbital interaction.

    Arguments:
        sktable_dict: sk tables dictionary.
        interpolator: sk interpolation method.
        interactions: orbital interactions, e.g. (0, 0, 0) for ss0 orbital.
        skf: object with SKF integrals data.
    """
    sk_interp = kwargs.get('interpolation', 'sk_interpolation')
    with_variable = kwargs.get('with_variable', False)
    # sktable_dict['variable'] = [] if with_variable else None

    # if sk_type != 'compression_radii':
    if sk_interp in ('sk_interpolation', 'numpy_interpolation'):

        for ii, name in enumerate(interactions):
            sktable_dict[(*skf.elements.tolist(), *name, 'H')] = \
                interpolator(skf.hs_grid, skf.hamiltonian.T[ii])
            sktable_dict[(*skf.elements.tolist(), *name, 'S')] = \
                interpolator(skf.hs_grid, skf.overlap.T[ii])

    elif sk_interp == 'bicubic_interpolation':
        compr_grid = kwargs.get('compression_radii_grid')

        for ii, name in enumerate(interactions):
            sktable_dict[(*skf.elements.tolist(), *name, 'H')] = \
                interpolator(compr_grid, skf.hamiltonian[..., ii], skf.hs_grid)
            sktable_dict[(*skf.elements.tolist(), *name, 'S')] = \
                interpolator(compr_grid, skf.overlap[..., ii], skf.hs_grid)

    elif sk_interp == 'spline':
        for ii, name in enumerate(interactions):
            _h = interpolator(skf.hs_grid, skf.hamiltonian.T[ii], **kwargs)
            _s = interpolator(skf.hs_grid, skf.overlap.T[ii], **kwargs)
            sktable_dict[(*skf.elements.tolist(), *name, 'H')] = _h
            sktable_dict[(*skf.elements.tolist(), *name, 'S')] = _s

            if with_variable:  # get ML optimizer variable
                sktable_dict[(*skf.elements.tolist(), *name, 'H', 'abcd')] = _h.abcd
                sktable_dict[(*skf.elements.tolist(), *name, 'S', 'abcd')] = _s.abcd
                sktable_dict['variable'].append(
                    sktable_dict[(*skf.elements.tolist(), *name, 'H', 'abcd')].requires_grad_(True))
                sktable_dict['variable'].append(
                    sktable_dict[(*skf.elements.tolist(), *name, 'S', 'abcd')].requires_grad_(True))

    return sktable_dict


def _get_onsite_dict(sktable_dict: dict, skf: object) -> dict:
    """Get sk tables for global element.

    Arguments:
        sktable_dict: sk tables dictionary.
        skf: object with SKF integrals data.
    """
    # add onsite, return onsite obey sequence s, p, d ...
    onsite = skf.onsite.unsqueeze(1)

    if _orb[skf.elements.tolist()[0]] == 's':
        sktable_dict[(*skf.elements.tolist(), 'onsite')] = torch.cat([onsite[-1]])

    elif _orb[skf.elements.tolist()[0]] == 'p':
        sktable_dict[(*skf.elements.tolist(), 'onsite')] = \
            torch.cat([onsite[-1], onsite[-2].repeat(3)])

    elif _orb[skf.elements.tolist()[0]] == 'd':
        sktable_dict[(*skf.elements.tolist(), 'onsite')] = \
            torch.cat([onsite[-1], onsite[-2].repeat(3), onsite[-3].repeat(5)])
    return sktable_dict


def _get_u_dict(sktable_dict: dict, skf: object, **kwargs) -> dict:
    """Get sk tables for global element.

    Arguments:
        sktable_dict: sk tables dictionary.
        skf: object with SKF integrals data.
    """
    if kwargs.get('orbital_resolve', False):
        raise NotImplementedError('Not implement orbital resolved U.')

    # add onsite, return onsite obey sequence s, p, d ...
    sktable_dict[(*skf.elements.tolist(), 'U')] = skf.U.unsqueeze(1)[-1]
    return sktable_dict


def _get_repulsive_dict(sktable_dict: dict, skf: object) -> dict:
    """Return repulsive parameter."""
    sktable_dict[(*skf.elements.tolist(), 'n_repulsive')] = skf.n_repulsive
    sktable_dict[(*skf.elements.tolist(), 'rep_cutoff')] = skf.rep_cutoff
    sktable_dict[(*skf.elements.tolist(), 'rep_table')] = skf.rep_table
    sktable_dict[(*skf.elements.tolist(), 'rep_grid')] = skf.rep_grid
    sktable_dict[(*skf.elements.tolist(), 'rep_short')] = skf.rep_short
    sktable_dict[(*skf.elements.tolist(), 'rep_long_c')] = skf.rep_long_c
    sktable_dict[(*skf.elements.tolist(), 'rep_long_grid')] = skf.rep_long_grid
    return sktable_dict


def _get_other_params_dict(sktable_dict: dict, skf: object) -> dict:
    """Return other parameters except HS, onsite, U and repulsive."""
    sktable_dict[(*skf.elements.tolist(), 'g_step')] = skf.g_step
    sktable_dict[(*skf.elements.tolist(), 'n_points')] = skf.n_points
    sktable_dict[(*skf.elements.tolist(), 'hs_cutoff')] = skf.hs_cutoff
    sktable_dict[(*skf.elements.tolist(), 'hs_grid')] = skf.hs_grid
    sktable_dict[(*skf.elements.tolist(), 'version')] = skf.version
    return sktable_dict


class LoadSKF:
    """Get integrals for given systems.

        Arguments:
            elements: global elements of single & multi systems.
            hamiltonian: Hamiltonian skf table data.
            overlap: Overlap skf table data.

        Keyword Args:
            repulsive: If read repulsive data.
    """

    def __init__(self, elements, hamiltonian: Tensor, overlap: Tensor,
                 hs_grid: Tensor, hs_cutoff, **kwargs):
        repulsive = kwargs.get('repulsive', False)
        # If a single element was specified, resolve it to a tensor
        if isinstance(elements, int):
            self.elements = torch.tensor([elements]*2)
        elif isinstance(elements, list):
            self.elements = torch.tensor(elements)
        else:
            self.elements = elements

        # homo or hetero
        self.homo = kwargs.get('homo', self.elements[0] == self.elements[1])
        if self.homo:
            self.onsite = kwargs['onsite']
            self.U = kwargs['U']

        self.hamiltonian = hamiltonian
        self.overlap = overlap
        self.hs_grid = hs_grid
        self.g_step = kwargs['g_step']
        self.n_points = kwargs['n_points']

        # Repulsion properties
        self.hs_cutoff = hs_cutoff

        if repulsive:
            self.n_repulsive = kwargs['n_repulsive']
            self.rep_cutoff = kwargs['rep_cutoff']
            self.rep_table = kwargs['rep_table']
            self.rep_grid = kwargs['rep_grid']
            self.rep_long_grid = kwargs['rep_long_grid']
            self.rep_short = kwargs['rep_short']
            self.rep_long_c = kwargs['rep_long_c']

        # for polynomial representation of
        self.rep_poly = kwargs.get('rep_poly', None)

        # identify the skf specification version
        self.version = kwargs.get('version', 'unclear')

    @classmethod
    def read(cls, path: str, element: list, element_number: list, **kwargs):
        """Read different type SKF files.

        Arguments:
            path: path to SKF files.
            sk_type: type of SKF files.
            element: element pair of SKF files.
            element_number: pair element numbers.
        """
        sk_type = kwargs.get('sk_type', 'normal')
        if sk_type == 'normal':
            return cls.read_normal(path, element, element_number, **kwargs)
        elif sk_type == 'h5py':
            return cls.read_hdf(path, element, element_number, **kwargs)
        elif sk_type == 'compression_radii':
            return cls.read_compression_radii(
                path, element, element_number, **kwargs)

    @classmethod
    def _get_version_number(cls, file: str, lines: list) -> str:
        """Return skf version, see: https://github.com/bhourahine/slako-doc."""
        if file.startswith('@'):  # If 1'st char is @, the version is 1.0e
            return '1.0e'
        elif len(lines[0].split()) == 2 and lines[0].split()[1].isnumeric():
            return '0.9'  # If no comment line; this must be version 0.9
        else:
            return 'unclear'

    @classmethod
    def _asterisk_to_repeat_tensor(cls, xx: list) -> Tensor:
        """Transfer 'int*float' to repetitive tensor float.repeat(int)."""
        return torch.cat([torch.tensor([float(ii.split('*')[1])]).repeat(
            int(ii.split('*')[0])) if '*' in ii else torch.tensor([float(ii)])
            for ii in xx])

    @classmethod
    def read_normal(cls, path: str, element: list,
                    element_number: list, **kwargs):
        """Read in a skf file and returns an SKF_File instance.

        File names should follow the naming convention X-Y.skf where X and
        Y are the chemical symbol's of the elements involved.

        Arguments:
            path_to_skf: Path to the target skf file.
            element_number: Input element number pair.
        """
        repulsive = kwargs.get('repulsive', False)
        homo = kwargs.get('homo', element_number[0] == element_number[1])

        # alias for common code structure
        lmf = lambda xx: list(map(float, xx.split()))
        _asterisk = lambda xx: list(map(str.strip, xx.split()))

        if 'path_to_skf' in kwargs:
            path_to_skf = kwargs.get('path_to_skf')
        else:
            path_to_skf = os.path.join(
                path, element[0] + '-' + element[1] + '.skf')

        file = open(path_to_skf, 'r').read()
        lines = file.split('\n')

        ver = cls._get_version_number(file, lines)
        lines = lines[1:] if ver in ['1.0e'] else lines

        # Get atomic numbers & identify if this is the homo case
        # homo = element_number[0] == element_number[1]
        is_asterisk = '*' in lines[2] or '*' in lines[2 + homo]

        if homo:
            if is_asterisk:
                homo_ln = cls._asterisk_to_repeat_tensor(_asterisk(
                    lines[1].replace(',', ' ')))
            else:
                homo_ln = torch.tensor(lmf(lines[1]))
            n = int((len(homo_ln) - 1) / 3)  # <- Number of shells specified
            onsite, _, U, occupations = homo_ln.split_with_sizes([n, 1, n, n])

        # grid distance and grid points number
        g_step, n_points = lines[0].replace(',', ' ').split()[:2]
        g_step, n_points = float(g_step), int(n_points)

        # construct distance list. Note; distance start at 1 * g_step not zero
        hs_grid = torch.arange(1, n_points + 1) * g_step

        # Fetch the H and S sk tables
        if is_asterisk:
            hamiltonian, overlap = torch.stack([
                cls._asterisk_to_repeat_tensor(_asterisk(ii.replace(',', ' ')))
                for ii in lines[2 + homo: 2 + n_points + homo]]).chunk(2, 1)
        else:
            hamiltonian, overlap = torch.tensor(
                [lmf(ii) for ii in lines[2 + homo: 2 + n_points + homo]]).chunk(2, 1)

        # 2 + homo line data, which contains mass, rcut and cutoff
        if is_asterisk:
            mass, *r_poly, cutoff = cls._asterisk_to_repeat_tensor(_asterisk(
                lines[2 + homo].replace(',', '')))[:10]
        else:
            mass, *r_poly, cutoff = torch.tensor(lmf(lines[2 + homo]))[:10]

        hs_cutoff = (n_points - 1) * g_step

        # Check if there is a spline representation
        if 'Spline' in file and repulsive:
            start = lines.index('Spline') + 1  # Identify spline section start

            # get number of spline sections & overwrite the r_cutoff previously
            nrep, r_cutoff = lines[start].split()
            nrep, r_cutoff = int(nrep), float(r_cutoff)

            r_tab = torch.tensor([lmf(line) for line in
                                  lines[start + 2: start + 1 + nrep]])
            rep = r_tab[:, 2:]  # -> repulsive tables
            r_grid = torch.tensor([*r_tab[:, 0], r_tab[-1, 1]])

            # Get the short and long range terms
            r_short = torch.tensor(lmf(lines[start + 1]))
            r_long_tab = torch.tensor(lmf(lines[start + 1 + nrep]))
            r_long_grid = r_long_tab[:2]
            r_long_c = r_long_tab[2:]

        # Build the parameter lists to pass to the in_grid_pointst method
        pos = (element_number, hamiltonian, overlap, hs_grid, hs_cutoff)

        # Those that are passed by keyword
        kwd = {'version': ver, 'rep_poly': r_poly, 'homo': homo,
               'g_step': g_step, 'n_points': n_points}

        if homo:  # Passed only if homo case
            kwd.update({'mass': mass, 'onsite': onsite, 'U': U,
                        'occupations': occupations})

        if 'Spline' in file and repulsive:
            kwd.update({'n_repulsive': nrep, 'rep_table': rep,
                        'rep_grid': r_grid, 'rep_short': r_short,
                        'rep_long_grid': r_long_grid, 'rep_long_c': r_long_c,
                        'rep_cutoff': r_cutoff, 'repulsive': True})

        return cls(*pos, **kwd)

    @classmethod
    def read_hdf(cls, path: str, element: list,
                 element_number: list, **kwargs):
        """Generate integral from h5py binary data."""
        repulsive = kwargs.get('repulsive', False)
        homo = kwargs.get('homo', element_number[0] == element_number[1])

        element_ij = element[0] + element[1]
        if not os.path.isfile(path):
            raise FileExistsError('dataset %s do not exist' % path)

        kwd = {}  # create empty dict
        with h5py.File(path, 'r') as f:
            hs_grid = torch.from_numpy(f[element_ij + '/hs_grid'][()])
            hs_cutoff = f[element_ij + '/hs_cutoff'][()]
            hamiltonian = torch.from_numpy(f[element_ij + '/hamiltonian'][()])
            overlap = torch.from_numpy(f[element_ij + '/overlap'][()])

            # ver = f[element_ij + '/version'][()]
            ver = f[element_ij].attrs['version']
            rep_poly = f[element_ij + '/rep_poly'][()]
            g_step = f[element_ij + '/g_step'][()]
            n_points = f[element_ij + '/n_points'][()]

            # return hamiltonian, overlap, and related data
            pos = (element_number, hamiltonian, overlap, hs_grid, hs_cutoff)
            kwd = {'version': ver, 'rep_poly': rep_poly, 'homo': homo,
                   'g_step': g_step, 'n_points': n_points}

            # homo SKF files
            if homo:
                onsite = torch.from_numpy(f[element_ij + '/onsite'][()])
                U = torch.from_numpy(f[element_ij + '/U'][()])
                kwd.update({'onsite': onsite, 'U': U})

            if repulsive:
                nrep = f[element_ij + '/n_repulsive'][()]
                rep = torch.from_numpy(f[element_ij + '/rep_table'][()])
                r_grid = torch.from_numpy(f[element_ij + '/rep_grid'][()])
                r_short = torch.from_numpy(f[element_ij + '/rep_short'][()])
                r_long_grid = torch.from_numpy(f[element_ij + '/rep_long_grid'][()])
                r_long_c = torch.from_numpy(f[element_ij + '/rep_long_c'][()])
                r_cutoff = f[element_ij + '/rep_cutoff'][()]
                kwd.update({'n_repulsive': nrep, 'rep_table': rep,
                            'rep_grid': r_grid, 'rep_short': r_short,
                            'rep_long_grid': r_long_grid, 'repulsive': True,
                            'rep_cutoff': r_cutoff, 'rep_long_c': r_long_c})

        return cls(*pos, **kwd)

    @classmethod
    def read_compression_radii(
            cls, path: str, element: list, element_number: list, **kwargs):
        """Read in a skf file and returns an SKF_File instance."""
        repulsive = kwargs.get('repulsive', False)
        homo = kwargs.get('homo', element[0] == element[1])

        name_prefix = element[0] + '-' + element[1] + '.skf'
        filenamelist = []
        filenames = os.listdir(path)
        for name in filenames:
            if name.startswith(name_prefix):
                filenamelist.append(name)
        n_compr = int(np.sqrt(len(filenamelist)))
        n_points = []

        ver, r_poly, ham, over, hs_grid, cutoff = [], [], [], [], [], []
        if homo:
            onsite, U, occupations, mass = _allocate_homo(n_compr)
        if repulsive:
            n_rep, rep_cutoff, rep_table, rep_grid = [], [], [], []
            rep_short, rep_long_c, rep_long_grid = [], [], []

        for icp, iname in enumerate(filenamelist):
            row = int(icp // n_compr)
            col = int(icp % n_compr)

            iname = os.path.join(path, iname)
            # call the normal read function
            skf = LoadSKF.read_normal(path, element, element_number,
                                      path_to_skf=iname, **kwargs)

            n_points.append(skf.n_points)
            ham.append(skf.hamiltonian), over.append(skf.overlap)
            hs_grid.append(skf.hs_grid), cutoff.append(skf.hs_cutoff)
            ver.append(skf.version), r_poly.append(skf.rep_poly)

            if homo:  # Passed only if homo case
                mass[row, col] = skf.mass
                onsite[row, col] = skf.onsite
                U[row, col] = skf.U
                occupations[row, col] = skf.occupations
            if repulsive:
                n_rep.append(skf.n_repulsive), rep_cutoff.append(skf.rep_cutoff)
                rep_table.append(skf.rep_table), rep_grid.append(skf.rep_grid)
                rep_short.append(skf.rep_short), rep_long_c.append(skf.rep_long_c)
                rep_long_grid.append(skf.rep_long_grid)

        g_step = skf.g_step  # g_step should be all the same
        hs_grid = torch.arange(1, int(max(n_points) + 5) + 1) * g_step

        hamiltonian = torch.zeros(n_compr, n_compr, max(n_points) + 5, 10)
        overlap = torch.zeros(n_compr, n_compr, max(n_points) + 5, 10)
        hamiltonian[:, :, :int(max(n_points))] = pack([pack(
            [ham[icp * n_compr + jj] for jj in range(n_compr)])
            for icp in range(n_compr)])
        overlap[:, :, :int(max(n_points))] = pack([pack(
            [over[icp * n_compr + jj] for jj in range(n_compr)])
            for icp in range(n_compr)])

        # smooth the tail
        hamiltonian = smooth_tail_batch(
            hs_grid.repeat(icp + 1, 1), hamiltonian.reshape(
                icp + 1, -1, hamiltonian.shape[-1]), torch.tensor(n_points)
            ).reshape(hamiltonian.shape[0], hamiltonian.shape[1], hamiltonian.shape[2], -1)
        overlap = smooth_tail_batch(
            hs_grid.repeat(icp + 1, 1), overlap.reshape(icp + 1, -1, overlap.shape[-1]),
            torch.tensor(n_points)).reshape(overlap.shape[0], overlap.shape[1],
                                            overlap.shape[2], -1)

        # Build the parameter lists to pass to the in_grid_pointst method
        pos = (element_number, hamiltonian, overlap, hs_grid, cutoff)

        # Those that are passed by keyword
        kwd = {'version': ver, 'rep_poly': r_poly, 'homo': homo,
               'g_step': g_step, 'n_points': n_points}

        if homo:  # Passed only if homo case
            kwd.update({'mass': mass, 'onsite': onsite, 'U': U,
                        'occupations': occupations})

        if repulsive:
            kwd.update({'n_repulsive': n_rep, 'rep_table': rep_table,
                        'rep_grid': rep_grid, 'rep_short': rep_short,
                        'rep_long_grid': rep_long_grid, 'rep_long_c': rep_long_c,
                        'rep_cutoff': rep_cutoff, 'repulsive': True})
        return cls(*pos, **kwd)


def _allocate_homo(n_compr):
    """Allocate empty tensor if homo."""
    onsite = torch.zeros(n_compr, n_compr, 3)
    U = torch.zeros(n_compr, n_compr, 3)
    occupations = torch.zeros(n_compr, n_compr, 3)
    mass = torch.zeros(n_compr, n_compr, 1)
    return onsite, U, occupations, mass
