# -*- coding: utf-8 -*-
"""A container to hold data associated with a chemical system's structure.

This module provides the `Geometry` data structure class and its associated
code. The `Geometry` class is intended to hold any & all data needed to fully
describe a chemical system's structure.
"""
from typing import Union, List, Optional
from operator import itemgetter
import torch
import numpy as np
from h5py import Group
from ase import Atoms
import ase.io as io
from tbmalt.common.batch import pack, merge
from tbmalt.data.units import length_units
from tbmalt.structures.cell import Pbc

from tbmalt.data import chemical_symbols
from tbmalt.data import atomic_numbers as data_numbers
Tensor = torch.Tensor


class Geometry:
    """Data structure for storing geometric information on molecular systems.

    The `Geometry` class stores any information that is needed to describe a
    chemical system; atomic numbers, positions, etc. This class also permits
    batch system representation. However, mixing of PBC & non-PBC systems is
    strictly forbidden.

    Arguments:
        atomic_numbers: Atomic numbers of the atoms.
        positions : Coordinates of the atoms.
        units: Unit in which ``positions`` were specified. For a list of
            available units see :mod:`.units`. [DEFAULT='bohr']

    Attributes:
        atomic_numbers: Atomic numbers of the atoms.
        positions : Coordinates of the atoms.
        n_atoms: Number of atoms in the system.

    Notes:
        When representing multiple systems, the `atomic_numbers` & `positions`
        tensors will be padded with zeros. Tensors generated from ase atoms
        objects or HDF5 database entities will not share memory with their
        associated numpy arrays, nor will they inherit their dtype.

    Warnings:
        At this time, periodic boundary conditions are not supported.

    Examples:
        Geometry instances may be created by directly passing in the atomic
        numbers & atom positions

        >>> from tbmalt import Geometry
        >>> H2 = Geometry(torch.tensor([1, 1]),
        >>>               torch.tensor([[0.00, 0.00, 0.00],
        >>>                             [0.00, 0.00, 0.79]]))
        >>> print(H2)
        Geometry(H2)

        Or from an ase.Atoms object

        >>> from ase.build import molecule
        >>> CH4_atoms = molecule('CH4')
        >>> print(CH4_atoms)
        Atoms(symbols='CH4', pbc=False)
        >>> CH4 = Geometry.from_ase_atoms(CH4_atoms)
        >>> print(CH4)
        Geometry(CH4)

        Multiple systems can be represented by a single ``Geometry`` instance.
        To do this, simply pass in lists or packed tensors where appropriate.

    """

    __slots__ = ['atomic_numbers', 'positions', 'n_atoms', 'updated_dist_vec',
                 'cell', 'is_periodic', 'periodic_list', 'frac_list', 'pbc',
                 '_n_batch', '_mask_dist', '__dtype', '__device']

    def __init__(self, atomic_numbers: Union[Tensor, List[Tensor]],
                 positions: Union[Tensor, List[Tensor]],
                 cell: Union[Tensor, List[Tensor]] = None,
                 frac: Union[float, List[float]] = None,
                 units: Optional[str] = 'bohr', **kwargs):

        # "pack" will only effect lists of tensors
        self.atomic_numbers: Tensor = pack(atomic_numbers)
        self.positions: Tensor = pack(positions)
        self.updated_dist_vec = kwargs.get('updated_dist_vec', None)

        # bool tensor is_periodic defines if there is solid
        if cell is None:
            self.is_periodic = False  # no system is solid
            self.cell = None
        else:
            cell = pack(cell)
            if cell.eq(0).all():
                self.is_periodic = False  # all cell is zeros
                self.cell = None
            else:
                _cell = Pbc(cell, frac, units)
                self.cell, self.periodic_list, self.frac_list, self.pbc = \
                    _cell.cell, _cell.periodic_list, _cell.frac_list, _cell.pbc
                self.is_periodic = True if self.periodic_list.any() else False

        # Mask for clearing padding values in the distance matrix.
        if (temp_mask := self.atomic_numbers != 0).all():
            self._mask_dist: Union[Tensor, bool] = False
        else:
            self._mask_dist: Union[Tensor, bool] = ~(
                temp_mask.unsqueeze(-2) * temp_mask.unsqueeze(-1))

        self.n_atoms: Tensor = self.atomic_numbers.count_nonzero(-1)

        # Number of batches if in batch mode (for internal use only)
        self._n_batch: Optional[int] = (None if self.atomic_numbers.dim() == 1
                                        else len(atomic_numbers))

        # Ensure the distances are in atomic units (bohr)
        if units != 'bohr':
            self.positions: Tensor = self.positions * length_units[units]

        # These are static, private variables and must NEVER be modified!
        self.__device = self.positions.device
        self.__dtype = self.positions.dtype

        # Check for size discrepancies in `positions` & `atomic_numbers`
        if self.atomic_numbers.ndim == 2:
            check = len(atomic_numbers) == len(positions)
            assert check, '`atomic_numbers` & `positions` size mismatch found'

        # Ensure tensors are on the same device (only two present currently)
        if self.positions.device != self.positions.device:
            raise RuntimeError('All tensors must be on the same device!')

    @property
    def device(self) -> torch.device:
        """The device on which the `Geometry` object resides."""
        return self.__device

    @device.setter
    def device(self, *args):
        # Instruct users to use the ".to" method if wanting to change device.
        raise AttributeError('Geometry object\'s dtype can only be modified '
                             'via the ".to" method.')

    @property
    def dtype(self) -> torch.dtype:
        """Floating point dtype used by geometry object."""
        return self.__dtype

    @property
    def distances(self) -> Tensor:
        """Distance matrix between atoms in the system."""
        if self.updated_dist_vec is None:
            dist = torch.cdist(self.positions, self.positions, p=2)
            # Ensure padding area is zeroed out
            dist[self._mask_dist] = 0
            torch.diagonal(dist, dim1=-2, dim2=-1).zero_()
            return dist
        else:
            return torch.sqrt((self.updated_dist_vec ** 2).sum(-1))

    @property
    def distance_vectors(self) -> Tensor:
        """Distance vector matrix between atoms in the system."""
        if self.updated_dist_vec is None:
            dist_vec = self.positions.unsqueeze(-2) - self.positions.unsqueeze(-3)
            dist_vec[self._mask_dist] = 0
            return dist_vec
        else:
            return self.updated_dist_vec

    @property
    def chemical_symbols(self) -> list:
        """Chemical symbols of the atoms present."""
        return batch_chemical_symbols(self.atomic_numbers)

    def unique_atomic_numbers(self) -> Tensor:
        """Identifies and returns a tensor of unique atomic numbers.

        This method offers a means to identify the types of elements present
        in the system(s) represented by a `Geometry` object.

        Returns:
            unique_atomic_numbers: A tensor specifying the unique atomic
                numbers present.
        """
        return torch.unique(self.atomic_numbers[self.atomic_numbers.ne(0)])

    @classmethod
    def from_ase_atoms(cls, atoms: Union[Atoms, List[Atoms]],
                       device: Optional[torch.device] = None,
                       dtype: Optional[torch.dtype] = None,
                       units: str = 'angstrom') -> 'Geometry':
        """Instantiates a Geometry instance from an `ase.Atoms` object.

        Multiple atoms objects can be passed in to generate a batched Geometry
        instance which represents multiple systems.

        Arguments:
            atoms: Atoms object(s) to instantiate a Geometry instance from.
            device: Device on which to create any new tensors. [DEFAULT=None]
            dtype: dtype to be used for floating point tensors. [DEFAULT=None]
            units: Length unit used by `Atoms` object. [DEFAULT='angstrom']

        Returns:
            geometry: The resulting ``Geometry`` object.

        Warnings:
            Periodic boundary conditions are not currently supported.

        Raises:
            NotImplementedError: If the `ase.Atoms` object has periodic
                boundary conditions enabled along any axis.
        """
        # If not specified by the user; ensure that the default dtype is used,
        # rather than inheriting from numpy. Failing to do this will case some
        # *very* hard to diagnose errors.
        dtype = torch.get_default_dtype() if dtype is None else dtype

        if not isinstance(atoms, list):  # If a single system
            return cls(  # Create a Geometry instance and return it
                torch.tensor(atoms.get_atomic_numbers(), device=device),
                torch.tensor(atoms.positions, device=device, dtype=dtype),
                torch.tensor(atoms.cell, device=device, dtype=dtype),
                units=units)

        else:  # If a batch of systems
            return cls(  # Create a batched Geometry instance and return it
                [torch.tensor(a.get_atomic_numbers(), device=device)
                 for a in atoms],
                [torch.tensor(a.positions, device=device, dtype=dtype)
                 for a in atoms],
                [torch.tensor(a.cell, device=device, dtype=dtype)
                 for a in atoms],
                units=units)

    @classmethod
    def from_hdf5(cls, source: Union[Group, List[Group]],
                  device: Optional[torch.device] = None,
                  dtype: Optional[torch.dtype] = None,
                  units: str = 'bohr') -> 'Geometry':
        """Instantiate a `Geometry` instances from an HDF5 group.

        Construct a `Geometry` entity using data from an HDF5 group. Passing
        multiple groups, or a single group representing multiple systems, will
        return a batched `Geometry` instance.

        Arguments:
            source: An HDF5 group(s) containing geometry data.
            device: Device on which to place tensors. [DEFAULT=None]
            dtype: dtype to be used for floating point tensors. [DEFAULT=None]
            units: Unit of length used by the data. [DEFAULT='bohr']

        Returns:
            geometry: The resulting ``Geometry`` object.

        """
        # If not specified by the user; ensure that the default dtype is used,
        # rather than inheriting from numpy. Failing to do this will case some
        # *very* hard to diagnose errors.
        dtype = torch.get_default_dtype() if dtype is None else dtype

        # If a single system or a batch system
        if not isinstance(source, list):
            # Read & parse a datasets from the database into a System instance
            # & return the result.
            return cls(torch.tensor(source['atomic_numbers'], device=device),
                       torch.tensor(source['positions'], dtype=dtype,
                                    device=device),
                       units=units)
        else:
            return cls(  # Create a batched Geometry instance and return it
                [torch.tensor(s['atomic_numbers'], device=device)
                 for s in source],
                [torch.tensor(s['positions'], device=device, dtype=dtype)
                 for s in source],
                units=units)

    def to_hdf5(self, target: Group):
        """Saves Geometry instance into a target HDF5 Group.

        Arguments:
            target: The hdf5 group to which the system's data should be saved.

        Notes:
            This function does not create its own group as it expects that
            ``target`` is the group into which data should be writen.
        """
        # Short had for dataset creation
        add_data = target.create_dataset

        # Add datasets for atomic_numbers, positions, lattice, and pbc
        add_data('atomic_numbers', data=self.atomic_numbers.cpu().numpy())
        pos = add_data('positions', data=self.positions.cpu().numpy())

        # Add units meta-data to the atomic positions
        pos.attrs['unit'] = 'bohr'

    def to_vasp(self, output: str, direct=False, selective_dynamics=False, zrange=(0, 10000)):
        assert self.cell is not None, 'VASP only support PBC system'

        positions = self.positions / length_units['angstrom']
        cells = self.cell / length_units['angstrom']
        cell_xyz = torch.sqrt((cells ** 2).sum(-2)).unsqueeze(-2)
        positions = positions / cell_xyz if direct else positions

        def write_single(number, position, cell):
            f = open(output, "w")
            f.write(self.__repr__() + "\n")
            f.write("1.0 \n")

            for i in range(3):
                for j in range(3):
                    f.write(" %15.10f" % cell[i, j])
                f.write("\n")

            for i in torch.unique(number):
                f.write(chemical_symbols[i] + " ")
            f.write("\n")

            for i in torch.unique(number):
                f.write(" %d" % (sum(number == i)) + " ")
            f.write("\n")

            if selective_dynamics:
                f.write("selective dynamics\n")

            if direct:
                f.write("Direct\n")
            else:
                f.write("Cartesian\n")
            print('selective_dynamics', selective_dynamics)

            if selective_dynamics:
                for i in torch.unique(number):
                    print(i)
                    pos = position[number == i]
                    for j in range(len(pos)):
                        for k in range(3):
                            f.write(" %.12f" % pos[j, k])

                        if pos[j, 2] > zrange[0] and pos[j, 2] < zrange[-1]:
                            select = ' False False False'
                        else:
                            select = ' True True True'
                        print('select', select, pos[j, 2])
                        f.write(select + "\n")
            else:
                for i in torch.unique(number):
                    pos = position[number == i]
                    for j in range(len(pos)):
                        for k in range(3):
                            f.write(" %.12f" % pos[j, k])
                        f.write("\n")

        # Write output file
        if self._n_batch is None:
            write_single(self.atomic_numbers, positions, cells)
        else:
            for number, position, cell in zip(self.atomic_numbers, positions, cells):
                write_single(number, position, cell)


    def remove_atoms(self, low_limit, up_limit):
        mask = self.positions[..., 2] < up_limit
        mask = mask * self.positions[..., 2] > low_limit

        if self._n_batch is None:
            atomic_numbers = self.atomic_numbers[mask]
            positions = self.positions[mask]
        else:
            atomic_numbers = pack([atom[imask] for atom, imask in zip(self.atomic_numbers, mask)])
            positions = pack([pos[imask] for pos, imask in zip(self.positions, mask)])

        return Geometry(atomic_numbers, positions, cell=self.cell)


    def to(self, device: torch.device) -> 'Geometry':
        """Returns a copy of the `Geometry` instance on the specified device

        This method creates and returns a new copy of the `Geometry` instance
        on the specified device "``device``".

        Arguments:
            device: Device to which all associated tensors should be moved.

        Returns:
            geometry: A copy of the `Geometry` instance placed on the
                specified device.

        Notes:
            If the `Geometry` instance is already on the desired device then
            `self` will be returned.
        """
        # Developers Notes: It is imperative that this function gets updated
        # whenever new attributes are added to the `Geometry` class. Otherwise
        # this will return an incomplete `Geometry` object.
        if self.atomic_numbers.device == device:
            return self
        else:
            return self.__class__(self.atomic_numbers.to(device=device),
                                  self.positions.to(device=device))

    def __getitem__(self, selector) -> 'Geometry':
        """Permits batched Geometry instances to be sliced as needed."""
        # Block this if the instance has only a single system
        if self.atomic_numbers.ndim != 2:
            raise IndexError(
                'Geometry slicing is only applicable to batches of systems.')

        if self.cell is None:
            return self.__class__(self.atomic_numbers[selector],
                                  self.positions[selector])
        else:
            return self.__class__(self.atomic_numbers[selector],
                                  self.positions[selector],
                                  cell=self.cell[selector])

    def __add__(self, other: 'Geometry') -> 'Geometry':
        """Combine two `Geometry` objects together."""
        if self.__class__ != other.__class__:
            raise TypeError(
                'Addition can only take place between two Geometry objects.')

        # Catch for situations where one or both systems are not batched.
        s_batch = self.atomic_numbers.ndim == 2
        o_batch = other.atomic_numbers.ndim == 2

        an_1 = torch.atleast_2d(self.atomic_numbers)
        an_2 = torch.atleast_2d(other.atomic_numbers)

        pos_1 = self.positions
        pos_2 = other.positions

        pos_1 = pos_1 if s_batch else pos_1.unsqueeze(0)
        pos_2 = pos_2 if o_batch else pos_2.unsqueeze(0)

        if self.cell is not None:
            cell_1 = self.cell if s_batch else self.cell.unsqueeze(0)
            cell_2 = other.cell if o_batch else other.cell.unsqueeze(0)

            return self.__class__(merge([an_1, an_2]), merge([pos_1, pos_2]),
                                  cell=merge([cell_1, cell_2]))
        else:
            return self.__class__(merge([an_1, an_2]), merge([pos_1, pos_2]))

    def __eq__(self, other: 'Geometry') -> bool:
        """Check if two `Geometry` objects are equivalent."""
        # Note that batches with identical systems but a different order will
        # return False, not True.

        if self.__class__ != other.__class__:
            raise TypeError(f'"{self.__class__}" ==  "{other.__class__}" '
                            f'evaluation not implemented.')

        def shape_and_value(a, b):
            return a.shape == b.shape and torch.allclose(a, b)

        return all([
            shape_and_value(self.atomic_numbers, other.atomic_numbers),
            shape_and_value(self.positions, other.positions)
        ])

    def __repr__(self) -> str:
        """Creates a string representation of the Geometry object."""
        # Return Geometry(CH4) for a single system & Geometry(CH4, H2O, ...)
        # for multiple systems. Only the first & last two systems get shown if
        # there are more than four systems (this prevents endless spam).

        def get_formula(atomic_numbers: Tensor) -> str:
            """Helper function to get reduced formula."""
            # If n atoms > 30; then use the reduced formula
            if len(atomic_numbers) > 30:
                return ''.join([f'{chemical_symbols[z]}{n}' if n != 1 else
                                f'{chemical_symbols[z]}' for z, n in
                                zip(*atomic_numbers.unique(return_counts=True))
                                if z != 0])  # <- Ignore zeros (padding)

            # Otherwise list the elements in the order they were specified
            else:
                return ''.join(
                    [f'{chemical_symbols[int(z)]}{int(n)}' if n != 1 else
                     f'{chemical_symbols[z]}' for z, n in
                     zip(*torch.unique_consecutive(atomic_numbers,
                                                   return_counts=True))
                     if z != 0])

        if self.atomic_numbers.dim() == 1:  # If a single system
            formula = get_formula(self.atomic_numbers)
        else:  # If multiple systems
            if self.atomic_numbers.shape[0] < 4:  # If n<4 systems; show all
                formulas = [get_formula(an) for an in self.atomic_numbers]
                formula = ' ,'.join(formulas)
            else:  # If n>4; show only the first and last two systems
                formulas = [get_formula(an) for an in
                            self.atomic_numbers[[0, 1, -2, -1]]]
                formula = '{}, {}, ..., {}, {}'.format(*formulas)

        # Wrap the formula(s) in the class name and return
        return f'{self.__class__.__name__}({formula})'

    def __str__(self) -> str:
        """Creates a printable representation of the System."""
        # Just redirect to the `__repr__` method
        return repr(self)


####################
# Helper Functions #
####################
def batch_chemical_symbols(atomic_numbers: Union[Tensor, List[Tensor]]
                           ) -> list:
    """Converts atomic numbers to their chemical symbols.

    This function allows for en-mass conversion of atomic numbers to chemical
    symbols.

    Arguments:
        atomic_numbers: Atomic numbers of the elements.

    Returns:
        symbols: The corresponding chemical symbols.

    Notes:
        Padding vales, i.e. zeros, will be ignored.

    """
    a_nums = atomic_numbers

    # Catch for list tensors (still faster doing it this way)
    if isinstance(a_nums, list) and isinstance(a_nums[0], Tensor):
        a_nums = pack(a_nums, value=0)

    # Convert from atomic numbers to chemical symbols via a itemgetter
    symbols = np.array(  # numpy must be used as torch cant handle strings
        itemgetter(*a_nums.flatten())(chemical_symbols)
    ).reshape(a_nums.shape)
    # Mask out element "X", aka padding values
    mask = symbols != 'X'
    if symbols.ndim == 1:
        return symbols[mask].tolist()
    else:
        return [s[m].tolist() for s, m in zip(symbols, mask)]


def unique_atom_pairs(geometry: Optional[Geometry] = None,
                      unique_atomic_numbers: Optional[Tensor] = None,
                      elements: list = None) -> Tensor:
    """Returns a tensor specifying all unique atom pairs.

    This takes `Geometry` instance and identifies all atom pairs. This use
    useful for identifying all possible two body interactions possible within
    a given system.

    Arguments:
         geometry: `Geometry` instance representing the target system.

    Returns:
        unique_atom_pairs: A tensor specifying all unique atom pairs.
    """
    if geometry is not None:
        uan = geometry.unique_atomic_numbers()
    elif unique_atomic_numbers is not None:
        uan = unique_atomic_numbers
    elif elements is not None:
        uan = torch.tensor([data_numbers[element] for element in elements])
    else:
        raise ValueError('Both geometry and unique_atomic_numbers are None.')

    n_global = len(uan)
    return torch.stack([uan.repeat(n_global),
                        uan.repeat_interleave(n_global)]).T

def to_atomic_numbers(species: list) -> Tensor:
    """Return atomic numbers from element species."""
    return torch.tensor([chemical_symbols.index(isp) for isp in species])


def to_element_species(atomic_numbers: Union[Tensor]) -> list:
    """Return element species from atomic numbers."""
    assert atomic_numbers.dim() in (
        1,
        2,
    ), f"get input dimension {atomic_numbers.dim()} not 1 or 2"
    if atomic_numbers.dim() == 1:
        return [chemical_symbols[int(ia)] for ia in atomic_numbers]
    else:
        return [
            [chemical_symbols[int(ia)] for ia in atomic_number]
            for atomic_number in atomic_numbers
        ]


class GeometryPbcOneCell(Geometry):
    """Transfer periodic boundary condition to molecule like system."""

    def __init__(self, geometry: Geometry, periodic):
        assert geometry.is_periodic, 'This class only works when PBC is True'
        self.geometry = geometry
        self.periodic = periodic
        # self.n_atoms = (self.geometry.n_atoms * self.geometry.periodic.ncell).long()
        self.n_atoms = (self.geometry.n_atoms * self.cell_mask.sum(-1)).long()

        self.pe_ind0 = self.periodic.ncell.repeat_interleave(self.geometry.n_atoms)
        self.pe_ind = self.cell_mask.sum(-1).repeat_interleave(self.geometry.n_atoms)

        # method 1, smaller size
        _atomic_numbers = self.geometry.atomic_numbers[self.geometry.atomic_numbers.ne(0)]
        _atomic_numbers = torch.repeat_interleave(_atomic_numbers, self.pe_ind)
        self.atomic_numbers = pack(_atomic_numbers.split(tuple(self.n_atoms)))

        # method 2
        # self.atomic_numbers = pack([number.repeat_interleave(mask) for number, mask in zip(
        #     self.geometry.atomic_numbers, self.cell_mask.sum(-1).long())])

        self.positions = self._to_onecell()

        # Mask for clearing padding values in the distance matrix.
        if (temp_mask := self.atomic_numbers != 0).all():
            self._mask_dist: Union[Tensor, bool] = False
        else:
            self._mask_dist: Union[Tensor, bool] = ~(
                temp_mask.unsqueeze(-2) * temp_mask.unsqueeze(-1)
            )

    @property
    def cell_mask(self):
        return self.periodic.neighbour.any(-1).any(-1)

    @property
    def n_cell(self):
        return (self.n_atoms / self.n_central_atoms).long()

    @property
    def n_central_atoms(self):
        return self.geometry.n_atoms

    @property
    def distances(self) -> Tensor:
        dist = torch.cdist(self.positions, self.positions, p=2)
        # Ensure padding area is zeroed out
        dist[self._mask_dist] = 0.0

        # cdist bug, sometimes distances diagonal is not zero
        _ind = torch.arange(dist.shape[-1])
        if not (dist[..., _ind, _ind].eq(0)).all():
            dist[..., _ind, _ind] = 0

        return dist

    @property
    def distance_vectors(self) -> Tensor:
        """Distance vector matrix between atoms in the system."""
        dist_vec = self.positions.unsqueeze(-2) - self.positions.unsqueeze(-3)
        dist_vec[self._mask_dist] = 0
        return dist_vec

    def _to_onecell(self):
        """Transfer periodic positions to molecule like positions."""
        # Permute positions to make sure atomic numbers and positions are consistent
        _pos = self.periodic.positions_pe[self.cell_mask]
        _pos = _pos.split(tuple(self.cell_mask.sum(-1).tolist()), 0)

        # method 1, smaller size
        return pack(torch.cat([ipos[:ind, :ia].flatten(0, 1) for ipos, ind, ia in zip(
            _pos, self.cell_mask.sum(-1), self.geometry.n_atoms)]).split(tuple(self.n_atoms)))

        # method 2
        # _pos = self.geometry.periodic.positions_pe[self.cell_mask]
        # return pack(_pos.split(tuple(self.cell_mask.sum(-1).tolist()), 0)).flatten(1, 2)

    @property
    def cell_mat(self) -> Tensor:
        """Return indices of cell vector of corresponding atoms."""
        # When write PBC cell to non-PBC like system, atoms in system come
        # from different cells, it's important to label the cell indices.
        # TODO, try to replace pack
        # method 1, smaller size
        cellvec = self.periodic.cellvec.permute(0, -1, 1)[self.cell_mask]
        cellvec = pack([icell.repeat_interleave(ind, dim=0) for icell, ind in zip(
            cellvec.split(tuple(self.cell_mask.sum(-1))), self.geometry.n_atoms)], value=self.pad_values)

        # method 2
        # cellvec = self.geometry.periodic.cellvec.permute(0, -1, 1)[self.cell_mask]
        # cellvec = pack(cellvec.split(tuple(self.cell_mask.sum(-1))), value=self.pad_values)
        # cellvec = pack(cellvec.repeat_interleave(self.geometry.n_atoms, dim=0)
        #                .split(tuple(self.geometry.n_atoms)), value=self.pad_values).flatten(1, 2)

        return cellvec

    @property
    def cell2d_mat(self) -> Tensor:
        """Return indices of cell and the shape is similar to distances."""
        tmp = self.cell_mat.unsqueeze(-2) + self.cell_mat.unsqueeze(-3)
        tmp1 = tmp - self.cell_mat.unsqueeze(-3)
        tmp2 = tmp - self.cell_mat.unsqueeze(-2)

        return torch.cat([tmp1, tmp2], dim=-1)

    @property
    def central_cell_ind(self):
        return (self.cell2d_mat[..., :3] == 0).sum(-1) == 3

    @property
    def is_periodic(self):
        """This is a quasi molecule-like geometry."""
        return False

    @property
    def pad_values(self):
        return 1e6
