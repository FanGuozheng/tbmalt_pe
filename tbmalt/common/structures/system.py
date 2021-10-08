"""Geometric, structural and orbital information."""
from typing import Union, List
import numpy as np
import torch
from tbmalt.common.batch import pack
from tbmalt.common.data import bohr, atom_name, val_elect, l_num
from tbmalt.common.structures.cell import Pbc
Tensor = torch.Tensor


class System:
    """Geometry object.

    This object will generate single geometry (molecule, unit cell)
    information, or a list of geometry information from dataset. The
    difference of single and multi geometry input will be the dimension.
    In general, the output information will include symbols, max quantum
    number, angular moments, magnetic moments, masses and atomic charges.

    Arguments:
        numbers: Atomic number of each atom in single or multi geometry.
            For multi geometry, if geometry sizes is not same, then use
            sequences tensor input.
        positions: :math:`(N, 3)` where `N = number of atoms`
            in single geometry, or :math:`(M, N, 3)` where `M = number of
            geometry, N = number of atoms` in multi geometry.
        cell: optional
            The lattice vectors of the cell, if applicable. This should be
            a 3x3 matrix. While a 1x3 vector can be specified it will be
            auto parsed into a 3x3 vector.
        pbc: True will enact periodic boundary conditions (PBC) along all
            cell dimensions (buck geometry), False will fully disable PBC
            (isolated molecules), & an array of booleans will only enact
            PBC on a subset of cell dimensions (slab). The first two can
            be auto-inferred from the presence of the ``cell`` parameter.
            (`bool`, `array_like` [`bool`], optional).

    Examples:
        >>> import torch
        >>> from ase.build import molecule as molecule_database
        >>> from tbmalt.common.structures.geometry import Geometry
        >>> molecule = molecule_database('CH4')
        >>> geometry = Geometry.from_ase_atoms(molecule)
        >>> geometry.numbers
        >>> tensor([[6, 1, 1, 1, 1]])
        >>> geometry = Geometry(torch.tensor([1, 1]), torch.tensor([
            [0., 0., 0.], [0.5, 0.5, 0.5]]))
        >>> geometry.distances
        >>> tensor([[[0.0000, 1.6366], [1.6366, 0.0000]]])

    Todo:
        Add periodic boundaries.

    """

    def __init__(self, numbers: Union[Tensor, List[Tensor]],
                 positions: Union[Tensor, List[Tensor]],
                 cell=None, frac=None, **kwargs):
        # bool tensor is_periodic defines which is solid and which is molecule
        if cell is not None:
            self._cell = Pbc(cell, frac, **kwargs)
            self.cell, self.is_periodic, self.is_frac, self.pbc =\
                self._cell.cell, self._cell.is_periodic, self._cell.is_frac, self._cell.pbc
            self.periodic = True if True in self.is_periodic else False
            self.positions, self.numbers, self.batch, self.is_periodic, self.is_frac = \
                self._check(numbers, positions, self.is_periodic, self.is_frac, **kwargs)
        else:
            self.periodic = False  # no system is solid
            self.positions, self.numbers, self.batch, self.is_periodic, self.is_frac = \
                self._check(numbers, positions, **kwargs)

        # size of each geometry
        self.size_system = self._get_size()

        # get distance
        self.distances = self._get_distances()

        # get symbols
        self.symbols = self._get_symbols()

        # size of batch size, size of each geometry (number of atoms)
        self.size_batch = len(self.numbers)

        # get max l of each atom, number of orbitals (l + 1) ** 2 of each atom
        # number of total orbitals of each geometry in batch
        self.l_max, self.atom_orbitals, self.geometry_orbitals = \
            self._get_l_orbital()

        # get Hamiltonian, overlap shape in batch of each geometry and batch
        self.single_hs_shape, self.hs_shape = self._get_hs_shape()

    def _check(self, numbers, positions, is_periodic=None, is_frac=None, **kwargs):
        """Check the type and dimension of numbers, positions."""
        unit = kwargs['unit'] if 'unit' in kwargs else 'angstrom'
        coord_type = kwargs['coord'] if 'coord' in kwargs else 'cartesian'

        # sequences of tensor
        if isinstance(numbers, list):
            numbers = pack(numbers)
        elif isinstance(numbers, Tensor) and numbers.dim() == 1:
            numbers = numbers.unsqueeze(0)

        # positions type check
        if isinstance(positions, list):
            positions = pack(positions)
        elif isinstance(positions, Tensor) and positions.dim() == 2:
            positions = positions.unsqueeze(0)

        # if there is no solid, build is_periodic as False tensor
        if is_periodic is None:
            is_periodic = torch.zeros(numbers.shape[0], dtype=torch.bool)
            if coord_type in ('fraction', 'Fraction'):
                raise ValueError('Coordinate type for non-periodic should be cartesian')
            elif coord_type not in ('cartesian', 'Cartesian'):
                raise ValueError('Please select either cartesian or fraction')
        else:
            # check whether specify the fraction system
            if is_frac is None and coord_type in ('fraction', 'Fraction'):
                raise ValueError('Please specify the fraction system')

        # transfer positions of non-periodic systems from angstrom to bohr
        if unit in ('angstrom', 'Angstrom'):
            positions[~is_periodic] = positions[~is_periodic] / bohr
        elif unit not in ('bohr', 'Bohr'):
            raise ValueError('Please select either angstrom or bohr')

        assert positions.shape[0] == numbers.shape[0]
        batch_ = True if numbers.shape[0] != 1 else False
        return positions, numbers, batch_, is_periodic, is_frac

    def _get_distances(self):
        """Return distances between a list of atoms for each geometry."""
        return pack([torch.sqrt(((ipos[:inat].repeat(inat, 1) -
                                  ipos[:inat].repeat_interleave(inat, 0))
                                 ** 2).sum(1)).reshape(inat, inat)
                     for ipos, inat in zip(self.positions, self.size_system)])

    def _get_symbols(self):
        """Get atom name for each geometry in batch."""
        return [[atom_name[ii - 1] for ii in inu[inu.ne(0.)]]
                for inu in self.numbers]

    def _get_size(self):
        """Get each geometry size (number of atoms) in batch."""
        return [len(inum[inum.ne(0.0)]) for inum in self.numbers]

    def _get_l_orbital(self):
        """Return the number of orbitals associated with each atom."""
        # max l for each atom
        l_max = pack([torch.tensor([l_num[ii - 1] for ii in inum[inum.ne(0)]])
                      for inum in self.numbers], value=-1)

        # max valence orbital number for each atom and each system
        atom_orbitals = pack([torch.tensor([(ii + 1) ** 2 for ii in lm])
                              for lm in l_max])
        system_orbitals = pack([sum(iao) for iao in atom_orbitals])

        return l_max, atom_orbitals, system_orbitals

    def _get_hs_shape(self):
        """Return shapes of each single and batch Hamiltonian and overlap."""
        maxorb = max(self.geometry_orbitals)
        single_hs_shape = [torch.Size([iorb, iorb]) for iorb in self.geometry_orbitals]
        hs_shape = torch.Size([self.size_batch, maxorb, maxorb])
        return single_hs_shape, hs_shape

    def get_positions_vec(self):
        """Get positions vector between atoms.

        Returns:
            positions_vector: Vectors between positions of each atom for batch.

        """
        return pack([ipo.unsqueeze(-2) - ipo.unsqueeze(-3)
                     for ipo in self.positions])

    def get_valence_electrons(self, dtype=torch.int64):
        """Get the number of orbitals associated with each atom.

        Returns:
            valence_electrons: Valence electrons of each system for batch.

        """
        return pack([torch.tensor([val_elect[ii - 1]
                                   for ii in inum[inum.ne(0)]], dtype=dtype)
                     for inum in self.numbers])

    def get_global_species(self):
        """Get global element information for single or multi geometry.

        Returns:
            element_name: Global atom names in batch.
            element_number: Global atom numbers in batch.
            element_name_pair: Global atom name pairs in batch.
            element_number_pair: Global atom number pairs in batch.

        """
        numbers_ = torch.unique(self.numbers)
        numbers = numbers_[numbers_.ne(0.)]
        element_name = [atom_name[ii - 1] for ii in numbers]
        element_number, nn = numbers.tolist(), len(numbers)
        element_name_pair = [[iel, jel] for iel, jel in zip(
            sorted(element_name * nn), element_name * nn)]
        element_number_pair = [
            [atom_name.index(ii[0]) + 1, atom_name.index(ii[1]) + 1]
            for ii in element_name_pair]
        return element_name, element_number, element_name_pair, element_number_pair

    def get_resolved_orbital(self):
        """Get resolved orbitals and accumulated orbitals.

        Returns:
            resolved_orbital: Resolved quantum number l of each orbital in
                atom, return empty Tensor for padding part.

        """
        return [[torch.arange(lm + 1, dtype=torch.int8).repeat_interleave(
            2 * torch.arange(lm + 1) + 1) for lm in ilm] for ilm in self.l_max]

    @classmethod
    def to_element_number(cls, element: list):
        """Return element number from element.

        Arguments:
            element: Element names.

        Returns:
            numbers: Atomic number of each element.

        """
        element = [element] if type(element[0]) is not list else element
        return pack([torch.tensor([atom_name.index(ii) + 1 for ii in iele])
                     for iele in element])

    @classmethod
    def to_element(cls, number: Union[Tensor, List[Tensor]]):
        """Return elements number from elements.

        Arguments:
            numbers: Atomic number of each element.

        Returns:
            element: Element names.

        """
        if isinstance(number, Tensor):
            number = number.unsqueeze(0) if number.dim() == 1 else number
        return [[atom_name[ii - 1] for ii in inum[inum.ne(0)]] for inum in number]

    @classmethod
    def from_ase_atoms(cls, atoms, **kwargs):
        """Instantiate a Geometry instance from an ase.Atoms object.

        Arguments:
            atoms: ASE Atoms object(s) to be converted into Geometry instance(s).

        Returns:
            Geometry : Geometry object.

        """
        # if isinstance(atoms, list):  # If multiple atoms objects supplied:
        #     # Recursively call from_ase_atoms and return the result
        #     numbers = [torch.from_numpy(iat.numbers) for iat in atoms]
        #     positions = [torch.from_numpy(iat.positions) for iat in atoms]
        #     return Geometry(numbers, positions)

        # return Geometry(torch.from_numpy(atoms.numbers),
        #                 torch.torch.from_numpy(atoms.positions))
        if isinstance(atoms, list):  # If multiple atoms objects supplied
            # Recursively call from_ase_atoms and return the result
            numbers = [torch.from_numpy(iat.numbers) for iat in atoms]
            positions = [torch.from_numpy(iat.positions) for iat in atoms]
            cell = [torch.from_numpy(np.asarray(iat.cell)) for iat in atoms]
            pbc = [torch.from_numpy(np.asarray(iat.pbc)) for iat in atoms]
            return System(numbers, positions, cell, pbc, **kwargs)

        elif isinstance(atoms, object):  # If single atoms objects supplied
            return System(torch.from_numpy(atoms.numbers),
                            torch.torch.from_numpy(atoms.positions),
                            torch.from_numpy(np.asarray(atoms.cell)),
                            torch.from_numpy(np.asarray(atoms.pbc)), **kwargs)

    def to_hd5(self, target):
        """Convert the Geometry instance to a set of hdf5 datasets.

        Return:
            target: The hdf5 entity to which the set of h5py.Dataset instances
                representing the geometry should be written.

        """
        # Short had for dataset creation
        add_data = target.create_dataset

        # Add datasets for numbers, positions, lattice, and pbc
        add_data('numbers', data=self.numbers)
        add_data('positions', data=self.positions.numpy())

    @staticmethod
    def from_hd5(source, **kwargs):
        """Convert an hdf5.Groups entity to a Geometry instance.

        Arguments:
            source : hdf5 File/Group containing the geometry's data.

        Return:
            Geometry : A Geometry instance representing the data stored.

        Notes:
            It should be noted that dtype will not be inherited from the
            database. Instead the default PyTorch dtype will be used.

        """
        # Get default dtype
        dtype = torch.get_default_dtype()

        # Read & parse datasets from the database into a Geometry instance
        # & return the result.
        return System(
            torch.tensor(source['numbers']),
            torch.tensor(source['positions'], dtype=dtype), **kwargs)
