"""Write skf to hdf5 binary file.

The skf include normal skf files or skf with a list of compression radii.
"""
import h5py
import torch
import numpy as np
from tbmalt.io.hdf import LoadHdf
from tbmalt import Geometry
from tbmalt.utils.ase.ase_aims import AseAims
from tbmalt.utils.ase.ase_dftbplus import AseDftb
from tbmalt.data.units import length_units
from tbmalt.structures.geometry import batch_chemical_symbols
from tbmalt.common.batch import deflate

class CalReference:
    """Transfer SKF files from skf files to hdf binary type.

    Arguments:
        path_to_input: Joint path and input files.
        input_type: The input file type, such as hdf, json, etc.
        reference_type: The type of reference, such as FHI-aims, DFTB+.

    Keyword Args:
        path_to_skf: Joint path and SKF files if reference is DFTB+.
        path_to_aims_specie: Joint path and FHI-aims specie files if reference
            is FHI-aims.
    """

    def __init__(self, path_to_input: str, input_type: str, size: int,
                 reference_type='dftbplus', **kwargs):
        """Calculate and write reference properties from DFT(B)."""
        self.path_input = path_to_input
        self.input_type = input_type
        self.reference_type = reference_type
        self.periodic = kwargs.get('periodic', False)

        if self.reference_type == 'dftbplus':
            self.path_to_dftbplus = kwargs.get('path_to_dftbplus', './dftb+')
            self.path_to_skf = kwargs.get('path_to_skf', './')

        elif self.reference_type == 'aims':
            self.path_to_aims = kwargs.get('path_to_aims', './aims.x')
            self.path_to_aims_specie = kwargs.get('path_to_aims_specie', './')

        dataset = self._load_input(size)
        self.numbers = dataset.numbers
        self.positions = dataset.positions
        self.symbols = dataset.symbols
        self.atom_specie_global = dataset.atom_specie_global
        self.latvecs = dataset.latvec if self.input_type == 'Si' else None

    def _load_input(self, size):
        """Load."""
        if self.input_type == 'ANI-1':
            return LoadHdf(self.path_input, size, self.input_type)
        elif self.input_type == 'Si':
            return LoadHdf(self.path_input, size, self.input_type)

    def __call__(self, properties: list, **kwargs):
        """Call WriteSK.

        Arguments:
            properties: Properties to be calculated.
            mode: mode of function, 'w' for writing and 'a' for appending.

        Keyword Args:

        """
        if self.reference_type == 'aims':
            aims = AseAims(self.path_to_aims, self.path_to_aims_specie, periodic='self.periodic')
            result = aims.run_aims(self.positions, self.symbols, self.latvecs, properties)

        elif self.reference_type == 'dftbplus':
            dftb = AseDftb(self.path_to_dftbplus, self.path_to_skf,
                           properties, **kwargs)
            result = dftb.run_dftb(self.positions, self.symbols, self.latvecs, properties)
        return result

    def to_hdf(results: dict, cal_reference: object = None,
               properties: list = ['charge'], geometry: Geometry = None, **kwargs):
        """Generate reference results to binary hdf file.

        Arguments:
            results: dict type which contains physical properties.
            symbols: list type which contains element symbols of each system.
            properties: list type which defines properties to be written.

        Keyword Args:
            mode: a: append, w: write into new output file.
        """
        input_type = kwargs.get('input_type', 'ANI-1')
        if cal_reference is not None:
            numbers = cal_reference.numbers
            symbols = cal_reference.symbols
            positions = cal_reference.positions
            atom_specie = cal_reference.atom_specie_global
        elif geometry is not None:
            numbers = geometry.atomic_numbers

            # symbols = geometry.chemical_symbols
            positions = geometry.positions / length_units['a']
            atom_specie = geometry.unique_atomic_numbers()
            unique_numbers = torch.unique(numbers, sorted=False, dim=0)
        global_group = kwargs.get('global_group', 'global_group')

        if input_type == 'Si':
            latvec = cal_reference.latvecs
        output_name = kwargs.get('output_name', 'reference.hdf')
        mode = kwargs.get('mode', 'a')  # -> if override output file

        with h5py.File(output_name, mode) as f:

            # write global parameters
            print('mode', mode)
            print('global_group in f', global_group, global_group in f)
            gg = f[global_group] if global_group in f else \
                f.create_group(global_group)
            if 'atom_specie' in gg.attrs:
                gg.attrs['atom_specie'] = np.unique(np.concatenate(
                    [gg.attrs['atom_specie'],
                     np.array(atom_specie)])).tolist()
            else:
                gg.attrs['atom_specie'] = atom_specie

            if 'geometry_specie' not in gg.attrs:
                gg.attrs['geometry_specie'] = []

            for number in unique_numbers:
                mask = (number == numbers).all(-1)
                symbol = batch_chemical_symbols(number)

                if ''.join(symbol) not in f.keys():  # -> new molecule specie

                    # add to molecule_specie_global in global group
                    gg.attrs['geometry_specie'] = np.unique(
                        np.concatenate([gg.attrs['geometry_specie'],
                                        np.array([''.join(symbol)])])).tolist()

                    g = gg.create_group(''.join(symbol))
                    g.attrs['label'] = symbol
                    g.attrs['numbers'] = number
                    g.attrs['size_geometry'] = torch.count_nonzero(number)
                    g.attrs['n_geometry'] = mask.sum()

                # n_geometry = g.attrs['n_geometry']  # each molecule specie number
                # g.attrs['n_geometry'] = n_geometry + 1

                    g.create_dataset('positions', data=deflate(positions[mask]))
                    if input_type == 'Si':
                        g.create_dataset('lattice vector', data=latvec[mask])

                    for pro in properties:
                        g.create_dataset(pro, data=deflate(results[pro][mask]))


            # write each system with symbol as label
            # for ii, isys in enumerate(symbols):
            #
            #     if ''.join(isys) not in f.keys():  # -> new molecule specie
            #
            #         # add to molecule_specie_global in global group
            #         gg.attrs['geometry_specie_global'] = np.unique(
            #             np.concatenate([gg.attrs['geometry_specie_global'],
            #                             np.array([''.join(isys)])])).tolist()
            #
            #         g = f.create_group(''.join(isys))
            #         g.attrs['specie'] = isys
            #         g.attrs['numbers'] = numbers[ii]
            #         g.attrs['size_geometry'] = len(isys)
            #         g.attrs['n_geometry'] = 0
            #     else:
            #         g = f[''.join(isys)]
            #
            #     n_system = g.attrs['n_geometry']  # each molecule specie number
            #     g.attrs['n_geometry'] = n_system + 1
            #
            #     g.create_dataset(str(n_system + 1) + 'position', data=positions[ii])
            #     if input_type == 'Si':
            #         g.create_dataset(str(n_system + 1) + 'lattice vector', data=latvec[ii])
            #
            #     for iproperty in properties:
            #         iname = str(n_system + 1) + iproperty
            #         g.create_dataset(iname, data=results[iproperty][ii])
