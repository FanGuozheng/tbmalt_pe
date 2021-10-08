"""Write skf to hdf5 binary file.

The skf include normal skf files or skf with a list of compression radii.
"""
import h5py
import torch
import numpy as np
from tbmalt.io.loadhdf import LoadHdf
from tbmalt.utils.ase.ase_aims import AseAims
from tbmalt.utils.ase.ase_dftbplus import AseDftb


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

    @classmethod
    def to_hdf(cls, results: dict, cal_reference: object,
               properties: list, **kwargs):
        """Generate reference results to binary hdf file.

        Arguments:
            results: dict type which contains physical properties.
            symbols: list type which contains element symbols of each system.
            properties: list type which defines properties to be written.

        Keyword Args:
            mode: a: append, w: write into new output file.
        """
        input_type = kwargs.get('input_type', 'ANI-1')
        numbers = cal_reference.numbers
        symbols = cal_reference.symbols
        positions = cal_reference.positions
        if input_type == 'Si':
            latvec = cal_reference.latvecs
        atom_specie_global = cal_reference.atom_specie_global
        output_name = kwargs.get('output_name', 'reference.hdf')
        mode = kwargs.get('mode', 'a')  # -> if override output file

        with h5py.File(output_name, mode) as f:

            # write global parameters
            gg = f['global_group'] if 'global_group' in f else \
                f.create_group('global_group')
            if 'atom_specie_global' in gg.attrs:
                gg.attrs['atom_specie_global'] = np.unique(np.concatenate(
                    [gg.attrs['atom_specie_global'],
                     np.array(atom_specie_global)])).tolist()
            else:
                gg.attrs['atom_specie_global'] = atom_specie_global

            if 'molecule_specie_global' not in gg.attrs:
                gg.attrs['molecule_specie_global'] = []

            # write each system with symbol as label
            for ii, isys in enumerate(symbols):

                if ''.join(isys) not in f.keys():  # -> new molecule specie

                    # add to molecule_specie_global in global group
                    gg.attrs['molecule_specie_global'] = np.unique(
                        np.concatenate([gg.attrs['molecule_specie_global'],
                                        np.array([''.join(isys)])])).tolist()

                    g = f.create_group(''.join(isys))
                    g.attrs['specie'] = isys
                    g.attrs['numbers'] = numbers[ii]
                    g.attrs['size_molecule'] = len(isys)
                    g.attrs['n_molecule'] = 0
                else:
                    g = f[''.join(isys)]

                n_system = g.attrs['n_molecule']  # each molecule specie number
                g.attrs['n_molecule'] = n_system + 1
                g.create_dataset(str(n_system + 1) + 'position', data=positions[ii])
                if input_type == 'Si':
                    g.create_dataset(str(n_system + 1) + 'lattice vector', data=latvec[ii])

                for iproperty in properties:
                    iname = str(n_system + 1) + iproperty
                    g.create_dataset(iname, data=results[iproperty][ii])
