"""Write skf to hdf5 binary file.

The skf include normal skf files or skf with a list of compression radii.
"""
import h5py
import torch
import numpy as np
from collections import Counter
from tbmalt.io.loadskf import LoadSKF
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
        mode: a: append, w: write into new output file.
        path_to_skf: Joint path and SKF files if reference is DFTB+.
        path_to_aims_specie: Joint path and FHI-aims specie files if reference
            is FHI-aims.
    """

    def __init__(self, path_to_input: str, input_type: str, size: int,
                 reference_type='dftbplus', **kwargs):
        """Read skf, smooth the tail, write to hdf."""
        self.path_in = path_to_input
        self.in_type = input_type
        self.mode = kwargs.get('mode', 'a')

        self.reference_type = reference_type
        if self.reference_type == 'dftbplus':
            self.path_to_dftbplus = kwargs.get('path_to_dftbplus', './dftb+')
            self.path_to_skf = kwargs.get('path_to_skf', './')
        elif self.reference_type == 'aims':
            self.path_to_aims = kwargs.get('path_to_aims', './aims.x')
            self.path_to_aims_specie = kwargs.get('path_to_aims_specie', './')

        in_instance = self._load_input(size)
        if self.in_type == 'ANI-1':
            self.numbers = in_instance.numbers
            self.positions = in_instance.positions
            self.symbols = in_instance.symbols
            self.atom_specie_global = in_instance.atom_specie_global

    def _load_input(self, size):
        """Load."""
        if self.in_type == 'ANI-1':
            return LoadHdf(self.path_in, size, self.in_type)

    def __call__(self, properties: list, **kwargs):
        """Call WriteSK.

        Arguments:
            properties: Properties to be calculated.
            mode: mode of function, 'w' for writing and 'a' for appending.

        Keyword Args:

        """
        if self.reference_type == 'aims':
            aims = AseAims(self.path_to_aims, self.path_to_aims_specie)
            result = aims.run_aims(self.positions, self.symbols, properties)
        elif self.reference_type == 'dftbplus':
            dftb = AseDftb(self.path_to_dftbplus, self.path_to_skf,
                           properties, **kwargs)
            result = dftb.run_dftb(self.positions, self.symbols, properties)
        return result

    @classmethod
    def to_hdf(cls, results: dict, numbers: list, symbols: list,
               positions: list, properties: list, **kwargs):
        """Generate reference results to binary hdf file.

        Arguments:
            results: dict type which contains physical properties.
            symbols: list type which contains element symbols of each system.
            properties: list type which defines properties to be written.

        Keyword Args:
            mode: a: append, w: write into new output file.
        """
        output = kwargs.get('output', 'reference.hdf')
        mode = kwargs.get('mode', 'a')  # -> if override output file
        atom_specie_global = kwargs.get('atom_specie_global', None)

        with h5py.File(output, mode) as f:

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
                print(g.attrs['specie'], g.attrs['numbers'], positions[ii])

                for iproperty in properties:
                    iname = str(n_system + 1) + iproperty
                    print("results[iproperty]", iname, results[iproperty][ii])
                    g.create_dataset(iname, data=results[iproperty][ii])
