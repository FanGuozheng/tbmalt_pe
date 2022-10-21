#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Load data."""
from typing import Tuple, Union, Literal
from abc import ABC
import random
import logging
import json
import os
import scipy
import scipy.io
import numpy as np
import torch
import ase
import ase.io as io
import h5py
from tbmalt.common.batch import pack
from tbmalt.structures.geometry import Geometry
ATOMNUM = {'H': 1, 'C': 6, 'N': 7, 'O': 8}
HIRSH_VOL = [10.31539447, 0., 0., 0., 0., 38.37861207, 29.90025370, 23.60491416]
Tensor = torch.Tensor


class Hdf(ABC):

    def __init__(self):
        pass


class ReadGeometry:
    """Transfer various input geometry to TBMaLT geometry or h5 files.

    Arguments:
        geometry_files: Single or batch input files.

    """

    def __init__(self, geometry_files: Union[str, list],
                 geometry_out_type: Literal['h5', 'ase', 'geometry'] = 'cif',
                 geometry_out_path: str = './'):
        self.geometry_files = geometry_files
        self.geometry_out_type = geometry_out_type
        self.geometry_out_path = geometry_out_path
        if os.path.isdir(geometry_out_path):
            os.system('rm -r ' + geometry_out_path)
            logging.getLogger(f'{geometry_out_path} exist, romove current dir')

        # Build output dir
        os.system('mkdir -p ' + self.geometry_out_path)

    def cif(self):
        """"""
        if isinstance(self.geometry_files, list):
            _in = [io.read(ii) for ii in self.geometry_files]
            if self.geometry_out_type == 'geometry':
                return Geometry.from_ase_atoms(_in)
            elif self.geometry_out_type == 'aims':
                [io.write('geometry.in.' + str(ii), iin, format='aims')
                 for ii, iin in enumerate(_in)]
                os.system('mv geometry.in.* ' + self.geometry_out_path)
            elif self.geometry_out_type == 'dftbplus':
                [io.write('geo.gen.' + str(ii), iin, format='dftb')
                 for ii, iin in enumerate(_in)]
                os.system('mv geo.gen.* ' + self.geometry_out_path)



def read_hdf(path_to_hdf: str) -> Tuple[list, list, list]:
    """Read positions, species, energies from `path_to_hdf`."""
    positions, species, energies = [], [], []
    # such as for ani_gdb_s01.h5, there are 3 species: CH4, NH3, H2O
    with h5py.File(path_to_hdf) as f:
        # there is usually only one key in path_to_hdf
        for ikey in f.keys():
            # each molecule specie in ikey
            for jkey in f[ikey]:
                coordinate = f[ikey][jkey]['coordinates'][()]
                energy = f[ikey][jkey]['energies'][()]
                specie = [a.decode('utf8') for a in f[ikey][jkey]['species'][()]]
                species.append(specie)
                positions.append(coordinate)
                energies.append(energy)
    return positions, species, energies


def to_geo(positions: list, species: list, number_mol: list):
    """Constructs ase.Atoms and write into DFTB+ geo type data."""
    # loop over oall the molecule species from the dataset
    for position, specie, num in zip(positions, species, number_mol):
        # loop over each molecule specie and write into DFTB geo type
        # you can define the name style and the path you prefer
        for ii, mol in enumerate(position[:num]):
            mol_obj = ase.Atoms(specie, mol)
            ase.io.write(str(mol_obj.symbols) + '.gen.' + str(ii), mol_obj, format='dftb')

# if __name__ == '__main__':
#     """Read from ANI-1 and write into DFTB+ geo type data."""
#     path_to_hdf = '/home/gz_fan/Public/tbmalt/dataset/ani_gdb_s03.h5'
#     positions, species, energies = read_hdf(path_to_hdf)
#     print(species, [ip.shape for ip in positions], len(positions))
#     # the [2] * len(species) is the number of molecules for each molecule specie
#     # to_geo(positions, species, [2] * len(species))


class AniDataloader:
    """Interface to ANI-1 data."""

    def __init__(self, store_file):
        if not os.path.exists(store_file):
            exit('Error: file not found - ' + store_file)
        self.store = h5py.File(store_file, 'r')

    def size(self):
        count = 0
        for g in self.store.values():
            count = count + len(g.items())
        return count


class LoadHdf(Hdf):
    """Load h5py binary dataset.

    Arguments:
        dataType: the data type, hdf, json...
        hdf_num: how many dataset in one hdf file

    Returns:
        positions: all the coordination of molecule
        symbols: all the Geometry in each molecule
    """

    def __init__(self, dataset, size, hdf_type, **kwargs):
        self.dataset = dataset
        self.size = size

        if hdf_type == 'ANI-1':
            self.numbers, self.positions, self.symbols, \
                self.atom_specie_global = self.load_ani1(**kwargs)
        elif hdf_type == 'hdf_reference':
            self.load_reference()

    def load_ani1(self, **kwargs):
        """Load the data from hdf type input files."""
        dtype = kwargs.get('dtype', np.float64)

        # define the output
        numbers, positions = [], []

        # symbols for each molecule, global atom specie
        symbols, atom_specie_global = [], []

        # temporal coordinates for all
        _coorall = []

        # temporal molecule species for all
        _specie, _number = [], []

        # temporal number of molecules in all molecule species
        n_molecule = []

        # load each ani_gdb_s0*.h5 data in datalist
        adl = AniDataloader(self.dataset)
        self.in_size = round(self.size / adl.size())  # each group size

        # such as for ani_gdb_s01.h5, there are 3 species: CH4, NH3, H2O
        for iadl, data in enumerate(adl):

            # get each molecule specie size
            size_ani = len(data['coordinates'])
            isize = min(self.in_size, size_ani)

            # global species
            for ispe in data['species']:
                if ispe not in atom_specie_global:
                    atom_specie_global.append(ispe)

            # size of each molecule specie
            n_molecule.append(isize)

            # selected coordinates of each molecule specie
            _coorall.append(torch.from_numpy(
                data['coordinates'][:isize].astype(dtype)))

            # add atom species in each molecule specie
            _specie.append(data['species'])
            _number.append(Geometry.to_element_number(data['species']).squeeze())

        for ispe, isize in enumerate(n_molecule):
            # get symbols of each atom
            symbols.extend([_specie[ispe]] * isize)
            numbers.extend([_number[ispe]] * isize)

            # add coordinates
            positions.extend([icoor for icoor in _coorall[ispe][:isize]])

        return numbers, positions, symbols, atom_specie_global

    @classmethod
    def load_reference(cls, dataset, size, properties, **kwargs):
        """Load reference from hdf type data."""
        out_type = kwargs.get('output_type', Tensor)
        test_ratio = kwargs.get('test_ratio', 1.0)
        version = kwargs.get('version', 'old')
        global_group = kwargs.get('global_group', 'global_group')

        data = {}
        for ipro in properties:
            data[ipro] = []

        positions, numbers = [], []

        with h5py.File(dataset, 'r') as f:

            gg = f[global_group]

            geo_specie = gg.attrs['molecule_specie_global'] \
                if version == 'old' else gg.attrs['geometry_specie']

            _size = int(size / len(geo_specie))

            # add atom name and atom number
            for imol_spe in geo_specie:
                if version == 'old':
                    g = f[imol_spe]
                    g_size = g.attrs['n_molecule']
                    isize = min(g_size, _size)
                    start = 0 if test_ratio == 1.0 else int(isize * (1 - test_ratio))
                    random_idx = random.sample(range(g_size), isize)

                    # for imol in range(start, isize):  # loop for the same molecule specie
                    for imol in random_idx:

                        for ipro in properties:  # loop for each property
                            idata = g[str(imol + 1) + ipro][()]
                            data[ipro].append(LoadHdf.to_out_type(idata, out_type))

                        _position = g[str(imol + 1) + 'position'][()]
                        positions.append(LoadHdf.to_out_type(_position, out_type))
                        numbers.append(LoadHdf.to_out_type(g.attrs['numbers'], out_type))
                else:
                    g = f[imol_spe]
                    g_size = g.attrs['n_geometry']
                    isize = min(g_size, _size)
                    random_idx = random.sample(range(g_size), isize)

                    # for imol in range(start, isize):  # loop for the same molecule specie
                    for ipro in properties:  # loop for each property
                        idata = g[ipro][()][random_idx]
                        data[ipro].append(torch.from_numpy(idata))

                    positions.append(torch.from_numpy(g['positions'][()][random_idx]))
                    number = torch.from_numpy(g.attrs['numbers']).repeat(len(random_idx), 1)
                    numbers.append(number)

                    # for imol in random_idx:

                        # for ipro in properties:  # loop for each property
                        #     idata = g[str(imol + 1) + ipro][()]
                        #     data[ipro].append(LoadHdf.to_out_type(idata, out_type))

                        # _position = g[str(imol + 1) + 'position'][()]
                        # positions.append(LoadHdf.to_out_type(_position, out_type))
                        # numbers.append(LoadHdf.to_out_type(g.attrs['numbers'], out_type))

        if out_type is Tensor and version == 'old':
            for ipro in properties:  # loop for each property
                data[ipro] = pack(data[ipro])
        elif out_type is Tensor and version == 'new':
            numbers = pack(numbers).flatten(0, 1)
            positions = pack(positions).flatten(0, 1)
            for ipro in properties:  # loop for each property
                data[ipro] = pack(data[ipro]).flatten(0, 1)

        return numbers, positions, data

    @classmethod
    def to_out_type(cls, data, out_type):
        """Transfer data type."""
        if out_type is torch.Tensor:
            if type(data) is torch.Tensor:
                return data
            elif type(data) in (float, np.float16, np.float32, np.float64):
                return torch.tensor([data])
            elif type(data) is np.ndarray:
                return torch.from_numpy(data)
            else:
                raise ValueError('not implemented data type')
        elif out_type is np.ndarray:
            pass

    @classmethod
    def get_info(cls, dataset):
        """Get general information from 'global_group' of h5py type dataset."""
        with h5py.File(dataset, 'r') as f:
            g = f['global_group']
            if 'molecule_specie_global' in g.attrs.keys():

                # print each subgroup information
                for imol in g.attrs['molecule_specie_global']:
                    print('molecule type:', imol)
                    print('numbers:', f[imol].attrs['numbers'])
                    print('number of molecules:', f[imol].attrs['n_molecule'])

            if 'atom_specie_global' in g.attrs.keys():
                print('global atom specie:', g.attrs['atom_specie_global'])


class LoadJson:

    def __init__(self):
        pass

    def load_json_data(self):
        """Load the data from json type input files."""
        dire = self.para['pythondata_dire']
        filename = self.para['pythondata_file']
        positions = []

        with open(os.path.join(dire, filename), 'r') as fp:
            fpinput = json.load(fp)

            if 'symbols' in fpinput['general']:
                symbols = fpinput['general']['symbols'].split()

            for iname in fpinput['geometry']:
                icoor = fpinput['geometry'][iname]
                positions.append(torch.from_numpy(np.asarray(icoor)))
        return positions, symbols


class LoadQM7:

    def __init__(self):
        pass

    def loadqm7(self, dataset):
        """Load QM7 type data."""
        dataset = scipy.io.loadmat(dataset)
        n_dataset_ = self.para['n_dataset'][0]
        coor_ = dataset['R']
        qatom_ = dataset['Z']
        positions = []
        specie = []
        symbols = []
        specie_global = []
        for idata in range(n_dataset_):
            icoor = coor_[idata]
            natom_ = 0
            symbols_ = []
            for iat in qatom_[idata]:
                if iat > 0.0:
                    natom_ += 1
                    idx = int(iat)
                    ispe = \
                        list(ATOMNUM.keys())[list(ATOMNUM.values()).index(idx)]
                    symbols_.append(ispe)
                    if ispe not in specie_global:
                        specie_global.append(ispe)

            number = torch.from_numpy(qatom_[idata][:natom_])
            coor = torch.from_numpy(icoor[:natom_, :])
            positions.append(coor)
            symbols.append(symbols_)
            specie.append(list(set(symbols_)))


class Split:
    """Split tensor according to chunks of split_sizes.

    Parameters
    ----------
    tensor : `torch.Tensor`
        Tensor to be split
    split_sizes : `list` [`int`], `torch.tensor` [`int`]
        Size of the chunks
    dim : `int`
        Dimension along which to split tensor

    Returns
    -------
    chunked : `tuple` [`torch.tensor`]
        List of tensors viewing the original ``tensor`` as a
        series of ``split_sizes`` sized chunks.

    Raises
    ------
    KeyError
        If number of elements requested via ``split_sizes`` exceeds hte
        the number of elements present in ``tensor``.
    """
    def __init__(tensor, split_sizes, dim=0):
        if dim < 0:  # Shift dim to be compatible with torch.narrow
            dim += tensor.dim()

        # Ensure the tensor is large enough to satisfy the chunk declaration.
        if tensor.size(dim) != split_sizes.sum():
            raise KeyError(
                'Sum of split sizes fails to match tensor length along specified dim')

        # Identify the slice positions
        splits = torch.cumsum(torch.Tensor([0, *split_sizes]), dim=0)[:-1]

        # Return the sliced tensor. use torch.narrow to avoid data duplication
        return tuple(tensor.narrow(int(dim), int(start), int(length))
                     for start, length in zip(splits, split_sizes))
