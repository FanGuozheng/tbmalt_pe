"""Load data."""
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import os
import scipy
import scipy.io
import numpy as np
import torch
import h5py
from tbmalt.common.batch import pack
from tbmalt.common.structures.system import System
ATOMNUM = {'H': 1, 'C': 6, 'N': 7, 'O': 8}
HIRSH_VOL = [10.31539447, 0., 0., 0., 0., 38.37861207, 29.90025370, 23.60491416]
Tensor = torch.Tensor


class AniDataloader:
    """ Contructor."""

    def __init__(self, store_file):
        if not os.path.exists(store_file):
            exit('Error: file not found - '+store_file)
        self.store = h5py.File(store_file, 'r')

    def h5py_dataset_iterator(self, g, prefix=''):
        """Group recursive iterator."""
        for key in g.keys():
            item = g[key]
            path = '{}/{}'.format(prefix, key)
            keys = [i for i in item.keys()]
            if isinstance(item[keys[0]], h5py.Dataset):  # test for dataset
                data = {'path': path}
                for k in keys:
                    if not isinstance(item[k], h5py.Group):
                        # dataset = np.array(item[k].value)
                        dataset = np.array(item[k][()])
                        if type(dataset) is np.ndarray:
                            if dataset.size != 0:
                                if type(dataset[0]) is np.bytes_:
                                    dataset = [a.decode('ascii') for a in dataset]

                        data.update({k: dataset})

                yield data
            else:
                yield from self.h5py_dataset_iterator(item, path)

    def __iter__(self):
        """Default class iterator (iterate through all data)."""
        for data in self.h5py_dataset_iterator(self.store):
            yield data

    def get_group_list(self):
        """Return a list of all groups in the file."""
        return [g for g in self.store.values()]

    def iter_group(self, g):
        """Allow interation through the data in a given group."""
        for data in self.h5py_dataset_iterator(g):
            yield data

    def get_data(self, path, prefix=''):
        """Return the requested dataset."""
        item = self.store[path]
        path = '{}/{}'.format(prefix, path)
        keys = [i for i in item.keys()]
        data = {'path': path}
        for k in keys:
            if not isinstance(item[k], h5py.Group):
                dataset = np.array(item[k].value)

                if type(dataset) is np.ndarray:
                    if dataset.size != 0:
                        if type(dataset[0]) is np.bytes_:
                            dataset = [a.decode('ascii') for a in dataset]

                data.update({k: dataset})
        return data

    def group_size(self):
        """Return the number of groups."""
        return len(self.get_group_list())

    def size(self):
        count = 0
        for g in self.store.values():
            print(g.items())
            count = count + len(g.items())
        return count

    def cleanup(self):
        """Close the HDF5 file."""
        self.store.close()


class LoadHdf:
    """Load h5py binary dataset.

    Arguments:
        dataType: the data type, hdf, json...
        hdf_num: how many dataset in one hdf file

    Returns:
        positions: all the coordination of molecule
        symbols: all the atoms in each molecule
    """

    def __init__(self, dataset, size, hdf_type, **kwargs):
        self.dataset = dataset
        self.size = size

        if hdf_type == 'ANI-1':
            self.numbers, self.positions, self.symbols, \
                self.atom_specie_global = self.load_ani1(**kwargs)
        elif hdf_type == 'Si':
            self.numbers, self.positions, self.symbols, \
                self.atom_specie_global, self.latvec = self.load_si(**kwargs)
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
            _number.append(System.to_element_number(data['species']).squeeze())

        for ispe, isize in enumerate(n_molecule):
            # get symbols of each atom
            symbols.extend([_specie[ispe]] * isize)
            numbers.extend([_number[ispe]] * isize)

            # add coordinates and cells
            positions.extend([icoor for icoor in _coorall[ispe][:isize]])

        return numbers, positions, symbols, atom_specie_global

    def load_si(self, **kwargs):
        """Load the data from hdf type input files."""
        dtype = kwargs.get('dtype', np.float64)

        # define the output
        numbers, positions, latvec = [], [], []

        # symbols for each molecule, global atom specie
        symbols, atom_specie_global = [], []

        # temporal coordinates for all
        _coorall = []

        # temporal cells for all
        _cellall = []

        # temporal molecule species for all
        _specie, _number = [], []

        # temporal number of molecules in all molecule species
        n_molecule = []

        # load each ani_gdb_s0*.h5 data in datalist
        adl = AniDataloader(self.dataset)
        # self.in_size = round(self.size / adl.size())  # each group size
        self.in_size = self.size

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

            # selected lattice vectors of each molecule specie
            _cellall.append(torch.from_numpy(
                data['cells'][:isize].astype(dtype)))

            # add atom species in each molecule specie
            _specie.append(data['species'])
            _number.append(System.to_element_number(data['species']).squeeze())

        for ispe, isize in enumerate(n_molecule):
            # get symbols of each atom
            symbols.extend([_specie[ispe]] * isize)
            numbers.extend([_number[ispe]] * isize)

            # add coordinates and cells
            positions.extend([icoor for icoor in _coorall[ispe][:isize]])
            latvec.extend([icell for icell in _cellall[ispe][:isize]])

        return numbers, positions, symbols, atom_specie_global, latvec

    @classmethod
    def load_reference(cls, dataset, size, properties, **kwargs):
        """Load reference from hdf type data."""
        _periodic = kwargs.get('periodic', False)
        out_type = kwargs.get('output_type', Tensor)
        data = {}
        for ipro in properties:
            data[ipro] = []

        positions, numbers = [], []
        if _periodic:
            latvecs = []

        with h5py.File(dataset, 'r') as f:
            gg = f['global_group']
            molecule_specie = gg.attrs['molecule_specie_global']
            _size = int(size / len(molecule_specie))

            # add atom name and atom number
            for imol_spe in molecule_specie:
                g = f[imol_spe]
                g_size = g.attrs['n_molecule']
                isize = min(g_size, _size)

                for imol in range(isize):  # loop for the same molecule specie

                    for ipro in properties:  # loop for each property
                        idata = g[str(imol + 1) + ipro][()]
                        data[ipro].append(LoadHdf.to_out_type(idata, out_type))

                    _position = g[str(imol + 1) + 'position'][()]
                    positions.append(LoadHdf.to_out_type(_position, out_type))
                    numbers.append(LoadHdf.to_out_type(g.attrs['numbers'], out_type))
                    if _periodic:
                        _latvec = g[str(imol + 1) + 'lattice vector'][()]
                        latvecs.append(LoadHdf.to_out_type(_latvec, out_type))

        if out_type is Tensor:
            for ipro in properties:  # loop for each property
                data[ipro] = pack(data[ipro])

        if _periodic:
            return numbers, positions, latvecs, data
        else:
            return numbers, positions, data

    @classmethod
    def load_reference_si(cls, dataset, index, properties, **kwargs):
        """Load reference from hdf type data."""
        _periodic = kwargs.get('periodic', False)
        out_type = kwargs.get('output_type', Tensor)
        data = {}
        for ipro in properties:
            data[ipro] = []

        positions, numbers = [], []
        if _periodic:
            latvecs = []

        with h5py.File(dataset, 'r') as f:
            gg = f['global_group']
            molecule_specie = gg.attrs['molecule_specie_global']

            # add atom name and atom number
            for imol_spe in molecule_specie:
                g = f[imol_spe]
                g_size = g.attrs['n_molecule']

                for imol in index:  # loop for the same molecule specie

                    for ipro in properties:  # loop for each property
                        idata = g[str(imol + 1) + ipro][()]
                        data[ipro].append(LoadHdf.to_out_type(idata, out_type))

                    _position = g[str(imol + 1) + 'position'][()]
                    positions.append(LoadHdf.to_out_type(_position, out_type))
                    numbers.append(LoadHdf.to_out_type(g.attrs['numbers'], out_type))
                    if _periodic:
                        _latvec = g[str(imol + 1) + 'lattice vector'][()]
                        latvecs.append(LoadHdf.to_out_type(_latvec, out_type))

        if out_type is Tensor:
            for ipro in properties:  # loop for each property
                data[ipro] = pack(data[ipro])

        if _periodic:
            return numbers, positions, latvecs, data
        else:
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
