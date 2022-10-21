#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""An interface to datset reading, loading and writing."""
from typing import List, Literal
import os
import pickle
import random
import scipy
import numpy as np
import torch
import h5py
import ase
import ase.io as io
from ase import Atoms
from ase.constraints import FixAtoms
from ase.build import surface
from torch import Tensor
from tbmalt import Geometry, Basis, Periodic
from torch.utils.data import Dataset as _Dataset
from tbmalt.structures.geometry import to_atomic_numbers
from tbmalt.common.batch import pack
from tbmalt.structures.geometry import to_element_species
from tbmalt.common.logger import get_logger

logger = get_logger(__name__)
n_mesh = {'gamma': [1, 1, 1, 1, 1], 'low': [1, 1, 2, 2, 3],
          'intermediate': [1, 3, 5, 7, 9], 'high': [2, 4, 8, 12, 16]}


class Dataset(_Dataset):
    """An interface to dataset inherited from `torch.utils.data.Dataset`.

    Arguments:
        properties: A list of atomic or geometric or electronic properties.

    """

    def __init__(self, numbers, positions, isperiodic, **kwargs):
        self.numbers = numbers
        self.positions = positions
        self.isperiodic = isperiodic

        for key in ('properties', 'symbols', 'atom_specie_global'):
            if key in kwargs.keys():
                setattr(self, key, kwargs[key])

        if self.isperiodic:
            setattr(self, 'cells', kwargs['cells'])
            setattr(self, 'kpoints', kwargs['kpoints'])
            setattr(self, 'klines', kwargs.get('klines', None))
            setattr(self, 'path', kwargs.get('path', None))

    @property
    def geometry(self) -> Geometry:
        """Create `Geometry` object in TBMaLT."""
        atomic_numbers = self.properties['atomic_numbers']
        assert atomic_numbers.dim() == 2, 'do not support single `Geometry`.'
        positions = self.properties['positions']
        cell = self.properties['cell'] if 'cell' in self.properties.keys() \
            else torch.zeros(len(atomic_numbers), 3, 3)

        return Geometry(atomic_numbers, positions, cell=cell, units='angstrom')

    @classmethod
    def qm9(cls):
        pass

    @classmethod
    def qm7(cls, path_to_data, n_dataset):
        dataset = scipy.io.loadmat(path_to_data)
        n_dataset_ = n_dataset
        coor_ = dataset['R']
        qatom_ = dataset['Z']
        positions = []
        for idata in range(n_dataset_):
            icoor = coor_[idata]
            natom_ = 0
            symbols_ = []

            number = torch.from_numpy(qatom_[idata][:natom_])
            coor = torch.from_numpy(icoor[:natom_, :])
            positions.append(coor)

    @classmethod
    def band(cls,
             path_to_data,
             shell_dict,
             batch_size,
             cutoff,
             groups: dict,
             logger,
             inverse=False,
             seed=0):
        """Generate band structure training dataset."""
        data = {}
        logger.info('begin to load data...')
        label_list = ['material', 'model', 'lattice']
        with h5py.File(path_to_data, 'r') as f:

            # The first order Group suggests different materials
            # train_dict, test_dict, general_dict = {}, {}, {}
            data_dict, general_dict = {}, {}
            general_dict['unique_atomic_number'] = []
            for ig, gkey in enumerate(groups.keys()):
                g = f[gkey]
                split_ratio = groups[gkey]
                subgs = [(subgkey, g[subgkey]) for subgkey in g.keys()
                         if isinstance(g[subgkey], h5py.Group)]
                logger.info(f'\n loading group {gkey}')

                # The second order Group suggests materials with different environment
                for subgkey, subg in subgs:
                    logger.info(f'load sub-group {subgkey}, train ratio: {split_ratio}')

                    numbers = torch.from_numpy(subg['numbers'][()])
                    for ii in np.unique(numbers):
                        if ii not in general_dict['unique_atomic_number'] and ii != 0:
                            general_dict['unique_atomic_number'].append(ii)
                    this_n_batch = len(numbers)

                    random.seed(seed)
                    tot_index = random.sample(torch.arange(this_n_batch).tolist(), this_n_batch)
                    n_epoch = int(this_n_batch / batch_size)
                    n_epoch = n_epoch if this_n_batch % batch_size == 0 else n_epoch + 1
                    n_load_batch = round(split_ratio * n_epoch)

                    # Skip if training or testing data size is not positive
                    if n_load_batch <= 0 or this_n_batch - n_load_batch <= 0:
                        continue

                    for ibatch in range(n_epoch):
                        ind0 = int(ibatch * batch_size)
                        ind1 = min((ibatch + 1) * batch_size, this_n_batch)
                        index = tot_index[ind0: ind1]
                        data[ibatch] = {}

                        data[ibatch]["numbers"] = numbers[index]
                        data[ibatch]["positions"] = torch.from_numpy(subg['positions'][()])[index]
                        data[ibatch]["cells"] = torch.from_numpy(subg['cells'][()])[index]
                        data[ibatch]["klines"] = torch.from_numpy(subg['klines'][()])[index]
                        data[ibatch]['vband_tot'] = torch.from_numpy(subg['vband'][()])[index]
                        data[ibatch]['cband_tot'] = torch.from_numpy(subg['cband'][()])[index]
                        data[ibatch]['band_tot'] = torch.cat([
                            data[ibatch]['vband_tot'], data[ibatch]['cband_tot']], -1)
                        data[ibatch]['n_vband'] = subg['n_vband'][()][index]
                        data[ibatch]['n_cband'] = subg['n_cband'][()][index]
                        data[ibatch]['gap'] = subg['gap'][()][index]
                        data[ibatch]['vbm_alignment'] = subg['vbm_alignment'][()][index]
                        data[ibatch]["group"] = gkey
                        data[ibatch]["subgroup"] = subgkey
                        data[ibatch].update({label_list[i]: ik for i, ik in enumerate(gkey.split('_'))})
                        data[ibatch]["labels"] = [ii.decode('utf-8') for ii in subg['labels'][()][index]]

                        data[ibatch]['geometry'] = Geometry(
                            atomic_numbers=data[ibatch]["numbers"],
                            positions=data[ibatch]["positions"],
                            cell=data[ibatch]["cells"],
                            units='angstrom',
                            dtype=torch.get_default_dtype())
                        data[ibatch]['basis'] = Basis(data[ibatch]["geometry"].atomic_numbers,
                                                      shell_dict)
                        data[ibatch]['periodic'] = Periodic(
                            data[ibatch]["geometry"],
                            data[ibatch]["geometry"].cell,
                            cutoff=cutoff)

                    # Add new epochs for batch training
                    random.seed(seed)
                    train_test_index = random.sample(torch.arange(n_epoch).tolist(), n_epoch)
                    train_pre_size = -1 if len(data_dict.keys()) == 0 else max(data_dict.keys())
                    # test_pre_size = -1 if len(train_dict.keys()) == 0 else max(train_dict.keys())

                    if not inverse:
                        data_dict.update({train_pre_size + ii + 1: data[ind]
                                          for ii, ind in enumerate(train_test_index[:n_load_batch])})
                    else:
                        data_dict.update({train_pre_size + ii + 1: data[ind]
                                          for ii, ind in enumerate(train_test_index[-n_load_batch:])})
                    # test_dict.update({test_pre_size + ii + 1: data[ind]
                    #                   for ii, ind in enumerate(train_test_index[n_load_batch:])})

                    if 'n_batch_list' not in general_dict.keys():
                        general_dict['n_batch_list'] = [n_load_batch]
                        # general_dict['n_test_batch_list'] = [this_n_batch - n_load_batch]
                    else:
                        general_dict['n_batch_list'].append(n_load_batch)
                        # general_dict['n_test_batch_list'].append(this_n_batch - n_load_batch)

        general_dict['unique_atomic_number'] = torch.tensor(general_dict['unique_atomic_number'])
        return data_dict, general_dict

    @classmethod
    def pkl(cls,
            path_to_data, type: str,
            properties: List[str],
            **kwargs) -> 'Dataset':
        """Read data from pkl object with geometry and properties.

        This input pkl files could be either from TBMaLT or the code with
        attributes including: atomic_numbers, positions, cell and the input
        properties.

        Arguments:
            path_to_data: Path to binary data which contains geometry
                information, atomic numbers, positions and atomic or
                geometric properties.
            properties: A list of atomic or geometric properties.
            to_geometry, If transfer atomic numbers and positions to
            `Geometry` object.
        """
        try:
            with open(path_to_data, 'rb') as f:
                data = pickle.load(f)
                atomic_numbers = pack(
                    [torch.from_numpy(ii) for ii in data.atomic_numbers])
                positions = pack(
                    [torch.from_numpy(ii) for ii in data.positions])
                cell = pack(
                    [torch.from_numpy(np.asarray(ii)) for ii in data.cell])
                properties = {
                    iproperty: pack([torch.from_numpy(ii)
                                     for ii in data.results[iproperty]])
                    for iproperty in properties}

            properties.update({
                'atomic_numbers': atomic_numbers,
                'positions': positions,
                'cell': cell
            })
            return cls(properties)
        except Exception:
            logger.error(f'Fails to open {path_to_data}')

    @classmethod
    def ani1(cls, dataset, size, **kwargs):
        """Load the data from ANI-1 dataset."""
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
        adl = AniDataloader(dataset)
        in_size = round(size / adl.size())  # each group size

        # such as for ani_gdb_s01.h5, there are 3 species: CH4, NH3, H2O
        for iadl, data in enumerate(adl):

            # get each molecule specie size
            size_ani = len(data['coordinates'])
            isize = min(in_size, size_ani)

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
            _number.append(to_atomic_numbers(data['species']).squeeze())

        for ispe, isize in enumerate(n_molecule):
            # get symbols of each atom
            symbols.extend([_specie[ispe]] * isize)
            numbers.extend([_number[ispe]] * isize)

            # add coordinates
            positions.extend([icoor for icoor in _coorall[ispe][:isize]])

        isperiodic = False
        kwargs = {'symbols': symbols, 'atom_specie_global': atom_specie_global}
        return cls(numbers, positions, isperiodic, **kwargs)

    @classmethod
    def anix(cls, dataset, size, **kwargs):
        """Load the data from hdf type input files."""
        dtype = kwargs.get('dtype', np.float64)
        min_size_molecule = kwargs.get('min_size_molecule', 100)

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
        adl = AniDataloader(dataset)

        # such as for ani_gdb_s01.h5, there are 3 species: CH4, NH3, H2O
        for iadl, data in enumerate(adl):

            # get each molecule specie size
            size_ani = len(data['coordinates'])
            if size_ani > min_size_molecule:
                isize = min(size, size_ani)
                _spe = to_element_species(
                    torch.from_numpy(data['atomic_numbers']))
                # global species
                for ispe in _spe:
                    if ispe not in atom_specie_global:
                        atom_specie_global.append(ispe)

                # size of each molecule specie
                n_molecule.append(isize)

                # selected coordinates of each molecule specie
                _coorall.append(torch.from_numpy(
                    data['coordinates'][:isize].astype(dtype)))

                # add atom species in each molecule specie
                _specie.append(_spe)
                _number.append(torch.from_numpy(
                    data['atomic_numbers']).squeeze())

        for ispe, isize in enumerate(n_molecule):
            # get symbols of each atom
            symbols.extend([_specie[ispe]] * isize)
            numbers.extend([_number[ispe]] * isize)

            # add coordinates
            positions.extend([icoor for icoor in _coorall[ispe][:isize]])

        isperiodic = False
        kwargs = {'symbols': symbols, 'atom_specie_global': atom_specie_global}
        return cls(numbers, positions, isperiodic, **kwargs)

    @classmethod
    def geometry(cls, dataset, format, **kwargs):
        """Read and build geometries from cif data."""
        band = kwargs.get('band', 10)
        n_kpoints = kwargs.get('n_kpoints', 10)
        kpoint_level = kwargs.get('kpoint_level', 'intermediate')
        numbers, positions = [], []
        cells, kpoints, kpts, path = [], [], [], []
        symbols, atom_specie_global = [], []

        if format != 'ase':
            isperiodic = True if ase.io.read(dataset[0]).cell is not None else False
        else:
            isperiodic = (dataset[0].cell != 0).any()

        for ii, data in enumerate(dataset):

            # get each molecule specie size
            idata = ase.io.read(data) if format != 'ase' else data
            numbers.append(torch.from_numpy(idata.numbers))
            positions.append(torch.from_numpy(idata.positions))
            if idata.cell is not None:
                cells.append(torch.from_numpy(idata.cell.array))
                _obj = idata.cell.bandpath(npoints=n_kpoints)

                # Return K-mesh according to cell size
                kpoints.append(Dataset.get_kpoints(idata.cell, kpoint_level))

                if band:
                    kpts.append(_obj.kpts)
                    path.append(_obj.path)

            if isperiodic:
                assert idata.cell is not None, 'do not support mixture molecule and solid'
            elif not isperiodic:
                assert idata.cell is None, 'do not support mixture molecule and solid'

            # global species
            symbols.append(to_element_species(torch.from_numpy(idata.numbers)))
            for ispe in symbols[-1]:
                if ispe not in atom_specie_global:
                    atom_specie_global.append(ispe)

        kwargs = {'symbols': symbols, 'atom_specie_global': atom_specie_global}
        if isperiodic:
            kwargs.update({"cells": cells, "kpoints": kpoints})
            if band:
                kwargs.update({'klines': kpts, 'path': path, 'band': True})

        return cls(numbers, positions, isperiodic, **kwargs)

    @classmethod
    def get_kpoints(cls, cell, kpoint_level: str) -> Tensor:
        """Return K-mesh automatically.

        Arguments:
            cell: Lattice cell.
            kpoint_level: K-mesh accurate level.
        """
        assert kpoint_level in n_mesh.keys(), f'kpoint_level is not in {n_mesh.keys()}'
        if isinstance(cell, np.ndarray):
            cell_size = torch.from_numpy((cell ** 2).sum(-1) ** 0.5)
        elif isinstance(cell, Tensor):
            cell_size = (cell ** 2).sum(-1) ** 0.5
        elif isinstance(cell, ase.cell.Cell):
            cell_size = torch.from_numpy((cell.array ** 2).sum(-1) ** 0.5)
        else:
            raise ValueError(f'do not support type: {type(cell)}')
        kpoint = torch.zeros(3).long()
        kpoint[cell_size.lt(2)] = n_mesh[kpoint_level][4]
        kpoint[cell_size.ge(2) * cell_size.lt(4)] = n_mesh[kpoint_level][3]
        kpoint[cell_size.ge(4) * cell_size.lt(6)] = n_mesh[kpoint_level][2]
        kpoint[cell_size.ge(6) * cell_size.lt(9)] = n_mesh[kpoint_level][1]
        kpoint[cell_size.ge(9)] = n_mesh[kpoint_level][0]

        return kpoint

    @staticmethod
    def get_eigenvalue(eigenvalue, ind_kpts, n_valence, n_conduction):
        """Select eigenvalue with indices."""
        assert eigenvalue.dim() == 3, 'only support batch eigenvalue with 3 dims'
        n_batch = eigenvalue.shape[0]
        ind_size = torch.arange(n_batch).repeat_interleave(
            len(ind_kpts) * (n_valence + n_conduction))

        _ind_kpts = ind_kpts.repeat_interleave(n_valence + n_conduction).repeat(n_batch)
        occ_band = eigenvalue.sum(1) < 0
        ind_band = torch.arange(-n_valence, n_conduction) + occ_band.sum(-1).unsqueeze(-1)
        ind_band = ind_band.repeat(1, 1, len(ind_kpts)).flatten()
        return eigenvalue[ind_size, _ind_kpts, ind_band].reshape(
            n_batch, len(ind_kpts), n_valence + n_conduction)

    @staticmethod
    def get_occ_eigenvalue(eigenvalue, ind_kpts, mask_v, n_valence, n_conduction, n_electron):
        """Select eigenvalue with indices."""
        assert eigenvalue.dim() == 3, 'only support batch eigenvalue with 3 dims'
        n_val = (n_electron / 2).long()

        # First, select all occupied states eigenvalue at certain K-points (ind_kpts)
        # Second, select valence states according to mask (mask_v)
        return pack([vals[ind_kpts, :n_val[ii]][..., mask_v[ii]]
                     for ii, vals in enumerate(eigenvalue)]),\
               pack([vals[ind_kpts, n_val[ii]: n_val[ii] + ic]
                     for ii, (vals, ic) in enumerate(zip(eigenvalue, n_conduction))])

    @classmethod
    def from_caculator(cls, calculator, properties):
        for ipro in properties:
            assert ipro in calculator.results.keys(), \
                f'{ipro} is not in {calculator.results.keys()}'
        for ipro in properties:
            return cls(calculator.results[ipro], ipro)

    @classmethod
    def load_reference(cls, dataset, size, properties, **kwargs):
        """Load reference from hdf type data."""
        out_type = kwargs.get('output_type', Tensor)
        test_ratio = kwargs.get('test_ratio', 1.0)

        # Choice about how to select data
        choice = kwargs.get('choice', 'squeeze')

        data = {}
        for ipro in properties:
            data[ipro] = []

        positions, numbers, data['cell'] = [], [], []

        with h5py.File(dataset, 'r') as f:
            gg = f['global_group']
            molecule_specie = gg.attrs['geometry_specie_global']
            try:
                data['n_band_grid'] = gg.attrs['n_band_grid']
            except:
                pass

            # Determine each geometry specie size
            if size < len(molecule_specie):
                molecule_specie = molecule_specie[:size]
                _size = 1
            else:
                _size = int(size / len(molecule_specie))

            # add atom name and atom number
            for imol_spe in molecule_specie:
                g = f[imol_spe]
                g_size = g.attrs['n_geometries']
                isize = min(g_size, _size)

                if choice == 'squeeze':
                    start = 0 if test_ratio == 1.0 else int(
                        isize * (1 - test_ratio))

                    # loop for the same molecule specie
                    for imol in range(start, isize):

                        for ipro in properties:  # loop for each property
                            idata = g[str(imol + 1) + ipro][()]
                            try:
                                if isinstance(idata, np.ndarray):
                                    data[ipro].append(
                                        Dataset.to_out_type(idata, out_type))
                                else:
                                    data[ipro].append(idata)
                            except:
                                pass

                        positions.append(Dataset.to_out_type(
                            g[str(imol + 1) + 'position'][()], out_type))
                        numbers.append(Dataset.to_out_type(
                            g.attrs['numbers'], out_type))
                        try:
                            data['cell'].append(Dataset.to_out_type(
                                g[str(imol + 1) + 'cell'][()], out_type))
                        except:
                            pass

                elif choice == 'random':
                    ind = random.sample(torch.arange(g_size).tolist(), isize)
                    # loop for the same molecule specie
                    for imol in ind:
                        for ipro in properties:  # loop for each property
                            idata = g[str(imol + 1) + ipro][()]
                            try:
                                if isinstance(idata, np.ndarray):
                                    data[ipro].append(
                                        Dataset.to_out_type(idata, out_type))
                                else:
                                    data[ipro].append(idata)
                            except:
                                data[ipro].append(idata)

                        positions.append(Dataset.to_out_type(
                            g[str(imol + 1) + 'position'][()], out_type))
                        numbers.append(Dataset.to_out_type(
                            g.attrs['numbers'], out_type))
                        try:
                            data['cell'].append(Dataset.to_out_type(
                                g[str(imol + 1) + 'cell'][()], out_type))
                        except:
                            pass

        if out_type is Tensor:
            for ipro in properties:  # loop for each property
                try:
                    data[ipro] = pack(data[ipro])
                except:
                    data[ipro] = data[ipro]
            try:
                data['cell'] = pack(data['cell'])
            except:
                pass

        return pack(numbers), pack(positions), data

    @classmethod
    def to_out_type(cls, data, out_type):
        """Transfer output data dtype."""
        if out_type is torch.Tensor:
            if type(data) is torch.Tensor:
                return data
            elif type(data) in (float, np.float16, np.float32, np.float64):
                return torch.tensor([data])
            elif type(data) is np.ndarray:
                return torch.from_numpy(data)
        elif out_type is np.ndarray:
            pass

    def __getitem__(self, index: Tensor) -> dict:
        """Return properties with selected indices from original samples."""
        return {key: val[index] for key, val in self.properties.items()}

    def __len__(self):
        return len(self.properties['numbers'])

    def __add__(self):
        pass

    def __repr__(self):
        """Representation of "Dataset" object."""
        if not self.geometry.isperiodic:
            _pe = 'molecule'
        elif not self.geometry.periodic_list.all():
            _pe = 'mixture'
        else:
            _pe = 'solid'

        return f'{self.__class__.__name__}({len(self.geometry.positions)} {_pe})'

    def metadata(self):
        pass


class AniDataloader:
    """Interface to ANI-1 data."""

    def __init__(self, input):
        if not os.path.exists(input):
            exit('Error: file not found - ' + input)
        self.input = h5py.File(input, 'r')

    def iterator(self, g, prefix=''):
        """Group recursive iterator."""
        for key in g.keys():
            item = g[key]
            path = '{}/{}'.format(prefix, key)
            keys = [i for i in item.keys()]
            if isinstance(item[keys[0]], h5py.Dataset):  # test for dataset
                data = {'path': path}
                for k in keys:
                    if not isinstance(item[k], h5py.Group):
                        dataset = np.array(item[k][()])
                        if type(dataset) is np.ndarray:
                            if dataset.size != 0:
                                if type(dataset[0]) is np.bytes_:
                                    dataset = [a.decode('ascii')
                                               for a in dataset]

                        data.update({k: dataset})

                yield data
            else:
                yield from self.iterator(item, path)

    def __iter__(self):
        """Default class iterator (iterate through all data)."""
        for data in self.iterator(self.input):
            yield data

    def size(self):
        count = 0
        for g in self.input.values():
            count = count + len(g.items())
        return count


class GeometryTo(object):
    """Transfer and write various input geometries.

    This class aims to generate batch of geometries and input easily.
    The geometries include various chemical environment, such as bulk,
    slab model, defect with selected optimized atoms, etc. The input
    also include various format, such as DFTB+, FHI-aims. For the
    input, a template is needed. With initialized geometries, further
    selected geometries and different format output could be realized
    by '__call__' function.

    Arguments:
        in_geometry_files: Single or batch input geometry files.
        to_geometry_type: Output geometry type.
        to_geometry_path: Output geometry path.

    """

    def __init__(self,
                 in_geometry_files: List[str],
                 labels,
                 # path_input_template: str,
                 to_geometry_type: Literal['h5', 'ase', 'geometry'] = 'cif',
                 to_geometry_path: str = './',
                 scale=False,
                 build_supercell=False,
                 **kwargs):
        self.in_geometry_files = in_geometry_files
        self.labels = labels
        # self.path_input_template = path_input_template
        self.to_geometry_type = to_geometry_type
        self.to_geometry_path = to_geometry_path
        self.scale = scale
        self.build_supercell = build_supercell
        self.build_slab = kwargs.get('build_slab', False)

    def __call__(self, idx: Tensor = None, form='dftbplus', build='bulk',
                 template: str = None, band_template: str = None,
                 **kwargs):
        """Transfer geometry."""

        if build == 'slab':
            self.slab_index = kwargs.get('slab_index')
            assert self.slab_index is not None, 'build_slab is True, slab_index should be defined'
            self.vacuum = kwargs.get('vacuum', [[10] * len(self.slab_index[0])] * len(self.slab_index))
            self.layer = kwargs.get('layer', [[2] * len(self.slab_index[0])] * len(self.slab_index))
            self.set_constraint = kwargs.get('set_constraint', None)
            self.constrain_extent = kwargs.get('constrain_extent', 2)

        elif build == 'scale':
            scale_params = kwargs.get('scale_params', [0.98, 0.99, 1.01, 1.02])
        elif build == 'supercell':
            supercell_params = kwargs.get(
                'supercell_params', [[[2, 0, 0], [0, 2, 0], [0, 0, 2]]] * len(self.in_geometry_files))

        elif build == 'band':
            assert band_template is not None, 'band_template should not be None'
            band_params = kwargs.get(
                'band_params', {'n_kpoints': 10, 'n_band_grid': 10})
        else:
            raise ValueError(f'invalid build value {build}')

        # Select geometries with input indices
        if idx is not None:
            self.in_geometry_files = [
                self.in_geometry_files[ii] for ii in idx]

        # Create geometric files and input files
        if isinstance(self.in_geometry_files, list):
            try:
                # To list ASE object
                _in = [io.read(geo) for geo in self.in_geometry_files]
                labels = [f'{self.labels[ii]}' for ii, _ in enumerate(_in)]

                if self.scale:
                    _in = [
                        Atoms(
                            positions=ia.positions * ii,
                            numbers=ia.numbers,
                            cell=ia.cell * ii,
                            pbc=ia.pbc,
                        )
                        for ia in _in for ii in self.scale_params
                    ]

                elif build == 'supercell':
                    _in = [
                        ase.build.sort(ase.build.make_supercell(geo, n_cell))
                        for geo, n_cell in zip(_in, supercell_params)
                    ]

                elif build == 'slab':
                    assert len(_in) == len(self.slab_index),\
                        'the size of slab_index should be equal to size of geometries'
                    _in = [surface(ii, ind, lay, vacuum=vac) for ii, sind, layer, vacuum in zip(
                        _in, self.slab_index, self.layer, self.vacuum)
                           for ind, lay, vac in zip(sind, layer, vacuum)]

                    labels = [f"{self.labels[ii]}_{''.join(tuple(map(str, ind)))}_{lay}_{str(vac)}"
                              for ii, (geo, sind, layer, vacuum) in enumerate(zip(
                            _in, self.slab_index, self.layer, self.vacuum))
                              for ind, lay, vac in zip(sind, layer, vacuum)]
            except:
                get_logger(self.__class__.__name__).error(
                    f'could not load {self.in_geometry_files}')

            self._obj_dict = {}

            if self.to_geometry_type == 'geometry':
                return Geometry.from_ase_atoms(_in)

            elif self.to_geometry_type == 'aims':
                for ii, iin in enumerate(_in):
                    # Generate geometry file and copy to target dir
                    if os.path.isdir(f'{self.to_geometry_path}/{labels[ii]}'):
                        if len(os.listdir(self.to_geometry_path)) != 0:
                            get_logger(self.__class__.__name__).info(
                                f'{self.to_geometry_path}/{labels[ii]} is not empty, all files will be removed')

                            # Remove and Build output dir
                            os.system('rm -r ' + f'{self.to_geometry_path}/{labels[ii]}')
                            os.system('mkdir -p ' + f'{self.to_geometry_path}/{labels[ii]}')
                    else:
                        get_logger(self.__class__.__name__).info(
                            f'{self.to_geometry_path}/{labels[ii]} do not exist, build now ...')
                        os.system('mkdir -p ' + f'{self.to_geometry_path}/{labels[ii]}')

                    _path = self.to_geometry_path + f'/{labels[ii]}'

                    if build != 'slab':
                        # io.write(os.path.join(_path, 'geometry.in'), iin, format='aims')
                        io.write(os.path.join(
                            self.to_geometry_path, 'geometry.in.' + str(ii)), iin, format='aims')
                    elif self.set_constraint is not None:
                        surface_z = max(iin.positions[:, -1])
                        c = FixAtoms(indices=[atom.index for atom in iin
                                              if atom.position[-1] < surface_z - self.constrain_extent])
                        iin.set_constraint(c)
                        io.write(os.path.join(_path, 'geometry.in'), iin, format='aims')

                    # Copy tempalte input files and modify
                    self._in_file = os.path.join(self.to_geometry_path, 'control.in.' + str(ii))

                    # Deal with band structures
                    if build == 'band':
                        _obj = self._aims_band(iin, band_template, band_params)
                        self._obj_dict.update({ii: _obj.todict()})

            elif self.to_geometry_type == 'dftbplus':
                for ii, iin in enumerate(_in):
                    io.write(os.path.join(
                        self.to_geometry_path, 'geo.gen.' + str(ii)), iin, format='dftb')

                    # Copy tempalte input files and modify
                    self._in_file = os.path.join(
                        self.to_geometry_path, 'dftb_in.hsd.' + str(ii))

                    # Deal with band structures
                    if build == 'band':
                        _obj = self._dftb_band(iin, band_template, band_params)
                        self._obj_dict.update({ii: _obj.todict()})
                    else:
                        [io.write(os.path.join(
                            self.to_geometry_path, 'geo.gen.' + str(ii)), ase.build.sort(iin), format='dftb')
                        for ii, iin in enumerate(_in)]


    def _aims_band(self, ase_aims_obj, band_template, band_params):
        """Modify control.in file and return ase object with band strucutres."""
        _obj = ase_aims_obj.cell.bandpath(npoints=band_params['n_kpoints'])
        kpts = _obj.kpts
        kpts_val = ''.join([
            'output band ' + str(ib)[1:-1] + ' ' + str(ie)[1:-1] + ' ' + str(
                band_params['n_band_grid']) + '\n' for ib, ie in zip(kpts[:-1], kpts[1:])])

        with open(band_template, 'r') as f:
            data = f.read()
            try:
                data = data.replace('kpts', kpts_val)
            except:
                logger.error('could not replace "kpts" with real band path,' +
                             ' check if there is keyword "kpts" in template')

        # Write modified data to new control.in
        with open(self._in_file, 'w') as f:
            f.write(data)

        return _obj

    def _dftb_band(self, ase_aims_obj, dftb_band_template, band_params):
        """Modify control.in file and return ase object with band strucutres."""
        _obj = ase_aims_obj.cell.bandpath(npoints=band_params['n_kpoints'])
        kpts = _obj.kpts
        kpts_val = ''.join([
            '1 ' + str(ib)[1:-1] + ' \n' + str(band_params['n_band_grid'] - 1) + ' '
            + str(ie)[1:-1] + '\n' for ib, ie in zip(kpts[:-1], kpts[1:])])
        with open(dftb_band_template, 'r') as f:
            data = f.read()
            try:
                data = data.replace('kpts', kpts_val)
            except:
                logger.error('could not replace "kpts" with real band path,' +
                             ' check if there is keyword "kpts" in template')

        # Write modified data to new control.in
        with open(self._in_file, 'w') as f:
            f.write(data)

        return _obj

    @property
    def obj_dict(self):
        """ASE object dictionary."""
        return self._obj_dict
