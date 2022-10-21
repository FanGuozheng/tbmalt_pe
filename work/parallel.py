#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Example to run training."""
import pickle
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from tbmalt.structures.geometry import Geometry
from tbmalt.ml.optim import OptHs, OptVcr, OptTvcr
from tbmalt.common.parameter import params
from tbmalt.io.hdf import LoadHdf
from tbmalt.physics.dftb.dftb import Dftb2
from tbmalt.ml.skfeeds import SkfFeed
from tbmalt.structures.basis import Basis
from tbmalt.physics.dftb.slaterkoster import hs_matrix
from tbmalt.ml.skfeeds import VcrFeed

from torch import Tensor
torch.set_printoptions(15)
torch.set_default_dtype(torch.float64)

###########
# general #
###########
target = 'optimize'
device = torch.device('cpu')

###################
# optimize params #
###################
size_opt = 1000
params['ml']['task'] = 'mlIntegral'
params['ml']['compression_radii_min'] = 2.0
params['ml']['compression_radii_max'] = 9.0
dataset_aims = './train/dataset/aims_6000_01.hdf'
version = 'old'
global_group0 = 'global_group'
global_group = 'train1'
dataset_dftb = './train/dataset/scc_6000_01.hdf'
params['ml']['targets'] = ['charge']  # charge, dipole, gap, cpa, homo_lumo
params['ml']['max_steps'] = 12
h_compr_feed = True
s_compr_feed = True
global_r = True
break_tolerance = True
write_train_data = True
params['ml']['charge_weight'], params['ml']['dipole_weight'] = 1.0, 1.0

parallel = True
device_ids = [0, 1, 2]

##################
# predict params #
##################
dataset_pred, size_pred = './train/dataset/aims_10000_03.hdf', 1000
params['ml']['ml_method'] = 'linear'  # nn, linear, random_forest
aims_property = ['dipole', 'charge']
dftb_property = ['dipole', 'charge']

# params['ml']['max_steps'] = 50
pickle_file = 'opt_compr.pkl'

max_l = {1: 0, 6: 1, 7: 1, 8: 1}
shell_dict = {1: [0], 6: [0, 1], 7: [0, 1], 8: [0, 1]}
# vcr = torch.tensor([1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5., 6., 8., 10.])
vcr = {1: torch.tensor([1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5., 6., 8., 10.]),
       6: torch.tensor([1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5., 6., 8., 10.]),
       7: torch.tensor([1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5., 6., 8., 10.]),
       8: torch.tensor([1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5., 6., 8., 10.])}
tvcr = torch.tensor([2., 2.5, 3., 4., 5., 7., 10.])


def optimize(dataset_ref, size, dataset_dftb=None, **kwargs):
    """Optimize spline parameters or compression radii."""
    params['ml']['lr'] = 0.001 if params['ml']['task'] == 'mlIntegral' else 0.01

    geo_opt, data_ref = _load_ref(dataset_ref, size, aims_property,
                                  version=version, global_group=global_group0)
    if dataset_dftb is not None:
        geo_dftb, data_dftb = _load_ref(dataset_dftb, size, dftb_property)

    if 'hirshfeld_volume_ratio' in data_ref.keys():
        data_ref['cpa'] = data_ref['hirshfeld_volume_ratio']

    # optimize integrals with spline parameters
    if params['ml']['task'] == 'mlIntegral':
        time0 = time.time()
        params['ml']['no_grad'] = False
        params['dftb']['path_to_skf'] = './train/slko/mio_new.hdf'
        opt = OptHs(geo_opt, data_ref, params, shell_dict)
        if parallel:
            # torch.nn.DataParallel(opt(break_tolerance=break_tolerance), device_ids=device_ids)
            DDP(opt(break_tolerance=break_tolerance), device_ids=device_ids)
        else:
            dftb = opt(break_tolerance=break_tolerance)

        # save training instance
        with open('opt_int.pkl', "wb") as f:
            pickle.dump(opt, f)
        time_end = time.time()
        print('running mlIntegral time:', time_end - time0)

    # optimize integrals with compression radii
    elif params['ml']['task'] == 'vcr':
        params['dftb']['path_to_skf'] = '../work/train/slko/vcr.h5'
        params['dftb']['path_to_skf2'] = '../slko/mio/'
        opt = OptVcr(geo_opt, data_ref, params, vcr, shell_dict,
                     h_compr_feed=h_compr_feed, s_compr_feed=s_compr_feed,
                     interpolation='BicubInterp', global_r=global_r)
        dftb = opt(break_tolerance=break_tolerance)

        # save training instance
        with open('opt_compr.pkl', "wb") as f:
            pickle.dump(opt, f)

    # optimize integrals with compression radii
    elif params['ml']['task'] == 'tvcr':
        params['dftb']['path_to_skf'] = '../work/train/slko/tvcr.h5'
        opt = OptTvcr(geo_opt, data_ref, params, tvcr, shell_dict,
                     h_compr_feed=h_compr_feed, s_compr_feed=s_compr_feed,
                     interpolation='BSpline', global_r=global_r)
        dftb = opt(break_tolerance=break_tolerance)

        # save training instance
        with open('opt_compr.pkl', "wb") as f:
            pickle.dump(opt, f)

    if 'cpa' in params['ml']['targets']:
        dftb_mio = _cal_cpa(geo_opt, params, path='../unittests/slko/mio-1-1/')
        _plot(data_ref['cpa'], dftb.cpa, dftb_mio.cpa, 'cpa')

    params['ml']['targets'] = ['charge', 'dipole']
    for target in params['ml']['targets']:
        _plot(data_ref[target], getattr(dftb, target), data_dftb[target], target)


def write_h5(dataset_ref, size, dataset_dftb=None, **kwargs):
    """Optimize spline parameters or compression radii."""
    import h5py
    import random
    from tbmalt.common.batch import pack
    from tbmalt.structures.geometry import batch_chemical_symbols

    size = 4200
    size_split = [[1000, 400], [1000, 400], [1000, 400]]
    seed_list = [[0, 1], [2, 3], [4, 5]]
    name_lay1 = ['run1', 'run2', 'run3']
    name_lay2 = ['train', 'test']

    fa = h5py.File('dataset.h5', 'a')

    with h5py.File(dataset_ref, 'r') as f:

        gg = f['global_group']
        geo_specie = gg.attrs['molecule_specie_global'] \
            if version == 'old' else gg.attrs['geometry_specie']

        # _size = int(size / len(geo_specie))
        # print(_size, geo_specie, size)

        for size, seeds, name1 in zip(size_split, seed_list, name_lay1):

            g1 = fa[name1] if name1 in fa else fa.create_group(name1)

            for isize, seed, name2 in zip(size, seeds, name_lay2):
                # add atom name and atom number
                g2 = g1[name2] if name2 in g1 else g1.create_group(name2)

                _size = int(isize / len(geo_specie))
                data = {}
                for ipro in ['dipole', 'charge', 'hirshfeld_volume_ratio']:
                    data[ipro] = []

                positions, numbers = [], []

                for imol_spe in geo_specie:
                    random.seed(seed)

                    g = f[imol_spe]

                    g_size = g.attrs['n_molecule']
                    this_size = min(g_size, _size)
                    random_idx = random.sample(range(g_size), this_size)

                    for imol in random_idx:
                        for ipro in ['dipole', 'charge', 'hirshfeld_volume_ratio']:
                            idata = g[str(imol + 1) + ipro][()]
                            data[ipro].append(torch.from_numpy(idata))

                        positions.append(torch.from_numpy(g[str(imol + 1) + 'position'][()]))
                        numbers.append(torch.from_numpy(g.attrs['numbers']))
                    # numbers.append(number)
                for ipro in ['dipole', 'charge', 'hirshfeld_volume_ratio']:
                    data[ipro] = pack(data[ipro])

                positions = pack(positions)
                numbers = pack(numbers)
                # unique_numbers = torch.unique(numbers, sorted=False, dim=0)
                unique_numbers = torch.unique(numbers)

                g2.attrs['unique_atoms'] = unique_numbers.tolist()

                # if 'geometry_specie' not in gg.attrs:
                #     gg.attrs['geometry_specie'] = []

                # for number in unique_numbers:
                #     mask = (number == numbers).all(-1)
                symbol = batch_chemical_symbols(numbers)
                # if ''.join(symbol) not in f.keys():  # -> new molecule specie

                # add to molecule_specie_global in global group
                # gg.attrs['geometry_specie'] = np.unique(
                #     np.concatenate([gg.attrs['geometry_specie'],
                #                     np.array([''.join(symbol)])])).tolist()
                #
                # g = gg.create_group(''.join(symbol))
                g2.attrs['label'] = [''.join(a) for a in symbol]
                g2.attrs['size'] = len(numbers)
                g2.create_dataset('atomic_numbers', data=numbers)
                g2.create_dataset('positions', data=positions)
                # g.attrs['size_geometry'] = torch.count_nonzero(number)
                # g.attrs['n_geometry'] = mask.sum()

                # n_geometry = g.attrs['n_geometry']  # each molecule specie number
                # g.attrs['n_geometry'] = n_geometry + 1
                # g.create_dataset('positions', data=deflate(positions[mask]))

                for pro in ['dipole', 'charge', 'hirshfeld_volume_ratio']:
                    # g.create_dataset(pro, data=deflate(results[pro][mask]))
                    g2.create_dataset(pro, data=data[pro])


def predict(pickle_file: str, dataset: str, size: int, **kwargs):
    """Test optimized results."""
    device = kwargs.get('device', torch.device('cpu'))

    # load optimized object
    loaded_model = pickle.load(open(pickle_file, 'rb'))

    # update training params in optimized object
    loaded_model.params['ml']['ml_method'] = params['ml']['ml_method']

    geo_dftb, data_dftb = _load_ref(dataset_dftb, size, params['ml']['targets'])
    geo_aims, data_aims = _load_ref(dataset_aims, size, params['ml']['targets'])

    dftb = loaded_model.predict(geo_aims)

    for target in params['ml']['targets']:
        _plot(data_aims[target], getattr(dftb, target), data_dftb[target], target)


def _load_ref(dataset, size, properties, units='angstrom', **kwargs):
    """Helper function to load dataset, return `Geometry` object, data."""
    numbers, positions, data = LoadHdf.load_reference(
        dataset, size, properties, **kwargs)
    cell = kwargs.get('cell', None)
    geo = Geometry(numbers, positions, units=units, cell=cell)

    return geo, data


def _cal_cpa(geometry, parameter, path):
    basis = Basis(geometry.atomic_numbers, shell_dict)
    h_feed, s_feed = SkfFeed.from_dir(
        path, shell_dict, geometry,
        interpolation='PolyInterpU', h_feed=True, s_feed=True)
    ham = hs_matrix(geometry, basis, h_feed)
    over = hs_matrix(geometry, basis, s_feed)
    dftb = Dftb2(parameter, geometry, ham, over, from_skf=True)
    dftb()
    return dftb


def dftb(dataset_dftb, size, params, device):
    params['dftb']['path_to_skf'] = './vcr.h5'
    params['dftb']['path_to_skf2'] = '../../../slko/mio-1-1/'

    geo, data = _load_ref(dataset_dftb, size, dftb_property,
                          version=version, global_group=global_group0)
    compr = torch.zeros(*geo.atomic_numbers.shape, 2)
    init_dict = {1: torch.tensor([ 3.0]),
                 6: torch.tensor([ 2.7]),
                 7: torch.tensor([ 2.2]),
                 8: torch.tensor([ 2.3])}
    for ii, iu in enumerate(torch.tensor([1, 6, 7, 8])):
        mask = geo.atomic_numbers == iu
        compr[mask] = init_dict[iu.tolist()]

    basis = Basis(geo.atomic_numbers, shell_dict)
    h_feed = VcrFeed.from_dir(
            params['dftb']['path_to_skf'], shell_dict, vcr,
            skf_type='h5', geometry=geo, integral_type='H',
            interpolation='BicubInterp')
    s_feed = VcrFeed.from_dir(
            params['dftb']['path_to_skf'], shell_dict, vcr,
            skf_type='h5', geometry=geo, integral_type='S',
            interpolation='BicubInterp')

    ham = hs_matrix(geo, basis, h_feed, multi_varible=compr)

    over = hs_matrix(geo, basis, s_feed, multi_varible=compr)

    # self.ham_list.append(ham.detach()), self.over_list.append(over.detach())
    dftb = Dftb2(geo, shell_dict, params['dftb']['path_to_skf'], repulsive=False)
    dftb(hamiltonian=ham, overlap=over)
    print(data['charge'] - dftb.charge)
    import matplotlib.pyplot as plt
    plt.plot(data['charge'].flatten(), dftb.charge.flatten())
    plt.show()
    plt.hist(data['charge'] - dftb.charge)
    plt.show()


def _plot(reference: Tensor, data1: Tensor, data2: Tensor, target: str,
          save: bool = True):
    """Plot single target with optimized value and DFTB value."""
    mae1 = (abs(reference - data1)).sum() / reference.shape[0]
    mae2 = (abs(reference - data2)).sum() / reference.shape[0]
    rmin, rmax = torch.min(reference), torch.max(reference)
    plt.plot(np.linspace(rmin, rmax), np.linspace(rmin, rmax), 'k')
    plot1 = plt.plot(reference, data1.detach(), 'rx')
    plot2 = plt.plot(reference, data2.detach(), 'bo')
    plt.xlabel(f'reference {target}')
    plt.ylabel(f'optimized {target} and DFTB {target}')
    plt.legend([plot1[0], plot2[0]], [f'opt MAE: {mae1}', f'DFTB MAE: {mae2}'])
    if save:
        plt.savefig(target, dpi=300)
    plt.show()


if __name__ == '__main__':
    """Main function."""
    if target == 'optimize':
        optimize(dataset_aims, size_opt, dataset_dftb, device=device)
    elif target == 'predict':
        predict(pickle_file, dataset_pred, size_pred, device=device)
    elif target == 'dftb':
        dftb(dataset_dftb, size_opt, params, device=device)
    elif target == 'write_h5':
        write_h5(dataset_aims, size_opt, dataset_dftb, device=device)
