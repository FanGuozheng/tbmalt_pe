#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Example to run training."""
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
from ase.build import molecule

from tbmalt.structures.geometry import Geometry
from tbmalt.ml.optim import OptHs, OptVcr, OptTvcr
from tbmalt.common.parameter import params
from tbmalt.io.hdf import LoadHdf
from tbmalt.physics.dftb.dftb import Dftb2
from tbmalt.ml.skfeeds import SkfFeed
from tbmalt.structures.basis import Basis
from tbmalt.physics.dftb.slaterkoster import hs_matrix
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
size_opt = 100
params['ml']['task'] = 'vcr'
params['ml']['compression_radii_min'] = 2.0
params['ml']['compression_radii_max'] = 9.0
dataset_aims = './dataset/aims_6000_01.hdf'
dataset_dftb = './dataset/scc_6000_01.hdf'
params['ml']['targets'] = ['charge']  # charge, dipole, gap, cpa, homo_lumo
params['ml']['max_steps'] = 12
h_compr_feed = True
s_compr_feed = True
global_r = True
break_tolerance = True
params['ml']['charge_weight'], params['ml']['dipole_weight'] = 1.0, 1.0

##################
# predict params #
##################
dataset_pred, size_pred = './dataset/aims_10000_03.hdf', 1000
params['ml']['ml_method'] = 'linear'  # nn, linear, random_forest
aims_property = ['charge', 'dipole', 'homo_lumo', 'hirshfeld_volume_ratio']
dftb_property = ['charge', 'dipole', 'homo_lumo']
# params['ml']['max_steps'] = 50
pickle_file = 'opt_compr.pkl'

max_l = {1: 0, 6: 1, 7: 1, 8: 1}
shell_dict = {1: [0], 6: [0, 1], 7: [0, 1], 8: [0, 1]}
vcr = torch.tensor([1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5., 6., 8., 10.])
tvcr = torch.tensor([2., 2.5, 3., 4., 5., 7., 10.])


def optimize(dataset_ref, size, dataset_dftb=None, **kwargs):
    """Optimize spline parameters or compression radii."""
    params['ml']['lr'] = 0.001 if params['ml']['task'] == 'mlIntegral' else 0.01

    geo_opt, data_ref = _load_ref(dataset_ref, size, [
        'charge', 'dipole', 'hirshfeld_volume_ratio'])
    data_ref['cpa'] = data_ref['hirshfeld_volume_ratio']
    if dataset_dftb is not None:
        geo_dftb, data_dftb = _load_ref(dataset_dftb, size, ['charge', 'dipole'])

    # optimize integrals with spline parameters
    if params['ml']['task'] == 'mlIntegral':
        params['dftb']['path_to_skf'] = './slko/mio_new.hdf'
        opt = OptHs(geo_opt, data_ref, params, shell_dict)
        dftb = opt(break_tolerance=break_tolerance)

        # save training instance
        with open('opt_int.pkl', "wb") as f:
            pickle.dump(opt, f)

    # optimize integrals with compression radii
    elif params['ml']['task'] == 'vcr':
        params['dftb']['path_to_skf'] = './vcr.h5'
        params['dftb']['path_to_skf2'] = '../../slko/mio-1-1/'
        opt = OptVcr(geo_opt, data_ref, params, vcr, shell_dict,
                     h_compr_feed=h_compr_feed, s_compr_feed=s_compr_feed,
                     interpolation='BicubInterp', global_r=global_r)
        dftb = opt(break_tolerance=break_tolerance)

        # save training instance
        with open('opt_compr.pkl', "wb") as f:
            pickle.dump(opt, f)

    # optimize integrals with compression radii
    elif params['ml']['task'] == 'tvcr':
        params['dftb']['path_to_skf'] = './tvcr.h5'
        opt = OptTvcr(geo_opt, data_ref, params, tvcr, shell_dict,
                     h_compr_feed=h_compr_feed, s_compr_feed=s_compr_feed,
                     interpolation='BSpline', global_r=global_r)
        dftb = opt(break_tolerance=break_tolerance)
        print('compr', opt.compr)

        # save training instance
        with open('opt_compr.pkl', "wb") as f:
            pickle.dump(opt, f)

    if 'cpa' in params['ml']['targets']:
        dftb_mio = _cal_cpa(geo_opt, params, path='../unittests/slko/mio-1-1/')
        _plot(data_ref['cpa'], dftb.cpa, dftb_mio.cpa, 'cpa')

    params['ml']['targets'] = ['charge', 'dipole']
    for target in params['ml']['targets']:
        _plot(data_ref[target], getattr(dftb, target), data_dftb[target], target)


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
        dataset, size, properties)
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
