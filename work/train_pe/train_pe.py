#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Example to run training."""
import os
import pickle
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import time
import logging
from sklearn.ensemble import RandomForestRegressor
import torch

from tbmalt import Geometry, Dftb2, Dftb1
from tbmalt.ml.optim import OptHsPe, Scale, VcrPe
from tbmalt.common.parameter import params
from tbmalt.io.hdf import LoadHdf
from tbmalt.io import Dataset
from tbmalt.physics.dftb.dftb import Dftb2
from tbmalt.ml.skfeeds import SkfFeed, VcrFeed, TvcrFeed
from tbmalt.structures.basis import Basis
from tbmalt.ml.acsf import Acsf
from torch import Tensor
from tbmalt.common.logger import get_logger
from tbmalt.common.batch import pack, merge
from tbmalt.ml.cacsf_pair import g_pe_pair, g_pe
torch.set_printoptions(4)
torch.set_default_dtype(torch.float64)
logger = get_logger(__name__)

###########
# general #
###########
task = 'train'
pickle_file = ['./opt.pkl',]
device = torch.device('cpu')


@dataclass
class Dataclass:
    """Data class aims to minimize the data storage."""

    num_opt: list
    pos_pe_opt: list
    dist_pe_opt: list
    onsite: list
    scale: dict
    geometry_opt: object
    train_onsite: bool
    orbital_resolved: bool
    scale_ham: bool


###################
# optimize params #
###################
params['ml']['task'] = 'scale'  # mlIntegral, scale, vcr
params['ml']['compression_radii_min'] = 2.0
params['ml']['compression_radii_max'] = 9.0
params['ml']['epoch'] = 1000
elements = ['Si', 'C']
kpoint_level = 'high'
loss = 'L1Loss'
tolerance = 1E-5
train_1der = False  # if training loss function include first derivative
inverse = False
skf_type = 'skf'
alignment = 'vbm'
loss_fn = 'MSELoss'
plot_std = False
params['ml']['targets'] = ['band']  # charge, dipole, gap, cpa, homo_lumo
params['ml']['max_steps'] = 40
h_compr_feed = True
s_compr_feed = True
global_r = False
break_tolerance = True
params['ml']['charge_weight'], params['ml']['dipole_weight'] = 1.0, 1.0
params['dftb']['path_to_skf'] = path_sk = './slko'

##################
# predict params #
##################
params['ml']['ml_method'] = 'linear'  # nn, linear, random_forest
aims_property = ['charge', 'dipole', 'homo_lumo', 'hirshfeld_volume_ratio']
dftb_property = ['charge', 'dipole', 'homo_lumo']

max_l = {1: 0, 6: 1, 7: 1, 8: 1}
shell_dict = {6: [0, 1, 2], 14: [0, 1, 2]}
vcr = {6: torch.tensor([2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0, 4.5]),
       14: torch.tensor([3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 9.0])}
tvcr = torch.tensor([2., 2.5, 3., 4., 5., 7., 10.])
n_interval = [5, 12]  # interval difference, interval number
shell_dict_std = {6: [0, 1], 14: [0, 1]}
shell_dict_list = [shell_dict_std, shell_dict]

params.update({"n_band0": torch.tensor([int(ii * n_interval[0]) for ii in range(n_interval[1])])})
params.update({"n_band1": torch.tensor([int(ii * n_interval[0] + 1) for ii in range(n_interval[1])])})
params.update({"n_valence": {6: 2, 14: 2}, "n_conduction": {6: 1, 14: 1},
               "train_e_low": -30, "train_e_high": 20})

# onsite_lr, lr, scale params if stop improve,
dataset_group = {
    'c_bulk_diamond2': 0.0,  # 1e-3, 3e-3, 0.1, 9.5
    'c_bulk_diamond8': 0.0,  # 1e-3, 3e-3, 0.1, 9.5
    'c_bulk_diamond64': 0.0,  # 1e-3, 3e-3, 0.1, 9.5
    'c_bulk_hex2': 0.0,  # 1e-3, 3e-3, 0.1, 9.5
    'c_bulk_hex4': 0.0,  # 5e-3, 1e-2, 0.02, 49
    'c_bulk_hex4_2layer': 0.0,
    'c_slab_diamond': 0.0,  # 1e-2, 3e-2, 0.01, 95
    'c_vac_diamond': 0.0,  # 1e-2, 3e-2, 0.01, 95
    'si_bulk_diamond2': 0.0,  # 1e-3, 1e-3, 0.1, 9.5
    'si_bulk_diamond8': 0.0,  # 1e-3, 1e-3, 0.1, 9.5
    'si_bulk_hex4': 0.0,  # 1e-3, 3e-3, 0.1, 9.5
    'si_bulk_tetrag4': 0.0,  # 2e-3, 3e-3, 0.02, 49
    # 'si_slab_diamond': 0.0,
    # 'si_vac_diamond': 0.0,  # 3e-3, 5e-3, 0.01, 95
    'sic_bulk_cubic2': 0.0,  # 2e-3, 3e-3, 0.02, 49
    'sic_bulk_diamond2': 0.0,  # 2e-3, 3e-3, 0.02, 49
    'sic_bulk_hex': 0.0,
    'sic_slab': 0.0,  # 4e-3 6e-3, 0.01, 95
    'sic_vac_diamond': 0.0  # 1e-2, 1e-2, 0.01 95
}

if params['ml']['task'] == 'scale':
    params['ml']['lr'] = 6e-3
    params['ml']['onsite_lr'] = 4e-3


def train_function(dataset: str,
                   subsets: list,
                   subsets_ratio: list,
                   batch_size: int,
                   train_onsite: bool,
                   orbital_resolved: bool,
                   scale_ham: bool,
                   cutoff: float = 10.0,
                   _params: dict = None):
    """Optimize spline parameters or compression radii."""
    # clean
    os.system('rm onsite.dat scale.dat')

    if _params is not None:
        def update_dict(dict1, dict2):
            for key1, value1 in dict2.items():
                if key1 in dict1 and isinstance(value1, dict) and isinstance(dict1[key1], dict):
                    update_dict(dict1[key1], value1)
                else:
                    dict1[key1] = value1

        update_dict(params, _params)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # prepare for training
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    for subset, ratio in zip(subsets, subsets_ratio):
        dataset_group[subset] = ratio

    train_dict, general_dict = Dataset.band(
        dataset, shell_dict, batch_size=batch_size, #params['ml']['batch_size'],
        cutoff=cutoff, groups=dataset_group, logger=logger, inverse=inverse)

    ml_variable = []
    for key, data in train_dict.items():
        geo = data['geometry']
        klines = data['klines']
        kpoints = pack([Dataset.get_kpoints(cell, kpoint_level) for cell in geo.cell])
        if params['ml']['task'] == 'scale':
            dftb2_scc = Dftb2(geo, shell_dict, path_sk, skf_type="skf",  kpoints=kpoints,
                              h_basis_type='normal', s_basis_type='normal', repulsive=False)
            train_dict[key]['dftb2_scc'] = dftb2_scc
            dftb1_band = Dftb1(geo, shell_dict, path_to_skf=path_sk, skf_type="skf", klines=klines,
                               h_basis_type='normal', s_basis_type='normal', repulsive=False,)
            dftb1_band.h_feed.gen_onsite(dftb1_band.geometry, dftb1_band.basis, orbital_resolved)

            train_dict[key]['dftb1_band'] = dftb1_band
            train_dict[key][(0, 0)] = torch.ones(dftb1_band.periodic.distances.shape).requires_grad_(True)
            train_dict[key][(0, 1)] = torch.ones(dftb1_band.periodic.distances.shape).requires_grad_(True)
            train_dict[key][(0, 2)] = torch.ones(dftb1_band.periodic.distances.shape).requires_grad_(True)
            train_dict[key][(1, 1)] = torch.ones(dftb1_band.periodic.distances.shape).requires_grad_(True)
            train_dict[key][(1, 2)] = torch.ones(dftb1_band.periodic.distances.shape).requires_grad_(True)
            train_dict[key][(2, 2)] = torch.ones(dftb1_band.periodic.distances.shape).requires_grad_(True)

            dftb1_band.h_feed.gen_onsite(dftb1_band.geometry, dftb1_band.basis, orbital_resolved)
            train_dict[key]['ml_onsite'] = dftb1_band.h_feed.on_site_dict["ml_onsite"]
            train_dict[key]['ml_onsite'].requires_grad_(True)
            ml_variable.append(
                {
                    "params": train_dict[key]['ml_onsite'],
                    "lr": params['ml']["onsite_lr"],
                }
            )
            ml_variable.extend([{'params': train_dict[key][(i, j)], 'lr': params['ml']["lr"]}
                                for i in range(0, 3) for j in range(0, 3) if i <= j])
        elif params['ml']['task'] == 'vcr':
            params['dftb']['path_to_skf'] = './band_slko/sic.h5'
            params['dftb']['path_to_skf2'] = './slko'
            compr0 = torch.ones(geo.atomic_numbers.shape)
            init_dict = {6: torch.tensor([2.5]),
                         14: torch.tensor([5.0])}
            for ii, iu in enumerate(geo.unique_atomic_numbers()):
                mask = geo.atomic_numbers == iu
                compr0[mask] = init_dict[iu.tolist()]
            compr0.requires_grad_(True)

            h_feed = VcrFeed.from_dir(
                params['dftb']['path_to_skf'], shell_dict, vcr,
                skf_type='h5', geometry=geo, integral_type='H',
                interpolation='BicubInterp')
            s_feed = VcrFeed.from_dir(
                params['dftb']['path_to_skf'], shell_dict, vcr,
                skf_type='h5', geometry=geo, integral_type='S',
                interpolation='BicubInterp')
            dftb2_scc = Dftb2(geo, shell_dict, path_sk, skf_type="skf",  kpoints=kpoints,
                              h_basis_type='vcr', s_basis_type='vcr', repulsive=False)

            train_dict[key]['dftb2_scc'] = dftb2_scc
            dftb1_band = Dftb1(geo, shell_dict, path_to_skf=path_sk, skf_type="skf", klines=klines,
                               h_basis_type='vcr', s_basis_type='vcr', repulsive=False,)
            dftb1_band.h_feed.gen_onsite(dftb1_band.geometry, dftb1_band.basis, orbital_resolved)

            train_dict[key]['dftb1_band'] = dftb1_band
            train_dict[key]['h_feed'] = h_feed
            train_dict[key]['s_feed'] = s_feed
            train_dict[key]['compr0'] = compr0
            ml_variable.append({'params': train_dict[key]['compr0'], 'lr': params['ml']["lr"]})

    # optimize integrals with spline parameters
    n_train_batch = sum(general_dict['n_batch_list'])
    n_train_batch_list = np.arange(n_train_batch)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # begin training
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    if params['ml']['task'] == 'mlIntegral':

        opt = OptHsPe(elements, params, shell_dict, skf_type='skf')
        for ii in range(params['ml']['epoch']):
            this_num = ii % n_train_batch

            this_ind = n_train_batch_list[this_num]
            this_train_dict = train_dict[this_ind]
            opt(this_train_dict)

        # save training instance
        with open('opt_int.pkl', "wb") as f:
            pickle.dump(opt, f)

    elif params['ml']['task'] == 'scale':
        opt = Scale(elements, params, shell_dict, train_dict, ml_variable, train_onsite, loss=loss,
                    orbital_resolved=orbital_resolved, scale_ham=scale_ham,
                    train_1der=train_1der, skf_type='skf', tolerance=tolerance)

    elif params['ml']['task'] == 'vcr':
        params['dftb']['path_to_skf'] = './band_slko/sic.h5'
        params['dftb']['path_to_skf2'] = './slko'
        opt = VcrPe(elements, params, shell_dict, train_dict, ml_variable, vcr, skf_type='skf')

    for ii in range(params['ml']['epoch']):
        this_num = ii % n_train_batch
        this_ind = n_train_batch_list[this_num]
        this_train_dict = opt.train_dict[this_ind]
        if params['ml']['task'] == 'scale':
            opt(this_train_dict, ii)
        elif params['ml']['task'] == 'vcr':
            opt(this_train_dict, ii)

    plt.plot(np.arange(len(opt._loss)), opt._loss)
    plt.show()

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # save training instance
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    with open('opt.pkl', "wb") as f:

        scale, onsite, num_opt, pos_pe_opt, dist_pe_opt = {}, [], [], [], []
        err0, err1 = [], []
        geometry_opt = None

        for key, data in opt.train_dict.items():
            dftb1_band = opt.train_dict[key]['dftb1_band']
            num_opt.append(dftb1_band.geometry.atomic_numbers)
            pos_pe_opt.append(dftb1_band.periodic.neighbour_pos)
            dist_pe_opt.append(dftb1_band.periodic.distances)
            err0.extend(torch.mean(opt.train_dict[key]['loss_mae0'][-1], -1))
            err1.extend(torch.mean(opt.train_dict[key]['loss_mae1'][-1], -1))

            if geometry_opt is None:
                geometry_opt = dftb1_band.geometry
                for i in range(0, 3):
                    for j in range(i, 3):
                        scale.update({(i, j): [data[(i, j)]]})
            else:
                geometry_opt = geometry_opt + dftb1_band.geometry
                for i in range(0, 3):
                    for j in range(i, 3):
                        scale[(i, j)].append(data[(i, j)])
            onsite.append(dftb1_band.h_feed.on_site_dict["ml_onsite"])

        # train_dict = opt.train_dict
        pickle.dump(Dataclass(num_opt, pos_pe_opt, dist_pe_opt, onsite, scale,
                              geometry_opt, train_onsite, orbital_resolved, scale_ham), f)

    return err0


def pred(pickle_file: list, dataset: str, subsets, subsets_ratio,
         batch_size, cutoff=10.0, **kwargs):
    """Test optimized results."""
    pred_log = 'pred.log'
    skf_list = kwargs.get('skf_list', ['./slko/pbc/', './slko/'])
    file_handler = logging.FileHandler(pred_log)

    # Set the log level for the file handler
    file_handler.setLevel(logging.DEBUG)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # prepare for testing
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    for subset, ratio in zip(subsets, subsets_ratio):
        dataset_group[subset] = ratio

    idx_dict = {inum: i for i, inum in enumerate(shell_dict.keys())}
    idx_dict.update({(inum, jnum): i + j
                     for i, inum in enumerate(shell_dict.keys())
                     for j, jnum in enumerate(shell_dict.keys())})
    params.update({"n_band0": torch.tensor([int(ii * n_interval[0]) for ii in range(n_interval[1])])})
    params.update({"n_band1": torch.tensor([int(ii * n_interval[0] + 1) for ii in range(n_interval[1])])})
    params.update({"n_valence": 'all', "n_conduction": {6: 1, 14: 1},
                   "train_e_low": -30, "train_e_high": 20})

    # load optimized object
    scale, onsite, num_opt, pos_pe_opt, dist_pe_opt = {}, [], [], [], []
    geometry_opt = None
    loaded_model = [pickle.load(open(file, 'rb')) for file in pickle_file]
    for model in loaded_model:
        num_opt.extend(model.num_opt)
        pos_pe_opt.extend(model.pos_pe_opt)
        dist_pe_opt.extend(model.dist_pe_opt)
        onsite.extend(model.onsite)

        if geometry_opt is None:
            geometry_opt = model.geometry_opt
        else:
            geometry_opt = geometry_opt + model.geometry_opt

        for i in range(0, 3):
            for j in range(i, 3):
                if (i, j) in model.scale.keys():
                    scale[(i, j)] = model.scale[(i, j)]
                else:
                    scale.update({(i, j): scale[(i, j)].extend(model.scale[(i, j)])})

    basis_opt = Basis(geometry_opt.atomic_numbers, shell_dict)
    num_opt = merge(num_opt)
    pos_pe_opt = merge(pos_pe_opt)
    dist_pe_opt = merge(dist_pe_opt)
    onsite_opt = torch.cat(onsite).detach()
    if not loaded_model[-1].orbital_resolved:
        shells_per_atom_opt = basis_opt.shells_per_atom[basis_opt.shells_per_atom.ne(0)]
        onsite_opt = pack(onsite_opt.split(tuple(shells_per_atom_opt)))
    elif loaded_model[-1].orbital_resolved:
        orbs_per_atom_opt = basis_opt.orbs_per_atom[basis_opt.orbs_per_atom.ne(0)]
        onsite_opt = pack(onsite_opt.flatten().split(tuple(orbs_per_atom_opt.flatten())))

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # build optimized features, one-body
    n_idx = len(shell_dict.keys())
    out_o11 = np.zeros((*num_opt.shape, n_idx))
    out_o12 = np.zeros((*num_opt.shape, n_idx))
    out_o14 = np.zeros((*num_opt.shape, len(idx_dict.keys()) - n_idx))
    out_o11, out_o12, out_o14 = g_pe(
        out_o11, out_o12, out_o14, geometry_opt.atomic_numbers.numpy(),
        geometry_opt.positions.numpy(), pos_pe_opt.numpy(),
        geometry_opt.n_atoms.numpy(),
        cutoff=10.0, eta=0.02, lamda=-1.0, zeta=1.0, idx_dict=idx_dict)

    # build optimized features, two-body
    out_o21 = np.zeros((*dist_pe_opt.shape, 1))
    out_o22 = np.zeros((*dist_pe_opt.shape, len(shell_dict.keys())))
    out_o23 = np.zeros((*dist_pe_opt.shape, len(shell_dict.keys())))
    out_o24 = np.zeros((*dist_pe_opt.shape, len(shell_dict.keys())))
    out_o21, out_o22, out_o23, out_o24 = g_pe_pair(
        out_o21, out_o22, out_o23, out_o24, geometry_opt.atomic_numbers.numpy(),
        geometry_opt.positions.numpy(), pos_pe_opt.numpy(),
        geometry_opt.n_atoms.numpy(),
        cutoff=10.0, eta=0.02, lamda=-1.0, zeta=1.0, idx_dict=idx_dict)

    x_train_1 = np.concatenate([out_o11, out_o12, out_o14], axis=-1)
    x_train = np.concatenate([out_o21, out_o22, out_o23, out_o24], axis=-1)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # update training params in optimized object, inverse is TRUE to avoid get
    # the same dataset in training
    test_dict, general_dict = Dataset.band(
        dataset, shell_dict, batch_size=batch_size,
        cutoff=cutoff, groups=dataset_group, logger=logger, inverse=inverse)

    for ii, (key, data) in enumerate(test_dict.items()):
        time_ii = time.time()
        geometry_pred = data['geometry']
        num_pred = geometry_pred.atomic_numbers
        kpoints = pack([Dataset.get_kpoints(cell, kpoint_level) for cell in geometry_pred.cell])
        dftb2_scc = Dftb2(geometry_pred, shell_dict, path_sk, skf_type="skf", kpoints=kpoints,
                          h_basis_type='normal', s_basis_type='normal', repulsive=False)
        data['dftb2_scc'] = dftb2_scc
        pe_pred = dftb2_scc.periodic
        dftb1_band = Dftb1(geometry_pred, shell_dict, path_to_skf=path_sk, skf_type="skf", klines=data['klines'],
                           h_basis_type='normal', s_basis_type='normal', repulsive=False, )
        dftb1_band.h_feed.gen_onsite(dftb1_band.geometry, dftb1_band.basis, loaded_model[-1].orbital_resolved)
        data['dftb1_band'] = dftb1_band
        time_i1 = time.time()
        logger.info(f'Running DFTB with geometry {geometry_pred} time: {time_i1 - time_ii}')

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # build pred features, one-body
        n_idx = len(shell_dict.keys())
        out_p11 = np.zeros((*num_pred.shape, n_idx))
        out_p12 = np.zeros((*num_pred.shape, n_idx))
        out_p14 = np.zeros((*num_pred.shape, len(idx_dict.keys()) - n_idx))
        out_p11, out_p12, out_p14 = g_pe(
            out_p11, out_p12, out_p14, geometry_pred.atomic_numbers.numpy(),
            geometry_pred.positions.numpy(), pe_pred.neighbour_pos.numpy(),
            geometry_pred.n_atoms.numpy(),
            cutoff=10.0, eta=0.02, lamda=-1.0, zeta=1.0, idx_dict=idx_dict)
        time_i2 = time.time()
        logger.info(f'Running one-body with geometry {geometry_pred} time: {time_i2 - time_i1}')

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # build pred features, two-body
        basis_pred = Basis(geometry_pred.atomic_numbers, shell_dict)
        out21 = np.zeros((*pe_pred.distances.shape, 1))
        out22 = np.zeros((*pe_pred.distances.shape, len(shell_dict.keys())))
        out23 = np.zeros((*pe_pred.distances.shape, len(shell_dict.keys())))
        out24 = np.zeros((*pe_pred.distances.shape, len(shell_dict.keys())))
        out21, out22, out23, out24 = g_pe_pair(
            out21, out22, out23, out24, geometry_pred.atomic_numbers.numpy(),
            geometry_pred.positions.numpy(), pe_pred.neighbour_pos.numpy(),
            geometry_pred.n_atoms.numpy(),
            cutoff=10.0, eta=0.02, lamda=-1.0, zeta=1.0, idx_dict=idx_dict)
        time_2b = time.time()
        logger.info(f'Running two-body with geometry {geometry_pred} time: {time_2b - time_i2}')

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # train and pred scale and onsite
        if params['ml']['task'] == 'scale':
            x_pred_1 = np.concatenate([out_p11, out_p12, out_p14], axis=-1)
            x_pred = np.concatenate([out21, out22, out23, out24], axis=-1)
            anp_train = basis_opt.atomic_number_matrix("atomic")
            anp_pred = basis_pred.atomic_number_matrix("atomic")

            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            # One-center targets to be predicted
            reg = RandomForestRegressor(n_estimators=100)
            onsite_pred = np.zeros((*num_pred.shape, onsite_opt.shape[-1]))
            for ia in shell_dict.keys():
                time_ia = time.time()
                mask_train_1 = num_opt == ia
                mask_pred_1 = num_pred == ia
                for i in range(onsite_opt.shape[-1]):
                    y_train = onsite_opt[num_opt[num_opt.ne(0)] == ia][..., i]

                    # predict onsite, make sure there is such atom and orbital in testing data
                    if x_pred_1[mask_pred_1].shape[0] > 0 and x_pred_1[mask_pred_1].shape[0] > 0:
                        reg.fit(x_train_1[mask_train_1], y_train)

                        pred = reg.predict(x_pred_1[mask_pred_1])
                        onsite_pred[mask_pred_1, i] = pred

                logger.info(f'Running time generate {ia} one-body pred time: {time.time() - time_ia}')

            data['ml_onsite'] = torch.from_numpy(onsite_pred)
            # if loaded_model[-1].orbital_resolved:
            #     data['ml_onsite'] = data['ml_onsite'].flatten()
            time_1b_pred = time.time()
            logger.info(f'Running one-body pred time: {time_1b_pred - time_2b}')

            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            # Two-body targets predictions
            for i in range(3):
                for j in range(i, 3):

                    y_pred_scale = np.zeros((x_pred.shape[:-1]))

                    for ia in shell_dict.keys():
                        for ja in shell_dict.keys():
                            time_ija = time.time()
                            mask_train = (anp_train[..., 0] == ia) * (anp_train[..., 1] == ja)
                            mask_pred = (anp_pred[..., 0] == ia) * (anp_pred[..., 1] == ja)

                            y_train = merge(scale[(i, j)]).detach()
                            mask_opt = mask_train.unsqueeze(-1) * dist_pe_opt.lt(cutoff)
                            mask_opt = mask_opt * y_train.ne(0.0) * y_train.ne(1.0)

                            mask_p = mask_pred.unsqueeze(-1) * pe_pred.distances.lt(cutoff)

                            if x_train[mask_opt].shape[0] == 0:
                                print(f'training data do not include {ia}-{ja}-{i}-{j}')
                            elif x_pred[mask_p].shape[0] == 0:
                                print(f'testing data do not include {ia}-{ja}-{i}-{j}')
                            else:
                                reg.fit(x_train[mask_opt], y_train[mask_opt])
                                y_pred = torch.from_numpy(reg.predict(x_pred[mask_p]))
                                y_pred_scale[mask_p] = y_pred

                            logger.info(f'Running time generate {ia}-{ja} two-body pred time: {time.time() - time_ija}')

                    data[(i, j)] = torch.from_numpy(y_pred_scale)
            time_2b_pred = time.time()
            logger.info(f'Running two-body pred time: {time_2b_pred - time_1b_pred}')

        if params['ml']['task'] == 'scale':
            loss_mae0, loss_mae1, std_err = Scale.pred(
                data, path_sk, params, shell_dict, elements, skf_type, alignment,
                train_onsite=loaded_model[-1].train_onsite,
                orbital_resolved=loaded_model[-1].orbital_resolved,
                scale_ham=loaded_model[-1].scale_ham,
                shell_dict_std=shell_dict_list,
                plot_std=plot_std, skf_list=skf_list, **kwargs)
            time_fin_pred = time.time()
            logger.info(f'Final pred time: {time_fin_pred - time_2b_pred}')

    return loss_mae0, loss_mae1, std_err


def re_calculate(pickle_file: list, dataset: str, subsets, subsets_ratio,
                 batch_size, cutoff=10.0, **kwargs):
    """Test optimized results."""
    pred_log = 'pred.log'
    skf_list = kwargs.get('skf_list', None)
    file_handler = logging.FileHandler(pred_log)

    # Set the log level for the file handler
    file_handler.setLevel(logging.DEBUG)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # prepare for testing
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    for subset, ratio in zip(subsets, subsets_ratio):
        dataset_group[subset] = ratio

    idx_dict = {inum: i for i, inum in enumerate(shell_dict.keys())}
    idx_dict.update({(inum, jnum): i + j
                     for i, inum in enumerate(shell_dict.keys())
                     for j, jnum in enumerate(shell_dict.keys())})
    params.update({"n_band0": torch.tensor([int(ii * n_interval[0]) for ii in range(n_interval[1])])})
    params.update({"n_band1": torch.tensor([int(ii * n_interval[0] + 1) for ii in range(n_interval[1])])})
    params.update({"n_valence": 'all', "n_conduction": {6: 1, 14: 1},
                   "train_e_low": -30, "train_e_high": 20})

    # load optimized object
    scale, onsite, num_opt, pos_pe_opt, dist_pe_opt = {}, [], [], [], []
    geometry_opt = None
    loaded_model = [pickle.load(open(file, 'rb')) for file in pickle_file]
    for model in loaded_model:
        num_opt.extend(model.num_opt)
        pos_pe_opt.extend(model.pos_pe_opt)
        dist_pe_opt.extend(model.dist_pe_opt)
        onsite.extend(model.onsite)

        if geometry_opt is None:
            geometry_opt = model.geometry_opt
        else:
            geometry_opt = geometry_opt + model.geometry_opt

        for i in range(0, 3):
            for j in range(i, 3):
                if (i, j) in model.scale.keys():
                    scale[(i, j)] = model.scale[(i, j)]
                else:
                    scale.update({(i, j): scale[(i, j)].extend(model.scale[(i, j)])})

    basis_opt = Basis(geometry_opt.atomic_numbers, shell_dict)
    num_opt = merge(num_opt)
    pos_pe_opt = merge(pos_pe_opt)
    dist_pe_opt = merge(dist_pe_opt)
    onsite_opt = torch.cat(onsite).detach()
    if not loaded_model[-1].orbital_resolved:
        shells_per_atom_opt = basis_opt.shells_per_atom[basis_opt.shells_per_atom.ne(0)]
        onsite_opt = pack(onsite_opt.split(tuple(shells_per_atom_opt)))
    elif loaded_model[-1].orbital_resolved:
        orbs_per_atom_opt = basis_opt.orbs_per_atom[basis_opt.orbs_per_atom.ne(0)]
        onsite_opt = pack(onsite_opt.flatten().split(tuple(orbs_per_atom_opt.flatten())))

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # update training params in optimized object, inverse is TRUE to avoid get
    # the same dataset in training
    test_dict, general_dict = Dataset.band(
        dataset, shell_dict, batch_size=batch_size,
        cutoff=cutoff, groups=dataset_group, logger=logger, inverse=inverse)

    for ii, (key, data) in enumerate(test_dict.items()):

        if ii == 0:
            geometry_pred = data['geometry']
            klines = data['klines']

            vband_tot = data['vband_tot']
            cband_tot = data['cband_tot']
            vbm_alignment = data['vbm_alignment']
            n_vband = data['n_vband']
        else:
            geometry_pred = geometry_pred + data['geometry']
            klines = torch.cat([klines, data['klines']])
            vband_tot = torch.cat([vband_tot, data['vband_tot']])
            cband_tot = torch.cat([cband_tot, data['cband_tot']])
            vbm_alignment = np.concatenate([vbm_alignment, data['vbm_alignment']])
            n_vband = np.concatenate([n_vband, data['n_vband']])

        data['klines'] = klines
        data['vband_tot'] = vband_tot
        data['cband_tot'] = cband_tot
        data['vbm_alignment'] = vbm_alignment
        data['n_vband'] = n_vband

    num_pred = geometry_pred.atomic_numbers
    kpoints = pack([Dataset.get_kpoints(cell, kpoint_level) for cell in geometry_pred.cell])
    dftb2_scc = Dftb2(geometry_pred, shell_dict, path_sk, skf_type="skf", kpoints=kpoints,
                      h_basis_type='normal', s_basis_type='normal', repulsive=False)
    data['dftb2_scc'] = dftb2_scc
    dftb1_band = Dftb1(geometry_pred, shell_dict, path_to_skf=path_sk, skf_type="skf", klines=klines,
                       h_basis_type='normal', s_basis_type='normal', repulsive=False, )
    dftb1_band.h_feed.gen_onsite(dftb1_band.geometry, dftb1_band.basis, loaded_model[-1].orbital_resolved)
    data['dftb1_band'] = dftb1_band

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # train and pred scale and onsite
    if params['ml']['task'] == 'scale':
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # One-center targets to be predicted
        data['ml_onsite'] = onsite_opt.reshape((*num_pred.shape, onsite_opt.shape[-1]))

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # Two-body targets predictions
        for i in range(3):
            for j in range(i, 3):
                data[(i, j)] = merge(scale[(i, j)]).detach()

    if params['ml']['task'] == 'scale':
        loss_mae0, loss_mae1, std_err = Scale.pred(
            data, path_sk, params, shell_dict, elements, skf_type, alignment,
            train_onsite=loaded_model[-1].train_onsite,
            orbital_resolved=loaded_model[-1].orbital_resolved,
            scale_ham=loaded_model[-1].scale_ham,
            skf_list=skf_list,
            shell_dict_std=shell_dict_list,
            plot_std=plot_std)

    return loss_mae0, loss_mae1, std_err


def _load_ref(dataset, size, properties, units='angstrom', cell: Tensor = None):
    """Helper function to load dataset, return `Geometry` object, data."""
    numbers, positions, data = LoadHdf.load_reference(
        dataset, size, properties)
    geo = Geometry(numbers, positions, units=units, cell=cell)

    return geo, data


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
