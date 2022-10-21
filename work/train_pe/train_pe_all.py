#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""List all training systems and the preparations."""
import os
import pickle

import numpy as np
import torch
import matplotlib.pyplot as plt

from train_pe import train_function, pred, re_calculate


task = 'c_vac_diamond'
skf_list = ['./slko/pbc/', './slko/']

def generate_dataset():
    pass

# the dataset from function generate_dataset
dataset = 'band_c_si.h5'


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Carbon systems

def c_bulk_diamond2():
    """Learning rate of onsite is onsite: 1e-3, int is 2e-3."""
    batch_size = 3
    param_list = []
    train_onsite = 'local'
    plot = True
    select_para = True  # hyperparameter generation
    train_slect_para = False  # get training errors of all hyperparameters
    test_slect_para = True  # get testing errors of all hyperparameters
    orbital_resolved = False
    scale_ham = True

    if select_para:
        param_list = []

        for i in [1e-3]:#, 2e-3, 3e-3, 5e-3, 1e-2]:
            for j in [2]:# 0.5, 1, 2, 3, 5, 10]:
                param_list.append((i * j, i))

    train_error = []
    if train_slect_para:
        for ii, param in enumerate(param_list):
            lr, onsite_lr = param
            param = {'ml': {'lr': lr, 'onsite_lr': onsite_lr}}
            loss = train_function(dataset, ['c_bulk_diamond2'], [0.4], batch_size, train_onsite,
                                  orbital_resolved, scale_ham, skf_list=skf_list, _params=param)
            train_error.append(np.average(loss[-5:]))
            os.system(f'mv opt.pkl opt-c_bulk_diamond2-{lr}-{onsite_lr}.pkl')
    else:
        for ii, param in enumerate(param_list):
            lr, onsite_lr = param
            err0, err1, _ = re_calculate([f'../data/opt-c_bulk_diamond2-{lr}-{onsite_lr}.pkl'],
                                         dataset, ['c_bulk_diamond2'], [0.4], batch_size, skf_list=skf_list)
            train_error.append(float(err0.numpy().mean()))

    test_error = []
    if test_slect_para:
        for ii, param in enumerate(param_list):
            lr, onsite_lr = param
            err0, err1, _ = pred([f'opt-c_bulk_diamond2-{lr}-{onsite_lr}.pkl'], dataset, ['c_bulk_diamond2'], [0.2], batch_size)
            test_error.append(float(err0.numpy().mean()))

    if plot:
        print('train_error, test_error', train_error, test_error)
        # 1e-3, 2e-3
        train_error = [0.30781288425326536, 0.25933154744946385, 0.20136252478644162, 0.22070303114131046, 0.23628214429718172,
                       0.3131140787241161, 0.24329493106456918, 0.23592683818228558, 0.2293298375769957, 0.28644393869521356,
                       0.3108886500807283, 0.25034543970233447, 0.2591552871895606, 0.30706074401928457, 0.3419042955302606,
                       0.3525912257512611, 0.30747343477650707, 0.2753264572638431, 0.31379675123851936, 0.5450014381883683,
                       0.384528409080867, 0.3088679225719824, 0.5089457079910561, 0.9882221326714561, 1.0393691533879057]
        test_error = [0.3120693448873019, 0.2621399410847088, 0.20526243745080078, 0.21793777501562317, 0.23144284835505285,
                      0.33300275645022953, 0.27705989836286077, 0.20853523156062542, 0.22163665160286744, 0.320720795102066,
                      0.3631611482960824, 0.2675228527981126, 0.19985273349578012, 0.26027539997099747, 0.3185054740111417,
                      0.3823029406564499, 0.2606881795774601, 0.23527867456890872, 0.3526077229227908, 0.6565668051523, 0.47376232492079945,
                      0.2419661185669313, 0.43795035705177054, 0.6285121359171832, 0.8901745773939764]
        x = np.array([1e-3, 2e-3, 3e-3, 5e-3, 1e-2])
        y = np.array([0.5, 1, 2, 3, 5])
        #y = y0 * x
        z1 = np.array(train_error).reshape(5, 5)
        z2 = np.array(test_error).reshape(5, 5)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6),)
        im = ax1.pcolormesh(x, y, z1, vmin=0.2, vmax=1.2)
        ax1.set_xlabel('learning rate of on-site')
        ax1.set_ylabel('learning rate of two-center integrals / learning rate of on-site')
        ax1.set_title('training error')
        ax2.set_title('predicting error')
        fig.colorbar(im, ax=ax1)
        im = ax2.pcolormesh(x, y, z2, vmin=0.2, vmax=1.2)
        ax2.set_xlabel('learning rate of on-site')
        fig.colorbar(im, ax=ax2,)
        plt.savefig('c_bulk_diamond2_lr.png', dpi=100)
        plt.show()
        plt.plot(np.arange(len(train_error)), train_error)
        plt.plot(np.arange(len(test_error)), test_error)
        plt.show()


def c_bulk_diamond8():
    """Learning rate of onsite is onsite: 1e-3, int is 5e-3."""
    batch_size = 3
    param_list = []
    train_onsite = 'local'
    select_para = True  # hyperparameter generation
    train_slect_para = False  # get training errors of all hyperparameters
    test_slect_para = False  # get testing errors of all hyperparameters
    orbital_resolved = False
    scale_ham = True
    plot = True

    if select_para:
        param_list = []

        for i in [1e-3, 2e-3, 3e-3, 5e-3]:
            for j in [0.5, 1, 2, 3, 5]:
                param_list.append((i * j, i))

    train_error = []
    if train_slect_para:
        for ii, param in enumerate(param_list):
            lr, onsite_lr = param
            param = {'ml': {'lr': lr, 'onsite_lr': onsite_lr}}
            loss = train_function(dataset, ['c_bulk_diamond8'], [0.4], batch_size, train_onsite,
                                  orbital_resolved, scale_ham, skf_list=skf_list, _params=param)
            train_error.append(np.average(loss[-5:]))
            os.system(f'mv opt.pkl opt-c_bulk_diamond8-{lr}-{onsite_lr}.pkl')

    test_error = []
    if test_slect_para:
        for ii, param in enumerate(param_list):
            lr, onsite_lr = param
            err0, err1, _ = pred([f'opt-c_bulk_diamond8-{lr}-{onsite_lr}.pkl'], dataset, ['c_bulk_diamond8'], [0.2], batch_size)
            test_error.append(float(err0.numpy().mean()))

    print(train_error, test_error)
    # 1e-3, 5e-3
    train_error = [0.32867498259430716, 0.2904039300820309, 0.24101710629067888, 0.22430095050681284, 0.1595133966015297,
                   0.3182330844763753, 0.2840927355938295, 0.24041156232189587, 0.2548116614548571, 0.14560628274856954,
                   0.3205122331694642, 0.2876435129062179, 0.2513348355161017, 0.28155279697235036, 0.269798319266429,
                   0.33519267644612344, 0.3302553094520252, 0.30058713107021445, 0.35912548744860456, 0.42819984566416036]
    test_error = [0.47992720607632866, 0.40883784600420736, 0.2683249245820064, 0.27560314885328135,
                  0.16800429494493935, 0.45641682523914573, 0.4042583380948479, 0.2761049296249113,
                  0.2676618801052718, 0.23061779047096082, 0.44908022600148073, 0.3868750365692531,
                  0.3020272692147931, 0.280359801913764, 0.17876434096899366, 0.4373512080099199,
                  0.39252676480393683, 0.38399374637462647, 0.33513270697055464, 0.38058678853539113]

    if plot:
        x = np.array([1e-3, 2e-3, 3e-3, 5e-3])
        y = np.array([0.5, 1, 2, 3, 5])
        z1 = np.array(train_error).reshape(len(x), len(y)).T
        z2 = np.array(test_error).reshape(len(x), len(y)).T
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6),)
        im = ax1.pcolormesh(x, y, z1, vmin=0.1, vmax=0.5)
        ax1.set_xlabel('learning rate of on-site')
        ax1.set_ylabel('learning rate of two-center integrals / learning rate of on-site')
        ax1.set_title('training error')
        ax2.set_title('predicting error')
        fig.colorbar(im, ax=ax1)
        im = ax2.pcolormesh(x, y, z2, vmin=0.1, vmax=0.5)
        ax2.set_xlabel('learning rate of on-site')
        fig.colorbar(im, ax=ax2,)
        plt.savefig('c_bulk_diamond8_lr.png', dpi=100)
        plt.show()
    print(train_error.index(min(train_error)))
    print(test_error.index(min(test_error)))
    plt.plot(np.arange(len(train_error)), train_error)
    plt.plot(np.arange(len(test_error)), test_error)
    plt.show()


def c_bulk_diamond64():
    """"""
    batch_size = 1
    param_list = []
    train_onsite = 'local'
    select_para = True  # hyperparameter generation
    train_slect_para = False  # get training errors of all hyperparameters
    test_slect_para = True  # get testing errors of all hyperparameters
    orbital_resolved = False
    scale_ham = True
    plot = True

    if select_para:
        param_list = []

        for i in [1e-3, 2e-3, 3e-3, 5e-3]:
            for j in [0.5, 1, 2, 3, 5]:
                if os.path.isfile(f'opt-c_bulk_diamond64-{i * j}-{i}.pkl'):
                    print(f'file opt-c_bulk_diamond64-{i * j}-{i}.pkl, skip')
                else:
                    print(f'calculate {i * j}-{i}')
                    param_list.append((i * j, i))

    train_error = []
    if train_slect_para:
        for ii, param in enumerate(param_list):
            lr, onsite_lr = param
            param = {'ml': {'lr': lr, 'onsite_lr': onsite_lr}}
            loss = train_function(dataset, ['c_bulk_diamond64'], [0.4], batch_size, train_onsite,
                                  orbital_resolved, scale_ham, skf_list=skf_list, _params=param)
            train_error.append(np.average(loss[-5:]))
            os.system(f'mv opt.pkl opt-c_bulk_diamond64-{lr}-{onsite_lr}.pkl')
    else:
        for ii, param in enumerate(param_list):
            lr, onsite_lr = param
            err0, err1, _ = re_calculate([f'../data/opt-c_bulk_diamond64-{lr}-{onsite_lr}.pkl'], dataset, ['c_bulk_diamond64'], [0.4], batch_size)
            train_error.append(float(err0.numpy().mean()))

    test_error = []
    if test_slect_para:
        for ii, param in enumerate(param_list):
            lr, onsite_lr = param
            err0, err1, _ = pred([f'opt-c_bulk_diamond64-{lr}-{onsite_lr}.pkl'], dataset, ['c_bulk_diamond64'], [0.2], batch_size)
            test_error.append(float(err0.numpy().mean()))
    print(train_error, test_error)

    if plot:
        x = np.array([1e-3, 2e-3, 3e-3, 5e-3])
        y = np.array([0.5, 1, 2, 3, 5])
        z1 = np.array(train_error).reshape(len(x), len(y)).T
        z2 = np.array(test_error).reshape(len(x), len(y)).T
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6),)
        im = ax1.pcolormesh(x, y, z1, vmin=0.1, vmax=0.5)
        ax1.set_xlabel('learning rate of on-site')
        ax1.set_ylabel('learning rate of two-center integrals / learning rate of on-site')
        ax1.set_title('training error')
        ax2.set_title('predicting error')
        fig.colorbar(im, ax=ax1)
        im = ax2.pcolormesh(x, y, z2, vmin=0.1, vmax=0.5)
        ax2.set_xlabel('learning rate of on-site')
        fig.colorbar(im, ax=ax2,)
        plt.savefig('c_bulk_diamond8_lr.png', dpi=100)
        plt.show()
    print(train_error.index(min(train_error)))
    print(test_error.index(min(test_error)))
    plt.plot(np.arange(len(train_error)), train_error)
    plt.plot(np.arange(len(test_error)), test_error)
    plt.show()


def c_bulk_hex2():
    batch_size = 3
    param_list = []
    train_onsite = 'local'
    select_para = True  # hyperparameter selection
    train_slect_para = True  # get training errors of all hyperparameters
    test_slect_para = True  # get testing errors of all hyperparameters
    orbital_resolved = False
    scale_ham = True

    if select_para:
        param_list = []

        for i in [1e-3, 2e-3, 3e-3, 5e-3]:
            for j in [0.5, 1, 2, 3, 5]:
                param_list.append((i * j, i))

    train_error = []
    if train_slect_para:
        for ii, param in enumerate(param_list):
            lr, onsite_lr = param
            param = {'ml': {'lr': lr, 'onsite_lr': onsite_lr}}
            loss = train_function(dataset, ['c_bulk_hex2'], [0.4], batch_size, train_onsite,
                                  orbital_resolved, scale_ham, skf_list=skf_list, _params=param)
            train_error.append(np.average(loss[-5:]))
            os.system(f'mv opt.pkl opt-c_bulk_hex2-{lr}-{onsite_lr}.pkl')

    test_error = []
    if test_slect_para:
        for ii, param in enumerate(param_list):
            lr, onsite_lr = param
            err0, err1, _ = pred([f'opt-c_bulk_hex2-{lr}-{onsite_lr}.pkl'], dataset, ['c_bulk_hex2'], [0.2], batch_size)
            test_error.append(float(err0.numpy().mean()))

    print(train_error, test_error)
    # 1e-3, 3e-3
    train_error = [1.017899631825004, 0.6128866114927319, 0.38226797296709186, 0.31778973613365846, 0.3237603715057058,
     0.9284368325060275, 0.5842599504913486, 0.33113127598255376, 0.3534697631905536, 0.3390805917552751,
     0.9956687343319081, 0.5640680401743559, 0.36121449464056843, 0.29001720201631676, 0.3424039790536035,
     0.659105565709116, 0.5241314946210707, 0.43271803870140413, 0.43579849508362206, 0.39817038127502863]
    test_error = [1.0153516097949102, 0.5381482493488775, 0.42017538049265823, 0.3764661904077468, 0.3789242592099074,
                  0.9223826015577038, 0.5811327995268081, 0.4142571534668145, 0.3513333548864208, 0.376677917706575,
                  0.9962666819411541, 0.5404895152185704, 0.4355156497807637, 0.40760119642453635, 0.4071248979645123,
                  0.6469538276105601, 0.5093883359465119, 0.37654150813868154, 0.449111393820868, 0.40261033064073914]

    print(train_error.index(min(train_error)))
    print(test_error.index(min(test_error)))
    plt.plot(np.arange(len(train_error)), train_error)
    plt.plot(np.arange(len(test_error)), test_error)
    plt.show()


def c_bulk_hex4_2layer():
    batch_size = 3
    param_list = []
    train_onsite = 'local'
    select_para = True  # hyperparameter selection
    train_slect_para = True  # get training errors of all hyperparameters
    test_slect_para = True  # get testing errors of all hyperparameters
    orbital_resolved = False
    scale_ham = True

    if select_para:
        param_list = []

        for i in [1e-3, 2e-3, 3e-3, 5e-3]:
            for j in [0.5, 1, 2, 3, 5]:
                param_list.append((i * j, i))

    train_error = []
    if train_slect_para:
        for ii, param in enumerate(param_list):
            lr, onsite_lr = param
            param = {'ml': {'lr': lr, 'onsite_lr': onsite_lr}}
            loss = train_function(dataset, ['c_bulk_hex4_2layer'], [0.4], batch_size, train_onsite,
                                  orbital_resolved, scale_ham, skf_list=skf_list, _params=param)
            train_error.append(np.average(loss[-5:]))
            os.system(f'mv opt.pkl opt-c_bulk_hex4_2layer-{lr}-{onsite_lr}.pkl')

    test_error = []
    if test_slect_para:
        for ii, param in enumerate(param_list):
            lr, onsite_lr = param
            err0, err1, _ = pred([f'opt-c_bulk_hex4_2layer-{lr}-{onsite_lr}.pkl'], dataset, ['c_bulk_hex4_2layer'], [0.2], batch_size)
            test_error.append(float(err0.numpy().mean()))

    print(train_error, test_error)
    # 1e-3, 3e-3

    print(train_error.index(min(train_error)))
    print(test_error.index(min(test_error)))
    plt.plot(np.arange(len(train_error)), train_error)
    plt.plot(np.arange(len(test_error)), test_error)
    plt.show()

def train_c_slab():
    batch_size = 3
    param_list = []
    train_onsite = 'local'
    select_para = True  # hyperparameter generation
    train_slect_para = True  # get training errors of all hyperparameters
    test_slect_para = True  # get testing errors of all hyperparameters
    orbital_resolved = False
    scale_ham = True

    if select_para:
        param_list = []

        for i in [5e-4, 1e-3, 2e-3, 5e-3]:
            for j in [0.5, 1, 2, 3, 5]:
                if os.path.isfile(f'opt-c_slab_diamond-{i * j}-{i}.pkl'):
                    print(f'file opt-c_slab_diamond-{i * j}-{i}.pkl, skip')
                else:
                    print(f'calculate {i * j}-{i}')
                    param_list.append((i * j, i))

    train_error = []
    if train_slect_para:
        for ii, param in enumerate(param_list):
            lr, onsite_lr = param
            param = {'ml': {'lr': lr, 'onsite_lr': onsite_lr}}
            loss = train_function(dataset, ['c_slab_diamond'], [0.4], batch_size, train_onsite,
                                  orbital_resolved, scale_ham, skf_list=skf_list, _params=param)
            train_error.append(np.average(loss[-5:]))
            os.system(f'mv opt.pkl opt-c_slab_diamond-{lr}-{onsite_lr}.pkl')

    test_error = []
    if test_slect_para:
        for ii, param in enumerate(param_list):
            lr, onsite_lr = param
            err0, err1, _ = pred([f'opt-c_slab_diamond-{lr}-{onsite_lr}.pkl'], dataset, ['c_slab_diamond'], [0.2], batch_size)
            test_error.append(float(err0.numpy().mean()))

    print(train_error, test_error)
    # 1e-3, 5e-3
    # train_error = [0.32867498259430716, 0.2904039300820309, 0.24101710629067888, 0.22430095050681284, 0.1595133966015297,
    #  0.3182330844763753, 0.2840927355938295, 0.24041156232189587, 0.2548116614548571, 0.14560628274856954,
    #  0.3205122331694642, 0.2876435129062179, 0.2513348355161017, 0.28155279697235036, 0.269798319266429,
    #  0.33519267644612344, 0.3302553094520252, 0.30058713107021445, 0.35912548744860456, 0.42819984566416036]
    # test_error = [0.47992720607632866, 0.40883784600420736, 0.2683249245820064, 0.27560314885328135, 0.16800429494493935, 0.45641682523914573, 0.4042583380948479, 0.2761049296249113, 0.2676618801052718, 0.23061779047096082, 0.44908022600148073, 0.3868750365692531, 0.3020272692147931, 0.280359801913764, 0.17876434096899366, 0.4373512080099199, 0.39252676480393683, 0.38399374637462647, 0.33513270697055464, 0.38058678853539113]

    print(train_error.index(min(train_error)))
    print(test_error.index(min(test_error)))
    plt.plot(np.arange(len(train_error)), train_error)
    plt.plot(np.arange(len(test_error)), test_error)
    plt.show()


def c_vac_diamond():
    """Train carbon defect systems."""
    batch_size = 1
    param_list = []
    train_onsite = 'local'
    select_para = True  # hyperparameter generation
    train_slect_para = False  # get training errors of all hyperparameters
    test_slect_para = True  # get testing errors of all hyperparameters
    orbital_resolved = False
    scale_ham = True

    if select_para:
        param_list = []

        for i in [1e-3,]:# 2e-3, 3e-3, 5e-3]:
            for j in [0.5, 1, 2, 3, 5]:
                if os.path.isfile(f'opt-c_vac_diamond-{i * j}-{i}.pkl'):
                    print(f'file opt-c_vac_diamond-{i * j}-{i}.pkl, skip')
                else:
                    print(f'calculate {i * j}-{i}')
                    param_list.append((i * j, i))

    train_error = []
    if train_slect_para:
        for ii, param in enumerate(param_list):
            lr, onsite_lr = param
            param = {'ml': {'lr': lr, 'onsite_lr': onsite_lr}}
            loss = train_function(dataset, ['c_vac_diamond'], [0.4], batch_size, train_onsite,
                                  orbital_resolved, scale_ham, skf_list=skf_list, _params=param)
            train_error.append(np.average(loss[-5:]))
            os.system(f'mv opt.pkl opt-c_vac_diamond-{lr}-{onsite_lr}.pkl')
    else:
        for ii, param in enumerate(param_list):
            lr, onsite_lr = param
            err0, err1, _ = re_calculate([f'../data/opt-c_vac_diamond-{lr}-{onsite_lr}.pkl'],
                                         dataset, ['c_vac_diamond'], [0.4], batch_size, skf_list=skf_list)
            train_error.append(float(err0.numpy().mean()))

    test_error = []
    if test_slect_para:
        for ii, param in enumerate(param_list):
            lr, onsite_lr = param
            err0, err1, _ = pred([f'../opt-c_vac_diamond-{lr}-{onsite_lr}.pkl'], dataset, ['c_vac_diamond'], [0.2],
                                 batch_size)
            test_error.append(float(err0.numpy().mean()))

    print(train_error, test_error)
    print(train_error.index(min(train_error)))
    print(test_error.index(min(test_error)))
    plt.plot(np.arange(len(train_error)), train_error)
    plt.plot(np.arange(len(test_error)), test_error)
    plt.show()


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Silicon systems
def si_bulk_diamond2():
    batch_size = 3
    train_onsite = 'local'
    orbital_resolved = False
    scale_ham = False

    plot = True
    select_para = True  # hyperparameter generation
    train_slect_para = True  # get training errors of all hyperparameters
    test_slect_para = True  # get testing errors of all hyperparameters
    orbital_resolved = False
    scale_ham = True

    if select_para:
        param_list = []

        for i in [1e-3, 2e-3, 3e-3, 5e-3, 1e-2]:
            for j in [0.5, 1, 2, 3, 5]:
                param_list.append((i * j, i))

    train_error = []
    if train_slect_para:
        for ii, param in enumerate(param_list):
            lr, onsite_lr = param
            param = {'ml': {'lr': lr, 'onsite_lr': onsite_lr}}
            loss = train_function(dataset, ['si_bulk_diamond2'], [0.4], batch_size, train_onsite,
                                  orbital_resolved, scale_ham, skf_list=skf_list, _params=param)
            train_error.append(np.average(loss[-5:]))
            os.system(f'mv opt.pkl opt-si_bulk_diamond2-{lr}-{onsite_lr}.pkl')
    # else:
    #     for ii, param in enumerate(param_list):
    #         lr, onsite_lr = param
    #         err0, err1, _ = re_calculate([f'../data/opt-si_bulk_diamond2-{lr}-{onsite_lr}.pkl'], dataset, ['si_bulk_diamond2'], [0.4], batch_size)
    #         train_error.append(float(err0.numpy().mean()))

    test_error = []
    if test_slect_para:
        for ii, param in enumerate(param_list):
            lr, onsite_lr = param
            err0, err1, _ = pred([f'opt-si_bulk_diamond2-{lr}-{onsite_lr}.pkl'], dataset, ['si_bulk_diamond2'], [0.2], batch_size)
            test_error.append(float(err0.numpy().mean()))

    if plot:
        print('train_error', train_error, '\n test_error', test_error)
        # 1e-3, 2e-3
        # train_error = []
        # test_error = []
        x = np.array([1e-3, 2e-3, 3e-3, 5e-3, 1e-2])
        y = np.array([0.5, 1, 2, 3, 5])
        #y = y0 * x
        z1 = np.array(train_error).reshape(5, 5)
        z2 = np.array(test_error).reshape(5, 5)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6),)
        im = ax1.pcolormesh(x, y, z1, vmin=0.2, vmax=1.2)
        ax1.set_xlabel('learning rate of on-site')
        ax1.set_ylabel('learning rate of two-center integrals / learning rate of on-site')
        ax1.set_title('training error')
        ax2.set_title('predicting error')
        fig.colorbar(im, ax=ax1)
        im = ax2.pcolormesh(x, y, z2, vmin=0.2, vmax=1.2)
        ax2.set_xlabel('learning rate of on-site')
        fig.colorbar(im, ax=ax2,)
        plt.savefig('c_bulk_diamond2_lr.png', dpi=100)
        plt.show()
        plt.plot(np.arange(len(train_error)), train_error)
        plt.plot(np.arange(len(test_error)), test_error)
        plt.show()

def si_bulk_hex4():
    batch_size = 3
    param_list = []
    train_onsite = 'local'
    select_para = True  # hyperparameter selection
    train_slect_para = True  # get training errors of all hyperparameters
    test_slect_para = True  # get testing errors of all hyperparameters
    orbital_resolved = False
    scale_ham = True

    if select_para:
        for i in [1e-3, 2e-3, 3e-3, 5e-3]:
            for j in [0.5, 1, 2, 3, 5]:
                param_list.append((i * j, i))

    train_error, test_error = [], []
    if train_slect_para:
        for ii, param in enumerate(param_list):
            lr, onsite_lr = param
            param = {'ml': {'lr': lr, 'onsite_lr': onsite_lr}}
            loss = train_function(dataset, ['si_bulk_hex4'], [0.4], batch_size, train_onsite,
                                  orbital_resolved, scale_ham, skf_list=skf_list, _params=param)
            train_error.append(np.average(loss[-5:]))
            os.system(f'mv opt.pkl opt-si_bulk_hex4-{lr}-{onsite_lr}.pkl')

    test_error = []
    if test_slect_para:
        for ii, param in enumerate(param_list):
            lr, onsite_lr = param
            err0, err1, _ = pred([f'opt-si_bulk_hex4-{lr}-{onsite_lr}.pkl'], dataset, ['si_bulk_hex4'], [0.2], batch_size)
            test_error.append(float(err0.numpy().mean()))

    print(train_error, test_error)
    # lr = 1e-3, onsite_lr = 5e-3
    train_error = [0.207921452387484, 0.19323919562765285, 0.17248056591224867, 0.1548262550219413, 0.14953808387572706,
     0.20988085273560397, 0.19561649142624637, 0.18348618848511222, 0.18183448160726498, 0.20688532011556293,
     0.21983923694610494, 0.19311779230865847, 0.19297734936914102, 0.20458271395171662, 0.18543699731238542,
     0.23845693094114484, 0.23399382156553106, 0.27936218841490884, 0.18423949358255584, 0.29695132775674693]
    test_error = [
        0.22148438788300695, 0.200351141613125, 0.16913004385118452, 0.1551401938330183, 0.151971385888368,
        0.21244050656109373, 0.20227072773194746, 0.19276081453986085, 0.19830599566864376, 0.16318411108724276,
        0.21871670239806418, 0.20429125800823678, 0.20585161637438817, 0.24021509120077522, 0.24326619085945353,
        0.24093594839214508, 0.2544091873799884, 0.3197596784922092, 0.3158245840775959, 0.11667180476222046]

    plt.plot(np.arange(len(train_error)), train_error)
    plt.plot(np.arange(len(test_error)), test_error)
    plt.show()


def train_si_tet():
    """Train Si tetragonal."""
    batch_size = 10
    select_para = False
    train_onsite = 'local'
    orbital_resolved = False
    scale_ham = False
    if select_para:
        param_list = []
        for i in [1e-3, 2e-3, 3e-3, 5e-3, 1e-2]:
            for j in [0.5, 1, 2, 3, 5]:
                param_list.append((i * j, i))
    else:
        # This param is selected based on the grid points of lr and onsite_lr
        param_list = [(2e-3, 1e-3),]

    train_error, test_error = [], []
    for ii, param in enumerate(param_list):
        lr, onsite_lr = param
        param = {'ml': {'lr': lr, 'onsite_lr': onsite_lr}}
        loss = train_function(dataset, ['si_bulk_tetrag4'], [0.4], batch_size, train_onsite,
                              orbital_resolved, scale_ham, skf_list=skf_list, _params=param)
        train_error.append(np.average(loss[-5:]))

        os.system(f'mv opt.pkl opt-si_bulk_tetrag4-{lr:f}-{onsite_lr:f}.pkl')

        err0, err1, _ = pred([f'opt-si_bulk_tetrag4-{lr:f}-{onsite_lr:f}.pkl'], dataset, ['si_bulk_tetrag4'], [0.2], batch_size)
        test_error.append(float(err0.numpy().mean()))

    print(train_error, test_error)
    # lr = 0.002, onsite_lr = 0.001
    train_error = [0.3873032246305138, 0.380708348198074, 0.33075828818035563, 0.31811409576013366, 0.3209774843694813,
     0.38518271556775907, 0.36405949198635945, 0.34375386018941667, 0.3078191606462515, 0.3077657320136672,
     0.3906884517334004, 0.3936489947452781, 0.33109889551770405, 0.3341049461577514, 0.28697675109019344,
     0.3850035327140721, 0.37578824626211416, 0.27173112849889824, 0.32051141111103654, 0.2907241615777081,
     0.35438926067033716, 0.33956190439408906, 0.2769561776440209, 0.25404473667084104, 0.34606163118823263]
    [0.4297033263483318, 0.4176501246006836, 0.4160383175039598, 0.41963394102767293, 0.448501369884853, 0.41541743332079617, 0.41605678435367915, 0.4315377925200185, 0.428107228345052, 0.49354518754293536, 0.4214014265419535, 0.4317514824796856, 0.41846859119838015, 0.45097306072473, 0.46771142131669063, 0.43577419771015496, 0.41079761966856476, 0.3948331257965656, 0.45898455825153883, 0.4460367074780044, 0.41087665782928945, 0.3728757980247848, 0.4231271101494701, 0.433348508075452, 0.6080244673144849]

    # train_error = [0.4887896299068711, 0.46907361621232546, 0.3206354300593969, 0.42571331818698555, 0.4575252146404244,
    #  0.7691603336788606, 0.48159892936356635, 0.5280704814566963, 0.40314084269362843, 0.6285093050975856,
    #  0.7274253537144739, 0.4603827487199427, 0.37570249087902946, 0.5374214673699327, 0.7162472279541404,
    #  0.7332870741170618, 0.4758658468440684, 0.5657358666717313, 0.446765087204406, 0.561875053217751,
    #  0.5985430494534748, 0.5109499812172208, 0.6382315337068206, 0.8704168098394668, 1.0125973104095887]

    print(train_error.index(min(train_error)))
    print(test_error.index(min(test_error)))
    plt.plot(np.arange(len(train_error)), train_error)
    plt.plot(np.arange(len(test_error)), test_error)
    plt.show()


def train_si_salb():
    pass


def train_si_salb():
    pass


def train_si_vac():
    pass


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Silicon Carbide systems
def sic_bulk_cubic2():
    batch_size = 3
    param_list = []
    train_onsite = 'local'
    select_para = True  # hyperparameter generation
    train_slect_para = True  # get training errors of all hyperparameters
    test_slect_para = True  # get testing errors of all hyperparameters
    orbital_resolved = False
    scale_ham = True

    if select_para:
        param_list = []

        for i in [1e-3, 2e-3, 3e-3, 5e-3]:
            for j in [0.5, 1, 2, 3, 5]:
                param_list.append((i * j, i))

    train_error = []
    if train_slect_para:
        for ii, param in enumerate(param_list):
            lr, onsite_lr = param
            param = {'ml': {'lr': lr, 'onsite_lr': onsite_lr}}
            loss = train_function(dataset, ['sic_bulk_cubic2'], [0.4], batch_size, train_onsite,
                                  orbital_resolved, scale_ham, skf_list=skf_list, _params=param)
            train_error.append(np.average(loss[-5:]))
            os.system(f'mv opt.pkl opt-sic_bulk_cubic2-{lr}-{onsite_lr}.pkl')

    test_error = []
    if test_slect_para:
        for ii, param in enumerate(param_list):
            lr, onsite_lr = param
            err0, err1, _ = pred([f'opt-sic_bulk_cubic2-{lr}-{onsite_lr}.pkl'], dataset, ['sic_bulk_cubic2'], [0.2], batch_size)
            test_error.append(float(err0.numpy().mean()))

    print(train_error, test_error)
    # 2e-3, 1e-2
    train_error = [1.2366667658165091, 1.249184456028257, 1.2467924789161202, 1.0227702128429244, 0.7127189975901304,
     1.2416679368901848, 1.2549283686765667, 0.9460204123663931, 1.0033151768856619, 0.6164507080000496,
     1.2500441799240871, 1.2595407161604664, 0.848780826986481, 0.7000942803108963, 0.4976869372736582,
     1.2652021686042607, 1.2615774475548371, 1.1077030261504681, 0.6869419108233139, 0.7704096808609211]
    test_error = [1.0441191063476636, 1.0648534410566515, 1.0014858838185556, 0.8833287111124917, 0.6782844047869139,
                  0.9696616914370566, 0.9056692487402858, 0.8589554387679549, 0.6311126439594383, 0.6379385130011803,
                  0.923330446477246, 0.9355173042061314, 0.7453809268748368, 0.6222900725176269, 0.868555959034392,
                  0.7899301957575917, 0.7545974036462879, 0.6779379049843861, 0.8426708209223167, 0.9107500332949611]

    print(train_error.index(min(train_error)))
    print(test_error.index(min(test_error)))
    plt.plot(np.arange(len(train_error)), train_error)
    plt.plot(np.arange(len(test_error)), test_error)
    plt.show()


def sic_bulk_diamond2():
    batch_size = 3
    param_list = []
    train_onsite = 'local'
    select_para = True  # hyperparameter generation
    train_slect_para = True  # get training errors of all hyperparameters
    test_slect_para = True  # get testing errors of all hyperparameters
    orbital_resolved = False
    scale_ham = True

    if select_para:
        param_list = []

        for i in [1e-3, 2e-3, 3e-3, 5e-3]:
            for j in [0.5, 1, 2, 3, 5]:
                param_list.append((i * j, i))

    train_error = []
    if train_slect_para:
        for ii, param in enumerate(param_list):
            lr, onsite_lr = param
            param = {'ml': {'lr': lr, 'onsite_lr': onsite_lr}}
            loss = train_function(dataset, ['sic_bulk_diamond2'], [0.4], batch_size, train_onsite,
                                  orbital_resolved, scale_ham, skf_list=skf_list, _params=param)
            train_error.append(np.average(loss[-5:]))
            os.system(f'mv opt.pkl opt-sic_bulk_diamond2-{lr}-{onsite_lr}.pkl')

    test_error = []
    if test_slect_para:
        for ii, param in enumerate(param_list):
            lr, onsite_lr = param
            err0, err1, _ = pred([f'opt-sic_bulk_diamond2-{lr}-{onsite_lr}.pkl'], dataset, ['sic_bulk_diamond2'], [0.2], batch_size)
            test_error.append(float(err0.numpy().mean()))

    print(train_error, test_error)
    # 2e-3, 1e-2
    train_error = [0.22017438840675077, 0.2315807565384159, 0.2506561359010829, 0.28991457864980535, 0.34420818990663016,
     0.23168317515207737, 0.23114372836740152, 0.2016076863858225, 0.2905323513289587, 0.40215623526917427,
     0.25575505147026595, 0.22901761344813626, 0.2272810417934969, 0.2699162836633603, 0.3870788387261677,
     0.29388863872106474, 0.32113248473252476, 0.34418221130871296, 0.4865187340820393, 0.7307416497069161]
    test_error = [
        0.4241398444682559, 0.7081319770623345, 0.5867330102302017, 0.5113811574986303, 0.6033848415956506,
        0.36336875224968046, 0.6432960910020447, 0.5216464391782542, 0.4645648824679359, 0.6822247587094886,
        0.3750846026544932, 0.48522822167715945, 0.6221501486309859, 0.8133327393627823, 0.39355765284392114,
        0.2422693476739036, 0.46274743029100257, 0.44747512863521416, 0.5476514212447766, 0.33037371900574236]

    print(train_error.index(min(train_error)))
    print(test_error.index(min(test_error)))
    plt.plot(np.arange(len(train_error)), train_error)
    plt.plot(np.arange(len(test_error)), test_error)
    plt.show()


def sic_bulk_diamond8():
    """Si-C 8."""
    batch_size = 3
    param_list = []
    train_onsite = 'local'
    select_para = True  # hyperparameter generation
    train_slect_para = True  # get training errors of all hyperparameters
    test_slect_para = True  # get testing errors of all hyperparameters
    orbital_resolved = False
    scale_ham = True

    if select_para:
        param_list = []

        for i in [1e-3, 2e-3, 3e-3, 5e-3]:
            for j in [0.5, 1, 2, 3, 5]:
                param_list.append((i * j, i))

    train_error = []
    if train_slect_para:
        for ii, param in enumerate(param_list):
            lr, onsite_lr = param
            param = {'ml': {'lr': lr, 'onsite_lr': onsite_lr}}
            loss = train_function(dataset, ['sic_bulk_diamond8'], [0.4], batch_size, train_onsite,
                                  orbital_resolved, scale_ham, skf_list=skf_list, _params=param)
            train_error.append(np.average(loss[-5:]))
            os.system(f'mv opt.pkl opt-sic_bulk_diamond8-{lr}-{onsite_lr}.pkl')

    test_error = []
    if test_slect_para:
        for ii, param in enumerate(param_list):
            lr, onsite_lr = param
            err0, err1, _ = pred([f'opt-sic_bulk_diamond8-{lr}-{onsite_lr}.pkl'], dataset, ['sic_bulk_diamond8'], [0.2], batch_size)
            test_error.append(float(err0.numpy().mean()))

    print(train_error, test_error)
    train_error = []
    test_error = []

    print(train_error.index(min(train_error)))
    print(test_error.index(min(test_error)))
    plt.plot(np.arange(len(train_error)), train_error)
    plt.plot(np.arange(len(test_error)), test_error)
    plt.show()


def sic_bulk_diamond64():
    """Si-C 64."""
    batch_size = 3
    param_list = []
    train_onsite = 'local'
    select_para = True  # hyperparameter generation
    train_slect_para = True  # get training errors of all hyperparameters
    test_slect_para = True  # get testing errors of all hyperparameters
    orbital_resolved = False
    scale_ham = True

    if select_para:
        param_list = []

        for i in [1e-3, 2e-3, 3e-3, 5e-3]:
            for j in [0.5, 1, 2, 3, 5]:
                param_list.append((i * j, i))

    train_error = []
    if train_slect_para:
        for ii, param in enumerate(param_list):
            lr, onsite_lr = param
            param = {'ml': {'lr': lr, 'onsite_lr': onsite_lr}}
            loss = train_function(dataset, ['sic_bulk_diamond64'], [0.4], batch_size, train_onsite,
                                  orbital_resolved, scale_ham, skf_list=skf_list, _params=param)
            train_error.append(np.average(loss[-5:]))
            os.system(f'mv opt.pkl opt-sic_bulk_diamond64-{lr}-{onsite_lr}.pkl')

    test_error = []
    if test_slect_para:
        for ii, param in enumerate(param_list):
            lr, onsite_lr = param
            err0, err1, _ = pred([f'opt-sic_bulk_diamond64-{lr}-{onsite_lr}.pkl'], dataset, ['sic_bulk_diamond64'], [0.2], batch_size)
            test_error.append(float(err0.numpy().mean()))

    print(train_error, test_error)
    # 2e-3, 1e-2
    train_error = []
    test_error = []

    print(train_error.index(min(train_error)))
    print(test_error.index(min(test_error)))
    plt.plot(np.arange(len(train_error)), train_error)
    plt.plot(np.arange(len(test_error)), test_error)
    plt.show()


def sic_bulk_diamond64_fix():
    """Si-C 64 with fixed training rate."""
    batch_size = 3
    param_list = []
    train_onsite = 'local'
    select_para = True  # hyperparameter generation
    train_slect_para = True  # get training errors of all hyperparameters
    test_slect_para = True  # get testing errors of all hyperparameters
    orbital_resolved = False
    scale_ham = True

    if select_para:
        param_list = []

        for i in [1e-3, 2e-3, 3e-3, 5e-3]:
            for j in [0.5, 1, 2, 3, 5]:
                param_list.append((i * j, i))

    train_error = []
    if train_slect_para:
        for ii, param in enumerate(param_list):
            lr, onsite_lr = param
            param = {'ml': {'lr': lr, 'onsite_lr': onsite_lr}}
            loss = train_function(dataset, ['sic_bulk_diamond64'], [0.4], batch_size, train_onsite,
                                  orbital_resolved, scale_ham, skf_list=skf_list, _params=param)
            train_error.append(np.average(loss[-5:]))
            os.system(f'mv opt.pkl opt-sic_bulk_diamond64-{lr}-{onsite_lr}.pkl')

    test_error = []
    if test_slect_para:
        for ii, param in enumerate(param_list):
            lr, onsite_lr = param
            err0, err1, _ = pred([f'opt-sic_bulk_diamond64-{lr}-{onsite_lr}.pkl'], dataset, ['sic_bulk_diamond64'], [0.2], batch_size)
            test_error.append(float(err0.numpy().mean()))

    print(train_error, test_error)
    # 2e-3, 1e-2
    train_error = []
    test_error = []

    print(train_error.index(min(train_error)))
    print(test_error.index(min(test_error)))
    plt.plot(np.arange(len(train_error)), train_error)
    plt.plot(np.arange(len(test_error)), test_error)
    plt.show()


def train_sic_slab():
    pass


def train_sic_vac():
    pass


def rename():
    import os
    param_list = []
    for i in [5e-4, 1e-3, 2e-3, 3e-3, 5e-3, 1e-2]:
        for j in [0.5, 1, 2, 3, 5]:
            param_list.append((i * j, i))

    for ii, param in enumerate(param_list):
        lr, onsite_lr = param
        name1 = f'opt-{lr:f}-{onsite_lr:f}.pkl'
        name2 = f'opt-c_bulk_diamond2-{lr:f}-{onsite_lr:f}.pkl'
        os.system(f'mv {name1} {name2}')


def load():
    import pickle
    loaded_model = pickle.load(open('../data/opt_c_bulk_diamond2_0.4.pkl', 'rb'))
    print(loaded_model.dist_pe_opt[-1].shape)
    print(loaded_model.scale[(0, 0)][-1].shape)


if __name__ == '__main__':
    """Main function."""
    eval(task + '()')
