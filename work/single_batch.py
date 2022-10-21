#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Testing the efficiency between single and batch."""
import time

import matplotlib.pyplot as plt
import torch

from tbmalt.structures.geometry import Geometry

from tbmalt.io.hdf import LoadHdf
from tbmalt.physics.dftb.dftb import Dftb2

torch.set_printoptions(15)
torch.set_default_dtype(torch.float64)

##########
# params #
##########
target = 'optimize'
device = torch.device('cpu')

data1 = './dataset/scc_6000_01.hdf'
data3 = './dataset/scc_6000_03.hdf'

dftb_property = ['dipole', 'charge']
properties = ['charge',]

shell_dict = {1: [0], 6: [0, 1], 7: [0, 1], 8: [0, 1]}
size_list = [60, 120, 300, 600, 1200]


def single_batch(device):
    """Optimize spline parameters or compression radii."""

    time_dict = {}
    for size in size_list:

        # ====== ANI1 =======
        numbers, positions, data = LoadHdf.load_reference(data1, size, properties)

        # batch
        # time0 = time.process_time()
        geometry = Geometry(numbers, positions, units='angstrom')
        path_to_skf = '../tests/unittests/data/slko/mio'
        dftb = Dftb2(geometry, shell_dict=shell_dict,
                     path_to_skf=path_to_skf, skf_type='skf')
        time0 = time.process_time()
        dftb()
        time_end = time.process_time()
        print(f'batch size {size}, ANI1 time:', time_end - time0)
        time_dict.update({str(size) + 'batch_ani1': time_end - time0})

        # single, ANI1
        time_sin = 0
        # time0 = time.process_time()
        for number, position in zip(numbers, positions):
            geometry = Geometry(number.unsqueeze(0), position.unsqueeze(0), units='angstrom')
            path_to_skf = '../tests/unittests/data/slko/mio'
            dftb = Dftb2(geometry, shell_dict=shell_dict,
                         path_to_skf=path_to_skf, skf_type='skf')
            time0 = time.process_time()
            dftb()
            time_end = time.process_time()
            time_sin += time_end - time0

        print(f'single, size {size} ANI1 time:', time_sin)
        time_dict.update({str(size) + 'single_ani1': time_sin})
        time_end = time.process_time()
        print(f'single, size {size} ANI1 time:', time_end - time0)
        time_dict.update({str(size) + 'single_ani1': time_end - time0})

    for size in size_list:
        # ====== ANI3 =======
        numbers, positions, data = LoadHdf.load_reference(data3, size, properties)

        # batch
        # time0 = time.process_time()
        geometry = Geometry(numbers, positions, units='angstrom')
        path_to_skf = '../tests/unittests/data/slko/mio'
        dftb = Dftb2(geometry, shell_dict=shell_dict,
                     path_to_skf=path_to_skf, skf_type='skf')
        time0 = time.process_time()
        dftb()
        time_end = time.process_time()
        print(f'batch, size {size} ANI3 time:', time_end - time0)
        time_dict.update({str(size) + 'batch_ani3': time_end - time0})

        # single, ANI3
        time_sin = 0
        time0 = time.process_time()
        for number, position in zip(numbers, positions):
            geometry = Geometry(number.unsqueeze(0), position.unsqueeze(0), units='angstrom')
            path_to_skf = '../tests/unittests/data/slko/mio'
            dftb = Dftb2(geometry, shell_dict=shell_dict,
                         path_to_skf=path_to_skf, skf_type='skf')
            time0 = time.process_time()
            dftb()
            time_end = time.process_time()
            time_sin += time_end - time0

        print(f'single, size {size} ANI3 time:', time_sin)
        time_dict.update({str(size) + 'single_ani3': time_sin})
        time_end = time.process_time()
        print(f'single, size {size} ANI3 time:', time_end - time0)
        time_dict.update({str(size) + 'single_ani3': time_end - time0})

    print(time_dict)
    # {'60batch_ani1': 0.26059699058532715, '60single_ani1': 2.694204092025757,
    # '120batch_ani1': 0.3314366340637207, '120single_ani1': 5.4519970417022705,
    # '300batch_ani1': 0.5616278648376465, '300single_ani1': 13.487237215042114,
    # '600batch_ani1': 0.9825789928436279, '600single_ani1': 28.260409832000732,
    # '1200batch_ani1': 1.7620341777801514, '1200single_ani1': 55.35037589073181,
    # '60batch_ani3': 0.5739607810974121, '60single_ani3': 5.892338991165161,
    # '120batch_ani3': 0.6532931327819824, '120single_ani3': 11.961782932281494,
    # '300batch_ani3': 1.2143080234527588, '300single_ani3': 31.77964687347412,
    # '600batch_ani3': 2.2885918617248535, '600single_ani3': 68.38412189483643,
    # '1200batch_ani3': 4.558136940002441, '1200single_ani3': 135.45029211044312}

    # ========= plot ==========
    ani1_sing_time = [time_dict[str(size) + 'single_ani1'] for size in size_list]
    ani1_bat_time = [time_dict[str(size) + 'batch_ani1'] for size in size_list]
    ani3_sing_time = [time_dict[str(size) + 'single_ani3'] for size in size_list]
    ani3_bat_time = [time_dict[str(size) + 'batch_ani3'] for size in size_list]

    plt.plot(size_list, ani1_sing_time, '-o', label=r'single, ANI-1$_1$')
    plt.plot(size_list, ani1_bat_time, '--o', label=r'batch, ANI-1$_1$')
    plt.plot(size_list, ani3_sing_time, '-x', label=r'single, ANI-1$_3$')
    plt.plot(size_list, ani1_bat_time, '--x', label=r'batch, ANI-1$_3$')

    plt.xlabel('data set size', fontsize='large')
    plt.ylabel('CPU time (s)', fontsize='large')
    plt.legend()
    plt.savefig('batchSingle.png', dpi=300)
    plt.show()


def parallel():
    device_ids = [0, 1, 2, 3]

    time_dict= {}

    for size in size_list:
        # ====== ANI3 =======
        numbers, positions, data = LoadHdf.load_reference(data3, size, properties)

        # batch
        time0 = time.time()
        geometry = Geometry(numbers, positions, units='angstrom')
        path_to_skf = '../tests/unittests/data/slko/mio'
        dftb = Dftb2(geometry, shell_dict=shell_dict,
                     path_to_skf=path_to_skf, skf_type='skf')
        dftb()
        time_end = time.time()
        print(f'batch, size {size} ANI3 time:', time_end - time0)
        time_dict.update({str(size) + 'batch_ani3': time_end - time0})

        # single, ANI3
        numbers, positions, data = LoadHdf.load_reference(data3, size, properties)

        # batch
        time0 = time.time()
        geometry = Geometry(numbers, positions, units='angstrom')
        path_to_skf = '../tests/unittests/data/slko/mio'
        dftb = Dftb2(geometry, shell_dict=shell_dict,
                     path_to_skf=path_to_skf, skf_type='skf')
        torch.nn.DataParallel(dftb(), device_ids=device_ids)
        time_end = time.time()
        print(f'batch, size {size} ANI3 time:', time_end - time0)
        time_dict.update({str(size) + 'batch_ani3': time_end - time0})


if __name__ == '__main__':
    """Main function."""
    single_batch(device=device)
    # parallel()
