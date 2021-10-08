"""Write SKF files to binary files."""
import os
import pytest
import torch
from tbmalt.utils.reference.write_reference import CalReference
from tbmalt.io.loadhdf import LoadHdf
from tbmalt.common.structures.system import System
torch.set_default_dtype(torch.float64)
torch.set_printoptions(15)


def test_aims_results_to_hdf():
    """Test writing h5py binary file from FHI-aims calculations."""
    # -> define all input parameters
    properties = ['charge', 'energy', 'homo_lumo', 'dos',
                  'dos_gamma', 'dos_raw', 'dos_raw_gamma', 'eigenvalue']
    path_to_input = '/home/wbsun/Downloads/data/Silicon_Data/Si8.h5py'
    input_type = 'Si'
    reference_size = 1
    reference_type = 'aims'
    output_name = 'test_aims_si64.hdf'
    path_to_aims_specie = '/home/wbsun/opt/fhi-aims.171221_1/species_defaults/tight'
    path_to_aims = '/home/wbsun/opt/fhi-aims.171221_1/bin/aims.171221_1.scalapack.mpi.x'

    w_aims_ani1 = CalReference(path_to_input, input_type,
                               reference_size, reference_type,
                               path_to_aims_specie=path_to_aims_specie,
                               path_to_aims=path_to_aims,
                               periodic='True')

    # calculate properties
    results = w_aims_ani1(properties)

    # write results (properties) to hdf
    CalReference.to_hdf(results, w_aims_ani1, properties, mode='w',
                        output_name=output_name, input_type='Si')

    # test the hdf reference
    numbers, positions, data = LoadHdf.load_reference(
        output_name, reference_size, properties)

    # make sure the data type consistency
    print('numbers', w_aims_ani1.numbers)
    # print('dipole', results['dipole'])
    LoadHdf.get_info(output_name)  # return dataset information


def test_dftbplus_nonscc_results_to_hdf():
    """Test repulsive of hdf."""
    # -> define all input parameters
    properties = ['dipole', 'charge', 'homo_lumo', 'energy',
                  'formation_energy']
    path_to_input = './dataset/ani_gdb_s03.h5'
    input_type = 'ANI-1'
    reference_size = 2000
    reference_type = 'dftbplus'
    path_to_dftbplus = os.path.join(_path, 'dftbplus/dftb+')
    path_to_skf = os.path.join(_path, './slko/init')

    w_dftb_ani1 = CalReference(path_to_input, input_type,
                               reference_size, reference_type,
                               path_to_skf=path_to_skf,
                               path_to_dftbplus=path_to_dftbplus)

    # calculate properties
    output_name = 'nonscc.hdf'
    results = w_dftb_ani1(properties, dftb_type='nonscc')

    # write results (properties) to hdf
    CalReference.to_hdf(results, w_dftb_ani1, properties, mode='w',
                        output_name=output_name)

    # load the the generated dataset
    numbers, positions, data = LoadHdf.load_reference(
        output_name, reference_size, properties)

    # make sure the data type consistency
    print('numbers', w_dftb_ani1.numbers)
    print('dipole', results['charge'])
    LoadHdf.get_info(output_name)  # return dataset information


def test_dftbplus_scc_results_to_hdf():
    """Test repulsive of hdf."""
    # -> define all input parameters
    properties = ['charge', 'energy', 'homo_lumo', 'eigenvalue', 'occupancy']
    path_to_input = '/home/wbsun/Downloads/data/Silicon_Data/Si8.h5py'
    input_type = 'Si'
    reference_size = 1
    reference_type = 'dftbplus'
    path_to_dftbplus = '/home/wbsun/anaconda3/bin/dftb+'
    path_to_skf = '/home/wbsun/dftbplus/slako/pbc/pbc-0-3'

    w_dftb_ani1 = CalReference(path_to_input, input_type,
                               reference_size, reference_type,
                               path_to_skf=path_to_skf,
                               path_to_dftbplus=path_to_dftbplus,
                               periodic='True')

    # calculate properties
    output_name = 'test_dftb_si64.hdf'
    results = w_dftb_ani1(properties, dftb_type='scc_si_pbc')

    # write results (properties) to hdf
    CalReference.to_hdf(results, w_dftb_ani1, properties, mode='w',
                        output_name=output_name, input_type='Si')

    # load the the generated dataset
    numbers, positions, data = LoadHdf.load_reference(
        output_name, reference_size, properties)

    # make sure the data type consistency
    print('numbers', w_dftb_ani1.numbers)
    # print('dipole', results['charge'])
    LoadHdf.get_info(output_name)  # return dataset information
