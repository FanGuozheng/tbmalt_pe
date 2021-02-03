"""Write SKF files to binary files."""
import os
import pytest
import torch
from tbmalt.utils.reference.write_reference import CalReference
from tbmalt.io.loadhdf import LoadHdf
from tbmalt.common.structures.system import System
torch.set_default_dtype(torch.float64)
torch.set_printoptions(15)


os.system('cp -r /home/gz_fan/Public/tbmalt/aims .')
os.system('cp -r /home/gz_fan/Public/tbmalt/dataset .')
_path = os.getcwd()


def test_aims_results_to_hdf():
    """Test writing h5py binary file from FHI-aims calculations."""
    # -> define all input parameters
    properties = ['dipole', 'charge', 'homo_lumo', 'total_energy', 'alpha_mbd',
                  'hirshfeld_volume', 'hirshfeld_volume_ratio']
    path_to_input = './dataset/ani_gdb_s01.h5'
    input_type = 'ANI-1'
    reference_size = 6
    reference_type = 'aims'
    path_to_aims_specie = os.path.join(_path, 'aims/species_defaults/tight')
    path_to_aims = os.path.join(_path, 'aims/aims.x')

    w_aims_ani1 = CalReference(path_to_input, input_type,
                               reference_size, reference_type,
                               path_to_aims_specie=path_to_aims_specie,
                               path_to_aims=path_to_aims)

    # calculate properties
    results = w_aims_ani1(properties)

    # write results (properties) to hdf
    CalReference.to_hdf(results, w_aims_ani1.numbers, w_aims_ani1.symbols,
                        w_aims_ani1.positions, properties, mode='w',
                        atom_specie_global=w_aims_ani1.atom_specie_global)

    # test the hdf reference
    numbers, positions, data = LoadHdf.load_reference(
        'reference.hdf', reference_size, properties)

    # make sure the data type consistency
    dist = System(numbers, positions).distances

def test_dftbplus_results_to_hdf():
    """Test repulsive of hdf."""
