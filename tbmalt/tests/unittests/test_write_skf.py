"""Write SKF files to binary files."""
import os
import pytest
import torch
from tbmalt.utils.skf.write_skf import WriteSK
from tbmalt.io.loadskf import IntegralGenerator
from ase.build import molecule as molecule_database
from tbmalt.common.structures.system import System
from tbmalt.tb.sk import SKT
torch.set_default_dtype(torch.float64)
torch.set_printoptions(15)
os.system('cp -r /home/gz_fan/Public/tbmalt/slko .')


def test_write_normal_skf_to_hdf():
    """Test writing h5py binary file.

    Need loadskf.py from sk repo."""
    path = './slko/auorg-1-1/'
    element = ['C', 'N', 'O', 'H']

    # generate h5py skf data
    WriteSK(path, element, sk_type='normal', repulsive=True)

    # read sk from generated h5py file and test tolerance
    path = './skf.hdf'
    molecule = molecule_database('CH4')
    system = System.from_ase_atoms(molecule)
    sk = IntegralGenerator.from_dir(path, system, sk_type='h5py')
    c_c_ref = torch.tensor([
        3.293893775138E-01, -2.631898290831E-01, 4.210227871585E-01,
        -4.705514912464E-01, -3.151402994035E-01, 3.193776711119E-01,
        -4.531014049627E-01, 4.667288655632E-01])
    atom_pair = torch.tensor([6, 6])
    distance = torch.tensor([2.0])
    c_c_sktable = torch.cat([
        sk(distance, atom_pair, torch.tensor([1, 1]), hs_type='H').squeeze(0),
        sk(distance, atom_pair, torch.tensor([0, 1]), hs_type='H').squeeze(0),
        sk(distance, atom_pair, torch.tensor([0, 0]), hs_type='H').squeeze(0),
        sk(distance, atom_pair, torch.tensor([1, 1]), hs_type='S').squeeze(0),
        sk(distance, atom_pair, torch.tensor([0, 1]), hs_type='S').squeeze(0),
        sk(distance, atom_pair, torch.tensor([0, 0]), hs_type='S').squeeze(0)])
    assert torch.max(abs(c_c_ref - c_c_sktable)) < 1E-14, 'Tolerance check'


def test_repulsive_hdf():
    """Test repulsive of hdf."""
    sk = IntegralGenerator.from_dir('./skf.hdf', elements=['C', 'H'],
                                    repulsive=True, sk_type='h5py')
    assert sk.sktable_dict[(6, 6, 'n_repulsive')] == 48
    assert sk.sktable_dict[(6, 1, 'rep_cutoff')] == 3.5
    assert torch.max(abs(
        sk.sktable_dict[(6, 6, 'rep_table')][4] - torch.tensor(
            [2.251634, -5.614025888725752, 6.723138065482665,
             -4.85914618348724]))) < 1E-14, 'Tolerance check'
    assert torch.max(abs(
        sk.sktable_dict[(1, 1, 'rep_grid')][3] - torch.tensor(
            [1.32]))) < 1E-14, 'Tolerance check'
    assert torch.max(abs(
        sk.sktable_dict[(6, 1, 'rep_long_c')] -
        torch.tensor([-0.01, 0.02007634639672507, -0.008500295606269857,
                      0.1099349367199619, -0.2904128801769102,
                      0.1912556086105955]))) < 1E-14


def test_write_compr_skf_to_hdf():
    """Test writing h5py binary file with various compression radii."""
    path = './slko/compr/'
    element = ['C', 'N', 'O', 'H']
    WriteSK(path, element, sk_type='compression_radii', homo=False)


def test_hdf_compr():
    """Test repulsive of hdf."""
    molecule = System.from_ase_atoms([molecule_database('CH4')])
    compression_r = torch.tensor([3., 3.1, 3.2, 3.3, 3.4])
    compression_radii_grid = torch.tensor([
        01.00, 01.50, 02.00, 02.50, 03.00, 03.50, 04.00,
        04.50, 05.00, 05.50, 06.00, 07.00, 08.00, 09.00, 10.00])
    sk = IntegralGenerator.from_dir(
        './skf.hdf', elements=['C', 'H'], sk_type='h5py',
        interpolation='bicubic_interpolation', homo=False,
        compression_radii_grid=compression_radii_grid)
    skt = SKT(molecule, sk, compression_radii=compression_r, fix_onsite=True,
              fix_U=True)
