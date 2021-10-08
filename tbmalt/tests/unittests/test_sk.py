"""Perform tests on functions which read SKF or SK transformations.
Reference is from DFTB+."""
import os
import re
import numpy as np
import torch
from ase.build import molecule as molecule_database
from tbmalt.common.structures.system import System
from tbmalt.common.structures.periodic import Periodic
from tbmalt.tb.sk import SKT
from tbmalt.io.loadskf import IntegralGenerator
from tbmalt.common.batch import pack
torch.set_default_dtype(torch.float64)
torch.set_printoptions(15)


# copy from public directory
os.system('cp /home/gz_fan/Public/tbmalt/skf.hdf .')
os.system('cp -r /home/gz_fan/Public/tbmalt/slko .')
os.system('cp -r /home/gz_fan/Public/tbmalt/sk .')


def test_read_skf_mio(device):
    """Test mio SKF data."""
    molecule = molecule_database('CH4')
    system = System.from_ase_atoms(molecule)
    sk = IntegralGenerator.from_dir('./slko/mio-1-1', system)

    # Hpp0 Hpp1, Hsp0, Hss0, Spp0 Spp1, Ssp0, Sss0 at distance 2.0
    atom_pair = torch.tensor([6, 6])
    distance = torch.tensor([2.0])
    c_c_ref = torch.tensor([
        3.293816666621e-01, -2.631899962890e-01, 4.210169476043e-01,
        -4.705475117630e-01, -3.151403220092e-01, 3.193776444242e-01,
        -4.531013858250e-01, 4.667288518272e-01])
    c_c_sktable = torch.cat([
        sk(distance, atom_pair, torch.tensor([1, 1]), hs_type='H').squeeze(0),
        sk(distance, atom_pair, torch.tensor([0, 1]), hs_type='H').squeeze(0),
        sk(distance, atom_pair, torch.tensor([0, 0]), hs_type='H').squeeze(0),
        sk(distance, atom_pair, torch.tensor([1, 1]), hs_type='S').squeeze(0),
        sk(distance, atom_pair, torch.tensor([0, 1]), hs_type='S').squeeze(0),
        sk(distance, atom_pair, torch.tensor([0, 0]), hs_type='S').squeeze(0)])
    assert torch.max(abs(c_c_ref - c_c_sktable)) < 1E-14, 'Tolerance check'


def test_repulsive_mio():
    """Test repulsive of mio."""
    sk = IntegralGenerator.from_dir('./slko/mio-1-1',
                                    elements=['C', 'N', 'O', 'H'], repulsive=True)
    assert sk.sktable_dict[(1, 1, 'n_repulsive')] == 16
    assert sk.sktable_dict[(1, 6, 'rep_cutoff')] == 3.5
    assert torch.max(abs(
        sk.sktable_dict[(6, 6, 'rep_table')][0] - torch.tensor(
            [3.344853000000000, -8.185615473079642, 8.803750000000022,
             1.681545674779360]))) < 1E-14, 'Tolerance check'
    assert torch.max(abs(
        sk.sktable_dict[(6, 7, 'rep_table')][-1] - torch.tensor(
            [0.05220000000000001, -0.07188183228314954, 0.01604256322595639,
             0.005489302378575448]))) < 1E-14, 'Tolerance check'
    assert torch.max(abs(sk.sktable_dict[(8, 8, 'rep_long_grid')] -
                         torch.tensor([3.28, 4.2]))) < 1E-14, 'Tolerance check'


def test_other_params():
    """Test other parameters except HS, onsite, U and repulsive."""
    sk = IntegralGenerator.from_dir('./slko/mio-1-1',
                                    elements=['C', 'H'], repulsive=True)
    assert sk.sktable_dict[(1, 6, 'g_step')] == 0.02
    assert sk.sktable_dict[(1, 6, 'n_points')] == 500
    assert sk.sktable_dict[(1, 1, 'hs_cutoff')] == (500 - 1) * 0.02
    assert (sk.sktable_dict[(6, 6, 'hs_grid')] == torch.arange(
        1, 500 + 1) * 0.02).all()
    sk2 = IntegralGenerator.from_dir('./slko/auorg-1-1',
                                     elements=['C', 'H'], repulsive=True)
    assert sk2.sktable_dict[(1, 6, 'version')] == '0.9'
    assert sk2.sktable_dict[(1, 6, 'g_step')] == 0.02
    assert sk2.sktable_dict[(6, 1, 'hs_cutoff')] == (500 - 1) * 0.02


def test_sk_mio_ase_single(device):
    """Test SK transformation values of single molecules from mio."""
    system = System.from_ase_atoms([molecule_database('CH3CH2NH2')])
    sktable = IntegralGenerator.from_dir('./slko/mio-1-1', system)
    skt = SKT(system, sktable)
    assert torch.max(abs(skt.H[0][:h_ch3ch2nh2.shape[0], :h_ch3ch2nh2.shape[1]]
                         - h_ch3ch2nh2)) < 1E-14, 'Tolerance check'
    assert torch.max(abs(skt.S[0][:s_ch3ch2nh2.shape[0], :s_ch3ch2nh2.shape[1]]
                         - s_ch3ch2nh2)) < 1E-14, 'Tolerance check'


def test_sk_mio_ase_batch(device):
    """Test SK transformation values of batch molecules from mio."""
    system = System.from_ase_atoms([molecule_database('CH3S'),
                                    molecule_database('PH3')])
    sktable = IntegralGenerator.from_dir('./slko/mio-1-1', system)
    skt = SKT(system, sktable)
    assert torch.max(abs(skt.H[0][:h_ch3s.shape[0], :h_ch3s.shape[1]] - h_ch3s)
                     ) < 1E-14, 'Tolerance check'
    assert torch.max(abs(skt.S[0][:s_ch3s.shape[0], :s_ch3s.shape[1]] - s_ch3s)
                     ) < 1E-14, 'Tolerance check'
    assert torch.max(abs(skt.H[1][:h_ph3.shape[0], :h_ph3.shape[1]] - h_ph3)
                     ) < 1E-14, 'Tolerance check'
    assert torch.max(abs(skt.S[1][:s_ph3.shape[0], :s_ph3.shape[1]] - s_ph3)
                     ) < 1E-14, 'Tolerance check'


def test_read_skf_auorg(device):
    """Read auorg type SKF files."""
    molecule = molecule_database('CH4')
    system = System.from_ase_atoms(molecule)
    sk = IntegralGenerator.from_dir('./slko/auorg-1-1', system)

    # Hpp0 Hpp1, Hsp0, Hss0, Spp0 Spp1, Ssp0, Sss0 at distance 2.0
    atom_pair = torch.tensor([6, 6])
    distance = torch.tensor([2.0])
    c_c_ref = torch.tensor([
        3.293893775138E-01, -2.631898290831E-01, 4.210227871585E-01,
        -4.705514912464E-01, -3.151402994035E-01, 3.193776711119E-01,
        -4.531014049627E-01, 4.667288655632E-01])
    c_c_sktable = torch.cat([
        sk(distance, atom_pair, torch.tensor([1, 1]), hs_type='H').squeeze(0),
        sk(distance, atom_pair, torch.tensor([0, 1]), hs_type='H').squeeze(0),
        sk(distance, atom_pair, torch.tensor([0, 0]), hs_type='H').squeeze(0),
        sk(distance, atom_pair, torch.tensor([1, 1]), hs_type='S').squeeze(0),
        sk(distance, atom_pair, torch.tensor([0, 1]), hs_type='S').squeeze(0),
        sk(distance, atom_pair, torch.tensor([0, 0]), hs_type='S').squeeze(0)])
    assert torch.max(abs(c_c_ref - c_c_sktable)) < 1E-14, 'Tolerance check'


def test_repulsive_auorg():
    """Test repulsive of mio."""
    sk = IntegralGenerator.from_dir('./slko/auorg-1-1',
                                    elements=['C', 'Au'], repulsive=True)
    assert sk.sktable_dict[(79, 79, 'n_repulsive')] == 51
    assert sk.sktable_dict[(79, 6, 'rep_cutoff')] == 6.69486306644
    assert torch.max(abs(
        sk.sktable_dict[(6, 79, 'rep_table')][-2] - torch.tensor(
            [0.000281334224069, -0.00110975431978, 0.00164091609277,
             -0.00105262626385]))) < 1E-14, 'Tolerance check'
    assert torch.max(abs(
        sk.sktable_dict[(79, 79, 'rep_grid')][-1] - torch.tensor(
            [7.19486306644]))) < 1E-14, 'Tolerance check'
    assert torch.max(abs(
        sk.sktable_dict[(79, 6, 'rep_long_c')] -
        torch.tensor([0.000185721978564, -0.00081275078063, 0.00133311038379,
                      -12.9871430346, 395.316431054, -3186.4064968]))) < 1E-14


def test_repulsive_hdf():
    """Test repulsive of hdf (originally from auorg)."""
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


def test_read_skf_h5py(device):
    """Read auorg type SKF files."""
    molecule = molecule_database('CH4')
    system = System.from_ase_atoms(molecule)
    sk = IntegralGenerator.from_dir('./skf.hdf', system, sk_type='h5py')
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


def test_sk_single(device):
    """Test SK transformation values of single molecule."""
    numbers = torch.tensor([6, 1, 1, 1, 1])
    positions = torch.tensor([
        [0., 0., 0.], [0.629118, 0.629118, 0.629118],
        [-0.629118, -0.629118, 0.629118], [0.629118, -0.629118, -0.629118],
        [-0.629118, 0.629118, -0.629118]])
    molecule = System(numbers, positions)
    sktable = IntegralGenerator.from_dir('./slko/auorg-1-1', molecule)
    skt = SKT(molecule, sktable)
    assert torch.max(abs(skt.H - h_ch4)) < 1E-14, 'Tolerance check'
    assert torch.max(abs(skt.S - s_ch4)) < 1E-14, 'Tolerance check'


def test_sk_single_d_orb(device):
    """Test SK transformation values of single molecule with d orbitals."""
    numbers = torch.tensor([79, 8])
    positions = torch.tensor([[0., 0., 0.], [1., 1., 0.]])
    molecule = System(numbers, positions)
    sktable = IntegralGenerator.from_dir('./slko/auorg-1-1/', molecule)
    skt = SKT(molecule, sktable)
    assert torch.max(abs(skt.H - h_auo)) < 1E-14, 'Tolerance check'
    assert torch.max(abs(skt.S - s_auo)) < 1E-14, 'Tolerance check'


def test_sk_batch(device):
    """Test SK transformation values of batch molecules.
    Test p-d, s-p, d-d orbitals."""
    numbers = [torch.tensor([79, 8]), torch.tensor([79, 79]),
               torch.tensor([6, 1, 1, 1, 1])]
    positions = [torch.tensor([[0., 0., 0.], [1., 1., 0.]]),
                 torch.tensor([[0., 0., 0.], [1., 1., 0.]]),
                 torch.tensor([[0., 0., 0.], [0.629118, 0.629118, 0.629118],
                               [-0.629118, -0.629118, 0.629118],
                               [0.629118, -0.629118, -0.629118],
                               [-0.629118, 0.629118, -0.629118]])]
    molecule = System(numbers, positions)
    sktable = IntegralGenerator.from_dir('./slko/auorg-1-1/', molecule)
    skt = SKT(molecule, sktable)
    assert torch.max(abs(skt.H[0][:h_auo.shape[0], :h_auo.shape[1]] - h_auo)
                     ) < 1E-14, 'Tolerance check'
    assert torch.max(abs(skt.S[0][:s_auo.shape[0], :s_auo.shape[1]] - s_auo)
                     ) < 1E-14, 'Tolerance check'
    assert torch.max(abs(skt.H[1][:h_auau.shape[0], :h_auau.shape[1]] - h_auau)
                     ) < 1E-14, 'Tolerance check'
    assert torch.max(abs(skt.S[1][:s_auau.shape[0], :s_auau.shape[1]] - s_auau)
                     ) < 1E-14, 'Tolerance check'
    assert torch.max(abs(skt.H[2][:h_ch4.shape[0], :h_ch4.shape[1]] - h_ch4)
                     ) < 1E-14, 'Tolerance check'
    assert torch.max(abs(skt.S[2][:s_ch4.shape[0], :s_ch4.shape[1]] - s_ch4)
                     ) < 1E-14, 'Tolerance check'


def test_sk_ase_single(device):
    """Test SK transformation values of single ASE molecule."""
    molecule = molecule_database('CH4')
    molecule = System.from_ase_atoms(molecule)
    sktable = IntegralGenerator.from_dir('./slko/auorg-1-1/', molecule)
    skt = SKT(molecule, sktable)
    assert torch.max(abs(skt.H - h_ch4)) < 1E-14, 'Tolerance check'
    assert torch.max(abs(skt.S - s_ch4)) < 1E-14, 'Tolerance check'


def test_sk_ase_batch(device):
    """Test SK transformation values of batch ASE molecules."""
    molecule = System.from_ase_atoms([
        molecule_database('H2'), molecule_database('N2'),
        molecule_database('CH4'), molecule_database('NH3'),
        molecule_database('H2O'), molecule_database('CO2'),
        molecule_database('CH3CHO')])
    sktable = IntegralGenerator.from_dir('./slko/auorg-1-1/', molecule)
    skt = SKT(molecule, sktable)
    assert torch.max(abs(skt.H[0][:h_h2.shape[0], :h_h2.shape[1]] - h_h2)
                     ) < 1E-14, 'Tolerance check'
    assert torch.max(abs(skt.S[0][:s_h2.shape[0], :s_h2.shape[1]] - s_h2)
                     ) < 1E-14, 'Tolerance check'
    assert torch.max(abs(skt.H[1][:h_n2.shape[0], :h_n2.shape[1]] - h_n2)
                     ) < 1E-14, 'Tolerance check'
    assert torch.max(abs(skt.S[1][:s_n2.shape[0], :s_n2.shape[1]] - s_n2)
                     ) < 1E-14, 'Tolerance check'
    assert torch.max(abs(skt.H[2][:h_ch4.shape[0], :h_ch4.shape[1]] - h_ch4)
                     ) < 1E-14, 'Tolerance check'
    assert torch.max(abs(skt.S[2][:s_ch4.shape[0], :s_ch4.shape[1]] - s_ch4)
                     ) < 1E-14, 'Tolerance check'
    assert torch.max(abs(skt.H[3][:h_nh3.shape[0], :h_nh3.shape[1]] - h_nh3)
                     ) < 1E-14, 'Tolerance check'
    assert torch.max(abs(skt.S[3][:s_nh3.shape[0], :s_nh3.shape[1]] - s_nh3)
                     ) < 1E-14, 'Tolerance check'
    assert torch.max(abs(skt.H[4][:h_h2o.shape[0], :h_h2o.shape[1]] - h_h2o)
                     ) < 1E-14, 'Tolerance check'
    assert torch.max(abs(skt.S[4][:s_h2o.shape[0], :s_h2o.shape[1]] - s_h2o)
                     ) < 1E-14, 'Tolerance check'
    assert torch.max(abs(skt.H[5][:h_co2.shape[0], :h_co2.shape[1]] - h_co2)
                     ) < 1E-14, 'Tolerance check'
    assert torch.max(abs(skt.S[5][:s_co2.shape[0], :s_co2.shape[1]] - s_co2)
                     ) < 1E-14, 'Tolerance check'
    assert torch.max(abs(skt.H[6][:h_ch3cho.shape[0], :h_ch3cho.shape[1]]
                         - h_ch3cho)) < 1E-14, 'Tolerance check'
    assert torch.max(abs(skt.S[6][:s_ch3cho.shape[0], :s_ch3cho.shape[1]]
                         - s_ch3cho)) < 1E-14, 'Tolerance check'


def test_sk_ase_batch_cubic(device):
    """Test batch molecule SK transformtion value.
    The interpolation of integral is cubic interpolation, which is different
    from DFTB+."""
    molecule = System.from_ase_atoms([
        molecule_database('H2'), molecule_database('N2'),
        molecule_database('CH4'), molecule_database('NH3'),
        molecule_database('H2O'), molecule_database('CN'),
        molecule_database('CO2'), molecule_database('CH3CHO')])
    sktable = IntegralGenerator.from_dir(
        './slko/auorg-1-1/', molecule, sk_interpolation='numpy_interpolation')
    skt = SKT(molecule, sktable)
    assert torch.max(abs(skt.H[0][:h_h2.shape[0], :h_h2.shape[1]] - h_h2)
                     ) < 1E-9, 'Tolerance check'
    assert torch.max(abs(skt.H[4][:h_h2o.shape[0], :h_h2o.shape[1]] - h_h2o)
                     ) < 1E-9, 'Tolerance check'


def test_sk_interpolation(device):
    """Test SK interpolation with random distance."""
    sktable = IntegralGenerator.from_dir('slko/mio-1-1/', elements=['C', 'H'])
    ch_00 = sktable(torch.tensor([2.0591670220427729]), torch.tensor([6, 1]),
                    torch.tensor([0, 0]), hs_type='H')
    assert abs(ch_00 - -0.33103366295265063) < 1E-14, 'distance < leng'
    cc_01 = sktable(torch.tensor([1.6365507123010095]), torch.tensor([6, 6]),
                    torch.tensor([0, 1]), hs_type='H')
    assert abs(cc_01 - 0.45308093203471983) < 1E-14, 'distance < leng'
    hc_00 = sktable(torch.tensor([10.473924558726461]), torch.tensor([1, 6]),
                    torch.tensor([0, 0]), hs_type='H')

    # second derivative: y2 + y0 - 2.0 * y1 is smaller than 1E-14, but
    # delta ** 2 will result in the difference > 1E-14, seems not bug
    assert abs(hc_00 - 8.066100641653708e-06) < 1E-11, 'distance < leng'
    hc_00_b = sktable(torch.tensor([10.473924558726461, 10.584152656903393]),
                      torch.tensor([1, 6]), torch.tensor([0, 0]), hs_type='H')
    assert torch.max(abs(hc_00_b - torch.tensor(
        [[8.066100641653708e-06], [4.4433154052067092E-006]]))) < 1E-11


def get_matrix(filename):
    """Read DFTB+ hamsqr1.dat and oversqr.dat."""
    text = ''.join(open(filename, 'r').readlines())
    string = re.search('(?<=MATRIX\n).+(?=\n)', text, flags=re.DOTALL).group(0)
    out = np.array([[float(i) for i in row.split()]
                    for row in string.split('\n')])
    return torch.from_numpy(out)


def test_spline_tail():
    """Test the interplolation for tail by spline."""
    latvec = torch.tensor([[6., 0., 0.], [0., 6., 0.], [0., 0., 6.]])
    positions = torch.tensor([
        [3., 3., 3.], [3.6, 3.6, 3.6], [2.4, 3.6, 3.6], [3.6, 2.4, 3.6], [3.6, 3.6, 2.4]])
    numbers = torch.tensor([6, 1, 1, 1, 1])
    molecule = System(numbers, positions, latvec)
    sk_spline = IntegralGenerator.from_dir('./slko/mio-1-1', molecule,
                                           interpolation='spline')
    sk_skinter = IntegralGenerator.from_dir('./slko/mio-1-1', molecule)

    # C-C pair
    atom_pair_cc = torch.tensor([6, 6])
    distance = torch.from_numpy(np.linspace(9.98, 10.98, 50))

    # pp orbitals
    tail_spline_cc_Hpp = sk_spline(distance, atom_pair_cc, torch.tensor([1, 1]), hs_type='H')
    tail_skinter_cc_Hpp = sk_skinter(distance, atom_pair_cc, torch.tensor([1, 1]), hs_type='H')
    tail_spline_cc_Spp = sk_spline(distance, atom_pair_cc, torch.tensor([1, 1]), hs_type='S')
    tail_skinter_cc_Spp = sk_skinter(distance, atom_pair_cc, torch.tensor([1, 1]), hs_type='S')

    # sp orbital
    tail_spline_cc_Hsp = sk_spline(distance, atom_pair_cc, torch.tensor([0, 1]), hs_type='H')
    tail_skinter_cc_Hsp = sk_skinter(distance, atom_pair_cc, torch.tensor([0, 1]), hs_type='H')
    tail_spline_cc_Ssp = sk_spline(distance, atom_pair_cc, torch.tensor([0, 1]), hs_type='S')
    tail_skinter_cc_Ssp = sk_skinter(distance, atom_pair_cc, torch.tensor([0, 1]), hs_type='S')

    # ss orbital
    tail_spline_cc_Hss = sk_spline(distance, atom_pair_cc, torch.tensor([0, 0]), hs_type='H')
    tail_skinter_cc_Hss = sk_skinter(distance, atom_pair_cc, torch.tensor([0, 0]), hs_type='H')
    tail_spline_cc_Sss = sk_spline(distance, atom_pair_cc, torch.tensor([0, 0]), hs_type='S')
    tail_skinter_cc_Sss = sk_skinter(distance, atom_pair_cc, torch.tensor([0, 0]), hs_type='S')

    # C-H pair
    atom_pair_ch = torch.tensor([6, 1])

    # sp orbital
    tail_spline_ch_Hsp = sk_spline(distance, atom_pair_ch, torch.tensor([0, 1]), hs_type='H')
    tail_skinter_ch_Hsp = sk_skinter(distance, atom_pair_ch, torch.tensor([0, 1]), hs_type='H')
    tail_spline_ch_Ssp = sk_spline(distance, atom_pair_ch, torch.tensor([0, 1]), hs_type='S')
    tail_skinter_ch_Ssp = sk_skinter(distance, atom_pair_ch, torch.tensor([0, 1]), hs_type='S')

    # ss orbital
    tail_spline_ch_Hss = sk_spline(distance, atom_pair_ch, torch.tensor([0, 0]), hs_type='H')
    tail_skinter_ch_Hss = sk_skinter(distance, atom_pair_ch, torch.tensor([0, 0]), hs_type='H')
    tail_spline_ch_Sss = sk_spline(distance, atom_pair_ch, torch.tensor([0, 0]), hs_type='S')
    tail_skinter_ch_Sss = sk_skinter(distance, atom_pair_ch, torch.tensor([0, 0]), hs_type='S')

    # H-H pair
    atom_pair_hh = torch.tensor([1, 1])

    # ss orbital
    tail_spline_hh_Hss = sk_spline(distance, atom_pair_hh, torch.tensor([0, 0]), hs_type='H')
    tail_skinter_hh_Hss = sk_skinter(distance, atom_pair_hh, torch.tensor([0, 0]), hs_type='H')
    tail_spline_hh_Sss = sk_spline(distance, atom_pair_hh, torch.tensor([0, 0]), hs_type='S')
    tail_skinter_hh_Sss = sk_skinter(distance, atom_pair_hh, torch.tensor([0, 0]), hs_type='S')

    tail_spline_cc = pack([tail_spline_cc_Hpp, tail_spline_cc_Spp, tail_spline_cc_Hsp,
                           tail_spline_cc_Ssp, tail_spline_cc_Hss, tail_spline_cc_Sss])
    tail_skinter_cc = pack([tail_skinter_cc_Hpp, tail_skinter_cc_Spp, tail_skinter_cc_Hsp,
                            tail_skinter_cc_Ssp, tail_skinter_cc_Hss, tail_skinter_cc_Sss])
    tail_spline_ch = pack([tail_spline_ch_Hsp, tail_spline_ch_Ssp,
                           tail_spline_ch_Hss, tail_spline_ch_Sss])
    tail_skinter_ch = pack([tail_skinter_ch_Hsp, tail_skinter_ch_Ssp,
                            tail_skinter_ch_Hss, tail_skinter_ch_Sss])
    tail_spline_hh = pack([tail_spline_hh_Hss, tail_spline_hh_Sss])
    tail_skinter_hh = pack([tail_skinter_hh_Hss, tail_skinter_hh_Sss])
    assert torch.max(abs(tail_spline_cc - tail_skinter_cc)) < 1E-10, 'Tolerance check'
    assert torch.max(abs(tail_spline_ch - tail_skinter_ch)) < 1E-10, 'Tolerance check'
    assert torch.max(abs(tail_spline_hh - tail_skinter_hh)) < 1E-10, 'Tolerance check'


def test_sk_pe_single():
    """Test SK transformation values of single periodic molecule."""
    latvec = torch.tensor([[4., 0., 0.], [0., 4., 0.], [0., 0., 4.]])
    cutoff = torch.tensor([9.98])
    positions = torch.tensor([[0., 0., 0.], [0., 2., 0.]])
    numbers = torch.tensor([1, 1])
    molecule = System(numbers, positions, latvec)
    sktable = IntegralGenerator.from_dir('./slko/mio-1-1', molecule)
    periodic = Periodic(molecule, molecule.cell, cutoff=cutoff)
    skt = SKT(molecule, sktable, periodic)
    ref_H = torch.tensor([[[-0.2400383294622024, -0.09890788917325767],
                           [-0.09890788917325767, -0.2400383294622024]]])
    ref_S = torch.tensor([[[1.001205766178933, 0.1364211414311132],
                           [0.1364211414311132, 1.001205766178933]]])
    assert torch.max(abs(skt.H - ref_H)) < 1E-10, 'Tolerance check'
    assert torch.max(abs(skt.S - ref_S)) < 1E-10, 'Tolerance check'


h_h2 = get_matrix('./sk/h2/hamsqr1.dat')
s_h2 = get_matrix('./sk/h2/oversqr.dat')
h_n2 = get_matrix('./sk/n2/hamsqr1.dat')
s_n2 = get_matrix('./sk/n2/oversqr.dat')
h_nh3 = get_matrix('./sk/nh3/hamsqr1.dat')
s_nh3 = get_matrix('./sk/nh3/oversqr.dat')
h_ch4 = get_matrix('./sk/ch4/hamsqr1.dat')
s_ch4 = get_matrix('./sk/ch4/oversqr.dat')
h_h2o = get_matrix('./sk/h2o/hamsqr1.dat')
s_h2o = get_matrix('./sk/h2o/oversqr.dat')
h_co2 = get_matrix('./sk/co2/hamsqr1.dat')
s_co2 = get_matrix('./sk/co2/oversqr.dat')
h_ch3cho = get_matrix('./sk/ch3cho/hamsqr1.dat')
s_ch3cho = get_matrix('./sk/ch3cho/oversqr.dat')
h_auo = get_matrix('./sk/auo/hamsqr1.dat')
s_auo = get_matrix('./sk/auo/oversqr.dat')
h_auau = get_matrix('./sk/auau/hamsqr1.dat')
s_auau = get_matrix('./sk/auau/oversqr.dat')
h_ch3s = get_matrix('./sk/ch3s/hamsqr1.dat')
s_ch3s = get_matrix('./sk/ch3s/oversqr.dat')
h_ph3 = get_matrix('./sk/ph3/hamsqr1.dat')
s_ph3 = get_matrix('./sk/ph3/oversqr.dat')
h_ch3ch2nh2 = get_matrix('./sk/ch3ch2nh2/hamsqr1.dat')
s_ch3ch2nh2 = get_matrix('./sk/ch3ch2nh2/oversqr.dat')
