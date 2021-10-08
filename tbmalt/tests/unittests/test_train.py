"""Perform tests on functions which read SKF or SK transformations.

Reference is from DFTB+."""
import os
import re
import numpy as np
import torch
from ase.build import molecule as molecule_database
from tbmalt.common.structures.system import System
from tbmalt.tb.sk import SKT
from tbmalt.common.parameter import Parameter
from tbmalt.io.loadskf import IntegralGenerator
from tbmalt.io.loadhdf import LoadHdf
from tbmalt.ml.train import CompressionRadii, Charge, Integal
torch.set_default_dtype(torch.float64)
torch.set_printoptions(15)


# copy from public directory
os.system('cp -r /home/gz_fan/Public/tbmalt/slko .')
os.system('cp -r /home/gz_fan/Public/tbmalt/sk .')
os.system('cp -r /home/gz_fan/Public/tbmalt/reference.hdf .')


def test_read_compr(device):
    """Test SKF data with various compression radii."""
    properties = ['dipole', 'charge']
    # sys = System.from_ase_atoms([molecule_database('CH4')])
    compression_radii_grid = torch.tensor([
        01.00, 01.50, 02.00, 02.50, 03.00, 03.50, 04.00,
        04.50, 05.00, 05.50, 06.00, 07.00, 08.00, 09.00, 10.00])

    numbers, positions, reference = LoadHdf.load_reference(
        './reference.hdf', 3, properties)
    sys = System(numbers, positions)
    variable = torch.ones(sys.positions.shape[0], sys.positions.shape[1]) * 3
    params = Parameter()
    # sk = IntegralGenerator.from_dir(
    #     './slko/compr', sys, repulsive=True,
    #     sk_type='compression_radii', homo=False, interpolation='bicubic_interpolation',
    #     compression_radii_grid=compression_radii_grid)
    sk = IntegralGenerator.from_dir(
        './slko/skf.hdf.compr', sys, repulsive=False,
        sk_type='h5py', homo=False, interpolation='bicubic_interpolation',
        compression_radii_grid=compression_radii_grid)

    train = CompressionRadii(sys, reference, variable, params, sk)
    train(properties)


def test_read_integral(device):
    """Test SKF data with various compression radii."""
    import time
    time0 = time.time()
    properties = ['dipole', 'charge']
    numbers, positions, reference = LoadHdf.load_reference(
        './reference.hdf', 3, properties)
    sys = System(numbers, positions)
    params = Parameter()
    # sk = IntegralGenerator.from_dir(
    #     './slko/mio-1-1', sys, repulsive=False, interpolation='spline', with_variable=True)
    sk = IntegralGenerator.from_dir(
        './slko/skf.hdf.mio', sys, repulsive=False, interpolation='spline',
        sk_type='h5py', with_variable=True)
    variable = sk.sktable_dict['variable']

    train = Integal(sys, reference, variable, params, sk)
    train(properties)
    time1 = time.time()
    print('time', time1 - time0)


# def test_read_compr_batch(device):
#     """Test SKF data with various compression radii."""
#     molecule = System.from_ase_atoms([molecule_database('CH4'),
#                                       molecule_database('NH3'),
#                                       molecule_database('C2H6'),
#                                       molecule_database('CH3CH2O')])
#     compression_radii_grid = torch.tensor([
#         01.00, 01.50, 02.00, 02.50, 03.00, 03.50, 04.00,
#         04.50, 05.00, 05.50, 06.00, 07.00, 08.00, 09.00, 10.00])
#     compression_r = torch.tensor([[3., 3.1, 3.2, 3.3, 3.4, 0, 0, 0],
#                                   [3., 3., 3., 3., 0., 0., 0., 0],
#                                   [3., 3., 3., 3., 3., 3.5, 3.5, 4.],
#                                   [3., 3., 3., 3., 3., 3.5, 3.5, 4.]])
#     sk = IntegralGenerator.from_dir(
#         './slko/compr', molecule, repulsive=True,
#         sk_type='compression_radii', homo=False, interpolation='bicubic_interpolation',
#         compression_radii_grid=compression_radii_grid)

#     skt = SKT(molecule, sk, compression_radii=compression_r, fix_onsite=True,
#               fix_U=True)


# def test_xlbomd():
#     """Test."""
#     # -> define all input parameters
#     properties = ['dipole', 'charge']
#     reference_size = 1000
#     params = Parameter()

#     # load the the generated dataset
#     numbers, positions, data_nonscc = LoadHdf.load_reference(
#         'nonscc.hdf', reference_size, properties)
#     # load the the generated dataset
#     _, _, data = LoadHdf.load_reference('scc.hdf', reference_size, properties)

#     sys = System(numbers, positions)

#     molecule = System(numbers, positions)
#     sk = IntegralGenerator.from_dir('./slko/mio-1-1', molecule)

#     variable = _get_charge(sys)
#     train = CompressionRadii(sys, data, variable, params, sk)
#     train(properties)


def _get_charge(system):
    pass


# def test_sk_ase_batch(device):
#     """Test SK transformation values of batch ASE molecules."""
#     molecule = System.from_ase_atoms([
#         molecule_database('H2'), molecule_database('N2'),
#         molecule_database('CH4'), molecule_database('NH3'),
#         molecule_database('H2O'), molecule_database('CO2'),
#         molecule_database('CH3CHO')])
#     sktable = IntegralGenerator.from_dir('./slko/auorg-1-1/', molecule)
#     skt = SKT(molecule, sktable)
#     assert torch.max(abs(skt.H[0][:h_h2.shape[0], :h_h2.shape[1]] - h_h2)
#                      ) < 1E-14, 'Tolerance check'
#     assert torch.max(abs(skt.S[0][:s_h2.shape[0], :s_h2.shape[1]] - s_h2)
#                      ) < 1E-14, 'Tolerance check'
#     assert torch.max(abs(skt.H[1][:h_n2.shape[0], :h_n2.shape[1]] - h_n2)
#                      ) < 1E-14, 'Tolerance check'
#     assert torch.max(abs(skt.S[1][:s_n2.shape[0], :s_n2.shape[1]] - s_n2)
#                      ) < 1E-14, 'Tolerance check'
#     assert torch.max(abs(skt.H[2][:h_ch4.shape[0], :h_ch4.shape[1]] - h_ch4)
#                      ) < 1E-14, 'Tolerance check'
#     assert torch.max(abs(skt.S[2][:s_ch4.shape[0], :s_ch4.shape[1]] - s_ch4)
#                      ) < 1E-14, 'Tolerance check'
#     assert torch.max(abs(skt.H[3][:h_nh3.shape[0], :h_nh3.shape[1]] - h_nh3)
#                      ) < 1E-14, 'Tolerance check'
#     assert torch.max(abs(skt.S[3][:s_nh3.shape[0], :s_nh3.shape[1]] - s_nh3)
#                      ) < 1E-14, 'Tolerance check'
#     assert torch.max(abs(skt.H[4][:h_h2o.shape[0], :h_h2o.shape[1]] - h_h2o)
#                      ) < 1E-14, 'Tolerance check'
#     assert torch.max(abs(skt.S[4][:s_h2o.shape[0], :s_h2o.shape[1]] - s_h2o)
#                      ) < 1E-14, 'Tolerance check'
#     assert torch.max(abs(skt.H[5][:h_co2.shape[0], :h_co2.shape[1]] - h_co2)
#                      ) < 1E-14, 'Tolerance check'
#     assert torch.max(abs(skt.S[5][:s_co2.shape[0], :s_co2.shape[1]] - s_co2)
#                      ) < 1E-14, 'Tolerance check'
#     assert torch.max(abs(skt.H[6][:h_ch3cho.shape[0], :h_ch3cho.shape[1]]
#                          - h_ch3cho)) < 1E-14, 'Tolerance check'
#     assert torch.max(abs(skt.S[6][:s_ch3cho.shape[0], :s_ch3cho.shape[1]]
#                          - s_ch3cho)) < 1E-14, 'Tolerance check'


# def test_sk_interpolation(device):
#     """Test SK interpolation with random distance."""
#     sktable = IntegralGenerator.from_dir('slko/mio-1-1/', elements=['C', 'H'])
#     ch_00 = sktable(torch.tensor([2.0591670220427729]), torch.tensor([6, 1]),
#                     torch.tensor([0, 0]), hs_type='H')
#     assert abs(ch_00 - -0.33103366295265063) < 1E-14, 'distance < leng'
#     cc_01 = sktable(torch.tensor([1.6365507123010095]), torch.tensor([6, 6]),
#                     torch.tensor([0, 1]), hs_type='H')
#     assert abs(cc_01 - 0.45308093203471983) < 1E-14, 'distance < leng'
#     hc_00 = sktable(torch.tensor([10.473924558726461]), torch.tensor([1, 6]),
#                     torch.tensor([0, 0]), hs_type='H')

#     # second derivative: y2 + y0 - 2.0 * y1 is smaller than 1E-14, but
#     # delta ** 2 will result in the difference > 1E-14, seems not bug
#     assert abs(hc_00 - 8.066100641653708e-06) < 1E-11, 'distance < leng'
#     hc_00_b = sktable(torch.tensor([10.473924558726461, 10.584152656903393]),
#                       torch.tensor([1, 6]), torch.tensor([0, 0]), hs_type='H')
#     assert torch.max(abs(hc_00_b - torch.tensor(
#         [[8.066100641653708e-06], [4.4433154052067092E-006]]))) < 1E-11


def get_matrix(filename):
    """Read DFTB+ hamsqr1.dat and oversqr.dat."""
    text = ''.join(open(filename, 'r').readlines())
    string = re.search('(?<=MATRIX\n).+(?=\n)', text, flags=re.DOTALL).group(0)
    out = np.array([[float(i) for i in row.split()]
                    for row in string.split('\n')])
    return torch.from_numpy(out)


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
