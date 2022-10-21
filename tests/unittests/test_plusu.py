import torch
from tbmalt import Geometry, Dftb1, Dftb2, Geometry
from tbmalt.common.parameter import params
from tbmalt.physics.force import DftbGradient
torch.set_default_dtype(torch.float64)
shell_dict = {1: [0], 6: [0, 1], 7: [0, 1], 8: [0, 1], 31: [0, 1, 2], 33: [0, 1]}

# Just for test, do not trust!!!
spin_params = {1: torch.tensor([[-0.072]]),
               6: torch.tensor([[-0.031, - 0.025], [-0.025, -0.023]]),
               8: torch.tensor([[-0.031, - 0.025], [-0.025, -0.023]]),
               31: torch.tensor([[-0.0173, -0.0128, -0.0030],
                                 [-0.0128, -0.0133, -0.0011],
                                 [-0.0030, -0.0011, -0.0213]]),
               33: torch.tensor([[-0.0179, -0.0138],
                                 [-0.0138, -0.0136]])}
# Just for test, do not trust!!!
plusu_params = {(6, 0): 0.1, (6, 1): 0.2,
                (8, 0): 0.1, (8, 1): 0.2, 'type': 'fll'}
path_to_skf = './tests/unittests/data/slko/mio'
# path_to_skf = '../../tests/unittests/data/slko/mio'


def test_h2o(device):
    """Test H2O molecule for the energy."""

    geometry = Geometry(atomic_numbers=torch.tensor([[8, 1, 1]]),
                        positions=torch.tensor([
                            [[0, -0.075791844, 0.1],
                             [0.866811829, 0.601435779, 0],
                             [-0.866811829, 0.601435779, 0]]]),
                        units='angstrom')
    geopbc = Geometry(atomic_numbers=torch.tensor([[8, 1, 1]]),
                      positions=torch.tensor([
                            [[0, -0.075791844, 0.1],
                             [0.866811829, 0.601435779, 0],
                             [-0.866811829, 0.601435779, 0]]]),
                      cell=torch.eye(3).unsqueeze(0) * 5.0,
                      units='angstrom')

    # 1.1 DFTB1+U FLL, molecule
    plusu_params.update({'type': 'fll'})
    dftb = Dftb1(geometry, shell_dict=shell_dict, path_to_skf=path_to_skf,
                 skf_type='skf', repulsive=True, plusu_params=plusu_params,)
    # dftb()
    ref = torch.tensor([6.79833930, 0.60083035, 0.60083035])

    # 1.2 DFTB1+U FLL, solid
    plusu_params.update({'type': 'fll'})
    dftb = Dftb1(geopbc, shell_dict=shell_dict, path_to_skf=path_to_skf,
                 skf_type='skf', repulsive=True, plusu_params=plusu_params,)
    # dftb()

    # 1.3 DFTB1+U pSIC, molecule
    plusu_params.update({'type': 'psic'})
    dftb = Dftb1(geometry, shell_dict=shell_dict, path_to_skf=path_to_skf,
                 skf_type='skf', repulsive=True, plusu_params=plusu_params,)
    # dftb()
    ref = torch.tensor([6.79833930, 0.60083035, 0.60083035])
    # assert torch.max(abs(dftb.charge - torch.tensor([
    #     6.6488640745375998, 0.67556796273120079, 0.67556796273120079
    # ]))) < 1E-10

    # 2.1 DFTB2+U FLL, molecule
    plusu_params.update({'type': 'fll'})
    dftb = Dftb2(geometry, shell_dict=shell_dict, path_to_skf=path_to_skf,
                 skf_type='skf', repulsive=True, plusu_params=plusu_params,)
    dftb()
    ref = torch.tensor([6.63589852, 0.68205074, 0.68205074])
    assert torch.max(abs(dftb.charge - ref)) < 1E-8

    # 2.2 DFTB2+U FLL, solid
    dftb = Dftb2(geopbc, shell_dict=shell_dict, path_to_skf=path_to_skf,
                 skf_type='skf', repulsive=True, plusu_params=plusu_params,)
    dftb()
    ref = torch.tensor([6.64441001, 0.67779500, 0.67779500])
    # print(dftb.charge)
    # assert torch.max(abs(dftb.charge - ref)) < 1E-8

    # 2.3 DFTB2+U FLL, with spin
    plusu_params.update({'type': 'fll'})
    dftb2 = Dftb2(geometry, shell_dict=shell_dict,
                  path_to_skf=path_to_skf,
                  skf_type='skf', repulsive=True,
                  plusu_params=plusu_params,
                  spin_params=spin_params,
                  )
    dftb2()
    ref = torch.tensor([6.63589852, 0.68205074, 0.68205074])
    print('dftb2.charge - ref', dftb2.charge - ref)
    assert torch.max(abs(dftb2.charge - ref)) < 1E-8, \
        'Tolerance error: with FLL U, with spin, no PBC'

    # 2.4 DFTB2+U FLL, with spin, with unpaired electrons
    plusu_params.update({'type': 'fll'})
    spin_params.update({'unpaired_electrons': torch.tensor([1.])})
    dftb2 = Dftb2(geometry, shell_dict=shell_dict,
                  path_to_skf=path_to_skf,
                  skf_type='skf', repulsive=True,
                  plusu_params=plusu_params,
                  spin_params=spin_params,
                  )
    dftb2()
    assert torch.max(abs(dftb2.charge - torch.tensor([
        6.44309709, 0.77845144, 0.77845144]))) < 1E-7, \
        'Tolerance error: with FLL U, with spin, no PBC'

    # 2.5 DFTB2+U pSIC, molecule
    plusu_params.update({'type': 'psic'})
    dftb = Dftb2(geometry, shell_dict=shell_dict, path_to_skf=path_to_skf,
                 skf_type='skf', repulsive=True, plusu_params=plusu_params,)
    dftb()
    ref = torch.tensor([6.87853938, 0.56073031, 0.56073031])
    assert torch.max(abs(dftb.charge - ref)) < 1E-8


def test_gaas(device):
    geometry = Geometry(
        atomic_numbers=torch.tensor([[31, 33]]),
        positions=torch.tensor([
            [[0, 0., 0.], [0.13567730000E+01, 0.13567730000E+01, 0.13567730000E+01]]]),
        cell=torch.tensor([[[0.27135460000E+01, 0.27135460000E+01, 0.00000000000E+00],
                            [0.00000000000E+00, 0.27135460000E+01, 0.27135460000E+01],
                            [0.27135460000E+01, 0.00000000000E+00, 0.27135460000E+01]]]),
        units='angstrom')
    path_to_skf = '/Users/gz_fan/Downloads/software/dftb/dftbplus/dftbplus/work/slko/hyb-0-2/'
    dftb2 = Dftb2(geometry, shell_dict=shell_dict,
                  path_to_skf=path_to_skf,
                  skf_type='skf', repulsive=True,
                  spin_params=spin_params, u_params=plusu_params,
                  )
    # dftb2()
    # print(dftb2.epsilon)


# test_h2o(torch.device('cpu'))
