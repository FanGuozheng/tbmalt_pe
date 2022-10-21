import torch
from tbmalt import Geometry, Dftb2, Md
from tbmalt.common.parameter import params
from tbmalt.physics.force import DftbGradient
torch.set_default_dtype(torch.float64)
torch.set_printoptions(15)
shell_dict = {1: [0], 6: [0, 1], 7: [0, 1], 8: [0, 1]}
path_to_skf = './tests/unittests/data/slko/mio'

spin_params = {1: torch.tensor([[-0.072]]),
               6: torch.tensor([[-0.031, - 0.025], [-0.025, -0.023]]),
               8: torch.tensor([[-0.031, - 0.025], [-0.025, -0.023]]),
               31: torch.tensor([[-0.0173, -0.0128, -0.0030],
                                 [-0.0128, -0.0133, -0.0011],
                                 [-0.0030, -0.0011, -0.0213]]),
               33: torch.tensor([[-0.0179, -0.0138],
                                 [-0.0138, -0.0136]])}


def test_h2o(device):
    """Test H2O molecule for the energy."""
    geometry = Geometry(atomic_numbers=torch.tensor([[8, 1, 1]]),
                        positions=torch.tensor([[
                            [0, -0.075791844, 0.1],
                            [0.866811829, 0.601435779, 0],
                            [-0.866811829, 0.601435779, 0]]]) * 1.8897259886,
                        )

    # 1. with spin, no PBC
    dftb2 = Dftb2(geometry, shell_dict=shell_dict,
                  path_to_skf=path_to_skf,
                  skf_type='skf', repulsive=True,
                  spin_params=spin_params,
                  )
    dftb2()
    assert torch.max(abs(dftb2.charge - torch.tensor([
        6.55045046, 0.72477477, 0.72477477]))) < 1E-8,\
        'Tolerance error: H2O test, with spin, no PBC'

    # 2. with spin, with unpaired electron, no PBC
    spin_params.update({'unpaired_electrons': torch.tensor([1.])})
    dftb2 = Dftb2(geometry, shell_dict=shell_dict,
                  path_to_skf=path_to_skf,
                  skf_type='skf', repulsive=True,
                  spin_params=spin_params,
                  )
    dftb2()
    assert torch.max(abs(dftb2.charge - torch.tensor([
        6.34125200, 0.82937398, 0.82937398]))) < 1E-7,\
        'Tolerance error: H2O test, with spin, with unpaired electron'


def test_fe4():
    pass
