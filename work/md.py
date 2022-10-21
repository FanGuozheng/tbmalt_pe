
import torch
from ase.build import molecule
import matplotlib.pyplot as plt

from tbmalt.physics.md import Md
from tbmalt import Geometry, Dftb2, Dftb1

torch.set_default_dtype(torch.float64)

shell_dict = {1: [0], 6: [0, 1], 7: [0, 1], 8: [0, 1], 14: [0, 1]}
init_ch4_v = torch.tensor([
    [-2.0967524627378050E-005, -3.3229234913795451E-005, -1.1565539949090987E-4],
    [9.9411452237828856E-4, -2.3313491707218517E-4, 5.5985119169391301E-4],
    [8.1888809940804867E-005, 5.5076844236628063E-4, 6.2229363545024830E-4],
    [-9.5454634678102872E-4, -1.0030594724123700E-3, -2.8665692437965072E-4],
    [1.2836441404012039E-4, 1.0813417321526829E-3, 4.8250946616984236E-4]])

path_to_skf = ''


def md(device):
    """Test CH4 molecule, including the bounds of repulsive grads."""
    geometry = Geometry.from_ase_atoms([molecule('CH4')], device=device)
    path_to_skf = '../unittests/data/slko/mio'

    md = Md(geometry, path_to_skf, shell_dict, skf_type='skf')
    md(1)
    # print(md.grad_instance.ham0_grad)
    # assert torch.allclose(md.grad, ch4_grad1), 'CH4 gradient error'

    md = Md(geometry, path_to_skf, shell_dict,
            init_velocity=init_ch4_v.unsqueeze(0), skf_type='skf')
    step = 10
    md(step)

    # assert torch.allclose(md.md_energy.squeeze(), ch4_e_md[: step], atol=1E-5,
    #                       ), 'CH4 MD kinetic energy failed'
    # assert torch.allclose(md.total_energy.squeeze(), ch4_e_tot[: step],
    #                       atol=2E-4,), 'CH4 total energy failed'


def test_si():
    geometry = Geometry(
        torch.tensor([[14, 14]]),
        torch.tensor([[[0., 0.,  0.], [1.356773, 1.356773, 1.356773]]]),
        cell=torch.tensor([[
            [2.713546, 2.713546, 0.0], [0.0, 2.713546, 2.713546],
            [2.713546, 0.0, 2.713546]]]),
        units='angstrom')

    klines = torch.tensor([[0.5, 0.5, -0.5, 0, 0, 0, 11],
                           [0, 0, 0, 0, 0, 0.5, 11],
                           [0, 0, 0.5, 0.25, 0.25, 0.25, 11]])
    path_to_skf = '/Users/gz_fan/Documents/ML/tbmalt/tbmalt/tests/unittests/data/slko/'
    dftb2 = Dftb2(geometry, shell_dict=shell_dict,
                  path_to_skf=path_to_skf, skf_type='skf', klines=klines)
    dftb2()
    plt.plot(torch.arange(len(dftb2.eigenvalue)), dftb2.eigenvalue.squeeze())
    plt.ylim(-5, 5)
    plt.show()


if __name__ == '__main__':
    # md(torch.device('cpu'))
    test_si()
