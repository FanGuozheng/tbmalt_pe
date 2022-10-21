#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for DFTB based MD simulations."""
import numpy as np
import torch
from ase.build import molecule
from tbmalt import Md, Geometry
from tbmalt.common.parameter import params
from tbmalt.common.batch import pack
torch.set_default_dtype(torch.float64)
shell_dict = {1: [0], 6: [0, 1], 7: [0, 1], 8: [0, 1]}


def test_h2(device):
    """Test H2 molecule for the gradient and energy."""
    geometry = Geometry.from_ase_atoms([molecule('H2')], device=device)
    path_to_skf = './tests/unittests/data/slko/mio'

    # Test the first MD step gradient
    md = Md(geometry, path_to_skf, shell_dict, skf_type='skf')
    md(1)
    assert torch.allclose(md.grad, h2_grad1), 'H2 gradient error'
    assert torch.allclose(md.a, h2_a_ref), 'H2 a fialed'

    # Get initial velocity from DFTB+ and test energy
    md = Md(geometry, path_to_skf, shell_dict, init_velocity=init_h2_v.unsqueeze(0),
            skf_type='skf')
    step = 10
    md(step)
    assert torch.allclose(md.md_energy.squeeze(),
                          h2_e_md[: step]), 'H2 MD kinetic energy failed'
    assert torch.allclose(md.total_energy.squeeze(),
                          h2_e_tot[: step]), 'H2 total energy failed'


def test_h2o(device):
    """Test H2O molecule for the gradient and energy."""
    geometry = Geometry.from_ase_atoms([molecule('H2O')], device=device)
    path_to_skf = './data/slko/mio'

    # Test the first MD step gradient
#     md = Md(geometry, path_to_skf, shell_dict, skf_type='skf')
#     md(1)
#     # assert torch.allclose(md.grad, h2_grad1), 'H2 gradient error'
#     # assert torch.allclose(md.a, h2_a_ref), 'H2 a fialed'
#
#     # Get initial velocity from DFTB+ and test energy
#     # md = Md(geometry, path_to_skf, shell_dict, init_velocity=init_h2_v.unsqueeze(0),
#     #         skf_type='skf')
#     step = 400
#     md(step)
#     import matplotlib.pyplot as plt
#     plt.plot(np.arange(step), md.md_energy)
#     plt.show()
#     # assert torch.allclose(md.md_energy.squeeze(),
#     #                       h2_e_md[: step]), 'H2 MD kinetic energy failed'
#     # assert torch.allclose(md.total_energy.squeeze(),
#     #                       h2_e_tot[: step]), 'H2 total energy failed'
#
#
# test_h2o(torch.device('cpu'))

def test_h3(device):
    """Test H3 molecule, including the bounds of repulsive grads."""
    geometry = Geometry(torch.tensor([[1, 1, 1]]), torch.tensor([[
        [0, 0, 0.], [0., 0., -0.25], [0., 0., 0.25]]]), units='angstrom')
    path_to_skf = './tests/unittests/data/slko/mio'

    md = Md(geometry, path_to_skf, shell_dict, skf_type='skf')
    md(1)

    assert torch.allclose(md.grad, h3_grad1), 'H3 gradient error'


def test_ch4(device):
    """Test CH4 molecule, including the bounds of repulsive grads."""
    geometry = Geometry.from_ase_atoms([molecule('CH4')], device=device)
    path_to_skf = './tests/unittests/data/slko/mio'

    md = Md(geometry, path_to_skf, shell_dict, skf_type='skf')
    md(1)
    # print(md.grad_instance.ham0_grad)
    assert torch.allclose(md.grad, ch4_grad1), 'CH4 gradient error'

    md = Md(geometry, path_to_skf, shell_dict,
            init_velocity=init_ch4_v.unsqueeze(0), skf_type='skf')
    step = 10
    md(step)

    assert torch.allclose(md.md_energy.squeeze(), ch4_e_md[: step], atol=1E-5,
                          ), 'CH4 MD kinetic energy failed'
    assert torch.allclose(md.total_energy.squeeze(), ch4_e_tot[: step],
                          atol=2E-4,), 'CH4 total energy failed'


def test_c2h6(device):
    """Test C2H6 molecule."""
    # Test single MD loop gradient
    geometry = Geometry.from_ase_atoms([molecule('C2H6')], device=device)
    path_to_skf = './tests/unittests/data/slko/mio'

    md = Md(geometry, path_to_skf, shell_dict, skf_type='skf')
    md(1)
    assert torch.allclose(md.grad, c2h6_grad1), 'CH4 gradient error'

    # Test MD energy
    md1 = Md(geometry, path_to_skf, shell_dict,
             init_velocity=init_c2h6_v.unsqueeze(0), skf_type='skf')
    step = 1
    md1(step)
    assert torch.allclose(md1.md_energy.squeeze(),
                          c2h6_e_md[: step]), 'C2H6 MD kinetic energy failed'
    assert torch.allclose(md1.total_energy.squeeze(),
                          c2h6_e_tot[: step]), 'C2H6 total energy failed'


def test_batch(device):
    """Test batch MD calculations."""
    # Test single MD step gradient
    geometry = Geometry.from_ase_atoms(
        [molecule('H2'), molecule('CH4'), molecule('C2H6')],
        device=device)
    path_to_skf = './tests/unittests/data/slko/mio'

    md = Md(geometry, path_to_skf, shell_dict, skf_type='skf')
    md(1)

    ref_grad = pack([h2_grad1, ch4_grad1, c2h6_grad1])
    assert torch.allclose(md.grad, ref_grad), 'batch gradient error'

    # Test MD energy and total energy
    init_v = pack([init_h2_v, init_ch4_v, init_c2h6_v])
    md2 = Md(geometry, path_to_skf, shell_dict, init_velocity=init_v, skf_type='skf')
    step = 10
    md2(step)
    e_md = pack([h2_e_md, ch4_e_md, c2h6_e_md])
    e_tot = pack([h2_e_tot, ch4_e_tot, c2h6_e_tot])
    import matplotlib.pyplot as plt
    plt.plot(torch.linspace(1, 10, 10), md2.total_energy, label='TBMaLT')
    plt.plot(torch.linspace(1, 10, 10), e_tot.T[:step], '--', label='DFTB+')
    plt.xlabel('MD step')
    plt.ylabel('total energy (Hartree)')
    plt.legend()

    assert torch.allclose(md2.md_energy, e_md.T[:step], atol=2E-5), \
        'batch MD kinetic energy failed'
    assert torch.allclose(md2.total_energy, e_tot.T[:step],
                          atol=2E-4), 'batch total energy failed'


##################
# Reference Data #
##################
# H2 molecule data
h2_grad1 = torch.tensor([
    [0, 0, -4.0665631856486861E-3], [0, 0, 4.0665631856486861E-3]])
h2_e_md = torch.tensor([
    0.0012975234, 0.0012727599, 0.0012393308, 0.0011981875, 0.0011505490,
    0.0010978712, 0.0010418066, 0.0009841530, 0.0009267945, 0.0008716354])
h2_e_tot = torch.tensor([
    -0.6736372400, -0.6736371888, -0.6736371196, -0.6736370333, -0.6736369318,
    -0.6736368181, -0.6736366954, -0.6736365674, -0.6736364387, -0.6736363141])
h2_a_ref = torch.tensor([[0., 0., 2.2131298255514482E-006],
                         [0., 0., -2.2131298255514482E-006]])
init_h2_v = torch.tensor(
    [[-5.4591943710368860E-4, -2.1798803882286827E-4, -6.0049920319015621E-4],
     [5.4591943710368860E-4, 2.1798803882286827E-4, 6.0049920319015610E-4]])

h3_grad1 = torch.tensor([
    [0., 0., 0.], [0., 0., 6.1825619136819565], [0., 0., -6.1825619136819512]])

# CH4 molecule data
ch4_grad1 = torch.tensor([
    [0, 0, 0],
    [3.0331945478077188E-4, 3.0331945478079964E-4, 3.0331945478079964E-4],
    [-3.0331945478081351E-4, -3.0331945478082045E-4, 3.0331945478086902E-4],
    [3.0331945478086209E-4, -3.0331945478082045E-4, -3.0331945478069555E-4],
    [-3.0331945478076494E-4, 3.0331945478070249E-4, -3.0331945478078576E-4]])
ch4_e_md = torch.tensor([
   0.0051900937, 0.0051845439, 0.0051732595, 0.0051563786, 0.0051340968,
   0.0051066643, 0.0050743807, 0.0050375888, 0.0049966683, 0.0049520320])
ch4_e_tot = torch.tensor([
    -3.2204808084, -3.2204807850, -3.2204807573, -3.2204807263, -3.2204806926,
    -3.2204806560, -3.2204806175, -3.2204805769, -3.2204805348, -3.2204804921])
init_ch4_v = torch.tensor([
    [-2.0967524627378050E-005, -3.3229234913795451E-005, -1.1565539949090987E-4],
    [9.9411452237828856E-4, -2.3313491707218517E-4, 5.5985119169391301E-4],
    [8.1888809940804867E-005, 5.5076844236628063E-4, 6.2229363545024830E-4],
    [-9.5454634678102872E-4, -1.0030594724123700E-3, -2.8665692437965072E-4],
    [1.2836441404012039E-4, 1.0813417321526829E-3, 4.8250946616984236E-4]])


# C2H6 molecule data
c2h6_grad1 = torch.tensor([
     [0, -4.8452850118619627E-006, 1.6021786820794880E-2],
     [0, 4.8452850118442496E-006, -1.6021786820794737E-2],
     [0, -2.8213210815240114E-3, -1.4230142553055963E-4],
     [2.4446789376713352E-3, 1.4130813082728552E-3, -1.4218394161596981E-4],
     [-2.4446789376712658E-3, 1.4130813082728066E-3, -1.4218394161592818E-4],
     [0, 2.8213210815239143E-3, 1.4230142553057351E-4],
     [2.4446789376712658E-3, -1.4130813082727858E-3, 1.4218394161588654E-4],
     [-2.4446789376712519E-3, -1.4130813082727650E-3, 1.4218394161587961E-4]])
init_c2h6_v = torch.tensor([
    [-9.3819961686513034E-005, -1.2563125908114458E-4, -2.0789390010224576E-4],
    [2.2088251791994089E-4, 3.0828911290961620E-008, 1.3827180751249985E-4],
    [1.0874273548431725E-005, 4.6880165026282452E-4, 5.4324126507628294E-4],
    [-1.0440805482343004E-3, -1.1127910237073151E-3, -3.8195098610863435E-4],
    [5.8180332515898017E-005, 1.0088555520811288E-3, 4.0095934571561422E-4],
    [6.7803724541918415E-4, 1.0571252737631620E-3, 2.0031871718134812E-005],
    [-1.7296435012174269E-4, 8.3140977603883266E-4, 3.5349778963701710E-4],
    [-1.0439569733293249E-3, -7.5691197611726273E-4, -1.0625415508320825E-4]
    ])
c2h6_e_md = torch.tensor([
    0.0090826639, 0.0090861308, 0.0090574522, 0.0089971387, 0.0089060710,
    0.0087855082, 0.0086370760, 0.0084627369, 0.0082647674, 0.0080457330])

c2h6_e_tot = torch.tensor([
    -5.6920129241, -5.6920128984, -5.6920128426, -5.6920127683, -5.6920126723,
    -5.6920125520, -5.6920124151, -5.6920122588, -5.6920120843, -5.6920118940])

ch3co_grad1 = torch.tensor([
    [-7.5312478814988024E-3, -1.0593319686872174E-4, 6.8955258170078082E-017],
    [-9.9285879855210979E-3, -1.5293684680079841E-2, -7.8645315086667278E-017],
    [-3.0860036628260756E-4, 5.0252183712128085E-3, -1.2983321015513916E-017],
    [1.2975112637778410E-3, 8.0482737807857033E-4, -3.0829151966930701E-3],
    [1.2975112637776953E-3, 8.0482737807861197E-4, 3.0829151966930285E-3],
    [1.5173413705747096E-2, 8.7647447495785777E-3, 5.4315140709734755E-017]])
