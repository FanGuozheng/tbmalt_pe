#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
from ase.build import molecule
from tbmalt import Md, Geometry, Basis
from tbmalt.physics.force import DftbGradient

torch.set_default_dtype(torch.float64)
shell_dict = {1: [0], 6: [0, 1], 7: [0, 1], 8: [0, 1]}


def test_h2(device):
    """Test H2 molecule for the gradient and energy."""
    geometry = Geometry.from_ase_atoms([molecule('H2')], device=device)
    path_to_skf = './tests/unittests/data/slko/mio'

    # # Test the first MD step gradient
    # md = Md(geometry, path_to_skf, shell_dict, skf_type='skf')
    # md(1)
    # assert torch.allclose(md.grad, h2_grad1), 'H2 gradient error'
    # assert torch.allclose(md.a, h2_a_ref), 'H2 a fialed'
    #
    # # Get initial velocity from DFTB+ and test energy
    # md = Md(geometry, path_to_skf, shell_dict, init_velocity=init_h2_v.unsqueeze(0),
    #         skf_type='skf')
    # step = 10
    # md(step)
    # assert torch.allclose(md.md_energy.squeeze(),
    #                       h2_e_md[: step]), 'H2 MD kinetic energy failed'
    # assert torch.allclose(md.total_energy.squeeze(),
    #                       h2_e_tot[: step]), 'H2 total energy failed'


def test_h2o(device):
    """Test H2 molecule for the gradient and energy."""
    geometry = Geometry.from_ase_atoms([molecule('H2')], device=device)
    basis = Basis(geometry.atomic_numbers, shell_dict)
    path_to_skf = './tests/unittests/data/slko/mio'

    # # 1.1 Test the band structure force
    # assert torch.allclose(md.grad, h2_grad1), 'H2 gradient error'
    # assert torch.allclose(md.a, h2_a_ref), 'H2 a fialed'
    #
    # # 1.2 Test the second order force
    # md = Md(geometry, path_to_skf, shell_dict, init_velocity=init_h2_v.unsqueeze(0),
    #         skf_type='skf')
    # step = 10
    # md(step)
    # assert torch.allclose(md.md_energy.squeeze(),
    #                       h2_e_md[: step]), 'H2 MD kinetic energy failed'
    # assert torch.allclose(md.total_energy.squeeze(),
    #                       h2_e_tot[: step]), 'H2 total energy failed'
    #
    # # 1.3 Test the Third order force
    #
    #
    # # 2.1 Test the band structure force of water in box
    #
    #
    # # 2.2 Test the second order force of water in box
    #
    #
    # # 2.3 Test the third order force of water in box


test_h2o(torch.device('cpu'))
