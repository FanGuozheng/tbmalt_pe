#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test repulsive potentials."""
import torch
from tbmalt import Geometry, Dftb1, Dftb2
from tbmalt.common.batch import pack

torch.set_default_dtype(torch.float64)
shell_dict = {1: [0], 6: [0, 1], 7: [0, 1], 8: [0, 1],
              14: [0, 1], 22: [0, 1, 2]}
h2o = Geometry(atomic_numbers=torch.tensor([[8, 1, 1]]),
               positions=torch.tensor([
                   [[0, -0.075791844, 0.1],
                    [0.866811829, 0.601435779, 0],
                    [-0.866811829, 0.601435779, 0]]]),
               units='angstrom'
               )
h2opbc = Geometry(atomic_numbers=torch.tensor([[8, 1, 1]]),
                  positions=torch.tensor([
                      [[0, -0.075791844, 0.1],
                       [0.866811829, 0.601435779, 0],
                       [-0.866811829, 0.601435779, 0]]]),
                  cell=torch.eye(3).unsqueeze(0) * 5.0,
                  units='angstrom'
                  )

si = Geometry(
    torch.tensor([[14, 14]]),
    torch.tensor([[[0., 0.,  0.], [1.356773, 1.356773, 1.356773]]]),
    cell=torch.tensor([[
        [2.713546, 2.713546, 0.0], [0.0, 2.713546, 2.713546],
        [2.713546, 0.0, 2.713546]]]),
    units='angstrom')

pos = pack([
    torch.tensor([[0., 0., 0.], [1.356773, 1.356773, 1.356773]]),
    torch.tensor([
        [0., 0., 0.], [1.356773, 1.356773, 1.356773],
        [2.713546, 2.713546, 0.], [4.070319, 4.070319, 1.356773],
        [2.713546,  0., 2.713546], [4.070319, 1.356773, 4.070319],
        [0., 2.713546, 2.713546], [1.356773, 4.070319, 4.070319]])])
latt = torch.tensor([
    [[2.713546, 2.713546, 0.], [0., 2.713546, 2.713546], [2.713546, 0., 2.713546]],
    [[5.427092, 0., 0.], [0., 5.427092, 0.], [0., 0., 5.427092]]])

batch = Geometry(
    torch.tensor([[14, 14, 0, 0, 0, 0, 0, 0], [14, 14, 14, 14, 14, 14, 14, 14]]),
    pos, cell=latt, units='angstrom')


def test_h2o(device):
    """Test H2O DFTB from ase input."""
    path_to_skf = './tests/unittests/data/slko/mio'
    kpoints = torch.tensor([1, 1, 1])

    dftb2 = Dftb2(h2o, shell_dict=shell_dict, repulsive=True,
                  path_to_skf=path_to_skf, skf_type='skf')
    dftb2()
    assert torch.max(abs(dftb2.repulsive_energy - torch.tensor(
        [0.0194647763]))) < 1E-8, 'Water repulsive potential test error'

    dftb2 = Dftb2(h2opbc, shell_dict=shell_dict, repulsive=True,
                  path_to_skf=path_to_skf, skf_type='skf', kpoints=kpoints)
    dftb2()
    assert torch.max(abs(dftb2.repulsive_energy - torch.tensor(
        [0.0194647763]))) < 1E-8, 'Water repulsive potential test error'


def test_si(device):
    """Test H2O DFTB from ase input."""
    path_to_skf = './tests/unittests/data/slko/'
    dftb2 = Dftb2(si, shell_dict=shell_dict, repulsive=True,
                  path_to_skf=path_to_skf, skf_type='skf',)
    dftb2()
    assert torch.max(abs(dftb2.repulsive_energy - torch.tensor(
        [0.0025222404]))) < 1E-8, 'Si repulsive potential test error'

    dftb = Dftb1(batch, shell_dict=shell_dict, repulsive=True,
                 path_to_skf=path_to_skf, skf_type='skf')
    dftb()
    assert torch.max(abs(dftb.repulsive_energy - torch.tensor(
        [0.0025222404, 0.0100889616]))) < 1E-8, 'Si repulsive potential test error'

    dftb2 = Dftb2(batch, shell_dict=shell_dict, repulsive=True,
                  path_to_skf=path_to_skf, skf_type='skf')
    dftb2()
    assert torch.max(abs(dftb2.repulsive_energy - torch.tensor(
        [0.0025222404, 0.0100889616]))) < 1E-8, 'Si repulsive potential test error'
