#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 15:30:05 2021

@author: gz_fan
"""
import torch
from torch import Tensor
import torch.nn.functional as F
from tbmalt.common.batch import pack


def fermi(eigenvalue: Tensor, nelectron: Tensor, kT=0.0, spin=None):
    """Fermi-Dirac distributions without smearing.
    Arguments:
        eigenvalue: Eigen-energies.
        nelectron: number of electrons.
        kT: temperature.
    Returns
        occ: occupancies of electrons
    """
    # make sure each system has at least one electron
    assert False not in torch.ge(nelectron, 1)
    if spin is None:
        # the number of full occupied state
        electron_pair = torch.true_divide(nelectron.clone().detach(), 2).int()
        # the left single electron
        electron_single = (nelectron.clone().detach() % 2).unsqueeze(1)

        # zero temperature
        if kT != 0:
            raise NotImplementedError('not implement smearing method.')

        # occupied state for batch, if full occupied, occupied will be 2
        # with unpaired electron, return 1
        occ_ = pack([
            torch.cat((torch.ones(electron_pair[i]) * 2, electron_single[i]), 0)
            for i in range(nelectron.shape[0])])

        # pad the rest unoccupied states with 0
        occ = F.pad(input=occ_, pad=(
            0, eigenvalue.shape[-1] - occ_.shape[-1]), value=0)

        # all occupied states (include full and not full occupied)
        nocc = (nelectron.clone().detach() / 2.).ceil()

        return occ, nocc
    else:
        electron_pair = torch.true_divide(nelectron[..., spin].clone().detach(), 1).int()
        # the left single electron
        electron_single = (nelectron[..., spin].clone().detach() % 1).unsqueeze(1)

        # zero temperature
        if kT != 0:
            raise NotImplementedError('not implement smearing method.')

        # occupied state for batch, if full occupied, occupied will be 2
        # with unpaired electron, return 1
        occ_ = pack([
            torch.cat((torch.ones(electron_pair[i]), electron_single[i]), 0)
            for i in range(nelectron.shape[0])])

        # pad the rest unoccupied states with 0
        occ = F.pad(input=occ_, pad=(
            0, eigenvalue.shape[-1] - occ_.shape[-1]), value=0)

        # all occupied states (include full and not full occupied)
        nocc = (nelectron.clone().detach() / 2.).ceil()

        return occ, nocc
