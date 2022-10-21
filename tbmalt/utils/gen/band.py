#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 21:15:46 2021

@author: gz_fan
"""
import torch
from torch import Tensor
import numpy as np


def read_band(file: str) -> Tensor:
    """Read `band.out` file from DFTB+."""
    try:
        with open(file) as f:
            text = f.readlines()
            band, number = [], []
            for ii in text:
                if 'KPT' not in ii and len(ii) > 1:
                    ii = ii.split()
                    number.append(int(ii[0]))
                    band.append(float(ii[1]))

            # return shape as: [n_kpoints, n_eigenstates]
            band = torch.from_numpy(np.asarray(band)).reshape(-1, max(number))
    except:
        raise IOError(f'can not read {file}')

    return band
