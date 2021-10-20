#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 22:50:40 2021

@author: gz_fan
"""

params = {
    'dftb':

        {
        'dftb': 'dftb2',

        'max_l':
            {1: 0, 6: 1, 7: 1, 8: 1},

        'path_to_skf': './slko/auorg-1-1',

        'dftb2':
            {'maxiter': 60,
             'gamma_type': 'exponential'
             },

        'mixer':
            {
            'mixer': 'Anderson',
            'mix_param': 0.2,  # -> mix factor
            'generations': 3,  # -> how many generations for mixing
            'tolerance': 1E-8,  # tolerance for convergence
            'sigma': 0.1,  # -> Gaussian smearing
            }

        },

    'ml': {
        'lr': 0.1,
        'min_steps': 60,  # -> min training steps
        'max_steps': 120,  # -> max training steps
        'loss_function': 'MSELoss',  # MSELoss, L1Loss
        'optimizer': 'Adam',  # SCG, Adam
        'compression_radii_min': 1.5,
        'compression_radii_max': 9.0,
        'method': 'linear',
        'tolerance': 1E-5
        }

    }
