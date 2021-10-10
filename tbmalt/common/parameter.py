# """Parameters."""
# from typing import Tuple, Union, Optional


# class Parameter:
#     """DFTB parameters."""

#     def __init__(self, dftb_params=None, constant_params=None, ml_params=None):
#         self.dftb_params = self.dftb_params(dftb_params)
#         self.constant_params = self.constant_params(constant_params)
#         if ml_params is not None:
#             self.ml_params = self.ml_params(ml_params)

#     def ml_params(self, ml_params: Union[bool, dict]) -> dict:
#         """Return machine learning parameters."""
#         _ml_params = {
#             'lr': 0.1,
#             'steps': 3,  # -> training steps
#             'loss_function': 'MSELoss',  # MSELoss, L1Loss
#             'optimizer': 'Adam',  # SCG, Adam
#             'compression_radii_min': 1.25,
#             'compression_radii_max': 9.0,
#             'ml_method': 'linear'
#             }

#         if type(ml_params) is dict:
#             _ml_params.update(ml_params)

#         return _ml_params

#     def constant_params(self, constant_params) -> dict:
#         """Return constant parameter."""
#         _constant_params = {'bohr': 0.529177249}

#         if constant_params is not None:
#             if type(constant_params) is dict:
#                 _constant_params.update(constant_params)
#             else:
#                 raise TypeError('constant_params should be None or dict')

#         return _constant_params

#     def dftb_params(self, dftb_params) -> dict:
#         """Parameter for DFTB."""
#         _dftb_params = {
#             'mix': 'Anderson',  # -> Anderson, Simple
#             'mix_param': 0.2,  # -> mix factor
#             'generations': 3,  # -> how many generations for mixing
#             'tolerance': 1E-10,  # tolerance for convergence
#             'scc': 'scc',  # 'scc', 'nonscc', 'xlbomd'
#             'maxiter': 60,  # -> max SCC loop
#             'sigma': 0.1  # -> Gaussian smearing
#             }

#         if dftb_params is not None:
#             print('dftb_params', type(dftb_params))
#             if type(dftb_params) is dict:
#                 _dftb_params.update(dftb_params)
#             else:
#                 raise TypeError('dftb_params should be None or dict')

#         return _dftb_params

#     @classmethod
#     def get_ml_params(cls):
#         """Return machine learning parameters."""
#         return {'lr': 0.1,

#                 # training steps
#                 'steps': 3,

#                 # get loss function type
#                 'loss_function': 'MSELoss',

#                 # get optimizer
#                 'optimizer': 'SCG'}
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
