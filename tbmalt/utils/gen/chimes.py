"""Analysis of forces.."""
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
from datetime import datetime

import torch
import numpy as np
from scipy.signal import find_peaks

from tbmalt.data.elements import chemical_masses
from tbmalt.structures.geometry import Geometry, batch_chemical_symbols
from tbmalt.data.units import _Bohr__AA

def gen_chimes_xyz(file, positions, force, labels, cells, n_atoms, mode='a'):
    labels = np.expand_dims(labels, -1)
    data = np.concatenate([labels, positions, force], -1)

    with open(file, mode) as f:
        for idata, inat, cell in zip(data, n_atoms, cells):
            np.savetxt(f, np.array([inat]), fmt='%i')
            np.savetxt(f, np.expand_dims(cell.diagonal(), 0))
            np.savetxt(f, idata, fmt="%s")


def write_fm_setup(file, TRJFILE: str, geometry: Geometry,
                   three_body: bool = False, **kwargs):

    num = geometry.atomic_numbers
    assert num.dim() == 2, 'only support batch system'

    uan = geometry.unique_atomic_numbers()
    n1 = num.unsqueeze(-1).repeat(1, 1, num.shape[-1])
    n2 = num.unsqueeze(-2).repeat(1, num.shape[-1], 1)
    num_pair = torch.stack([n1, n2]).permute(1, 2, 3, 0)
    if three_body:
        poly_orders = [12, 5]
    else:
        poly_orders = [12, 0]
    NATMTYP = len(uan)
    element = batch_chemical_symbols(uan)
    element_pair = [[element[ii] + element[jj], element[ii] + element[kk], element[jj] + element[kk]]
                    for ii in range(len(element))
                    for jj in range(ii, len(element))
                    for kk in range(jj, len(element))]
    ATMCHRG = [0] * NATMTYP
    ATMMASS = [chemical_masses[int(ii)] for ii in uan]
    S_DELTA = kwargs.get('S_DELTA', 0.01)
    FCUTTYP = kwargs.get('FCUTTYP', 'CUBIC')

    with open(file, 'w') as f:
        f.write('# ================================= \n')
        f.write('# Created using TBMaLT \n')
        f.write('# Date %s \n' % datetime.now())
        f.write('# ================================= \n')

        f.write('\n####### CONTROL VARIABLES #######\n\n')
        # trajectory file name
        f.write('# TRJFILE # \n    ' + TRJFILE + ' \n')

        # Number of frames
        f.write('# NFRAMES # \n    ' + str(geometry._n_batch) + ' \n')

        # Number of replicate layers
        f.write('# NLAYERS # \n' + kwargs.get('NLAYERS', '    1 \n'))

        # Fit charges
        f.write('# FITCOUL # \n' + kwargs.get('FITCOUL', '    false \n'))

        # Fit stresses
        f.write('# FITSTRS # \n' + kwargs.get('FITSTRS', '    false \n'))

        # Fit energies
        f.write('# FITENER # \n' + kwargs.get('FITENER', '    false \n'))

        # Polynomial orders
        f.write('# PAIRTYP # \n  ' + '  CHEBYSHEV ' + ' '.join(map(str, poly_orders)) + ' \n')

        # Coordinate transformation style
        f.write('# CHBTYPE # \n    ' + 'MORSE \n\n')

        f.write('\n####### TOPOLOGY VARIABLES #######\n\n')

        # element type
        f.write('# NATMTYP # \n' + f'    {NATMTYP} \n\n')

        # specify the type index, atom type, fixed charge, and mass.
        f.write('# TYPEIDX # # ATM_TYP # # ATMCHRG # # ATMMASS # \n')
        for i in range(NATMTYP):
            f.write(f'       {i + 1}       {element[i]}          {ATMCHRG[i]}      {ATMMASS[i]}\n')

        # Specify the pair type index, constituent atom types,
        # inner cutoff (s_minim), outer cutoff (s_maxim), Morse Transformation variable
        f.write('\n')
        f.write(
            '{:15s} {:15s} {:15s} '.format('# PAIRIDX #', '# ATM_TY1 #', '# ATM_TY1 #') +
            '{:15s} {:15s} {:15s} '.format('# S_MINIM #', '# S_MAXIM #', '# S_DELTA #') +
            '{:15s} {:15s} '.format('# MORSE_LAMBDA #', '# USEOVRP #') +
            '{:15s} {:15s} {:15s} '.format('# NIJBINS #', '# NJKBINS #', '# NIJBINS #'))

        n_pair = 0
        for i in range(NATMTYP):
            for j in range(i, NATMTYP):
                n_pair += 1
                mask = (num_pair[..., 0] == uan[i]) * (num_pair[..., 1] == uan[j])
                dist = geometry.distances[mask] * _Bohr__AA
                dist = (dist[dist.ne(0)]).tolist()
                peaks, _ = find_peaks(dist)
                f.write('\n')
                f.write(
                    '{:15s} {:15s} {:15s}'.format(str(n_pair), element[i], element[j]) +
                    '{:15s} {:15s} {:15s}'.format(str(min(dist) - 0.02), str(7.5), str(S_DELTA)) +
                    '{:15s} {:15s}'.format(str(dist[peaks[0]]), 'false') +
                    '{:15s} {:15s} {:15s}'.format('0.0', '0.0', '0.0'))

        if three_body:
            _chimes_three_body(f, element_pair)

        # cutoff type
        f.write('\n\n # FCUTTYP # \n' + f'    {FCUTTYP} \n\n')
        f.write('# ENDFILE # \n\n')


def _chimes_three_body(f, element_pair):
    """Private function to write three body input."""
    f.write(f'\n SPECIAL 3B S_MAXIM: SPECIFIC {len(element_pair)}')
    for pair in element_pair:
        f.write('\n' + f'{"".join(pair)} {pair[0]} {pair[1]} {pair[2]} 5.0 5.0 5.0')
