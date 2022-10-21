import os
import re
from typing import Union, List

import ase.io as io
import torch
import numpy as np
import matplotlib.pyplot as plt

from tbmalt import Dftb2


def dftbplus(path, alignment, properties=['band']):
    """Read DFTB+ band structure and DOS data."""
    eigenvalues = np.loadtxt(os.path.join(path, 'band_tot.dat'))[..., 1:]

    if os.path.isfile(path_de := os.path.join(path, 'detailed.out.scc')):
        detail = read_detailed_out(path_de)
        print('use scc detailed file')
    elif os.path.isfile(path_de := os.path.join(path, 'detailed.out')):
        detail = read_detailed_out(path_de)
        print('make sure detailed file is from MP-K-points SCC-DFTB')

    # geo = io.read(path + '/geo.gen')
    # kpt = geo.cell.bandpath(npoints=10)

    if alignment == 'vbm':
        ne = detail[..., 1]
        if ne % 2 == 0:
            n_band = int(ne / 2)
        else:
            n_band = int(ne / 2) + 1

        vbm = np.expand_dims(np.max(eigenvalues[..., n_band - 1], axis=-1),
                             axis=(-2, -1))
        eigenvalues -= vbm
    elif alignment == 'fermi':
        eigenvalues -= detail[0]

    if 'dos' not in properties and 'pdos' not in properties:
        return {'eigenvalues': eigenvalues}
    else:
        return {'eigenvalues': eigenvalues, 'dos': dftbplus_dos(path, alignment)}


def dftbplus_dos(path, alignment):
    files = [os.path.join(path, file) for file in sorted(os.listdir(path))
             if file.startswith('dos')]
    dos_dict = {}
    if os.path.isfile(path_de := os.path.join(path, 'detailed.out.scc')):
        detail = read_detailed_out(path_de)
        print('use scc detailed file')
    elif os.path.isfile(path_de := os.path.join(path, 'detailed.out')):
        detail = read_detailed_out(path_de)
        print('make sure detailed file is from MP-K-points SCC-DFTB')

    for file in files:
        with open(file, 'r') as f:
            data0 = f.readlines()
            weight, data = [], []

            data0 = list(filter(('\n').__ne__, data0))  # remove \n
            for ii in data0:
                if ii.startswith('KPT'):
                    weight.append(float(ii.split()[-1]))
                else:
                    data.append(ii)
            data = [ii for ii in data0 if not ii.startswith(' KPT')]
            # dos = np.concatenate([np.fromstring(
            #     ii.replace('\n', ''), sep=' ') for ii in data]).\
            #     reshape(len(weight), -1, 2).T
            # dos = (dos * np.array(weight)).sum(-1).T
            # key = file.split('_')[-1][:-4]
            # dos_dict.update({key: dos})
            dos = np.concatenate([np.fromstring(
                ii.replace('\n', ''), sep=' ') for ii in data]).\
                reshape(-1, 2).T
            # dos = (dos * np.array(weight)).sum(-1).T
            key = file.split('_')[-1][:-4]
            dos_dict.update({key: dos})

        dos = np.loadtxt(file)
        if alignment == 'fermi':
            dos -= detail[..., 0]

        key = file.split('_')[-1][:-4]
        dos_dict.update({key: dos})

    return dos_dict


def aims(path, alignment, properties):
    """Read FHI-aims band structure and DOS data."""
    data_dict = _aims_band(path, alignment) if 'band' in properties else {}

    if 'dos' in properties:
        data_dict.update({'dos': _aims_dos(path, alignment, data_dict['vbm'])})

    return data_dict


def _aims_dos(path, alignment, vbm):
    def read_dos(file):
        data_dict = {}

        with open(file, 'r') as f:
            # skip first 4 lines
            [f.readline() for i in range(4)]

            # read the remaining lines into a list of lists
            data = np.array([line.strip().split() for line in f], dtype=float)
            if alignment == 'vbm':
                data = data - vbm
            data_dict.update({'E': data[..., 0]})
            data_dict.update({'total': data[..., 1]})
            data_dict.update({'pdos': data[..., 2:]})
        return data_dict

    files = [os.path.join(path, file) for file in sorted(os.listdir(path))
             if file.endswith('dos.dat')]
    file_tot = [os.path.join(path, file) for file in os.listdir(path)
                if file.endswith('KS_DOS_total.dat')]
    assert len(file_tot) == 1, ''

    # data_dict = {file.split('/')[-1].split('_')[0]: read_dos(file)
    #              for file in files if file.endswith('dos.dat')}
    data_dict = {}
    for file in files:
        if file.endswith('dos.dat'):
            data_tmp = read_dos(file)
            name = file.split('/')[-1].split('_')[0]
            data_dict[name] = {}
            print(data_tmp['pdos'].shape, data_tmp.keys(), file.split('/')[-1])
            data_dict[name].update({
                'E': data_tmp['E'], 'total': data_tmp['total'],
                's': data_tmp['pdos'][..., 0], 'orbs': ['s']})
            if data_tmp['pdos'].shape[1] > 1:
                data_dict[name].update({'p':  data_tmp['pdos'][..., 1], 'orbs': ['s', 'p']})
            if data_tmp['pdos'].shape[1] > 2:
                data_dict[name].update({'d':  data_tmp['pdos'][..., 2], 'orbs': ['s', 'p', 'd']})

    data_dict.update({'total': read_dos(file_tot[0])})
    return data_dict


def _aims_band(path, alignment):
    files = [os.path.join(path, file) for file in sorted(os.listdir(path))
             if file.startswith('band')]
    data = [np.loadtxt(file) for file in files]
    data = np.concatenate(data)[..., 4:]
    occ = data[..., ::2]
    ne = occ.sum(0) / np.max(occ.sum(0))

    geo = io.read(path + '/geometry.in')
    kpt = geo.cell.bandpath(npoints=10)

    # This is dangerous code !!! use Occupation instead
    ne_mask = ne < 3E-1

    if ((ne < 5E-1) * (ne > 1E-1)).any():
        print('There is occupation of electrons between 0.1~1, which is ignored')
    eigenvalues = data[..., 1:: 2]

    if alignment == 'vbm':
        vbm = np.max(eigenvalues[..., ~ne_mask])
        bandgap = np.min(eigenvalues[..., ne_mask]) - vbm
        eigenvalues -= vbm

    elif alignment == 'fermi':
        vbm = None

    return {'eigenvalues': eigenvalues, 'vbm': vbm, 'kpt': kpt}


def tbmalt(dftb, alignment, properties=['band']):
    """Read TBMaLT band structure and DOS data."""
    eigenvalues = dftb['band'].eigenvalue.squeeze().numpy()

    if alignment == 'vbm':
        vbm = np.max(eigenvalues[..., :(dftb['band'].nelectron / 2).long().tolist()[0]])
        eigenvalues -= vbm
    elif alignment == 'fermi':
        eigenvalues -= dftb['scc'].E_fermi.numpy()

    if 'band' in properties and len(properties) == 1:
        return {'eigenvalues': eigenvalues}
    elif 'dos' in properties and 'band' in properties:
        return {'eigenvalues': eigenvalues, 'dos': tbmalt_dos(dftb['scc'])}
    else:
        raise NotImplementedError()


def tbmalt_dos(dftb_scc,):
    return dftb_scc.dos


class BandPlot:
    """Plot band structure or density of states (DOS)."""

    def __init__(self, alignment: str = 'vbm',
                 e_min: float = -5, e_max: float = 5,
                 save: bool = True, dpi: int = 300):
        self.alignment = alignment
        self.save = save
        self.e_min = e_min
        self.e_max = e_max

    def __call__(self,
                 file1: dict = {
                     'path': './band.dat', 'form': 'dftbplus', 'properties': ['band'],
                    'label': 'band-1', 'band_xlabel': None, 'select_pdos': None},
                 file2: dict = {
                     'path': None, 'form': None, 'properties': None,
                     'label': None, 'select_pdos': None},
                 file='band_dos.png'
                 ):

        # if properties is None:
        #     properties = [['band']] if path2 is None else [['band'], ['band']]
        properties = file1['properties']
        if file2['properties'] is not None:
            properties = properties + [pro for pro in file2['properties'] if pro not in properties]

        data1 = globals()[file1['form']](file1['path'], self.alignment, file1['properties'])
        data2 = None if file2['path'] is None else globals()[file2['form']](
            file2['path'], self.alignment, file2['properties'])
        if self.alignment == 'vbm':
            ylabel = {'band': r"$E - E_{VBM}$ (eV)"}
        elif self.alignment == 'fermi':
            ylabel = {r'band': "E - E$_{fermi}$ (eV)"}

        if len(properties) >= 2:
            f, (ab, ad) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [3, 2]})
        elif 'band' in properties:
            fig, ab = plt.subplots(1)
        elif 'dos' in properties:
            fig, ad = plt.subplots(1)

        len_kpath = None
        if 'band' in properties:
            if data1['eigenvalues'].ndim == 2:
                ab.plot(np.arange(len(data1['eigenvalues'])), data1['eigenvalues'], "r")
                label1 = file1['label']
                ab.plot([100], [100], "r", label=label1)
                len_kpath = len(data1['eigenvalues'])

            if file2['path'] is not None:
                ab.plot(np.arange(len(data2['eigenvalues'])), data2['eigenvalues'], "c--")
                label2 = file2['label']
                ab.plot([100], [100], "c--", label=label2)
                if len_kpath is not None:
                    assert len(data2['eigenvalues']) == len_kpath
                else:
                    len_kpath = len(data2['eigenvalues'])

            if len(properties) == 1:
                plt.ylabel(ylabel['band'])
                plt.ylim(self.e_min, self.e_max)
            else:
                ab.set_ylabel(ylabel['band'])
                ab.set_ylim(self.e_min, self.e_max)

            ab.set_xticks(np.arange(len_kpath + 1)[::10], file1['band_xlabel'])
            # ab.set_xticklabels(file1['band_xlabel'])
            ab.set_xlim(0, len_kpath)
            ab.legend()

        if 'dos' in file1['properties']:
            ad.plot(data1['dos']['total']['total'], data1['dos']['total']['E'],
                    label='total DOS')
            ad.set_ylim(self.e_min, self.e_max)
            ad.set_xticks([])

        # if 'dos' in file2['properties']:
        #     print( data2['dos'])
        #     ad.plot(data2['dos'][0].squeeze(), data2['dos'][1].squeeze())
        #     ad.set_ylim(self.e_min, self.e_max)
        #     ad.set_xticks([])
        if 'dos' in properties:
            ad.set_xlabel('DOS')
            ad.legend()

        orbs_idx = {}
        if 'pdos' in file1['properties']:
            if file1['select_pdos'] is None:
                pdos1 = {key: val for key, val in data1['dos'].items if key != 'total'}
            else:
                pdos1 = {key: data1['dos'][key] for key in file1['select_pdos'].keys()}

            for key, val in pdos1.items():
                # if key == 'total':
                #     continue

                # ad.plot(val[..., 1], val[..., 0], )
                orbs = val['orbs'] if file1['select_pdos'] is None else file1['select_pdos'][key]
                for orb in orbs:
                    ad.plot(val[orb], val['E'], label=key + ': ' +orb)

            # if 'dos' in file1['properties']:
            #     ad.plot(data2['dos'][0].squeeze(), data2['dos'][1].squeeze())
            ad.set_xlabel('DOS (a.u.)')
            ad.set_yticks([])
            ad.legend()

        plt.subplots_adjust(wspace=0)
        plt.savefig(file, dpi=300)
        plt.show()


def read_detailed_out(file):
    """Read DFTB+ output file detailed.out."""
    text = "".join(open(file, "r").readlines())

    E_f_ = re.search(
        "(?<=Fermi level).+(?=\n)", text, flags=re.DOTALL | re.MULTILINE
    ).group(0)
    E_f = re.findall(r"[-+]?\d*\.\d+", E_f_)[1]

    elect = re.search(
        "(?<=of electrons).+(?=\n)", text, flags=re.DOTALL | re.MULTILINE
    ).group(0)

    ne = re.findall(r"[-+]?\d*\.\d+", elect)[0]

    return np.array([float(E_f), float(ne)])
