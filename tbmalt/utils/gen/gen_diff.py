import os
import re
from typing import Union, List
import subprocess

import torch
import numpy as np
from torch import Tensor


class BandDiff:
    """This class works for a single system."""

    def __init__(self, band_files: Union[List, str],
                 ref_files: Union[List, str],
                 detailed_files: Union[List, str],
                 n_vband: int, n_cband: int, alignment: str,
                 gen_dftb: str = 'dp_band',
                 ref_form: str = 'aims', weigh0=1.0, weight1=0.0,
                 average=True):
        """
        Arguments:
            eigenvalues: Testing eigenvalues with VBM alignment.
            ref_eigenvalues: Reference eigenvalues with VBM alignment.
            mask_eigenvalues: Mask used to select eigenvalues.
        """
        self.n_vband = n_vband
        self.n_cband = n_cband
        self.alignment = alignment
        self.average = average # return average errors

        if isinstance(band_files, list):
            assert len(band_files) == len(detailed_files)
        self.band_files = band_files

        fail_read_files = []
        data, detail_data = [], []
        for file, detail in zip(band_files, detailed_files):
            try:
                print('file', file)
                data.append(np.loadtxt(file))
                detail_data.append(read_detailed_out_old(detail))
                fail_read_files.append(False)
            except:
                print('file fails', file)
                data.append(np.array(None))
                detail_data.append(None)
                fail_read_files.append(True)

        ref_data = [np.loadtxt(file) for file in ref_files]

        if isinstance(band_files, list):
            self.batch = True

            eigenvalues = None
            for ii, idata in enumerate(data):
                # if idata is not None and eigenvalues is None:
                if not fail_read_files[ii] and eigenvalues is None:
                    eigenvalues = np.empty((len(band_files), *idata.shape))
                    fermi_data = np.empty((len(detail_data), *detail_data[ii].shape))
                    eigenvalues[ii] = idata
                    fermi_data[ii] = detail_data[ii]
                # elif idata is not None and eigenvalues is not None:
                elif not fail_read_files[ii] and eigenvalues is not None:
                    eigenvalues[ii] = idata
                    fermi_data[ii] = detail_data[ii]

            if eigenvalues is None:
                self.error0 = np.nan
                return

            # The first number is not eigenvalues if generate from dp_tools
            if gen_dftb == 'dp_band':
                eigenvalues = eigenvalues[..., 1:]

            fermi_data = np.expand_dims(fermi_data, axis=(-3, -2))
        else:
            self.batch = False
            fermi_data = read_detailed_out_old(detailed_files)

        # DFTB+ eigenvalue, single system with different parameters
        ne = fermi_data[..., 1]

        if ne[0].squeeze() % 2 == 0:
            n_band = int(ne[0].squeeze() / 2)
        else:
            n_band = int(ne[0].squeeze() / 2) + 1
            print('the valence band is not fully occupied, n_band = ceil(ne/2)')

        if self.alignment == 'vbm':
            vbm = np.expand_dims(np.max(eigenvalues[..., n_band - 1], axis=-1),
                                 axis=(-2, -1))
            eigenvalues -= vbm
        elif self.alignment == 'fermi':
            fermi = fermi_data[..., 0]
            eigenvalues -= fermi

        if ref_form == 'aims':
            ref_data = np.concatenate(ref_data)[..., 4:]
            ref_occ = ref_data[..., ::2]
            ref_ne = ref_occ.sum(0) / np.max(ref_occ.sum(0))
            # This is dangerous code !!! use Occupation instead
            ref_ne_mask = ref_ne < 3E-1

            if ((ref_ne < 5E-1) * (ref_ne > 1E-1)).any():
                print('There is occupation of electrons between 0.1~1, which is ignored')
            ref_eigenvalues = ref_data[..., 1:: 2]
            if alignment == 'vbm':
                ref_vbm = np.max(ref_eigenvalues[..., ~ref_ne_mask])
                ref_eigenvalues -= ref_vbm
            else:
                pass
        else:
            raise NotImplemented(f'{ref_form} is not implemented')

        self.v, self.c, self.v_ref, self.c_ref, self.error0 = self.band0(
            eigenvalues, n_band, ref_eigenvalues, ref_ne_mask)
        self.sort_idx = np.argsort(self.error0)
        print('band file sort_idx', np.array(band_files)[self.sort_idx[:10]], self.error0)
        print('error0', self.error0[self.sort_idx[:3]])

        self.error1 = self.band1(self.v, self.c, self.v_ref, self.c_ref)
        self.error = self.error0 * weigh0 + self.error1 * weight1
        print('max error0', np.max(self.error0[~np.isnan(self.error0)]),
              'min error0', np.min(self.error0[~np.isnan(self.error0)]))
        # print('max error1', np.min(self.error1[~np.isnan(self.error1)]),
        #       'min error1', np.max(self.error1[~np.isnan(self.error1)]))
        # print('max error', np.max(self.error[~np.isnan(self.error)]),
        #       'min error', np.max(self.error[~np.isnan(self.error)]))

    def band0(self, eigenvalues, n_band, ref_eigenvalues, ref_cb_mask):
        """Zero order band structure energy difference."""
        v = eigenvalues[..., :n_band]
        c = eigenvalues[..., n_band:]

        v_ref = ref_eigenvalues[..., ~ref_cb_mask]
        c_ref = ref_eigenvalues[..., ref_cb_mask]
        if self.average:
            error_v = np.abs(v[..., -self.n_vband:] - v_ref[..., -self.n_vband:]).mean(-1)
            error_c = np.abs(c[..., :self.n_cband] - c_ref[..., :self.n_cband]).mean(-1)
        else:
            error_v = np.abs(v[..., -self.n_vband:] - v_ref[..., -self.n_vband:]).sum(-1)
            error_c = np.abs(c[..., :self.n_cband] - c_ref[..., :self.n_cband]).sum(-1)

        if self.batch:
            if self.average:
                error_v = error_v.mean(-1)
                error_c = error_c.mean(-1)
            else:
                error_v = error_v.sum(-1)
                error_c = error_c.sum(-1)

        return v, c, v_ref, c_ref, error_c + error_v

    def band1(self, v, c, v_ref, c_ref):
        """First order band structure difference."""
        vg = v[..., 1:] - v[..., :-1]
        cg = c[..., 1:] - c[..., :-1]
        vg_ref = v_ref[..., 1:] - v_ref[..., :-1]
        cg_ref = c_ref[..., 1:] - c_ref[..., :-1]
        if not self.average:
            error_v = np.abs(vg[..., -self.n_vband:] - vg_ref[..., -self.n_vband:]).sum(-1)
            error_c = np.abs(cg[..., -self.n_vband:] - cg_ref[..., -self.n_vband:]).sum(-1)
        else:
            error_v = np.abs(vg[..., -self.n_vband:] - vg_ref[..., -self.n_vband:]).mean(-1)
            error_c = np.abs(cg[..., -self.n_vband:] - cg_ref[..., -self.n_vband:]).mean(-1)

        if self.batch:
            if self.average:
                error_v = error_v.mean(-1)
                error_c = error_c.mean(-1)
            else:
                error_v = error_v.sum(-1)
                error_c = error_c.sum(-1)

        return error_v + error_c

def read_detailed_out_old(input):
    """Read DFTB+ output file detailed.out."""
    text = "".join(open(input, "r").readlines())

    E_f_ = re.search(
        "(?<=Fermi level).+(?=\n)", text, flags=re.DOTALL | re.MULTILINE
    ).group(0)
    E_f = re.findall(r"[-+]?\d*\.\d+", E_f_)[1]

    elect = re.search(
        "(?<=of electrons).+(?=\n)", text, flags=re.DOTALL | re.MULTILINE
    ).group(0)
    ne = re.findall(r"[-+]?\d*\.\d+", elect)[0]

    E_elect_ = re.search(
        "(?<=Total Electronic energy).+(?=\n)", text, flags=re.DOTALL | re.MULTILINE
    ).group(0)
    E_elect = re.findall(r"[-+]?\d*\.\d+", E_elect_)[1]

    return np.array([float(E_f), float(ne), float(E_elect)])


def read_detailed_out(input, properties=['tot_e']):
    """Read DFTB+ output file detailed.out."""
    text = "".join(open(input, "r").readlines())
    pro_dict = {}

    if 'fermi' in properties:
        E_f_ = re.search(
            "(?<=Fermi level).+(?=\n)", text, flags=re.DOTALL | re.MULTILINE
        ).group(0)
        E_f = re.findall(r"[-+]?\d*\.\d+", E_f_)[1]
        pro_dict.update({'fermi': float(E_f)})

    if 'n_elect' in properties:
        elect = re.search(
            "(?<=of electrons).+(?=\n)", text, flags=re.DOTALL | re.MULTILINE
        ).group(0)
        ne = re.findall(r"[-+]?\d*\.\d+", elect)[0]
        pro_dict.update({'n_elect': float(ne)})

    if 'e_elect' in properties:
        E_elect_ = re.search(
            "(?<=Total Electronic energy).+(?=\n)", text, flags=re.DOTALL | re.MULTILINE
        ).group(0)
        E_elect = re.findall(r"[-+]?\d*\.\d+", E_elect_)[1]
        pro_dict.update({'e_elect': float(E_elect)})

    if 'e_tot' in properties:
        E_tot_ = re.search(
            "(?<=Total energy).+(?=\n)", text, flags=re.DOTALL | re.MULTILINE
        ).group(0)
        E_tot = re.findall(r"[-+]?\d*\.\d+", E_tot_)[1]
        pro_dict.update({'e_tot': float(E_tot)})

    # return np.array([float(E_f), float(ne), float(E_elect)])
    return pro_dict


def read_aims_out(input, properties=['e_tot']):
    """Read DFTB+ output file detailed.out."""
    text = "".join(open(input, "r").readlines())
    pro_dict = {}

    if 'fermi' in properties:
        E_f_ = re.search(
            "(?<=Fermi level).+(?=\n)", text, flags=re.DOTALL | re.MULTILINE
        ).group(0)
        E_f = re.findall(r"[-+]?\d*\.\d+", E_f_)[1]
        pro_dict.update({'fermi': E_f})

    if 'e_tot' in properties:
        comme = "grep 'Total energy                  :' " + input + \
                " | tail -n 1 | awk '{print $7}'"
        e_tot = float(subprocess.check_output(comme, shell=True).decode('utf-8'))
        pro_dict.update({'e_tot': e_tot})

    return pro_dict
