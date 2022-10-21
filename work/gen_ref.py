#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate dataset for PyTorch training.

This example includes the following part:
    gen_reference: With FHI-aims input, perform DFT calculations.
    2. Read data from FHI-aims calculations.
    3. Write data as the TBMaLT and PyTorch format.
    gen_pyscf: generate data from PySCF

"""
import os
import re
import pickle
from typing import Union

import numpy as np
import matplotlib.pyplot as plt
import torch
from ase import io, Atoms
import h5py
import pyscf.pbc.tools.pyscf_ase as pyscf_ase
import pyscf.pbc.gto as gto
import pyscf.pbc.dft as pbcdft

from tbmalt.io.dataset import Dataset
from tbmalt import Geometry
from tbmalt.io.dataset import GeometryTo
from tbmalt.common.batch import pack
# from tbmalt.physics.ksampling import Ksampling

au2ev = 27.21139
torch.set_default_dtype(torch.float64)
torch.set_printoptions(4)

# Set parameters
# task: ref_tbmalt, ref_pyscf  >>> run reference (DFT)
# scale_geo, read_scale_energy, plot_dft_dftb  >>> tools
# gen_band, write_reference  >>> IO interface
params = {'task': ['write_reference']}

# `gen_reference` parameters
if 'gen_reference' in params['task']:
    geometry_path = "../data/si"
    to_geometry_path = os.path.join(os.getcwd(), "./geometry")
    aims_bin = os.path.join(os.getcwd(), "./aims.x")
    calculation_properties = ["band"]
    path_sk = "../../../tests/unittests/data/slko/mio"
    descriptor_package = "tbmalt"  # tbmalt, scikit-learn
    feature_type = "acsf"
    shell_dict = {1: [0], 3: [0, 1], 6: [0, 1], 14: [0, 1]}
    orbital_resolve = True
    neig_resolve = True
    add_k = False
    gen_ref, train = False, True

# gen_band >>>>>>>>
if 'gen_band' in params['task']:
    path_geoin = './data/raw/sic'
    scale_band = False
    scale_params_band = [0.97, 0.98, 0.99, 1.0, 1.01, 1.02, 1.03]
    path_input_template = "./data/control.in.band.light"
    to_geometry_type = "aims"
    to_geometry_path = "./data/aims_band"
    labels = os.listdir(path_geoin)
    set_constraint = None

# gen_vac >>>>>>>>
if 'gen_vac' in params['task']:
    geometry_in_file_vac = ['./data/raw/si/diamond/si_diamond_8.cif']
    to_geometry_path = "./data/aims_band_vac"
    build_supercell_vac = True
    supercell_vac = [[[2, 0, 0], [0, 2, 0], [0, 0, 2]]]

# gen_slab >>>>>>>>
if 'gen_slab' in params['task']:
    geometry_in_file_slab = ['./data/raw/c_r/C_diamond_8.cif']
    labels = ['c_diamond_8']
    to_geometry_path_slab = "./data/dft_band/c_slab"
    to_geometry_type_slab = "aims"
    path_input_template_slab = "./data/control.in.band.light"
    build_supercell_slab = True
    slab_index = [[(1, 0, 0), (1, 1, 0), (1, 1, 1)]]
    slab_layer = [[3, 3, 3]]
    slab_vacuum = [[7.5, 7.5, 7.5]]
    build_supercell_vac = True
    supercell_vac = [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]
    set_constraint = [0.3]
    constrain_extent = 2  # to control the extent of surface atoms to be relaxed

# write_reference >>>>>>>>
if 'write_reference' in params['task']:
    path_ref = '/Users/gz_fan/Documents/work/mlband/data/dft_band_out'
    output_w = 'band_c_si.h5'
    # path_ref_w = sorted([ii.path for path in path_ref for ii in os.scandir(path) if ii.is_dir()])
    # group_w = sorted([(os.path.split(path)[-1], *os.path.split(ii)[-1].rsplit('_', 1))
    #                   for path in path_ref for ii in os.scandir(path) if ii.is_dir()])

    # path_ref_w = sorted([ii.path for path in path_ref for ii in os.scandir(path) if ii.is_dir()])
    # return group, subgroup, path_to_subgroup
    group_w = sorted([(os.path.split(path.path)[-1], os.path.split(ii.path)[-1], ii.path)
                      for path in os.scandir(path_ref) if os.path.isdir(path.path)
                      for ii in os.scandir(path.path) if os.path.isdir(ii)])



def ref_tbmalt(device):
    """Generate reference for traning."""
    # 1. Generate geometry
    # geometry_in_files = [os.path.join(geometry_path, ii)
    #                      for ii in os.listdir(geometry_path)]
    geometry1 = [
        "cubic_2.cif",
        "cubic_8.cif",
        "hexagonal_4.cif",
        "monoclinic_4.cif",
        "monoclinic_8.cif",
        "monoclinic_16.cif",
        "orthorhombic_2.cif",
        "orthorhombic_4.cif",
        "orthorhombic_8.cif",
        "tetragonal_4.cif",
        "tetragonal_8.cif",
    ]
    geometry_in_files = [os.path.join(geometry_path, ii) for ii in geometry1]
    geot = GeometryTo(
        geometry_in_files,
        path_to_input_template=os.path.join(os.getcwd(), "control.in"),
        to_geometry_type="aims",
        to_geometry_path=to_geometry_path,
        calculation_properties=calculation_properties,
    )
    geot()
    with open("geometry.pkl", "wb") as f:
        pickle.dump(geot, f)

    # 2. Run FHI-aims calculations
    control_in = []
    geometry_in = []
    for file in os.listdir(to_geometry_path):
        if file.startswith("control.in"):
            control_in.append(file)
        elif file.startswith("geometry.in"):
            geometry_in.append(file)

    # 3. Run FHI-aims and read calculated results
    calculator = Calculator.aims(
        control_in=control_in,
        geometry_in=geometry_in,
        aims_bin=aims_bin,
        env_path=to_geometry_path,
        properties=calculation_properties,
        obj_dict=geot.obj_dict,
    )
    calculator.save(calculator)
    print([ii.shape for ii in calculator.results["band"]])
    print(calculator.results["occ"])

    # 3. Save calculated data
    with open("ref.pkl", "wb") as f:
        pickle.dump(calculator, f)


def ref_pyscf():
    path = './cif/c/'
    bulk_list = [os.path.join(path, ii) for ii in os.listdir(path)]
    dataset = Dataset.cif(bulk_list, kpoint_level='high')

    k_size = 10
    klines = pack([torch.from_numpy(kline[:k_size+1]) for kline in dataset.klines])
    kpath = [kpt[:k_size] for kpt in dataset.path]
    klines = torch.cat([klines[:, :-1], klines[:, 1:], 10 * torch.ones(klines.shape[0], k_size, 1)], -1)
    atomic_numbers = pack([ii for ii in dataset.numbers])
    ksample = Ksampling(len(dataset.numbers), atomic_numbers, klines=klines)

    vband, cband, kpt_list = [], [], []
    gap_list = []

    for ii, c in enumerate(bulk_list):
        ase_c = io.read(c)
        cell = gto.Cell()
        cell.atom = pyscf_ase.ase_atoms_to_pyscf(ase_c)
        cell.a = ase_c.cell
        cell.basis = 'gth-szv'
        cell.pseudo = 'gth-pade'
        cell.verbose = 5
        cell.build(None, None)

        # kpath = dataset.path[ii][:k_size]
        # band_kpts, _kpath, _ = get_bandpath(kpath, ase_c.cell, npoints=10*len(kpath))
        # band_kpts = cell.get_abs_kpts(band_kpts)
        band_kpts = ksample.kpoints[ii] @ cell.reciprocal_vectors()

        mf = pbcdft.KRKS(cell, cell.make_kpts([2, 5, 5]))
        mf.xc = 'hse06'
        mf.kernel()
        efermi = mf.get_fermi()

        e_kn = pack([torch.from_numpy(ii) for ii in mf.get_bands(band_kpts)[0]])
        ind = cell.nelectron // 2 - 1
        _gap = (torch.min(e_kn[..., ind + 1]) - torch.max(e_kn[..., ind])) * au2ev
        print('_gap', _gap)
        if _gap > 0:
            gap_list.append(_gap)
            vbmax = torch.max(e_kn[..., ind])
            e_kn = (e_kn - vbmax) * au2ev
        else:
            gap_list.append(0)
            e_kn = (e_kn - efermi) * au2ev

        # for n in range(cell.nao_nr()):
        #     plt.plot(torch.arange(len(band_kpts)), [e[n] for e in e_kn])
        # plt.show()

        vband.append(e_kn[..., : ind + 1])
        cband.append(e_kn[..., ind + 1:])

    write_h5('c_hse.h5',
             atomic_numbers,
             pack(dataset.positions),
             properties={'vband': vband, 'cband': cband, 'gap': gap_list,
                         'klines': klines, 'kpath': kpath},
             cells=pack(dataset.cells))


def write_h5(output: str,
             atomic_numbers: np.ndarray,
             positions: np.ndarray,
             properties: dict,
             cells: np.ndarray = None,
             labels: list = None,
             group=['new_group', 'new_subgroup'],
             mode: str = "a"):
    assert atomic_numbers.dim() == 2, 'atomic_numbers should be 2D dimension'
    assert positions.dim() == 3, 'positions should be 3D dimension'

    with h5py.File(output, mode) as f:
        g = f.create_group(group[0]) if group[0] not in f else f[group[0]]

        if group[1] in g:
            del g[group[1]]
            print(f'{group[1]} exist, delete this sub-group')
        print('group', group)
        subg = g.create_group(group[1])
        subg.attrs["n_geometries"] = len(atomic_numbers)
        subg.create_dataset("labels", data=labels)
        subg.create_dataset("numbers", data=atomic_numbers)
        subg.create_dataset("positions", data=positions)
        subg.create_dataset("cells", data=cells)

        for key, vals in properties.items():
            if key == "vband":
                subg.create_dataset(key, data=vals)
                subg.create_dataset("n_vband", data=[ii.shape[-1] for ii in vals])
            elif key == "cband":
                subg.create_dataset("n_cband", data=[ii.shape[-1] for ii in vals])
                subg.create_dataset(key, data=vals)
            else:
                subg.create_dataset(key, data=vals)


def scale(template: Union[list, Geometry],
          scale: list,
          to_path: Union[str, list],
          to_type: str
          ):
    geo = io.read(template)
    _in = [
        Atoms( positions=geo.positions * ii,
               numbers=geo.numbers,
               cell=geo.cell * ii,
               pbc=True,
               )
        for ii in scale
    ]

    if isinstance(to_path, str):
        to_path = [to_path] * len(scale)

    if to_type == "aims":
        os.system("rm " + to_path[0] + "/geometry.in.*")
        [
            io.write(os.path.join(path, "geometry.in." + str(ii)), iin, format="aims")
            for ii, (iin, path) in enumerate(zip(_in, to_path))
        ]
    elif to_type == "dftb":
        os.system("rm " + to_path[0] + "/geo.gen.*")
        [
            io.write(os.path.join(path, "geo.gen." + str(ii)), iin, format="dftb")
            for ii, (iin, path) in enumerate(zip(_in, to_path))
        ]

    return _in


def get_energy_single(detail, unit='H'):
    text = "".join(open(detail, "r").readlines())
    E_tot_ = re.search(
        "(?<=Total energy:).+(?=\n)", text, flags=re.DOTALL | re.MULTILINE
    ).group(0)
    E_tot = re.findall(r"[-+]?\d*\.\d+", E_tot_)

    E_rep_ = re.search(
        "(?<=Repulsive energy:).+(?=\n)", text, flags=re.DOTALL | re.MULTILINE
    ).group(0)
    E_rep = re.findall(r"[-+]?\d*\.\d+", E_rep_)

    if unit == 'H':
        return float(E_tot[0]), float(E_rep[0])
    else:
        return float(E_tot[1]), float(E_rep[1])


def get_energy_dft_single(aims, unit='H'):
    """Read FHI-aims output."""
    text = "".join(open(aims, "r").readlines())
    E_tot_ = re.findall("^.*\| Total energy                  :.*$", text, re.MULTILINE)[-1]
    E_tot = re.findall(r"[-+]?\d*\.\d+", E_tot_)

    return float(E_tot[0]) if unit == 'H' else float(E_tot[1])


def get_dft_energy(path, scal_params, unit='H'):
    energy = []
    for num in range(len(scal_params)):
        file = os.path.join(path, 'aims.out.' + str(num))
        energy.append(get_energy_dft_single(file, unit))
    return energy


def get_dftb_energy(path, scal_params, unit='H'):
    rep, tot = [], []
    for ii in range(len(scal_params)):
        detail = path + "/detailed.out." + str(ii)
        try:
            it, ir = get_energy_single(detail, unit)
        except:
            it, ir = 0, 0
        rep.append(ir)
        tot.append(it)
    return tot, rep


if __name__ == "__main__":
    scal_params = np.linspace(0.95, 1.05, 11)  # 11, 51

    if 'scale_geo' in params["task"]:
        template = "/home/gz_fan/Documents/ML/train/battery/CCS/CCS/test/LiCl/LiCl_mp-22905_conventional_standard.cif"
        geo_type = "aims"
        scal_params = np.linspace(0.8, 1.1, 11)  # 11, 51
        to_path = "/home/gz_fan/Documents/ML/train/battery/CCS/CCS/test/LiCl/aims_geo"
        geometries = scale(template, scal_params, to_path, geo_type)

    if 'read_scale_energy' in params["task"]:
        tot, rep = get_dftb_energy(path, scal_params)

    if 'plot_dft_dftb' in params["task"]:
        path_to_dft = '/Users/gz_fan/Documents/dftb/CCS/test/Li5PS4Cl2_trans/aims2'
        path_to_dftb = '/Users/gz_fan/Downloads/software/dftbplus/dftbplus/work/battery/Li5PS4Cl2/li_trans2'
        scal_params_test = np.linspace(0.95, 1.05, 11)  # 11, 51

        ref_e = get_dft_energy(path_to_dft, scal_params)
        plt.show()
        tot, rep = get_dftb_energy(path_to_dftb, scal_params_test)
        plt.plot(scal_params, np.array(ref_e) - min(ref_e), "rx", label="FHI-aims")
        plt.plot(scal_params_test, np.array(tot) - min(tot), label="DFTB+")
        plt.ylim(-0.1, 0.8)
        plt.ylabel("E (eV)")
        plt.legend()
        plt.show()
        plt.plot(np.arange(len(rep)), rep, label="rep")
        plt.xlabel("Grid points with scaling params")
        plt.ylabel("E (eV)")
        plt.legend()
        plt.show()

    if 'gen_band' in params["task"]:
        # generate band structures data
        path_dftb = (
            "/home/gz_fan/Downloads/software/dftbplus/dftbplus/work/test/battery"
        )
        geometry_in_files = [os.path.join(path_geoin, ii) for ii in os.listdir(path_geoin)]
        path_input_template = path_input_template
        to_geometry_type = to_geometry_type
        to_geometry_path = to_geometry_path

        geot = GeometryTo(
            in_geometry_files=geometry_in_files,
            path_to_input_template=path_input_template,
            to_geometry_type=to_geometry_type,
            to_geometry_path=to_geometry_path,
            calculation_properties=["band"],
            labels=labels,
            set_constraint=set_constraint,
        )
        geot()

    if 'gen_vac' in params["task"]:
        path_input_template = path_input_template
        to_geometry_type = to_geometry_type
        to_geometry_path = to_geometry_path

        geot = GeometryTo(
            in_geometry_files=geometry_in_file_vac,
            labels=labels,
            path_to_input_template=path_input_template,
            to_geometry_type=to_geometry_type,
            to_geometry_path=to_geometry_path,
            calculation_properties=["band"],
            build_supercell=build_supercell_vac,
            supercell_params=supercell_vac,
        )
        geot()

    if 'gen_slab' in params["task"]:
        geot = GeometryTo(
            in_geometry_files=geometry_in_file_slab,
            labels=labels,
            path_to_input_template=path_input_template_slab,
            to_geometry_type=to_geometry_type_slab,
            to_geometry_path=to_geometry_path_slab,
            calculation_properties=["band"],
            build_supercell=build_supercell_slab,
            supercell_params=supercell_vac,
            build_slab=True,
            slab_index=slab_index,
            layer=slab_layer,
            vacuum=slab_vacuum,
            set_constraint=set_constraint,
            constrain_extent=constrain_extent,
        )
        geot()

    if 'write_reference' in params["task"]:

        for group, subgroup, path in group_w:

            atomic_numbers, positions, cells = [], [], []
            klines, band_list, vband, cband, gap_list, vbm_alignment, _labels = \
                [], [], [], [], [], [], []

            for ipath in os.scandir(path):
                if not os.path.isdir(ipath):
                    continue

                # Read geometry.in and control.in
                geo = io.read(os.path.join(ipath.path, 'geometry.in'))
                control = os.path.join(os.path.join(ipath.path, 'control.in'))
                atomic_numbers.append(geo.numbers)
                positions.append(geo.positions)
                cells.append(geo.cell.array)

                lines = open(control).readlines()
                band_from_control = [line for line in lines if 'output band' in line]
                kline = np.array([ii for line in band_from_control for ii in line.split()
                                  if ii != 'output' and ii != 'band']).reshape(-1, 7).astype(float)
                klines.append(kline)

                # read band files
                band, occ = [], []
                for bandfile in [os.path.join(ipath.path, 'band100' + str(jj) + '.out')
                                 for jj in (1, 2, 3, 4, 5, 6, 7)]:
                    data = np.loadtxt(bandfile)[..., 4:]
                    ind = np.arange(1, data.shape[1], 2)
                    band.append(data[..., ind])
                    occ.append(data[..., ind - 1])

                band = np.concatenate(band)
                occ = np.concatenate(occ)
                _occ = np.round(occ.sum(0) / len(occ) / 2) != 0

                ne = int(geo.numbers.sum() / 2)
                # Warning: This may only works for FHI-aims, since the energy states = n_elec / 2
                vband.append(band[..., :ne])
                cband.append(band[..., ne:])
                band_list.append(band)
                _labels.append(os.path.split(ipath.path)[-1])

                vbm_align_value = np.max(vband[-1][..., -1])
                gap = np.min(cband[-1][..., 0]) - vbm_align_value

                # The data from FHI-aims band sometimes are close to zero,
                # But do not equal to zero, therefore we assume all VBM close
                # to zero are VBM alignment
                vbm_align = np.allclose(vbm_align_value, 0, atol=4E-1)

                if gap > 0 or vbm_align:
                    vband[-1] = vband[-1] - vbm_align_value
                    cband[-1] = cband[-1] - vbm_align_value
                    vbm_align = True

                gap = 0 if gap <= 0 else gap
                gap_list.append(gap)
                vbm_alignment.append(vbm_align)

                # if gap:
                #     vbm = np.max(vband[-1])
                #     vband[-1] = vband[-1] - vbm
                #     cband[-1] = cband[-1] - vbm
                #     print('vbm', vbm, np.min(cband[-1]))
                #     band_list[-1] = band_list[-1] - vbm

            # Write for each sub-dataset
            write_h5(output_w, torch.tensor(atomic_numbers), torch.tensor(positions),
                     group=[group, subgroup], cells=np.stack(cells),
                     labels=np.array(_labels, dtype='S'),
                     properties={
                         'band': np.asarray(band_list), 'vband': vband,
                         'cband': cband, 'gap': gap_list, 'klines': klines,
                         'vbm_alignment': vbm_alignment
                     },
                     )
