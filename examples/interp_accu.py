# Test Bicubic interpolation and spline interpolation
import numpy as np
import torch
from ase.build import molecule
import h5py
from torch.nn.functional import normalize
import matplotlib.pyplot as plt

from tbmalt import Dftb2, Periodic, Geometry, Basis
from tbmalt.physics.dftb.slaterkoster import hs_matrix
from tbmalt.common.maths.interpolation import BicubInterp, Spline1d, PolyInterpU
from tbmalt.ml.skfeeds import SkfFeed, VcrFeed
from tbmalt.io.hdf import LoadHdf

from torch import Tensor
torch.set_default_dtype(torch.float64)
torch.set_printoptions(10)
shell_dict = {1: [0], 6: [0, 1], 7: [0, 1], 8: [0, 1]}



def test_bicubic(device):
    dataset = '../work/train/dataset/scc_6000_01.hdf'
    # ================ #
    # std calculations #
    # ================ #
    atomic_numbers = (torch.ones(100) * 6).long()
    positions = torch.cat([torch.linspace(1.0, 8.0, 100).unsqueeze(-1),
                           torch.zeros(100, 2)], -1)
    geometry = Geometry(atomic_numbers, positions)
    path_to_skf = ['2.75', '3.75', '7.0']
    distance = geometry.distances[0][geometry.distances[0].gt(1.0)]

    int_h = []
    for path in path_to_skf:
        h_feed = SkfFeed.from_dir(
            path, shell_dict, elements=['H', 'C', 'N', 'O'], skf_type='skf',
            interpolation='PolyInterpU', integral_type='H')
        int_h.append(h_feed.off_site(
            atom_pair=torch.tensor([6, 6]),
            shell_pair=torch.tensor([0, 0]),
            distances=distance,
        ))

    # ================ #
    # vcr calculations #
    # ================ #
    path_to_vcr = './vcr.h5'
    vcr_grid = torch.tensor([1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5., 6., 8., 10.])

    h_feed_vcr = VcrFeed.from_dir(
                path_to_vcr, shell_dict, vcr_grid,
                skf_type='h5', elements=['H', 'C', 'N', 'O'], integral_type='H',
                interpolation='BicubInterp')

    int_vcr = []
    for path in path_to_skf:
        compr = torch.ones(len(distance), 2) * float(path)
        int_vcr.append(h_feed_vcr.off_site(
            atom_pair=torch.tensor([6, 6]),
            shell_pair=torch.tensor([0, 0]),
            distances=distance,
            variables=compr))

    numbers, positions, data = LoadHdf.load_reference(
        dataset, 1000, ['charge', 'dipole'],)
    geo = Geometry(numbers, positions, units='angstrom', cell=None)

    compr = torch.zeros(*geo.atomic_numbers.shape)
    init_dict = {1: torch.tensor([3.0]),
                 6: torch.tensor([2.7]),
                 7: torch.tensor([2.2]),
                 8: torch.tensor([2.3])}
    for ii, iu in enumerate(torch.tensor([1, 6, 7, 8])):
        mask = geo.atomic_numbers == iu
        compr[mask] = init_dict[iu.tolist()]

    # ================= #
    # DFTB calculations #
    # ================= #
    basis = Basis(geo.atomic_numbers, shell_dict)
    h_feed = VcrFeed.from_dir(
            path_to_vcr, shell_dict, vcr_grid,
            skf_type='h5', geometry=geo, integral_type='H',
            interpolation='BicubInterp')
    s_feed = VcrFeed.from_dir(
            path_to_vcr, shell_dict, vcr_grid,
            skf_type='h5', geometry=geo, integral_type='S',
            interpolation='BicubInterp')
    ham = hs_matrix(geo, basis, h_feed, multi_varible=compr)
    over = hs_matrix(geo, basis, s_feed, multi_varible=compr)
    dftb = Dftb2(geo, shell_dict, path_to_vcr, repulsive=False)
    dftb(hamiltonian=ham, overlap=over)

    # ================ #
    # Plot error #
    # ================ #
    charge_error = data['charge'] - dftb.charge
    density, dist = torch.histogram(charge_error, bins=50, range=(-0.001, 0.001))
    print('density', density)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ii, jj, r in zip(int_h, int_vcr, path_to_skf):
        axes[0].plot(distance, torch.abs(ii-jj), label='WFCR: '+r, linewidth=3)
    axes[0].set_yscale("log")
    axes[0].set_xlabel('C-C distances')
    axes[0].set_ylabel('MAE of Hamiltonian integrals')
    axes[0].legend(loc="best", fontsize="large")
    axes[1].plot((dist[1:] + dist[:-1]) / 2, density, linewidth=3)
    axes[1].set_ylabel('Counts')
    axes[1].set_xlabel('Errors of Mulliken charges')
    axes[1].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    plt.savefig("bicub_error.png", dpi=300)
    plt.show()



def test_spl(device):
    # ================ #
    # std calculations #
    # ================ #
    geometry = Geometry.from_ase_atoms([molecule('CH4')], device=device)
    path_to_skf = './mio_new.hdf'
    basis = Basis(geometry.atomic_numbers, shell_dict)

    h_feed = SkfFeed.from_dir(
        path_to_skf, shell_dict, geometry=geometry,
        interpolation='PolyInterpU', integral_type='H')
    s_feed = SkfFeed.from_dir(
        path_to_skf, shell_dict, geometry=geometry,
        interpolation='PolyInterpU', integral_type='S')

    h_spl = hs_matrix(geometry, basis, h_feed)
    s_vcr = hs_matrix(geometry, basis, s_feed)
    dftb2 = Dftb2(geometry, shell_dict=shell_dict,
                  path_to_skf=path_to_skf, skf_type='skf')
    dftb2(hamiltonian=h_spl, overlap=s_vcr)

    # ================ #
    # spl calculations #
    # ================ #
    geometry = Geometry.from_ase_atoms([molecule('CH4')], device=device)
    path_to_skf = './mio_new.hdf'
    basis = Basis(geometry.atomic_numbers, shell_dict)

    h_feed = SkfFeed.from_dir(
        path_to_skf, shell_dict, geometry=geometry,
        interpolation='Spline1d', integral_type='H')
    s_feed = SkfFeed.from_dir(
        path_to_skf, shell_dict, geometry=geometry,
        interpolation='Spline1d', integral_type='S')

    h_spl = hs_matrix(geometry, basis, h_feed)
    s_vcr = hs_matrix(geometry, basis, s_feed)
    dftb2 = Dftb2(geometry, shell_dict=shell_dict,
                  path_to_skf=path_to_skf, skf_type='skf')
    dftb2(hamiltonian=h_spl, overlap=s_vcr)


test_bicubic(torch.device('cpu'))
# test_interp()
