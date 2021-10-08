"""Test periodic Hamiltonian and overlap.

The second derivative in SK integral result in slightly accuracy decrease."""
import os
import re
import torch
import numpy as np
from tbmalt.common.structures.geometry import Geometry
from ase import Atoms
from ase.build import molecule as molecule_database
from tbmalt.common.structures.periodic import Periodic
from tbmalt.tb.sk import SKT
from tbmalt.io.loadskf import IntegralGenerator
from tbmalt.tb.dftb.scc import Scc
from tbmalt.common.parameter import Parameter
torch.set_default_dtype(torch.float64)
torch.set_printoptions(15)
_bohr = 0.529177249
os.system('cp -r /home/gz_fan/Public/tbmalt/slko .')
os.system('cp -r /home/gz_fan/Public/tbmalt/sk .')


def _get_cell_trans(latVec, cutoff, negExt=1, posExt=1, latvec_unit='bohr'):
    """Reproduce code originally from DFTB+ for test TBMaLT.

    This code is for single geometry and not vectorized, retain loop as DFTB+,
    to act as a reference for cell translation code in TBMaLT."""
    if latvec_unit == 'angstrom':
        latVec = latVec / _bohr
    recVec = torch.inverse(latVec)

    # get ranges of periodic boundary condition from negative to positive
    ranges = torch.zeros((2, 3), dtype=torch.int8)
    for ii in range(3):
        iTmp = torch.floor(cutoff * torch.sqrt(sum(recVec[:, ii] ** 2)))
        ranges[0, ii] = -negExt - iTmp
        ranges[1, ii] = posExt + iTmp

    # Length of the first, second and third column in ranges
    leng1, leng2, leng3 = ranges[1, :] - ranges[0, :] + 1
    ncell = leng1 * leng2 * leng3  # -> Number of lattice cells

    # Cell translation vectors in relative coordinates
    cellvec = torch.zeros(ncell, 3)
    col3 = torch.linspace(ranges[0, 2], ranges[1, 2], leng3)
    col2 = torch.linspace(ranges[0, 1], ranges[1, 1], leng2)
    col1 = torch.linspace(ranges[0, 0], ranges[1, 0], leng1)
    cellvec[:, 2] = col3.repeat(int(ncell.numpy() / leng3.numpy()))
    col2 = col2.repeat(leng3, 1)
    col2 = torch.cat([(col2[:, ii]) for ii in range(leng2)])
    cellvec[:, 1] = col2.repeat(int(ncell.numpy() / (leng2 * leng3).numpy()))
    col1 = col1.repeat(leng3 * leng2, 1)
    cellvec[:, 0] = torch.cat([(col1[:, ii]) for ii in range(leng1)])

    # Cell translation vectors in absolute units
    rcellvec = torch.stack([torch.matmul(
        torch.transpose(latVec, 0, 1), cellvec[ii]) for ii in range(ncell)])

    return cellvec.T, rcellvec


def test_cell_nonpe_h2():
    positions = torch.tensor([[0., 0., 0.368583], [0., 0., -0.368583]])
    numbers = torch.tensor([1, 1])
    geo = Geometry(numbers, positions)
    sktable = IntegralGenerator.from_dir('./slko/auorg-1-1', geo)
    skt = SKT(geo, sktable)
    assert torch.max(abs(skt.H - h_h2)) < 1E-14, 'Tolerance check'
    assert torch.max(abs(skt.S - s_h2)) < 1E-14, 'Tolerance check'


def test_pe_normal_h2_mio():
    """Test H2 Hamiltonian and ovelap in periodic geometry."""
    latvec = torch.tensor([[6., 0., 0.], [0., 6., 0.], [0., 0., 6.]])
    cutoff = torch.tensor([9.98])
    positions = torch.tensor([[0., 0., 0.], [0., 0., 0.2]])
    numbers = torch.tensor([1, 1])
    cellvec_ref, rcellvec_ref = _get_cell_trans(latvec / _bohr, cutoff + 1)

    geo = Geometry(numbers, positions, latvec)
    sktable = IntegralGenerator.from_dir('./slko/mio-1-1', geo)
    periodic = Periodic(geo, geo.cell, sktable.cutoff, unit='bohr')
    assert torch.max(abs(periodic.cellvec[0] - cellvec_ref)) < 1E-14
    assert torch.max(abs(periodic.rcellvec[0] - rcellvec_ref)) < 1E-14

    skt = SKT(geo, sktable, periodic)
    print(skt.H, h_h2_pe, geo.positions, positions)
    assert torch.max(abs(skt.H[0] - h_h2_pe[:, torch.tensor([0, 2])]
                         )) < 1E-14, 'Tolerance check'
    assert torch.max(abs(skt.S[0] - s_h2_pe[:, torch.tensor([0, 2])]
                         )) < 1E-14, 'Tolerance check'


def test_pe_h2_ase():
    """Test ASE H2 Hamiltonian and ovelap in periodic geometry."""
    h2 = Atoms('HH', positions=[[0., 0., 0.], [0., 0., 0.2]],
               cell=[6., 6., 6.], pbc=[1, 1, 1])
    geo = Geometry.from_ase_atoms(h2)
    sktable = IntegralGenerator.from_dir('./slko/mio-1-1', geo)
    periodic = Periodic(geo, geo.cell, sktable.cutoff, unit='bohr')

    skt = SKT(geo, sktable, periodic)
    assert torch.max(abs(skt.H[0] - h_h2_pe[:, torch.tensor([0, 2])]
                         )) < 1E-14, 'Tolerance check'
    assert torch.max(abs(skt.S[0] - s_h2_pe[:, torch.tensor([0, 2])]
                         )) < 1E-14, 'Tolerance check'


def test_pe_normal_ch4():
    """Test CH4 Hamiltonian and ovelap in periodic geometry."""
    latvec = torch.tensor([[6., 0., 0.], [0., 6., 0.], [0., 0., 6.]])
    positions = torch.tensor([[.5, .5, .5], [.6, .6, .6], [.4, .6, .6],
                              [.6, .4, .6], [.6, .6, .4]])
    cutoff = torch.tensor([9.98])
    numbers = [torch.tensor([6, 1, 1, 1, 1])]
    cellvec_ref, rcellvec_ref = _get_cell_trans(latvec / _bohr, cutoff + 1)
    geo = Geometry(numbers, positions, latvec)
    sktable = IntegralGenerator.from_dir('./slko/mio-1-1', geo)
    periodic = Periodic(geo, geo.cell, sktable.cutoff, unit='bohr')
    skt = SKT(geo, sktable, periodic)
    shape_ch4 = torch.arange(0, h_ch4_pe.shape[0]) * 2
    assert torch.max(abs(periodic.cellvec[0] - cellvec_ref)) < 1E-14
    assert torch.max(abs(periodic.rcellvec[0] - rcellvec_ref)) < 1E-14
    assert torch.max(abs(skt.H[0, :8, :8] - h_ch4_pe[:, shape_ch4]
                         )) < 1E-11, 'Tolerance check'
    assert torch.max(abs(skt.S[0, :8, :8] - s_ch4_pe[:, shape_ch4]
                         )) < 1E-10, 'Tolerance check'


def test_pe_normal_co2():
    """Test CO2 Hamiltonian and ovelap in periodic geometry."""
    latvec = torch.tensor([[6., 0., 0.], [0., 6., 0.], [0., 0., 6.]])
    positions = torch.tensor([[.5, .5, .5], [.55, .55, .55], [.45, .45, .45]])
    numbers = torch.tensor([6, 8, 8])
    cutoff = torch.tensor([9.98])
    cellvec_ref, rcellvec_ref = _get_cell_trans(latvec / _bohr, cutoff + 1)
    geo = Geometry(numbers, positions, latvec)
    sktable = IntegralGenerator.from_dir('./slko/mio-1-1', geo)
    periodic = Periodic(geo, geo.cell, sktable.cutoff, unit='bohr')
    skt = SKT(geo, sktable, periodic)
    shape_co2 = torch.arange(0, h_co2_pe.shape[0]) * 2
    assert torch.max(abs(periodic.cellvec[0] - cellvec_ref)) < 1E-14
    assert torch.max(abs(periodic.rcellvec[0] - rcellvec_ref)) < 1E-14
    assert torch.max(abs(skt.H - h_co2_pe[:, shape_co2])) < 1E-11
    assert torch.max(abs(skt.S - s_co2_pe[:, shape_co2])) < 1E-11


def test_pe_normal_batch():
    """Test batch Hamiltonian and ovelap in periodic geometry."""
    latvec = [torch.tensor([[6., 0., 0.], [0., 6., 0.], [0., 0., 6.]]),
              torch.tensor([[6., 0., 0.], [0., 6., 0.], [0., 0., 6.]]),
              torch.tensor([[6., 0., 0.], [0., 6., 0.], [0., 0., 6.]])]
    positions = [torch.tensor([[.5, .5, .5], [.6, .6, .6], [.4, .6, .6],
                              [.6, .4, .6], [.6, .6, .4]]),
                 torch.tensor([[0., 0., 0.], [0., 0., .2]]),
                 torch.tensor([[.5, .5, .5], [.55, .55, .55], [.45, .45, .45]])]
    numbers = [torch.tensor([6, 1, 1, 1, 1]), torch.tensor([1, 1]),
               torch.tensor([6, 8, 8])]
    geo = Geometry(numbers, positions, latvec)
    sktable = IntegralGenerator.from_dir('./slko/mio-1-1', geo)
    periodic = Periodic(geo, geo.cell, sktable.cutoff, unit='bohr')
    skt = SKT(geo, sktable, periodic)
    shape_ch4 = torch.arange(0, h_ch4_pe.shape[0]) * 2
    assert torch.max(abs(skt.H[0, :len(shape_ch4), :len(shape_ch4)] -
                         h_ch4_pe[:, shape_ch4])) < 1E-11, 'Tolerance check'
    assert torch.max(abs(skt.S[0, :len(shape_ch4), :len(shape_ch4)] -
                         s_ch4_pe[:, shape_ch4])) < 1E-10, 'Tolerance check'
    shape_h2 = torch.arange(0, h_h2_pe.shape[0]) * 2
    assert torch.max(abs(skt.H[1, :2, :2] - h_h2_pe[:, shape_h2])) < 1E-14
    assert torch.max(abs(skt.S[1, :2, :2] - s_h2_pe[:, shape_h2])) < 1E-14
    shape_co2 = torch.arange(0, h_co2_pe.shape[0]) * 2
    assert torch.max(abs(skt.H[2, :len(shape_co2), :len(shape_co2)] -
                         h_co2_pe[:, shape_co2])) < 1E-11
    assert torch.max(abs(skt.S[2, :len(shape_co2), :len(shape_co2)] -
                         s_co2_pe[:, shape_co2])) < 1E-11


def get_matrix(filename):
    """Read DFTB+ hamsqr1.dat and oversqr.dat."""
    text = ''.join(open(filename, 'r').readlines())
    string = re.search('(?<=MATRIX\n).+(?=\n)', text, flags=re.DOTALL).group(0)
    out = np.array([[float(i) for i in row.split()]
                    for row in string.split('\n')])
    return torch.from_numpy(out)


h_h2 = get_matrix('./sk/h2/hamsqr1.dat')
s_h2 = get_matrix('./sk/h2/oversqr.dat')
h_h2_pe = get_matrix('./sk/h2/hamsqr1.dat.pe')
s_h2_pe = get_matrix('./sk/h2/oversqr.dat.pe')
h_ch4 = get_matrix('./sk/ch4/hamsqr1.dat')
s_ch4 = get_matrix('./sk/ch4/oversqr.dat')
h_ch4_pe = get_matrix('./sk/ch4/hamsqr1.dat.pe')
s_ch4_pe = get_matrix('./sk/ch4/oversqr.dat.pe')
h_co2 = get_matrix('./sk/co2/hamsqr1.dat')
s_co2 = get_matrix('./sk/co2/oversqr.dat')
h_co2_pe = get_matrix('./sk/co2/hamsqr1.dat.pe')
s_co2_pe = get_matrix('./sk/co2/oversqr.dat.pe')
