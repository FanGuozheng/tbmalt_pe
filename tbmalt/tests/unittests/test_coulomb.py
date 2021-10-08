"""Test alpha and 1/R matrix calculation for coulomb interaction in periodic system."""

import torch
import numpy as np
from tbmalt.common.batch import pack
from tbmalt.common.structures.system import System
from tbmalt.tb.coulomb import Coulomb
from tbmalt.common.structures.periodic import Periodic
torch.set_default_dtype(torch.float64)
torch.set_printoptions(15)


def test_get_alpha():
    """Test optimizing alpha for the Ewald sum."""
    latvec = torch.tensor([[0, 4., 0], [2., 0, 0], [0, 0, 2.]])
    coord = torch.tensor([[0, 0, 0], [0, 2., 0]])
    atompair = torch.tensor([1, 1])
    geo = System(atompair, coord, latvec)
    periodic = Periodic(geo, geo.cell, cutoff=10.)
    coulomb = Coulomb(geo, periodic)
    assert torch.max(abs(coulomb.alpha - alpha_from_dftbplus)) < 1E-14, 'Tolerance check'


def test_get_alpha_batch():
    """Test optimizing alphas with different lattice vectors for the Ewald sum."""
    latvec = [torch.tensor([[0, 4., 0], [2., 0, 0], [0, 0, 2.]]),
              torch.tensor([[0, 8., 0], [4., 0, 0], [0, 0, 4.]])]
    positions = torch.tensor([[[0, 0, 0], [0, 2., 0]], [[0, 0, 0], [0, 2., 0]]])
    atompair = [torch.tensor([1, 1]), torch.tensor([1, 1])]
    geo = System(atompair, positions, latvec)
    periodic = Periodic(geo, geo.cell, cutoff=10.)
    coulomb = Coulomb(geo, periodic)
    assert torch.max(abs(coulomb.alpha - alpha_batch_from_dftbplus)) < 1E-14, 'Tolerance check'


def test_get_alpha_batch_trigonal():
    """Test optimizing alphas with different triclinic lattice vectors for the Ewald sum."""
    latvec = [torch.tensor([[3., 3., 0], [0, 3., 3.], [3., 0, 3.]]),
              torch.tensor([[5., 5., 0], [0, 5., 5.], [5., 0, 5.]]),
              torch.tensor([[3., -3., 3.], [-3., 3., 3.], [3., 3., -3.]])]
    positions = torch.tensor([[[0, 0, 0], [0, 2., 0]], [[0, 0, 0], [0, 2., 0]],
                              [[0, 0, 0], [0, 2., 0]]])
    atompair = [torch.tensor([1, 1]), torch.tensor([1, 1]), torch.tensor([1, 1])]
    geo = System(atompair, positions, latvec)
    periodic = Periodic(geo, geo.cell, cutoff=10.)
    coulomb = Coulomb(geo, periodic)
    assert torch.max(abs(coulomb.alpha - alpha_batch_from_dftbplus_trigonal)) < 1E-14, 'Tolerance check'


def test_get_alpha_batch_triclinc():
    """Test optimizing alphas with different triclinic lattice vectors for the Ewald sum."""
    latvec = [torch.tensor([[1., 1., 0], [0, 3., 3.], [2., 0, 2.]]),
              torch.tensor([[-2., 2., 2.], [3., -3., 3.], [4., 4., -4.]]),
              torch.tensor([[2., 0, 0], [3., 3., 3.], [0, 0, 4.]])]
    positions = torch.tensor([[[0, 0, 0], [0, 2., 0]], [[0, 0, 0], [0, 2., 0]],
                              [[0, 0, 0], [0, 2., 0]]])
    atompair = [torch.tensor([1, 1]), torch.tensor([1, 1]), torch.tensor([1, 1])]
    geo = System(atompair, positions, latvec)
    periodic = Periodic(geo, geo.cell, cutoff=10.)
    coulomb = Coulomb(geo, periodic)
    assert torch.max(abs(coulomb.alpha - alpha_batch_from_dftbplus_triclinc)) < 1E-14, 'Tolerance check'


def test_get_invr_h2():
    """Test 1/R matrix calculation in periodic system."""
    latvec = torch.tensor([[0, 4., 0], [2., 0, 0], [0, 0, 2.]])
    coord = torch.tensor([[0, 0, 0], [0, 2., 0]])
    atompair = torch.tensor([1, 1])
    geo = System(atompair, coord, latvec)
    periodic = Periodic(geo, geo.cell, cutoff=10.)
    coulomb = Coulomb(geo, periodic)
    assert torch.max(abs(coulomb.invrmat[0] - invr_h2_from_dftbplus)) < 1E-14, 'Tolerance check'


def test_get_invr_batch():
    """Test 1/R matrix batch calculation in periodic system."""
    latvec = torch.tensor([[0, 8., 0], [4., 0, 0], [0, 0, 4.]]).repeat(3, 1, 1)

    # Batch calculation for h2, ch4 and co2 periodic systems
    atompair = [torch.tensor([1, 1]), torch.tensor([6, 1, 1, 1, 1]), torch.tensor([8, 6, 8])]
    positions = [torch.tensor([[0, 0, 0], [0, 4., 0]]),
                 torch.tensor([[0, 0, 0], [0.6287614522, 0.6287614522, 0.6287614522],
                               [-0.6287614522, -0.6287614522, 0.6287614522],
                               [-0.6287614522, 0.6287614522, -0.6287614522],
                               [0.6287614522, -0.6287614522, -0.6287614522]]),
                 torch.tensor([[-2.0357279573e-03, -1.7878314480e-02, 1.1467019320e+00],
                               [5.4268823005e-03,  4.7660354525e-02, 7.7558560297e-03],
                               [-2.0357279573e-03, -1.7878314480e-02, -1.1525206566e+00]])]
    geo = System(atompair, positions, latvec)
    periodic = Periodic(geo, geo.cell, cutoff=10.)
    coulomb = Coulomb(geo, periodic)
    assert torch.max(abs(coulomb.invrmat - invr_from_dftbplus)) < 1E-14, 'Tolerance check'


def test_get_invr_batch_difflatvec():
    """Test 1/R matrix batch calculation with different lattice vectors as input in periodic system."""
    latvec = [torch.tensor([[0, 4., 0], [2., 0, 0], [0, 0, 2.]]),
              torch.tensor([[0, 8., 0], [4., 0, 0], [0, 0, 4.]])]

    atompair = [torch.tensor([1, 1]), torch.tensor([8, 6, 8])]
    coord = [torch.tensor([[0, 0, 0], [0, 2., 0]]),
             torch.tensor([[-2.0357279573e-03, -1.7878314480e-02, 1.1467019320e+00],
                           [5.4268823005e-03,  4.7660354525e-02, 7.7558560297e-03],
                           [-2.0357279573e-03, -1.7878314480e-02, -1.1525206566e+00]])]
    geo = System(atompair, coord, latvec)
    periodic = Periodic(geo, geo.cell, cutoff=10.)
    coulomb = Coulomb(geo, periodic)
    assert torch.max(abs(coulomb.invrmat - invr2_from_dftbplus)) < 1E-14, 'Tolerance check'


def test_get_invr_batch_trigonal_latvec():
    """Test 1/R matrix batch calculation with different trigonal lattice vectors as input in periodic system."""
    latvec = [torch.tensor([[3., 3., 0], [0, 3., 3.], [3., 0, 3.]]),
              torch.tensor([[5., 5., 0], [0, 5., 5.], [5., 0, 5.]]),
              torch.tensor([[3., -3., 3.], [-3., 3., 3.], [3., 3., -3.]])]

    atompair = [torch.tensor([1, 1]), torch.tensor([1, 1]), torch.tensor([1, 1])]
    coord = [torch.tensor([[0, 0, 0], [0, 2., 0]]),
             torch.tensor([[0, 0, 0], [0, 2., 0]]),
             torch.tensor([[0, 0, 0], [0, 2., 0]])]
    geo = System(atompair, coord, latvec)
    periodic = Periodic(geo, geo.cell, cutoff=10.)
    coulomb = Coulomb(geo, periodic)
    assert torch.max(abs(coulomb.invrmat - invr3_from_dftbplus)) < 1E-14, 'Tolerance check'


def test_get_invr_batch_triclinc_latvec():
    """Test 1/R matrix batch calculation with different trigonal lattice vectors as input in periodic system."""
    latvec = [torch.tensor([[1., 1., 0], [0, 3., 3.], [2., 0, 2.]]),
              torch.tensor([[-2., 2., 2.], [3., -3., 3.], [4., 4., -4.]]),
              torch.tensor([[2., 0, 0], [3., 3., 3.], [0, 0, 4.]])]

    atompair = [torch.tensor([1, 1]), torch.tensor([1, 1]), torch.tensor([8, 6, 8])]
    coord = [torch.tensor([[0, 0, 0], [0, 2, 0]]),
             torch.tensor([[0, 0, 0], [0, 2, 0]]),
             torch.tensor([[-2.0357279573e-03, -1.7878314480e-02, 1.1467019320e+00],
                           [5.4268823005e-03,  4.7660354525e-02, 7.7558560297e-03],
                           [-2.0357279573e-03, -1.7878314480e-02, -1.1525206566e+00]])]
    geo = System(atompair, coord, latvec)
    periodic = Periodic(geo, geo.cell, cutoff=10.)
    coulomb = Coulomb(geo, periodic)
    assert torch.max(abs(coulomb.invrmat - invr4_from_dftbplus)) < 1E-14, 'Tolerance check'


# Alpha values from dftb+
alpha_from_dftbplus = torch.tensor([0.47513600000000000])
alpha_batch_from_dftbplus = torch.tensor([0.47513600000000000, 0.23756800000000000])
alpha_batch_from_dftbplus_trigonal = torch.tensor([0.41943039999999998, 0.20971519999999999, 0.29360127999999996])
alpha_batch_from_dftbplus_triclinc = torch.tensor([0.61521919999999986, 0.30408703999999998, 0.54525952000000011])

# 1/R matrix for h2 with [[0, 4, 0], [2, 0, 0], [0, 0, 2]] as lattice vector
invr_h2_from_dftbplus = torch.tensor([[-0.47780521676190824, -0.27291142548538699],
                                      [-0.27291142548538699, -0.47780521676190824]])

# 1/R matrix for h2 with [[0, 8, 0], [4, 0, 0], [0, 0, 4]] as lattice vector
invr2_h2_from_dftbplus = torch.tensor([[-0.23890261448525388, -0.13645572591320607],
                                       [-0.13645572591320607, -0.23890261448525388]])

# 1/R matrix for h2 with [[3, 3, 0], [0, 3, 3], [3, 0, 3]] as lattice vector
invr3_h2_from_dftbplus = torch.tensor([[-0.40436750235111368, -6.8028764696638461E-002],
                                       [-6.8028764696638461E-002, -0.40436750235111368]])

# 1/R matrix for h2 with [[5, 5, 0], [0, 5, 5], [5, 0, 5]] as lattice vector
invr4_h2_from_dftbplus = torch.tensor([[-0.24262047344764978, 3.8984627392439006E-002],
                                       [3.8984627392439006E-002, -0.24262047344764978]])

# 1/R matrix for h2 with [[3, -3, 3], [-3, 3, 3], [3, 3, -3]] as lattice vector
invr5_h2_from_dftbplus = torch.tensor([[-0.32096660462527188, -1.7963256710504416E-002],
                                       [-1.7963256710504416E-002, -0.32096660462527188]])

# 1/R matrix for h2 with [[1, 1, 0], [0, 3, 3], [2, 0, 2]] as lattice vector
invr6_h2_from_dftbplus = torch.tensor([[-0.49711993867470139, -0.20209287655651773],
                                       [-0.20209287655651773, -0.49711993867470139]])

# 1/R matrix for h2 with [[-2, 2, 2], [3, -3, 3], [4, 4, -4]] as lattice vector
invr7_h2_from_dftbplus = torch.tensor([[-0.30660822168849816, -1.3395616647138307E-002],
                                       [-1.3395616647138307E-002, -0.30660822168849816]])

# 1/R matrix for ch4 with [[0, 8, 0], [4, 0, 0], [0, 0, 4]] as lattice vector
invr_ch4_from_dftbplus = torch.tensor([[-0.23890261448525388, 0.25618239418944322, 0.25618239418944333, 0.25618239418944333, 0.25618239418944333],
                                       [0.25618239418944322, -0.23890261448525388, 6.7967077421936944E-002, 0.11166535647645823, 6.7967077421936944E-002],
                                       [0.25618239418944322, 6.7967077421936944E-002, -0.23890261448525388, 6.7967077421936889E-002, 0.11166535647645823],
                                       [0.25618239418944322, 0.11166535647645823, 6.7967077421936944E-002, -0.23890261448525388, 6.7967077421936944E-002],
                                       [0.25618239418944322, 6.7967077421936944E-002, 0.11166535647645823, 6.7967077421936944E-002, -0.23890261448525388]])

# 1/R matrix for co2 with [[0, 8, 0], [4, 0, 0], [0, 0, 4]] as lattice vector
invr_co2_from_dftbplus = torch.tensor([[-0.23890261448525388, 0.25142867181718609, 0.13878449166434717],
                                       [0.25142867181718609, -0.23890261448525388, 0.24402908422714895],
                                       [0.13878449166434717, 0.24402908422714895, -0.23890261448525388]])

# 1/R matrix for co2 with [[2, 0, 0], [3, 3, 3], [0, 0, 4]] as lattice vector
invr2_co2_from_dftbplus = torch.tensor([[-0.45947292231582787, -7.7500649798791788E-003, -0.14118322393701194],
                                       [-7.7500649798791788E-003, -0.45947292231582787, -1.8028562581767889E-002],
                                       [-0.14118322393701194, -1.8028562581767889E-002, -0.45947292231582787]])

# 1/R matrix from dftb+
invr_from_dftbplus = pack([invr2_h2_from_dftbplus, invr_ch4_from_dftbplus, invr_co2_from_dftbplus])
invr2_from_dftbplus = pack([invr_h2_from_dftbplus, invr_co2_from_dftbplus])
invr3_from_dftbplus = pack([invr3_h2_from_dftbplus, invr4_h2_from_dftbplus, invr5_h2_from_dftbplus])
invr4_from_dftbplus = pack([invr6_h2_from_dftbplus, invr7_h2_from_dftbplus, invr2_co2_from_dftbplus])
