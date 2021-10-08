"""K points sampling for periodic system."""
import torch
import numpy as np
from tbmalt.common.batch import pack
Tensor = torch.Tensor
_bohr = 0.529177249
tol = 1e-4
minlim = - tol
maxlim = 1 - tol

def build_supercell(coeffsandshifts: Tensor):
    """Build the super structure and genearte a set of special K-points.

    Arguments:
        coeffsandshifts: Parameters for building the super structure, consisting of coefficients and shifts.

    Return:
        nkpoints: Number of the K-points.
        kpoints: K-points for periodic system.
        kweights: The weights of each K-point.

    Examples:
        >>> import torch
        >>> from ksampling import build_supercell
        >>> torch.set_default_dtype(torch.float64)
        >>> torch.set_printoptions(15)
        >>> coeffsandshifts = torch.tensor([[[2, 0, 0], [0, 2, 0], [0, 0, 3], [0.5, 0.5, 0]]])
        >>> nkpoints, kpoints, kweights = build_supercell(coeffsandshifts)
        >>> print(nkpoints)
        tensor([12.])

    """
    # Coefficients used to build supercell vectors from original lattice vectors
    coeffs = coeffsandshifts[:, :3]

    # Shift of the grid along the three reciprocal lattice vectors
    shifts = coeffsandshifts[:, 3]

    # Number of Kpoints without reducing
    nkpoints = torch.det(coeffs)

    # Inverse coefficients
    invcoeffs = torch.transpose(torch.solve(torch.eye(
                coeffs.shape[-1]), coeffs)[0], -1, -2)

    # Get ranges for building supercell vectors
    imgrange1 = torch.min(torch.min(coeffs, dim=1).values, torch.tensor([0, 0, 0]))
    imgrange2 = torch.sum(coeffs, dim=1) - 1
    imgrange = torch.stack((imgrange1, imgrange2), 0)

    # rr: Relative coordinate with respect to the original reciprocal lattice
    leng = imgrange[1, :].long() - imgrange[0, :].long() + 1
    transvec = pack([torch.stack([
            torch.linspace(iran[0, 0], iran[1, 0],
                           ile[0]).repeat_interleave(ile[2] * ile[1]),
            torch.linspace(iran[0, 1], iran[1, 1],
                           ile[1]).repeat(ile[0]).repeat_interleave(ile[2]),
            torch.linspace(iran[0, 2], iran[1, 2],
                           ile[2]).repeat(ile[0] * ile[1])])
            for ile, iran in zip(leng, imgrange.transpose(1, 0))])
    rr = pack([(icf.transpose(0, 1) @ itv.T.unsqueeze(-1)).squeeze(-1)
                         for icf, itv in zip(invcoeffs, transvec)])

    # Mask for rr
    mask_rr = torch.all(torch.all(torch.all(rr.le(maxlim) * rr.ge(minlim), dim=-1), dim=1), dim=0)
    if mask_rr == False:
        raise ValueError('Failed to find all K-points.')

    # Add shift and obtain Kpoints
    ss = torch.stack([torch.matmul(icf, ishif) for icf, ishif in zip(invcoeffs, shifts)])
    kpoints = torch.stack([(irr + iaa) for irr, iaa in zip(rr, ss)])

    # Kweights
    iweight = 1 / nkpoints
    kweights = pack([torch.cat((int(nkpoints[ii]) * [torch.unsqueeze(torch.tensor([iweight[ii]]), dim=0)]))
                    for ii in range(nkpoints.size(0))])

    return nkpoints, kpoints, kweights

def reduce_supercell(nkpoints: Tensor, kpoints: Tensor, kweights: Tensor):
    """Reduce the K-points obtained by supercell method through inversion.

     Arguments:
        nkpoints: Number of the K-points.
        Kpoints: K-points for periodic system.
        Kweights: The weights of each K-point.

    Return:
        nkpoints_re: Number of the K-points after reducing.
        kpoints_re: K-points for periodic system after reducing.
        kweights_re: The weights of each K-point after reducing.

    Examples:
        >>> import torch
        >>> from ksampling import build_supercell
        >>> from ksampling import reduce_supercell
        >>> torch.set_default_dtype(torch.float64)
        >>> torch.set_printoptions(15)
        >>> coeffsandshifts = torch.tensor([[[2, 0, 0], [0, 2, 0], [0, 0, 3], [0.5, 0.5, 0]]])
        >>> nkpoints, kpoints, kweights = build_supercell(coeffsandshifts)
        >>> nkpoints_re, kpoints_re, kweights_re = reduce_supercell(nkpoints, kpoints, kweights)
        >>> print(nkpoints_re)
        tensor([6.])
        >>> print(kpoints_re)
        tensor([[0.250000000000000, 0.250000000000000, 0.000000000000000],
        [0.250000000000000, 0.250000000000000, 0.333333333333333],
        [0.250000000000000, 0.250000000000000, 0.666666666666667],
        [0.250000000000000, 0.750000000000000, 0.000000000000000],
        [0.250000000000000, 0.750000000000000, 0.333333333333333],
        [0.250000000000000, 0.750000000000000, 0.666666666666667]])
        >>> print(kweights_re)
        tensor([[0.166666666666667],
        [0.166666666666667],
        [0.166666666666667],
        [0.166666666666667],
        [0.166666666666667],
        [0.166666666666667]])

    """
    # Inversion
    kp_in = torch.remainder((- 1.0 * kpoints), 1.0)

    # Build a mask for reducing
    mask_reduce = torch.ones_like(kpoints, dtype=bool)

    # Number of the K-points after reducing
    nkpoints_re = torch.clone(nkpoints)

    # Reducing
    for ibatch in range(nkpoints.size(0)):
        for ikp in range(int(nkpoints[ibatch])):
            for jkp in range(ikp + 1, int(nkpoints[ibatch])):
                if torch.all(abs(kpoints[ibatch, jkp] - kp_in[ibatch, ikp]) < tol):
                    mask_reduce[ibatch, jkp] = False
                    kweights[ibatch, ikp] = kweights[ibatch, ikp] + kweights[ibatch, jkp]
                    nkpoints_re[ibatch] -= 1

    # Build a mask according to kweight for batch calculation
    mask_kweight = torch.cat((3 * [torch.tensor(kweights, dtype=bool)]), dim=-1)

    # Mask for K-points
    mask_kpoint = torch.all(mask_reduce & mask_kweight, dim=-1)

    # K-points for periodic system after reducing
    kpoints_re = torch.clone(kpoints[mask_kpoint])

    # The weights of each K-point after reducing
    kweights_re = torch.clone(kweights[mask_kpoint])

    return nkpoints_re, kpoints_re, kweights_re


def specify_kpoint(kpointsandweights: Tensor, latvec: Tensor, **kwargs):
    """Explicitly specify K-points for the periodic system.

    Arguments:
        kpointsandweights: The coordinates and weights of K-points.
        latvec: Lattice vector describing the geometry of periodic system.

    Return:
        nkpoints: Number of the K-points.
        kpoints: K-points for periodic system.
        kweights: The weights of each K-point.

    """
    if kpointsandweights.dim() != 2:
        raise ValueError('Specified kpointsandweights dimension should be 2')

    if kpointsandweights.size(1) != 4:
        raise ValueError('Please use 4 values to specify the coordinates and weights of K-points')

    # Default coordinate of K-points is fraction coordinate
    coor = kwargs.get('coor', 'fraction')

    # Default unit of K-points is bohr
    unit = kwargs.get('unit', 'bohr')

    # Number of K-points
    nkpoints = torch.tensor([kpointsandweights.size(0)])

    # Read input coordinates of K-points
    kpoints = kpointsandweights[:, :3]

    # Normalize the input weights of K-points
    kweights = torch.unsqueeze((1. / sum(kpointsandweights[:, 3])) * kpointsandweights[:, 3], dim=-1)

    if unit in ('angstrom', 'Angstrom'):
        kpoints = kpoints / _bohr
        latvec = latvec / _bohr
    elif unit not in ('bohr', 'Bohr'):
        raise ValueError('Unit is either angstrom or bohr')

    if coor in ('absolute', 'Absolute'):
        kpoints = absolute_to_fraction(kpoints, latvec)
    elif coor not in('fraction', 'Fraction'):
        raise ValueError('Coordinate is either absolute or fraction')

    return nkpoints, kpoints, kweights


def fraction_to_absolute(kpoints: Tensor, invlatvec: Tensor):
    """Transfer K-points in fraction coordinates to absolute space."""
    return torch.stack([torch.matmul(invlatvec, kpoints[ikp])
                        for ikp in range(kpoints.size(0))])


def absolute_to_fraction(kpoints: Tensor, latvec: Tensor):
    """Transfer K-points in absolute space to fraction coordinates."""
    return torch.stack([torch.matmul(latvec, kpoints[ikp].double())
                        for ikp in range(kpoints.size(0))])
