"""Slater-Koster integrals related."""
import numpy as np
import torch
from tbmalt.common.structures.basis import Basis
from tbmalt.common.batch import pack
from numbers import Real
from typing import Union
Tensor = torch.Tensor
Size = torch.Size
_sqr3 = np.sqrt(3.)
_hsqr3 = 0.5 * np.sqrt(3.)


class SKT:
    """Construct a Hamiltonian or overlap matrices for both single and batch.

    Construct a Hamiltonian or overlap matrices from integral values retrieved
    from SK tables using batch-wise operable Slater-Koster transformations.

    Arguments:
        system: The molecule's Basis instance. This stores the data needed to
            perform the necessary masking operations.
        sktable: An object which can be called with distances list, atomic
            number pair, azimuthal number pair.

    Returns:
        HS: A tensor holding Hamiltonian or Overlap matrices associated.

    Examples:
        >>> from ase.build import molecule as molecule_database
        >>> from tbmalt.common.structures.system import System
        >>> from tbmalt.io.loadskf import IntegralGenerator
        >>> from tbmalt.tb.sk import SKT
        >>> system = System.from_ase_atoms([molecule_database('CH4')])
        >>> sktable = IntegralGenerator.from_dir('../tests/unittests/slko//mio-1-1', system)
        >>> skt = SKT(system, sktable)
        >>> skt.H.shape
        >>> torch.Size([1, 8, 8])
    """

    def __init__(self, system: object, sktable: object, periodic=None,
                 kpoint=None, **kwargs):
        self.system = system
        self.periodic = periodic
        self.H = torch.zeros(self.system.hs_shape)
        self.S = torch.zeros(self.system.hs_shape)

        compr = kwargs.get('compression_radii', None)

        if compr is not None:
            compr = compr if compr.dim() == 2 else compr.unsqueeze(0)

        basis = Basis(self.system)

        # create mask matrices which show full/block orbital information
        if not periodic:
            l_mat_f = basis.azimuthal_matrix(mask=True)
            l_mat_b = basis.azimuthal_matrix(block=True, mask=True)

        # return all terms without mask for on-site terms when periodic
        else:
            l_mat_f = basis.azimuthal_matrix(mask=False)
            l_mat_b = basis.azimuthal_matrix(block=True, mask=False)

        # atom indices and atomic numbers
        i_mat_b = basis.index_matrix('block')
        an_mat_a = basis.atomic_number_matrix('atom')
        vec_mat_a = self._position_vec()

        # Loop over each type of azimuthal-pair interaction
        for l_pair, f in zip(_SK_interactions, _SK_functions):
            index_mask_b = torch.where((l_mat_b == l_pair).all(dim=-1))

            if len(index_mask_b[0]) == 0:
                continue

            # gather the the atomic index mask.
            index_mask_a = tuple(i_mat_b[index_mask_b].T)

            # Add system indices back in to index_mask_a
            index_mask_a = (index_mask_b[0],) + index_mask_a

            # gather atomic numbers, distances, directional cosigns
            gathered_an = an_mat_a[index_mask_a]
            gathered_dists, _ncell = self._select_distance(index_mask_a)

            # mask for on-site terms when distances are zero
            mask_os = gathered_dists == 0

            # replace the zero distances by a large value, return zero for integral
            gathered_dists[mask_os] = 999

            # mask for on-site terms
            gathered_vecs = self._select_position_vec(vec_mat_a, index_mask_a)
            mask_os2 = torch.all(gathered_vecs == 0, -1)
            gathered_vecs[mask_os2] = 999

            compr_pair = self._get_compr_pair(compr, index_mask_a)

            if compr is not None:  # Construct compression radii pair
                compr = compr if compr.dim() == 2 else compr.unsqueeze(0)
                compr_pair = torch.stack([
                    compr[index_mask_a[0], index_mask_a[1]],
                    compr[index_mask_a[0], index_mask_a[2]]]).T
            else:
                compr_pair = None

            # request integrals from the integrals
            gathered_h, gathered_s = integral_retrieve(
                gathered_dists, gathered_an, sktable, l_pair, compr_pair, **kwargs)

            # call relevant Slater-Koster function to get the sk-block
            h_data = f(gathered_vecs, gathered_h)
            s_data = f(gathered_vecs, gathered_s)

            if l_pair[0] != 0:
                nr, nc = l_pair * 2 + 1  # № of rows/columns of this sub-block
                # № of sub-blocks in each system.
                nl = index_mask_b[0].unique(return_counts=True)[1]
                # Indices of each row
                r_offset = torch.arange(nr).expand(len(index_mask_b[-1]), nc).T
                # Index list to order the rows of all ℓ₁-ℓ₂ sub-blocks so that
                # the results can be assigned back into the H/S tensors without
                # mangling.
                r = (r_offset + index_mask_b[-2]).T.flatten().split((
                    nr * nl).tolist())
                r, _mask = pack(r, value=99, return_mask=True)
                r = r.cpu().sort(stable=True).indices
                # Correct the index list.
                r[1:] = r[1:] + (nl.cumsum(0)[:-1] * nr).unsqueeze(
                    -1).repeat_interleave((r.shape[-1]), dim=-1)
                r = r[_mask]
                # The "r" tensor only takes into account the central image, thus
                # the other images must now be taken into account.
                n = int(h_data.nelement() / (r.nelement() * nr))
                r = (r + (torch.arange(n) * len(r)).view(-1, 1)).flatten()
                # Perform the reordering
                h_data, s_data = h_data.view(-1, nc)[r], s_data.view(-1, nc)[r]

            h_data_shaped, s_data_shaped = self._reshape_hs(
                h_data.flatten(), s_data.flatten(), self.periodic, kpoint,
                index_mask_a[0], _ncell)

            # Create the full size index mask which will assign the results
            index_mask_f = torch.where((l_mat_f == l_pair).all(dim=-1))
            self.H[index_mask_f] = h_data_shaped
            self.H.transpose(-1, -2)[index_mask_f] = h_data_shaped
            self.S[index_mask_f] = s_data_shaped
            self.S.transpose(-1, -2)[index_mask_f] = s_data_shaped

        # get all the onsite mask for batch system
        mask_onsite = (
            torch.arange(self.H.shape[0]).repeat_interleave(self.H.shape[1]),
            torch.arange(self.H.shape[1]).repeat(self.H.shape[0]),
            torch.arange(self.H.shape[2]).repeat(self.H.shape[0]))
        self.H[mask_onsite] = self.H[mask_onsite] + onsite_retrieve(an_mat_a, sktable,
                                                                    self.H.shape, **kwargs)
        self.S[mask_onsite] = self.S[mask_onsite] + 1.

        # return U Hubbert
        self.U = U_retrieve(an_mat_a, sktable, self.system.numbers.shape,
                            **kwargs)

    def _select_distance(self, index_mask_a):
        """Gather distance."""
        if self.periodic is None:  # -> molecule
            return self.system.distances[index_mask_a], torch.tensor([1])
        else:  # -> solid, first dim is batch, second is cell
            dis = self.periodic.neighbour_dis
            return dis.permute(0, 2, 3, 1)[index_mask_a].T, dis.size(-3)

    def _position_vec(self):
        if self.periodic is None:
            vec_mat_a = self.system.positions.unsqueeze(-3) - \
                self.system.positions.unsqueeze(-2)
            return torch.nn.functional.normalize(vec_mat_a, p=2, dim=-1)
        else:
            vec_mat_a = self.periodic.neighbour_vec
            return torch.nn.functional.normalize(vec_mat_a, p=2, dim=-1)

    def _select_position_vec(self, vec_mat_a, index_mask_a):
        if self.periodic is None:
            return vec_mat_a[index_mask_a]
        else:
            # permute to transfer cell dimension to the last to match mask
            _vec_selec = vec_mat_a.permute(0, 2, 3, 4, 1)[index_mask_a]
            return _vec_selec.permute(2, 0, 1).reshape(-1, _vec_selec.shape[1])

    def _select_kpoint(self, kpoint, periodic, mask):
        """Select kpoint for each interactions."""
        if kpoint is None:
            batch_size = periodic.periodic_distances.shape[0]
            cell_vec = periodic.cellvec
            # kpoint = np.pi * torch.ones(batch_size, 1, 3)
            kpoint = torch.zeros(batch_size, 1, 3)

            # 1st dim of returned dot_product(kpoint, cellvec) is cellvec size
            return torch.bmm(kpoint[mask], cell_vec[mask]).squeeze(1).T
        else:
            cell_vec = periodic.cellvec
            return torch.bmm(kpoint.unsqueeze(1)[mask], cell_vec[mask]).squeeze(1).T

    def _get_compr_pair(self, compr, index_mask_a):
        """Return compression radii pair."""
        if compr is not None:  # Construct compression radii pair
            compr = compr if compr.dim() == 2 else compr.unsqueeze(0)
            return torch.stack([
                compr[index_mask_a[0], index_mask_a[1]],
                compr[index_mask_a[0], index_mask_a[2]]]).T
        else:
            return None

    def _reshape_hs(self, h_data_shaped, s_data_shaped, periodic, kpoint, mask, cell_size):
        """Reshape periodic H and S and sum over cell dimension."""
        if periodic is None:
            return h_data_shaped, s_data_shaped
        else:
            # sum along cell vector dimension
            _h = (h_data_shaped.reshape(cell_size, -1)).sum(0)
            _s = (s_data_shaped.reshape(cell_size, -1)).sum(0)

            return _h, _s


def split_by_size(tensor: Tensor, split_sizes: Tensor, dim=0):
    """Splits tensor according to chunks of split_sizes.

    Arguments:
        tensor: Tensor to be split
        split_sizes: Size of the chunks
        dim: Dimension along which to split tensor

    Returns:
        chunked: List of tensors viewing the original ``tensor`` as a
            series of ``split_sizes`` sized chunks.
    """
    if dim < 0:  # Shift dim to be compatible with torch.narrow
        dim += tensor.dim()

    # Ensure the tensor is large enough to satisfy the chunk declaration.
    if tensor.size(dim) != split_sizes.sum():
        if tensor.size(dim) % split_sizes.sum().numpy() == 0:
            expand = int(tensor.size(dim) / split_sizes.sum().numpy())
            split_sizes = split_sizes.repeat(expand)
        else:
            raise KeyError(
                'Sum of split sizes fails to match tensor length along specified dim')

    # Identify the slice positions
    splits = torch.cumsum(torch.Tensor([0, *split_sizes]), dim=0)[:-1]

    # Return the sliced tensor. use torch.narrow to avoid data duplication
    return tuple(tensor.narrow(int(dim), int(start), int(length))
                 for start, length in zip(splits, split_sizes))


def integral_retrieve(distances: Tensor, atom_pairs: Tensor, integral: object,
                      l_pair: Tensor, compr_pair=None, **kwargs):
    """Integral retrieval operation."""
    if distances.dim() == 2:  # -> periodic
        atom_pairs = atom_pairs.repeat(distances.shape[0], 1)
        sktable_h = torch.zeros(len(atom_pairs), l_pair.min() + 1)
        sktable_s = torch.zeros(len(atom_pairs), l_pair.min() + 1)
        distances = distances.flatten()

        if compr_pair is not None:  # repeat the cell dimension
            compr_pair = compr_pair.repeat(distances.shape[0], 1)
    else:
        sktable_h = torch.zeros(len(atom_pairs), l_pair.min() + 1)
        sktable_s = torch.zeros(len(atom_pairs), l_pair.min() + 1)

    unique_atom_pairs = atom_pairs.unique(dim=0)
    with_variable = kwargs.get('with_variable', False)

    # loop over each of the unique atom_pairs
    for atom_pair in unique_atom_pairs:

        # index mask for gather & scatter operations
        index_mask = torch.where((atom_pairs == atom_pair).all(1))

        if compr_pair is None and not with_variable:
            sktable_h[index_mask] = integral(distances[index_mask], atom_pair,
                                             l_pair, hs_type='H')
            sktable_s[index_mask] = integral(distances[index_mask], atom_pair,
                                             l_pair, hs_type='S')

        elif compr_pair is not None and not with_variable:
            sktable_h[index_mask] = integral(compr_pair[index_mask],
                                             atom_pair, l_pair, hs_type='H',
                                             input2=distances[index_mask])
            sktable_s[index_mask] = integral(compr_pair[index_mask],
                                             atom_pair, l_pair, hs_type='S',
                                             input2=distances[index_mask])
            # print('distance', distances[index_mask], '\n compr_pair[index_mask]', compr_pair[index_mask])

        elif with_variable:  # Hamiltonian as ML variable
            sktable_h[index_mask] = integral(
                distances[index_mask], atom_pair, l_pair, hs_type='H', get_abcd='abcd')
            sktable_s[index_mask] = integral(
                distances[index_mask], atom_pair, l_pair, hs_type='S', get_abcd='abcd')

    return sktable_h, sktable_s


def onsite_retrieve(
        an_mat_a: Tensor, integral: object, shape: Size, **kwargs) -> Tensor:
    """Onsite retrieval operation.

    Arguments:
        an_mat_a: Gather atomic number index for single or batch system.
        integral: Object of reading SK tables.
        shape: Shape of Hamiltonian and overlap.
    """
    # get the diagonals of the atomic identity matrices
    onsite_data = torch.zeros(shape[0], shape[1])
    onsite_element_block = an_mat_a.diagonal(dim1=-3, dim2=-2)[:, 0, :]

    # loop for onsite of each system in batch
    for ii, onsite_block in enumerate(onsite_element_block):
        ionsite = integral.get_onsite(
            onsite_block[onsite_block.ne(0)], **kwargs)
        onsite_data[ii, :ionsite.shape[0]] = ionsite

    return onsite_data.flatten()


def U_retrieve(an_mat_a: Tensor, integral: object, shape: Size, **kwargs) -> Tensor:
    """Onsite retrieval operation.

    Arguments:
        an_mat_a: Gather atomic number index for single or batch system.
        integral: Object of reading SK tables.
        shape: Shape of Hamiltonian and overlap.
    """
    # get the diagonals of the atomic identity matrices
    u_element_block = an_mat_a.diagonal(dim1=-3, dim2=-2)[:, 0, :]
    return pack([integral.get_U(iu[iu.ne(0)], **kwargs) for iu in u_element_block])


def _skt_ss(r: Tensor, integrals: Union[Tensor, Real]):
    """Perform Slater-Koster transformations for ss interactions.

    Arguments:
        r: The unit difference vector between a pair of orbitals. Or an
            array representing a set of such differences.
        integrals: ss0 integral evaluated at the inter-atomic associated with
            the specified distance(s).

    Returns:
        sub_block: The ss matrix sub-block, or a set thereof.
    """
    # No transformation is actually required so just return the integrals
    return integrals.unsqueeze(1)


def _skt_sp(r: Tensor, integrals: Union[Tensor, Real]):
    """Perform Slater-Koster transformations for sp interactions.

    Arguments:
        r: The unit difference vector between a pair of orbitals. Or an
            array representing a set of such differences.
        integrals: sp0 integral evaluated at the inter-atomic associated with
            the specified distance(s).

    Return:
        sub_block: The sp matrix sub-block, or a set thereof.

    """
    return (integrals * r).unsqueeze(1).roll(-1, -1)


def _skt_sd(r: Tensor, integrals: Union[Tensor, Real]):
    """Perform Slater-Koster transformations for sd interactions.

    Arguments:
        r: The unit difference vector between a pair of orbitals. Or an
            array representing a set of such differences.
        integrals: sd0 integral evaluated at the inter-atomic associated with
            the specified distance(s).

    Returns:
        sub_block: The sd matrix sub-block, or a set thereof.

    """
    # Unpack unit vector into its components. The transpose is needed when
    # `r` contains multiple vectors.
    x, y, z = r.T
    # Pre calculate squares, square routes, etc
    x2, y2, z2 = r.T ** 2

    # Construct the rotation matrix:
    #       ┌              ┐
    #       │ σs-d_xy      │
    #       │ σs-d_yz      │
    # rot = │ σs-d_z^2     │
    #       │ σs-d_xz      │
    #       │ σs-d_x^2-y^2 │
    #       └              ┘
    #
    # For a single instance the operation would be:
    # rot = np.array([
    #     [_sqr3 * x * y],
    #     [_sqr3 * y * z],
    #     [z2 - 0.5 * (x2 + y2)],
    #     [_sqr3 * x * z],
    #     [0.5 * _sqr3 * (x2 - y2)]
    # ])
    rot = torch.stack([
        _sqr3 * x * y,
        _sqr3 * y * z,
        z2 - 0.5 * (x2 + y2),
        _sqr3 * x * z,
        0.5 * _sqr3 * (x2 - y2)
    ])
    # Apply the transformation and return the result
    return (rot.T * integrals).unsqueeze(1)


def _skt_sf(r: Tensor, integrals: Union[Tensor, Real]):
    """Perform Slater-Koster transformations for sf interactions.

    Arguments:
        r: The unit difference vector between a pair of orbitals. Or an
            array representing a set of such differences.
        integrals: sf0 integral evaluated at the inter-atomic associated with
            the specified distance(s).

    Returns:
        sub_block: The sf matrix sub-block, or a set thereof.

    """
    raise NotImplementedError()


def _skt_pp(r: Tensor, integrals: Union[Tensor, Real]):
    """Perform Slater-Koster transformations for pp interactions.

    Arguments:
        r: The unit difference vector between a pair of orbitals. Or an
            array representing a set of such differences.
        integrals: pp0 & pp1 integrals evaluated at the inter-atomic associated
            with the specified distance(s).

    Returns:
        sub_block: The pp matrix sub-block, or a set thereof.
    """
    # Construct the rotation matrix:
    #       ┌                    ┐
    #       │ σp_y-p_y, πp_y-p_y │
    #       │ σp_y-p_z, πp_y-p_z │
    #       │ σp_y-p_z, πp_y-p_z │
    #       │ σp_z-p_y, πp_z-p_y │
    # rot = │ σp_z-p_z, πp_z-p_z │
    #       │ σp_z-p_x, πp_z-p_x │
    #       │ σp_x-p_y, πp_x-p_y │
    #       │ σp_x-p_z, πp_x-p_z │
    #       │ σp_x-p_x, πp_x-p_x │
    #       └                    ┘
    #
    # For a single instance the operation would be:
    # rot = np.array([
    #     [y2, 1 - y2],
    #     [yz, -yz],
    #     [xy, -xy],
    #
    #     [yz, -yz],
    #     [z2, 1 - z2],
    #     [xz, -xz],
    #
    #     [xy, -xy],
    #     [xz, -xz],
    #     [x2, 1 - x2]])
    r = r.T[[1, 2, 0]].T  # Reorder positions to mach physics notation
    outer = torch.bmm(r.unsqueeze(2), r.unsqueeze(1)).view(-1, 9)
    tmp = torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=r.dtype)
    rot = torch.stack((outer, tmp-outer), 2)
    return torch.bmm(rot, integrals.unsqueeze(2)).view(r.shape[0], 3, 3)


def _skt_pd(r: Tensor, integrals: Union[Tensor, Real]):
    """Perform Slater-Koster transformations for pd interactions.

    Arguments:
        r: The unit difference vector between a pair of orbitals. Or an
            array representing a set of such differences.
        integrals: pd0 & pd1 integrals evaluated at the inter-atomic associated
            with the specified distance(s).

    """
    # Unpack unit vector into its components. The transpose is needed when
    # `r` contains multiple vectors.
    x, y, z = r.T

    # Pre calculate squares, square routes, etc
    x2, y2, z2 = r.T ** 2
    alpha, beta = x2 + y2, x2 - y2
    z2_h_a = z2 - 0.5 * alpha
    xyz = r.prod(-1)

    # Construct the rotation matrix:
    #       ┌                                    ┐
    #       │ σp_y-d_xy,        πp_y-d_xy        │
    #       │ σp_y-d_yz,        πp_y-d_yz        │
    #       │ σp_y-d_z^2,       πp_y-d_z^2       │
    #       │ σp_y-d_xz,        πp_y-d_xz        │
    #       │ σp_y-d_(x^2-y^2), πp_y-d_(x^2-y^2) │
    #       │ σp_z-d_xy,        πp_z-d_xy        │
    #       │ σp_z-d_yz,        πp_z-d_yz        │
    # rot = │ σp_z-d_z^2,       πp_z-d_z^2       │
    #       │ σp_z-d_xz,        πp_z-d_xz        │
    #       │ σp_z-d_(x^2-y^2), πp_z-d_(x^2-y^2) │
    #       │ σp_x-d_xy,        πp_x-d_xy        │
    #       │ σp_x-d_yz,        πp_x-d_yz        │
    #       │ σp_x-d_z^2,       πp_x-d_z^2       │
    #       │ σp_x-d_xz,        πp_x-d_xz        │
    #       │ σp_x-d_(x^2-y^2), πp_x-d_(x^2-y^2) │
    #       └                                    ┘
    #
    # For a single instance the operation would be:
    # rot = np.array([
    #     [_sqr3 * y2 * x, x * (1 - 2 * y2)],
    #     [_sqr3 * y2 * z, z * (1 - 2 * y2)],
    #     [y * z2_h_a, -_sqr3 * y * z2],
    #     [_sqr3 * xyz, -2 * xyz],
    #     [_hsqr3 * y * beta, -y * (1 + beta)],
    #
    #     [_sqr3 * xyz, -2 * xyz],
    #     [_sqr3 * z2 * y, y * (1 - 2 * z2)],
    #     [z * z2_h_a, _sqr3 * z * alpha],
    #     [_sqr3 * z2 * x, x * (1 - 2 * z2)],
    #     [_hsqr3 * z * beta, -z * beta],
    #
    #     [_sqr3 * x2 * y, y * (1 - 2 * x2)],
    #     [_sqr3 * xyz, -2 * xyz],
    #     [x * z2_h_a, -_sqr3 * x * z2],
    #     [_sqr3 * x2 * z, z * (1 - 2 * x2)],
    #     [_hsqr3 * x * beta, x * (1 - beta)]
    # ])
    # There must be a nicer, vectorised and more elegant way to do this
    column_1 = torch.stack((
        _sqr3 * y2 * x,
        _sqr3 * y2 * z,
        y * z2_h_a,
        _sqr3 * xyz,
        _hsqr3 * y * beta,

        _sqr3 * xyz,
        _sqr3 * z2 * y,
        z * z2_h_a,
        _sqr3 * z2 * x,
        _hsqr3 * z * beta,

        _sqr3 * x2 * y,
        _sqr3 * xyz,
        x * z2_h_a,
        _sqr3 * x2 * z,
        _hsqr3 * x * beta
    ), -1)

    column_2 = torch.stack((
        x * (1 - 2 * y2),
        z * (1 - 2 * y2),
        -_sqr3 * y * z2,
        -2 * xyz,
        -y * (1 + beta),

        -2 * xyz,
        y * (1 - 2 * z2),
        _sqr3 * z * alpha,
        x * (1 - 2 * z2),
        -z * beta,

        y * (1 - 2 * x2),
        -2 * xyz,
        -_sqr3 * x * z2,
        z * (1 - 2 * x2),
        x * (1 - beta)
    ), -1)

    # Combine the two columns to create the final rotation matrix
    rot = torch.stack((column_1, column_2), -1)

    # Calculate, reshape and return the SK blocks
    return (rot @ integrals.unsqueeze(2)).view(-1, 3, 5)


def _skt_pf(r: Tensor, integrals: Union[Tensor, Real]):
    """Perform Slater-Koster transformations for pf interactions.

    Arguments:
        r: The unit difference vector between a pair of orbitals. Or an
            array representing a set of such differences.
        integrals: sf0 & sf1 integrals evaluated at the inter-atomic
            associated with the specified distance(s).

    Returns:
        sub_block: The pf matrix sub-block, or a set thereof.
    """
    raise NotImplementedError()


def _skt_dd(r: Tensor, integrals: Union[Tensor, Real]):
    """Perform Slater-Koster transformations for dd interactions.

    Arguments:
        r: The unit difference vector between a pair of orbitals. Or an
            array representing a set of such differences.
        integrals:  dd0, dd1 & dd2 integrals evaluated at the inter-atomic
            associated with the specified distance(s).

    Returns:
        sub_block: The dd matrix sub-block, or a set thereof.
    """
    # There are some tricks that could be used to reduce the size, complexity
    # and overhead of this monster. (this should be done at some point)

    # Unpack unit vector into its components. The transpose is needed when
    # `r` contains multiple vectors.
    x, y, z = r.T

    # Pre calculate squares, square routes, etc
    x2, xy, xz, y2, yz, z2 = r.T[[0, 0, 0, 1, 1, 2]] * r.T[[0, 1, 2, 1, 2, 2]]
    x2y2, y2z2, x2z2 = xy ** 2, yz ** 2, xz ** 2
    alpha, beta = x2 + y2, x2 - y2
    xyz = r.prod(-1)
    a_m_z2 = alpha - z2
    beta2 = beta ** 2
    sqr3_beta = _sqr3 * beta
    z2_h_a = z2 - 0.5 * alpha

    # Construct the rotation matrix:
    # ┌                                                                      ┐
    # │σd_xy-d_xy,             πd_xy-d_xy,             δd_xy-σd_xy           │
    # │σd_xy-d_yz,             πd_xy-d_yz,             δd_xy-σd_yz           │
    # │σd_xy-d_z^2,            πd_xy-d_z^2,            δd_xy-d_z^2           │
    # │σd_xy-d_xz,             πd_xy-d_xz,             δd_xy-d_xz            │
    # │σd_xy-(x^2-y^2),        πd_xy-(x^2-y^2),        δd_xy-(x^2-y^2)       │
    # │σd_yz-d_xy,             πd_yz-d_xy,             δd_yz-σd_xy           │
    # │σd_yz-d_yz,             πd_yz-d_yz,             δd_yz-σd_yz           │
    # │σd_yz-d_z^2,            πd_yz-d_z^2,            δd_yz-d_z^2           │
    # │σd_yz-d_xz,             πd_yz-d_xz,             δd_yz-d_xz            │
    # │σd_yz-(x^2-y^2),        πd_yz-(x^2-y^2),        δd_yz-(x^2-y^2)       │
    # │σd_z^2-d_xy,            πd_z^2-d_xy,            δd_z^2-σd_xy          │
    # │σd_z^2-d_yz,            πd_z^2-d_yz,            δd_z^2-σd_yz          │
    # │σd_z^2-d_z^2,           πd_z^2-d_z^2,           δd_z^2-d_z^2          │
    # │σd_z^2-d_xz,            πd_z^2-d_xz,            δd_z^2-d_xz           │
    # │σd_z^2-(x^2-y^2),       πd_z^2-(x^2-y^2),       δd_z^2-(x^2-y^2)      │
    # │σd_xz-d_xy,             πd_xz-d_xy,             δd_xz-d_xy            │
    # │σd_xz-d_yz,             πd_xz-d_yz,             δd_xz-d_yz            │
    # │σd_xz-d_z^2,            πd_xz-d_z^2,            δd_xz-d_z^2           │
    # │σd_xz-d_xz,             πd_xz-d_xz,             δd_xz-d_xz            │
    # │σd_xz-d_xy-(x^2-y^2),   πd_xz-d_xy-(x^2-y^2),   δd_xz-d_xy(x^2-y^2)   │
    # │σd_(x^2-y^2)-d_xy,      πd_(x^2-y^2)-d_xy,      δd_(x^2-y^2)-d_xy     │
    # │σd_(x^2-y^2)-d_yz,      πd_(x^2-y^2)-d_yz,      δd_(x^2-y^2)-d_yz     │
    # │σd_(x^2-y^2)-d_z^2,     πd_(x^2-y^2)-d_z^2,     δd_(x^2-y^2)-d_z^2    │
    # │σd_(x^2-y^2)-d_xz,      πd_(x^2-y^2)-d_xz,      δd_(x^2-y^2)-d_xz     │
    # │σd_(x^2-y^2)-(x^2-y^2), πd_(x^2-y^2)-(x^2-y^2), δd_(x^2-y^2)-(x^2-y^2)│
    # └                                                                      ┘
    # For a single instance the operation would be:
    # rot = np.array(
    # [
    # [3 * x2y2,                 alpha - 4 * x2y2,     z2 + x2y2],
    # [3 * xyz * y,              xz * (1 - 4 * y2),    xz * (y2 - 1)],
    # [_sqr3 * xy * z2_h_a,      -2 * _sqr3 * xyz * z, _hsqr3 * xy * (1 + z2)],
    # [3 * xyz * x,              yz * (1 - 4 * x2),    yz * (x2 - 1)],
    # [1.5 * xy * beta,          -2 * xy * beta,       0.5 * xy * beta],
    #
    # [3 * xyz * x,              zx * (1 - 4 * y2),    xz * (y2 - 1)],
    # [3 * y2z2,                 y2 + z2 - 4 * y2z2,   x2 + y2z2],
    # [_sqr3 * yz * z2_h_a,      _sqr3 * yz * a_m_z2,  -_hsqr3 * yz * alpha],
    # [3 * xyz * z,              xy * (1 - 4 * z2),    xy * (z2 - 1)],
    # [1.5 * yz * beta,          -yz * (1 + 2 * beta), yz * (1 + 0.5 * beta)],
    #
    # [_sqr3 * xy * z2_h_a,      -2 * _sqr3 * xyz * z, _hsqr3 * xy * (1 + z2)],
    # [_sqr3 * yz * z2_h_a,      _sqr3 * yz * a_m_z2,  -_hsqr3 * yz * alpha],
    # [z2_h_a ** 2,              3 * z2 * alpha,       0.75 * alpha ** 2],
    # [_sqr3 * xz * z2_h_a,      _sqr3 * xz * a_m_z2,  -_hsqr3 * xz * alpha],
    # [0.5 * sqr3_beta * z2_h_a, -z2 * sqr3_beta,      0.25 * sqr3_beta * (1 + z2)],
    #
    # [3 * xyz * x,              yz * (1 - 4 * x2),    yz * (x2 - 1)],
    # [3 * xyz * z,              xy * (1 - 4 * z2),    xy * (z2 - 1)],
    # [_sqr3 * xz * z2_h_a,      _sqr3 * xz * a_m_z2,  -_hsqr3 * xz * alpha],
    # [3 * x2z2,                 z2 + x2 - 4 * x2z2,   y2 + x2z2],
    # [1.5 * xz * beta,          xz * (1 - 2 * beta),  -xz * (1 - 0.5 * beta)],
    #
    # [1.5 * xy * beta,          -2 * xy * beta,       0.5 * xy * beta],
    # [1.5 * yz * beta,          -yz * (1 + 2 * beta), yz * (1 + 0.5 * beta)],
    # [0.5 * sqr3_beta * z2_h_a, -z2 * sqr3_beta,      0.25 * sqr3_beta * (1 + z2)],
    # [1.5 * xz * beta,          xz * (1 - 2 * beta),  -xz * (1 - 0.5 * beta)],
    # [0.75 * beta2,             alpha - beta2,        z2 + 0.25 * beta2]
    # ]
    # )
    # Ths is starting to get a little out of hand
    column_1 = torch.stack([
        3 * x2y2,
        3 * xyz * y,
        _sqr3 * xy * z2_h_a,
        3 * xyz * x,
        1.5 * xy * beta,

        3 * xyz * y,
        3 * y2z2,
        _sqr3 * yz * z2_h_a,
        3 * xyz * z,
        1.5 * yz * beta,

        _sqr3 * xy * z2_h_a,
        _sqr3 * yz * z2_h_a,
        z2_h_a ** 2,
        _sqr3 * xz * z2_h_a,
        0.5 * sqr3_beta * z2_h_a,

        3 * xyz * x,
        3 * xyz * z,
        _sqr3 * xz * z2_h_a,
        3 * x2z2,
        1.5 * xz * beta,

        1.5 * xy * beta,
        1.5 * yz * beta,
        0.5 * sqr3_beta * z2_h_a,
        1.5 * xz * beta,
        0.75 * beta2
    ], -1)

    column_2 = torch.stack([
        alpha - 4 * x2y2,
        xz * (1 - 4 * y2),
        -2 * _sqr3 * xyz * z,
        yz * (1 - 4 * x2),
        -2 * xy * beta,

        xz * (1 - 4 * y2),
        y2 + z2 - 4 * y2z2,
        _sqr3 * yz * a_m_z2,
        xy * (1 - 4 * z2),
        -yz * (1 + 2 * beta),

        -2 * _sqr3 * xyz * z,
        _sqr3 * yz * a_m_z2,
        3 * z2 * alpha,
        _sqr3 * xz * a_m_z2,
        -z2 * sqr3_beta,

        yz * (1 - 4 * x2),
        xy * (1 - 4 * z2),
        _sqr3 * xz * a_m_z2,
        z2 + x2 - 4 * x2z2,
        xz * (1 - 2 * beta),

        -2 * xy * beta,
        -yz * (1 + 2 * beta),
        -z2 * sqr3_beta,
        xz * (1 - 2 * beta),
        alpha - beta2
    ], -1)

    column_3 = torch.stack([
        z2 + x2y2,
        xz * (y2 - 1),
        _hsqr3 * xy * (1 + z2),
        yz * (x2 - 1),
        0.5 * xy * beta,

        xz * (y2 - 1),
        x2 + y2z2,
        -_hsqr3 * yz * alpha,
        xy * (z2 - 1),
        yz * (1 + 0.5 * beta),

        _hsqr3 * xy * (1 + z2),
        -_hsqr3 * yz * alpha,
        0.75 * alpha ** 2,
        -_hsqr3 * xz * alpha,
        0.25 * sqr3_beta * (1 + z2),

        yz * (x2 - 1),
        xy * (z2 - 1),
        -_hsqr3 * xz * alpha,
        y2 + x2z2,
        -xz * (1 - 0.5 * beta),

        0.5 * xy * beta,
        yz * (1 + 0.5 * beta),
        0.25 * sqr3_beta * (1 + z2),
        -xz * (1 - 0.5 * beta),
        z2 + 0.25 * beta2
    ], -1)

    rot = torch.stack((column_1, column_2, column_3), -1)
    return (rot @ integrals.unsqueeze(2)).view(-1, 5, 5)


def _skt_df(r: Tensor, integrals: Union[Tensor, Real]):
    """Perform Slater-Koster transformations for df interactions.

    Arguments:
        r: The unit difference vector between a pair of orbitals. Or an
            array representing a set of such differences.
        integrals: df0, df1 & df2 integrals evaluated at the inter-atomic
            associated with the specified distance(s).

    Returns:
        sub_block: The df matrix sub-block, or a set thereof.
    """
    raise NotImplementedError()


def _skt_ff(r: Tensor, integrals: Union[Tensor, Real]):
    """Perform Slater-Koster transformations for ff interactions.

    Arguments:
        r: The unit difference vector between a pair of orbitals. Or an
            array representing a set of such differences.
        integrals: ff0, ff1, ff2 & ff3 integrals evaluated at the inter-atomic
            associated with the specified distance(s).

    Returns:
        sub_block: The ff matrix sub-block, or a set thereof.
    """
    raise NotImplementedError()


# known sk interactions and their associated functions
_SK_interactions = torch.tensor([[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2]])
_SK_functions = [_skt_ss, _skt_sp, _skt_sd, _skt_pp, _skt_pd, _skt_dd]
