"""Get basis or orbital information."""
import torch
from tbmalt.common.batch import pack


class Basis:
    """Generate orbital, atomic, block information.

    This class is of most use when converting from orbital to atom, block and
    orbital resolved data, i.e. orbital/block resolved quantum number l
    matrix (azimuthal_matrix); Atom/orbital/block resolved atom numbers, etc.
    All the code is designed for batch calculations.

    Arguments:
        geometry: The class which contains geometry and orbital information.

    Examples:
        >>> from ase.build import molecule as molecule_database
        >>> from tbmalt.common.structures.geometry import Geometry
        >>> from tbmalt.common.structures.basis import Basis
        >>> molecule = molecule_database('CH4')
        >>> geometry = Geometry.from_ase_atoms(molecule)
        >>> basis = Basis(Geometry)
        >>> subblock = basis._sub_blocks()
        >>> subblock
        >>> [[tensor([[True, True], [True, True]]), tensor([[True]]),
              tensor([[True]]), tensor([[True]]), tensor([[True]])]]
    """

    def __init__(self, geometry: object):
        self.numbers = geometry.numbers
        self.l_max = geometry.l_max
        self.n_orbital = geometry.geometry_orbitals
        self.n_batch = geometry.size_batch
        self.n_atom = geometry.size_system

        # cached properties see class's docstring for more information.
        self._basis_list = geometry.get_resolved_orbital()
        self._basis_blocks = self._blocks()

        self._sub_basis_list = self._sub_basis_list()
        self._sub_basis_blocks = self._sub_blocks()

        self.hsshape, self.shape = geometry.hs_shape, geometry.single_hs_shape
        self.subshape = [torch.Size([iss, iss]) for iss in self._sub_shells()]

    def _blocks(self):
        """Get blocks of True, the size is the total orbital numbers."""
        block_ = [torch.full((ii, ii), True, dtype=torch.bool)
                  for ii in (torch.arange(torch.max(self.l_max) + 1) + 1) ** 2]
        return [[block_[iat] for iat in isy[isy.ge(0)]] for isy in self.l_max]

    def _sub_shells(self):
        return [sum(ii) + len(ii) for ii in self.l_max]

    def _sub_basis_list(self):
        block_ = [torch.arange(ii + 1, dtype=torch.int8)
                  for ii in range(torch.max(self.l_max) + 1)]
        return [[block_[iat] for iat in isy[isy.ge(0)]] for isy in self.l_max]

    def _sub_blocks(self):
        """Get blocks of True, the size is the total l numbers."""
        block_ = [torch.full((ii + 1, ii + 1), True, dtype=torch.bool)
                  for ii in range(torch.max(self.l_max) + 1)]
        return [[block_[iat] for iat in isy[isy.ge(0)]] for isy in self.l_max]

    def azimuthal_matrix(self, block=False, sort=False, mask=True,
                         char=False, mask_on_site=True, mask_diag=True):
        """Get the azimuthal quantum numbers.

        Arguments:
            block: If return block wise or full block matrix.
            sort: Sort the quantum number l in the last dimension.
            mask: Return mask or not.
            char: Switch the dtype to satisfy PyTorch indices type.
            mask_on_site: Return mask on the onsite part or not.
            mask_diag: Return diagonal mask or not.

        Returns:
            l_mat: A NxNx2 quantum number l, N will depend on input parameters.
                The padding part will return -2. If with mask, the mask part
                will return -1.
        """
        # full/block mode agnostic to reduce code duplication
        if not block:
            shape = self.shape
            basis_list = self._basis_list
            basis_blocks = self._basis_blocks
        else:
            shape = self.subshape
            basis_list = self._sub_basis_list
            basis_blocks = self._sub_basis_blocks

        # Repeat basis list to get ℓ-values for the 1'st orbital in each
        # interaction. Expand function is used as it is faster than repeat.
        l_mat = [torch.cat(ibl).expand(ish) for ibl, ish in zip(basis_list, shape)]

        # Convert from an NxNx1 matrix into the NxNx2 azimuthal matrix
        l_mat = [torch.stack((ilm.T, ilm), -1) for ilm in l_mat]

        if mask:
            # mask with on-site blocks, otherwise create a blank mask.
            mask = [torch.block_diag(*ibl) for ibl in basis_blocks]

            if not mask_on_site:
                for ii, im in enumerate(mask):
                    mask[ii][:] = mask_on_site

            # if mask_diag True; the diagonal will be masked
            if not mask_diag:
                for ii, im in enumerate(mask):
                    mask[ii].diagonal()[:] = mask_diag

            # apply the mask and set all the masked values to -1
            for ii, ilm in enumerate(l_mat):
                ilm[mask[ii], :] = -1

        # sort last dim, e.g. [1, 0] ==> [0, 1]
        if sort:
            l_mat = [ilm.sort(-1)[0] for ilm in l_mat]

        if not char:
            l_mat = [ilm.long() for ilm in l_mat]

        # return the azimuthal_matrix tensor and padding with -2
        return pack(l_mat, value=-2)

    def atomic_number_matrix(self, atom_number_type='atom'):
        """Get atomic numbers associated with each orbital pair.

        Arguments:
            atom_number_type: Return atom number types.
                there are three options, 'atom', 'block', 'orbital'. If
                atom_number_type is atom, atom numbers will repeat according
                to number of atoms. If it is block, atom number will repeat
                according to block size. If it is orbital, atom number will
                repeat according to orbital size.

        Returns:
            atomic_number_matrix: A NxNx2 tensor specifying the atomic numbers.
        """
        # Construct the first NxN slice of the matrix
        if atom_number_type == 'orbital':
            an_mat = [inu[inu.ne(0)].repeat_interleave(
                (il[il.ge(0)] + 1) ** 2).expand(isu)
                for inu, il, isu in zip(self.numbers, self.l_max, self.shape)]

        elif atom_number_type == 'block':
            an_mat = [inu[inu.ne(0)].repeat_interleave(
                il[il.ge(0)] + 1).expand(isu) for inu, il, isu in
                zip(self.numbers, self.l_max, self.subshape)]

        elif atom_number_type == 'atom':
            an_mat = pack([num[num.ne(0)].expand(num[num.ne(0)].shape * 2)
                           for num in self.numbers])

        else:
            raise ValueError("Please select from 'atom', 'block', 'orbital'.")

        return pack([torch.stack((am.T, am), -1) for am in an_mat])

    def index_matrix(self, index_type='atom'):
        """Specify the indices of the atoms associated with each orbital pair.

        Arguments:
            index_type: Return index types.
                There are three options, 'atom', 'block', 'orbital'. If
                atom_number_type is atom, atom numbers will return each atom
                indices. If it is block, atom number will return atom indices
                of each block. If it is orbital, atom number will the atom
                indices of each orbital.

        Returns:
            index_matrix: A NxNx2 tensor specifying the indices of the atoms
                associated with each interaction.
        """
        # Construct the first NxN slice of the matrix
        if index_type == 'orbital':
            i_mat = [torch.arange(len(il[il.ge(0)])).repeat_interleave(
                (il[il.ge(0)] + 1) ** 2).expand(isu)
                for il, isu in zip(self.l_max, self.shape)]

        elif index_type == 'block':
            i_mat = [torch.arange(len(il[il.ge(0)])).repeat_interleave(
                il[il.ge(0)] + 1).expand(isu)
                for il, isu in zip(self.l_max, self.subshape)]

        elif index_type == 'atom':
            n_atom = [len(il) for il in self.l_max]
            i_mat = [torch.arange(ina).expand(ina, ina) for ina in n_atom]

        else:
            raise ValueError("Please select from 'atom', 'block', 'orbital'.")

        return pack([torch.stack((im.T, im), -1) for im in i_mat])
