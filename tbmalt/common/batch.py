"""Helper functions for batch operations.

This module contains classes and helper functions associated with batch
construction, handling and maintenance.
"""

import torch


def pack(tensors, axis=0, size=None, value=0):
    """Pad and pack a sequence of tensors together.

    Pad a list of variable length tensors with zeros, or some other value, and
    pack them into a single tensor.

    Arguments:
        tensors (list of torch.Tensor):
            List of tensors to be packed, all with identical dtypes.
        axis (int, optional):
            Axis along which tensors will be concatenated; 0 for first axis
            -1 for last axis, etc. [DEFAULT=0]
        size (int, optional):
            Specifies the size to which tensors should be padded. By default
            tensors are padded to the size of the largest tensor. However,
            ``max_size`` can be used to overwrite this behaviour.
        value (Any, optional):
            The value with which the tensor is to be padded. [DEFAULT=0]

    Returns:
        packed_tensors (torch.Tensor):
            The input tensors padded and packed into a single tensor.

    Notes:
        ``packed_tensors`` maintains the same order as ``tensors``. This
        is faster & more flexable than the internal pytorch p6ck & pad
        functions (at this particuarl task).

    Examples:

        >>> from dftbmalt.utils.batch import pack
        >>> import torch
        >>> a, b, c = torch.rand(2,2), torch.rand(3,3), torch.rand(4,4)
        >>> abc_packed_a = pack([a, b, c])
        >>> print(abc_packed_a.shape)
        torch.Size([3, 4, 4])
        >>> abc_packed_b = pack([a, b, c], axis=1)
        >>> print(abc_packed_b.shape)
        torch.Size([4, 3, 4])
        >>> abc_packed_c = pack([a, b, c], axis=-1)
        >>> print(abc_packed_c.shape)
        torch.Size([4, 4, 3])
    """

    # If "size" unspecified; the maximum observed size along each axis is used
    if size is None:
        size = torch.max(torch.tensor([i.shape for i in tensors]), 0)[0]

    # Create a tensor to pack into & fill with padding value. Work under the
    # assumption that "axis" == 0 and permute later on (easier this way).
    padded = torch.empty(len(tensors), *size,
                         dtype=tensors[0].dtype).fill_(value)

    # Loop over tensors & the dimension of "padded" it is to be packed into.
    # This loop is not an elegant solution, but it is fast.
    for n, t in enumerate(tensors):
        # Pack via slice operations to allow code to be dimension agnostic.
        slices = (slice(0, s) for s in t.shape)
        padded[(n, *slices)] = t

    # If "axis" was anything other than 0, then padded must be permuted
    if axis != 0:
        # Resolve negative "axis" values to their 'positive' equivalents to
        # maintain expected slicing behaviour when using the insert function.
        if axis < 0:
            axis = padded.dim() + 1 + axis

        # Build a list of axes indices; but omit the axis on which the data
        # was concatenated (i.e. 0).
        ax = list(range(1, padded.dim()))

        # Re-insert the concatenation axis as specified
        ax.insert(axis, 0)

        # Perform the permeation
        padded = padded.permute(ax)

    # Return the packed tensor
    return padded

