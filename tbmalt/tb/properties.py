"""Physical properties routine."""
import torch
import numpy as np
from numbers import Real
from typing import Optional, Tuple, Union
from tbmalt.common.batch import pack
Tensor = torch.Tensor
ndarray = np.ndarray


def mulliken(overlap: Tensor, density: Tensor, atom_orbitals=None,
             **kwargs) -> Tensor:
    """Calculate Mulliken charge for both batch system.

    Arguments:
        overlap: 2D or 3D overlap matrix.
        density: 2D or 3D density matrix.
        atom_orbitals: Orbital number for each atom, which is (l + 1) ** 2.

    """
    orbital_resolve = kwargs.get('orbital_resolve', False)

    # sum overlap and density by hadamard product, get charge by orbital
    charge_orbital = (density * overlap).sum(dim=1)

    if orbital_resolve:

        # return orbital resolved charge
        return charge_orbital
    else:
        assert atom_orbitals is not None

        # get a list of accumulative orbital indices
        ind_cumsum = [torch.cat((
            torch.zeros(1), torch.cumsum(ii[ii.ne(0)], -1))).long()
            for ii in atom_orbitals]

        # return charge of each atom for batch system
        return pack([torch.stack([(icharge[ii: jj]).sum() for ii, jj in zip(
            icumsum[:-1], icumsum[1:])])
            for icumsum, icharge in zip(ind_cumsum, charge_orbital)])


def _generate_broadening(energies: Tensor, eps: Tensor,
                         sigma: Union[Real, Tensor] = 0.0) -> Tensor:
    """Construct the gaussian broadening terms.

    This is used to calculate the gaussian broadening terms used when
    calculating the DoS/PDoS.

    Arguments:
        energies: Energy values to evaluate the DoS/PDoS at.
        eps: Energy eigenvalues (epsilon).
        sigma: Smearing width for gaussian broadening function. [DEFAULT=0]

    Returns:
        g: Gaussian broadening terms.

    """

    def _gaussian_broadening(energy_in: Tensor, eps_in: Tensor, sigma: Real
                             ) -> Tensor:
        """Gaussian broadening factor used when calculating the DoS/PDoS."""
        return torch.erf((energy_in[..., :, None] - eps_in[..., None, :]).T
                         / (np.sqrt(2) * sigma)).T

    # Construct gaussian smearing terms.
    de = energies[..., 1] - energies[..., 0]
    ga = _gaussian_broadening((energies.T - (de / 2)).T, eps, sigma)
    gb = _gaussian_broadening((energies.T + (de / 2)).T, eps, sigma)
    return ((gb - ga).T / (2.0 * de)).T


def dos(eps: Tensor, energies: Tensor, sigma: Union[Real, Tensor] = 0.0,
        offset: Optional[Union[Real, Tensor]] = None,
        mask: Optional[Tensor] = None, scale: bool = False) -> Tensor:
    r"""Calculates the density of states for one or more systems.

    This calculates and returns the Density of States (DoS) for one or more
    systems. If desired, all but a selection of specific states can be masked
    out via the ``mask`` argument.

    Arguments:
        eps: Energy eigenvalues (epsilon).
        energies: Energy values to evaluate the DoS at. These are assumed to
            be relative to the ``offset`` value, if it is specified.
        sigma: Smearing width for gaussian broadening function. [DEFAULT=0]
        offset: Indicates that ``energies`` are given with respect to a offset
            value, e.g. the fermi energy.
        mask: Used to control which states are used in constricting the DoS.
            Only unmasked (True) states will be used, all others are ignored.
        scale: Scales the DoS to have a maximum value of 1. [DEFAULT=False]

    Returns:
        dos: The densities of states.

    Notes:
        The DoS is calculated via an equation equivalent to:

        .. math::
            g(E)=\delta(E-\epsilon_{i})

        Where g(E) is the density of states at an energy value E, and δ(E-ε)
        is the smearing width calculated as:

        .. math::
            \delta(E-\epsilon)=\frac{
                erf\left(\frac{E-\epsilon+\frac{\Delta E}{2}}{\sqrt{2}\sigma}\right)-
                erf\left(\frac{E-\epsilon+\frac{\Delta E}{2}}{\sqrt{2}\sigma}\right)}
                {2\Delta E}

        Where ΔE is the difference in energy between neighbouring points.
        It may be useful, such as in the creation of a cost function, to have
        only specific states (i.e. HOMO, HOMO-1, etc.) used to construct the
        PDoS. State selection can be achieved via the ``mask`` argument. This
        should be a boolean tensor with a shape matching that of ``eps``. Only
        states whose mask value is True will be included in the DoS, e.g.

            mask = torch.Tensor([True, False, False, False])

        would only use the first state when constructing the DoS.

    Warnings:
        It is imperative that padding values are masked out when operating on
        batches of systems! Failing to do so will result in the emergence of
        erroneous state occupancies. Care must also be taken to ensure that
        the number of sample points is the same for each system; i.e. all
        systems have the same number of elements in ``energies``. As padding
        values will result in spurious behaviour.

    Examples:
        Density of states constructed for an H2 molecule using test data:

        >>> from tbmalt.tests.unittests.data.properties.dos import H2
        >>> from tbmalt.physics.properties import dos
        >>> eps = H2['eigenvalues']
        >>> energies = H2['dos']['energy']
        >>> sigma = H2['sigma']
        >>> dos_values = dos(eps, energies, sigma)
        >>> plt.plot(energies, dos_values)
        >>> plt.xlabel('Energy [Ha]')
        >>> plt.ylabel('DoS [Ha]')
        >>> plt.show()

    """
    if mask is not None:  # Mask out selected eigen-states if requested.
        eps = eps.clone()  # Use a clone to prevent altering the original
        # Move masked states towards +inf
        eps[~mask] = torch.finfo(eps.dtype).max

    if offset is not None:  # Apply the offset, if applicable.
        # Offset must be applied differently for batches.
        if isinstance(offset, (Tensor, np.ndarray)) and len(offset.shape) > 0:
            eps = eps - offset[:, None]
        else:
            eps = eps - offset

    g = _generate_broadening(energies, eps, sigma)  # Gaussian smearing terms.
    distribution = torch.sum(g, -1)  # Compute the densities of states

    # Rescale the DoS so that it has a maximum peak value of 1.
    if scale:
        distribution = distribution / distribution.max(-1, keepdim=True)[0]

    return distribution


def pdos(C: Tensor, S: Tensor, eps: Tensor, energies: Tensor,
         sigma: Real = 0.1, offset: Optional[Real] = None,
         mask: Optional[Tensor] = None,
         scale: bool = False,
         ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    r"""Calculates the projected density of states for one or more systems.

    This calculates and returns the Projected Density of States (PDoS) for
    one or more systems. If desired, all but a selection of specific states
    can be masked out via the ``mask`` argument. The returned distributions
    can then be aggregated and resolved using ``resolve_distributions``.

    Arguments:
        C: Coefficient matrix with eigenvectors represented as columns.
        S: Overlap matrix.
        eps: Eigenvalues.
        energies: Energy values to evaluate the PDoS at. These are assumed to
            be relative to the ``offset`` value, if it is specified.
        sigma: Smearing width for gaussian broadening function. [DEFAULT=0.0]
        offset: Indicates that ``energies`` are given with respect to a offset
            value, e.g. the fermi energy.
        mask: Used to control which states are used in constricting the DoS.
            Only unmasked (True) states will be used, all others are ignored.
        scale: scales the distributions so that the maximum value of the sum
            of the distributions (DoS) is equal to 1.

    Returns:
        pdos: The projected densities of states, one per basis.

    Notes:
        The PDoS is calculated via an equation equivalent to:

        .. math::
            g(\mu, E)=\sum_i \sum_v C_{\nu i}^* S_{\mu v} C_{\mu i} \delta(E-\epsilon_{i})

        Where g(μ, E) is the density of states of orbital μ at an energy value
        E and δ(E-ε) is the smearing width calculated as:

        .. math::
            \delta(E-\epsilon)=\frac{
                erf\left(\frac{E-\epsilon+\frac{\Delta E}{2}}{\sqrt{2}\sigma}\right)-
                erf\left(\frac{E-\epsilon+\frac{\Delta E}{2}}{\sqrt{2}\sigma}\right)}
                {2\Delta E}

        Where ΔE is the difference in energy between neighbouring points. It
        is worth pointing out that the PDoS distributions should be treated as
        qualitative rather than quantitative.

        Masking operations are conducted in a manner identical to the ``dos``
        function; see ``dos`` function's documentation for more information.

        Unlike the ``dos`` function, ``pdos`` does not require faux states
        caused by zero padding to be masked out.

    Examples:
        Projected density of states constructed for a CH4 molecule:

        >>> import matplotlib.pyplot as plt
        >>> from tbmalt.tests.unittests.data.properties.dos import CH4
        >>> from tbmalt.physics.properties import pdos
        >>> C = CH4['eigenvectors']
        >>> S = CH4['overlap']
        >>> eps = CH4['eigenvalues']
        >>> energies = CH4['dos']['energy']
        >>> sigma = CH4['sigma']
        >>> pdos_values = pdos(C, S, eps, energies, sigma)
        >>> for pd in pdos_values:
        >>>     plt.plot(energies, pd, '-', linewidth=0.5)
        >>> plt.xlabel('Energy [Ha]')
        >>> plt.ylabel('PDoS [Ha]')
        >>> plt.show()

    """
    # First code blocks are uncommented as they are analogous to those of dos
    if mask is not None:
        eps, C = eps.clone(), C.clone()
        eps[~mask] = torch.finfo(eps.dtype).max
        # transpose(0,-2) will only permute batch tensors
        C.transpose(0, -2)[..., ~mask] = 0.

    if offset is not None:
        if isinstance(offset, (Tensor, np.ndarray)) and len(offset) > 1:
            eps = eps - offset[:, None]
        else:
            eps = eps - offset

    g = _generate_broadening(energies, eps, sigma)

    # Compute the projected densities of states
    distributions = torch.einsum('...vi,...ui,...vu,...ei->...ue', C, C, S, g)

    # Rescale distributions so the total DoS has a maximum value of 1.
    if scale:
        # distributions = (distributions / distributions.flatten(
        #     -2).max(-1, True)[0].unsqueeze(1))
        distributions = distributions / distributions.sum(-2, keepdim=True).amax(-1).unsqueeze(-1)

    return distributions


def resolve_distributions(distributions: Tensor, resolve_by: Tensor
                           ) -> Tuple[Tensor, Tensor]:
    """Aggregate distributions into a set of resolved PDoS distributions.

    This takes in a set of PDoS distributions and resolves them to the degree
    specified using the data in ``resolve_by``. This can be used to produce
    species, atom, etc. resolved PDoS distributions. Multiple

    Arguments:
        distributions: Projected densities of states distributions.
        resolve_by: Aggregates distributions by the features specified in
            argument. Each row of this array should correspond to a basis and
            each column to a property.

    Returns:
        distributions_agr: aggregated PDoS distributions.
        labels: Labels for the resolved distributions.

    Examples:
        Calculate the projected densities of states for a CH4 molecule then
        resolve the distrubitons by atomic number.

        >>> from tbmalt.tests.unittests.data.properties.dos import CH4
        >>> from tbmalt.physics.properties import pdos, resolve_distributions
        >>> C, S = CH4['eigenvectors'], CH4['overlap']
        >>> eps, sigma = CH4['eigenvalues'], CH4['sigma']
        >>> energies = CH4['dos']['energy']
        >>> zs = CH4['bases']['z']
        >>> pdos_values = pdos(C, S, eps, energies, sigma)
        >>> pdos_resolved, z_label = resolve_distributions(pdos_values, zs)
        >>> for pd, z in zip(pdos_resolved, z_label):
        >>>     plt.plot(energies, pd, '-', linewidth=0.5, label=f'Element: {z}')
        >>> plt.legend(loc="upper left")
        >>> plt.xlabel('Energy [Ha]')
        >>> plt.ylabel('PDoS [Ha]')
        >>> plt.show()

    """
    # Developers Notes
    # np.unique must be used as torch.unique false on multi-column systems.
    # As there is no easy way to *efficiently* vectorised this operation, for-
    # loops and separate handling of batch/non-batch data is required.

    def aggregate(dist, ind):
        # Helper function, this is a row-wise version of torch.scatter_add.
        return torch.stack([dist[np.where(ind == i)[0]].sum(0)
                            for i in ind.unique()])

    # If this is a single system, i.e. not a batch operation.
    if distributions.dim() == 2:
        # Identify all unique labels & the indices associated with them.
        labels, indices_u = np.unique(resolve_by.detach().cpu(),
                                      axis=0, return_inverse=True)
        # Aggregate distributions by summation according to their labels.
        # indices_u must be converted to a tensor to prevent GPU problems.
        distributions_agr = aggregate(distributions, torch.tensor(indices_u))
        # Return the resolved distributions and their associated labels
        return distributions_agr, torch.tensor(labels, device=resolve_by.device)

    # batch operation is identical to the non-batch case, but with a for-loop.
    else:
        labels, distributions_agr = [], []
        for d, r in zip(distributions, resolve_by.detach().cpu()):
            labels_u, indices_u = np.unique(r, axis=0, return_inverse=True)
            distributions_agr.append(aggregate(d, torch.tensor(indices_u)))
            labels.append(torch.tensor(labels_u))

        # Pack label & distribution lists into a single tensors & return them.
        return pack(distributions_agr), pack(labels, value=-1)


def band_pass_state_filter(eps: Tensor, n_homo: Union[int, Tensor],
                           n_lumo: Union[int, Tensor],
                           fermi: Union[Real, Tensor]) -> Tensor:
    """Generates masks able to filter out states too far from the fermi level.

     This function returns a mask for each ``eps`` system that can filter out
     all but a select number of states above and below the fermi level.

    Arguments:
        eps: Eigenvalues.
        n_homo: n states below the fermi level to retain, including the HOMO.
        n_lumo: n states above the fermi level to retain.
        fermi: Fermi level.

    Returns:
        mask: A boolean mask which is True for selected states.

    Notes:
        For each system, a ``n_homo``, ``n_lumo``, and ``fermi`` value must be
        provided. This assumes that all states, ``eps`` are ordered from
        lowest to highest.

    Warnings:
        It is down to the user to ensure that the number of requested HOMOs &
        LUMOs for each system is valid. This cannot be done safely here due to
        the effects of zero-padded packing; i.e. this function sees padding
        zeros as valid LUMO states.

    Raises:
        RuntimeError: If multiple systems have been provided but not multiple
            ``n_homo``, ``n_lumo``, and ``fermi`` values.

    Examples:
        Here, all but three states below and two states above the fermi level
        are masked out:

        >>> from tbmalt.physics import band_pass_state_filter
        >>> eps = torch.arange(-4., 6.)
        >>> mask = band_pass_state_filter(eps, 3, 2, 0.)
        >>> print(eps)
        tensor([-4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.,  5.])
        >>> print(mask)
        tensor([False, False, True,  True,  True,  True,  True,  False,
                False, False])

    """
    def index_list(eps_in, fermi_in):
        """Generate a state index list offset relative to the HOMO level.

        e.g: [-n, ..., -2, -1, 0, 1, 2, ..., +n], where 0 is at the index of
        the HOMO. An issue is encountered when operating on zero-padded packed
        data, as the padding 0s can get miss-identified as the homo. We want:
        [True, True, True, False, False, True, True]
                 this ↑,   but not these: ↑     ↑, to be the zero point. Thus,
        a more involved method must be used to get the HOMO state's index.
        """
        le_fermi = torch.where(eps_in <= fermi_in)[0]  # Find values ≤ fermi
        if len(le_fermi) == 1:  # If there's 1 value; then it is homo e.g. H2.
            homo = le_fermi[0]
        else:
            # Construct a difference array to highlight non-sequential states.
            # The Homo will be the last entry of the first sequential block.
            diff = le_fermi[1:] - le_fermi[:-1]
            homo = le_fermi[torch.unique_consecutive(diff, return_counts=True)[1][0]]


        # Generate and return the index list
        return torch.arange(eps.shape[-1], device=eps.device) - homo

    # If multiple systems were specified then ensure that multiple n_homo,
    # n_lumo, and fermi values were also specified.
    if eps.dim() == 2:
        if not all([isinstance(i, Tensor) and i.dim() != 0 for i in
                    [n_homo, n_lumo, fermi]]):
            raise RuntimeError('n_homo, n_lumo, and fermi values must be '
                               'provided for each system.')

    # Build relative index list, batch & non-batch must be handled differently.
    if eps.dim() == 1:
        ril = index_list(eps, fermi)
    else:
        ril = torch.stack([index_list(e, f) for e, f in zip(eps, fermi)])

    # Create & return a mash that masks out states outside of the band filter
    return ((-n_homo < ril.T) & (ril.T <= n_lumo)).T
