"""Simple, Anderson mixer modules."""
import torch
from abc import ABC, abstractmethod
Tensor = torch.Tensor

class _Mixer(ABC):
    """This is the abstract base class up on which all mixers are based.

    Arguments:
        mix_param: Mixing parameter, controls the extent of charge mixing.
        tolerance: Converged tolerance.

    Keyword Args:
        mask: Mask for dynamic SCC DFTB convergence.
        generations: Store values of how many steps.

    """

    def __init__(self, q_init: Tensor, mix_param=0.1, tolerance=1E-8, **kwargs):
        self.return_convergence = kwargs.get('return_convergence', False)
        self.generations = kwargs.get('generations', 3)
        q_init = q_init.unsqueeze(0) if q_init.dim() == 0 else q_init  # H atom
        q_init = q_init.unsqueeze(0) if q_init.dim() == 1 else q_init  # -> batch
        assert q_init.dim() == 2

        self.q_init = q_init
        self.mix_param = mix_param
        self.tolerance = tolerance
        self._step_number = 0
        self._delta = None

    @abstractmethod
    def __call__(self, q_new: Tensor, q_old=None, **kwargs):
        """Perform the mixing operation & returns the newly mixed charges.

        The single system will be transformed to batch. Adaptive mask.

        Arguments:
            q_new: The new charge that is to be mixed.
            q_old: Optional old charge.

        Keyword Args:
            mask: Converged mask for batch mixing.
        """
        q_new = q_new.unsqueeze(0) if q_new.dim() == 0 else q_new  # H atom
        q_new = q_new.unsqueeze(0) if q_new.dim() == 1 else q_new  # -> batch
        assert q_new.dim() == 2
        self.q_new, self.this_size = q_new, q_new.shape[-1]

        self.mask = kwargs.get('mask', None)
        if self.return_convergence is True and self.mask is None:
            self.mask = self.conv_mat[1] == False
        elif self.return_convergence is False and self.mask is None:
            self.mask = torch.tensor([True]).repeat(self.q_new.shape[0])

        # make qnew input more flexible, either full batch or batch with mask
        if self.q_new.shape[0] == self._F[0].shape[0]:
            self.qnew = self.q_new[self.mask]
        elif self.q_new.shape[0] == self._F[0, self.mask].shape[0]:
            self.qnew = self.q_new
        else:
            raise ValueError('wrong batch dimension of q_new.')

        # overwrite last entry if q_old is not None
        if q_old is not None:
            if q_old.shape[0] == self._F[0].shape[0]:
                self._dQ[0, self.mask] = q_old[self.mask]
            elif self.q_new.shape[0] == self._F[0, self.mask].shape[0]:
                self._dQ[0, self.mask] = self.q_old
            else:
                raise ValueError('wrong batch dimension of q_new.')

    @abstractmethod
    def __convergence__(self):
        """Get converged property.

        Returns:
            converged: Boolien indicating convergence status.
        """
        # convergence can be controlled either in mixer or out of mixer
        return torch.max(torch.abs(self._delta), -1).values < self.tolerance


class Simple(_Mixer):
    """Implementation of simple mixer.

    Arguments:
        q_init: Very first initial charge.
        mix_param: Mixing parameter, controls the extent of charge mixing.
        generations: Number of generations to consider during mixing.

    Examples:
        >>> from tbmalt.common.maths.mixer import Simple
        >>> import torch
        >>> qinit = torch.tensor([4, 1, 1, 1, 1])
        >>> mixer = Simple(qinit)
        >>> qnew = torch.tensor([4.4, 0.9, 0.9, 0.9, 0.9])
        >>> qmix = mixer(qnew)
        >>> qmix
        >>> tensor([[4.080000000000000, 0.980000000000000, 0.980000000000000,
                     0.980000000000000, 0.980000000000000]])
    """

    def __init__(self, q_init: Tensor, mix_param=0.2, generations=3, **kwargs):
        super().__init__(q_init, mix_param, **kwargs)
        self.mix_param = mix_param
        self._build_matrices(self.q_init)
        self._dQ[0] = self.q_init

    def _build_matrices(self, q_init: Tensor):
        """Build dQs, which stores all the mixing charges.

        Arguments:
            q_init: The first dQ tensor on which the new are to be based.

        """
        size = (self.generations + 1, *tuple(q_init.shape))
        self._F = torch.zeros(size)
        self._dQ = torch.zeros(size)
        if self.return_convergence:
            self.conv_mat = torch.zeros(self.generations + 1, q_init.shape[0],
                                        dtype=bool)

    def __call__(self, q_new: Tensor, q_old=None, **kwargs):
        """Perform the simple mixing operation & returns the mixed charge.

        Arguments:
            q_new: The new charge that is to be mixed.
            q_old: Previous charge to be mixed.

        Returns:
            q_mix: Mixed charge.

        """
        super().__call__(q_new, q_old, **kwargs)
        self._step_number += 1

        self._F[0, self.mask, :self.this_size] = \
            self.qnew - self._dQ[0, self.mask, :self.this_size]
        q_mix = self._dQ[0, self.mask] + self._F[0, self.mask] * self.mix_param

        self._dQ[:, self.mask] = torch.roll(self._dQ[:, self.mask], 1, 0)
        self._F[:, self.mask] = torch.roll(self._F[:, self.mask], 1, 0)
        self._dQ[0, self.mask] = q_mix

        # Save the last difference to _delta for convergence
        if self.return_convergence:
            self._delta = self._F[1]
            self.conv_mat[0] = self.__convergence__()
            self.conv_mat = torch.roll(self.conv_mat, 1, 0)
            return q_mix, self.conv_mat[1]
        else:
            return q_mix

    def __convergence__(self):
        """Return convergence."""
        return super().__convergence__()


class Anderson(_Mixer):
    """Anderson mixing.

    Arguments:
        q_init: Very first initial charge.
        mix_param: Mixing parameter, controls the extent of charge mixing.
        generations: Number of generations to consider during mixing.

    Keyword Args:
        gamma: Value added to the equation system's diagonal elements to help
            prevent the emergence of linear dependencies.

    Examples:
        >>> from tbmalt.common.maths.mixer import Anderson
        >>> import torch
        >>> qinit = torch.tensor([4, 1, 1, 1, 1])
        >>> mixer = Anderson(qinit)
        >>> qnew = torch.tensor([4.4, 0.9, 0.9, 0.9, 0.9])
        >>> qmix = mixer(qnew)
        >>> qmix
        >>> tensor([[4.080000000000000, 0.980000000000000, 0.980000000000000,
                     0.980000000000000, 0.980000000000000]])

    References:
        .. [1] Eyert, V. (1996). Journal of Computational Physics, 124(2), 271.
        .. [2] Hourahine, B., Aradi, B., Blum, V., Frauenheim, T. et al.,
            (2020). The Journal of Chemical Physics, 152(12), 124101.
        .. [3] Anderson, D. G. M. (2018). 80(1), 135–234.

    """

    def __init__(self, q_init: Tensor, mix_param=0.2, generations=3, **kwargs):
        super().__init__(q_init, mix_param, **kwargs)
        self.generations = generations
        self.gamma = kwargs.get('gamma', 0.01)
        self._build_matrices(self.q_init)
        self._dQ[0] = self.q_init

    def _build_matrices(self, q_init: Tensor):
        """Builds the F and dQs matrices.

        F is the charge difference of latest charge and previous charge. dQs
        store all the mixing charges.

        Arguments:
            q_init: The first dQ tensor on which the new are to be based.

        """
        size = (self.generations + 1, *tuple(q_init.shape))
        self._F = torch.zeros(size)
        self._dQ = torch.zeros(size)
        if self.return_convergence:
            self.conv_mat = torch.zeros(self.generations + 1, q_init.shape[0],
                                        dtype=bool)

    def __call__(self, q_new: Tensor, q_old=None, **kwargs):
        """Perform the actual Anderson mixing operation.

        Arguments:
            q_new: The newly calculated, pure, dQ vector that is to be mixed.
            q_old: The previous, mixed charges. If q_old is available,
                assign q_old to dQs[0].

        """
        super().__call__(q_new, **kwargs)
        self._step_number += 1  # increment _step_number

        self._F[0, self.mask, :self.this_size] = \
            self.qnew - self._dQ[0, self.mask, :self.this_size]

        if self._step_number >= 2:
            previous_step = self._step_number if self._step_number < \
                self.generations + 1 else self.generations + 1

            # solve the linear equation system: equation 4.1 ~ 4.3
            # aa_ij = <F(m)|F(m)-F(m-i)> - <F(m-j)|F(m)-F(m-i)>
            # b_i = <F(m)|F(m)-F(m-i)>
            tmp1 = self._F[0, self.mask] - self._F[1: previous_step, self.mask]
            bb = torch.bmm(tmp1.transpose(1, 0),
                           torch.unsqueeze(self._F[0, self.mask], -1))
            aa = bb.permute(0, 2, 1) - torch.matmul(
                self._F[1: previous_step, self.mask].transpose(1, 0),
                tmp1.permute(1, 2, 0))

            # to equation 8.2 (Eyert) for linear dependent problem
            if self.gamma is not None:
                aa = aa * (1 + torch.eye(aa.shape[-1]) * self.gamma ** 2)

            # Solve for the coefficients, use mask to avoid singular U error
            thetas = torch.solve(bb, aa)[0].squeeze(-1)
            q_bar = (thetas * self._dQ[1: previous_step, self.mask].transpose(
                2, 0)).transpose(2, 0).sum(0)
            F_bar = (thetas * self._F[1: previous_step, self.mask].transpose(
                2, 0)).transpose(2, 0).sum(0)

            # replace first terms of equations 4.1 and 4.2 with DFTB+ format
            theta_0 = 1 - thetas.sum(1)
            q_bar = q_bar + (theta_0 * self._dQ[0, self.mask].T).T
            F_bar = F_bar + (theta_0 * self._F[0, self.mask].T).T

            # Calculate the new mixed dQ following equation 4.4 (Eyert):
            q_mix = q_bar + (self.mix_param * F_bar)

        # If there is insufficient history for Anderson; use simple mixing
        else:
            q_mix = self._dQ[0, self.mask] + (self._F[0, self.mask] * self.mix_param)

        # Shift F & dQ histories to avoid a pytorch inplace error
        self._F[:, self.mask] = torch.roll(self._F[:, self.mask], 1, 0)
        self._dQ[:, self.mask] = torch.roll(self._dQ[:, self.mask], 1, 0)

        # assign the mixed dQ to the dQs history array
        self._dQ[0, self.mask] = q_mix

        # Save the last difference to _delta for convergence
        if self.return_convergence:
            self._delta = self._F[1]
            self.conv_mat[0] = self.__convergence__()
            self.conv_mat = torch.roll(self.conv_mat, 1, 0)
            return q_mix, self.conv_mat[1]
        else:
            return q_mix

    def __convergence__(self):
        """Return convergence."""
        return super().__convergence__()
