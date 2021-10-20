"""Train code."""
from typing import Literal
import numpy as np
import torch
import matplotlib.pyplot as plt
from tbmalt import Geometry, SkfParamFeed
from tbmalt.common.maths import hellinger
from tbmalt.common.batch import pack
from tbmalt.physics.dftb.dftb import Dftb2
from tbmalt.ml.skfeeds import SkfFeed, VcrFeed, TvcrFeed
from tbmalt.structures.basis import Basis
from tbmalt.physics.dftb.slaterkoster import hs_matrix
from tbmalt.ml.feature import Dscribe
from tbmalt.ml.scikitlearn import SciKitLearn
from tbmalt.structures.periodic import Periodic

Tensor = torch.Tensor


class Optim:
    """Optimizer."""

    def __init__(self, geometry: Geometry, reference: dict, variables: list,
                 params: dict, tolerance: float = 1E-7, **kwargs):
        self.geometry = geometry
        self.batch_size = self.geometry._n_batch
        self.reference = reference

        self.variable = variables

        self.params = params
        self.tolerance = tolerance

        # Initialize all targets with None
        for target in self.params['ml']['targets']:
            setattr(self, target, None)

        self.lr = self.params['ml']['lr']

        # get loss function
        self.criterion = getattr(
            torch.nn, self.params['ml']['loss_function'])(reduction='mean')

        # get optimizer
        self.optimizer = getattr(
            torch.optim, self.params['ml']['optimizer'])(self.variable, lr=self.lr)

    def __call__(self, **kwargs):
        """Call train class with properties."""
        self.loss_list = []
        self.loss_list.append(0)
        for target in self.params['ml']['targets']:
            self.params['dftb'][target] = True

        # self.properties = properties
        self.steps = self.params['ml']['max_steps']

    def __loss__(self, results, scc=True):
        """Get loss function for single step."""
        self.loss = 0.0

        # add properties (ML targetss) to loss function
        for target in self.params['ml']['targets']:
            self.loss = self.loss + self.criterion(
                results.__getattribute__(target), self.reference[target]) * \
                self.params['ml'][target+'_weight']

            setattr(self, target, results.__getattribute__(target).detach())
        self.loss_list.append(self.loss.detach())
        self.reach_convergence = abs(
            self.loss_list[-1] - self.loss_list[-2]) < self.tolerance

    def __predict__(self, system):
        """Predict with training results."""
        pass

    def __plot__(self, steps, loss, **kwargs):
        """Visualize training results."""
        compression_radii = kwargs.get('compression_radii', None)

        # plot loss
        plt.plot(np.linspace(1, steps, steps), loss)
        plt.xlabel('steps')
        plt.show()

        # plot compression radii
        if compression_radii is not None:
            compr = pack(compression_radii)
            for ii in range(compr.shape[1]):
                for jj in range(compr.shape[2]):
                    plt.plot(np.linspace(1, steps, steps), compr[:, ii, jj])
            plt.show()

    def _dos(self, dos: Tensor, refdos: Tensor):
        """Construct loss of dos or pdos."""
        return hellinger(dos, refdos[..., 1])


class OptHs(Optim):
    """Optimize integrals with spline interpolation."""

    def __init__(self, geometry: Geometry, reference, parameter, shell_dict, **kwargs):
        self.basis = Basis(geometry.atomic_numbers, shell_dict)
        self.shell_dict = shell_dict
        build_abcd_h = kwargs.get('build_abcd_h', True)
        build_abcd_s = kwargs.get('build_abcd_s', True)
        self.h_feed = SkfFeed.from_dir(
            parameter['dftb']['path_to_skf'], shell_dict, geometry=geometry,
            interpolation='Spline1d', integral_type='H',
            build_abcd=build_abcd_h)
        self.s_feed = SkfFeed.from_dir(
            parameter['dftb']['path_to_skf'], shell_dict, geometry=geometry,
            interpolation='Spline1d', integral_type='S',
            build_abcd=build_abcd_s)

        self.ml_variable = []
        if build_abcd_h:
            self.ml_variable.extend(self.h_feed.off_site_dict['variable'])
        if build_abcd_s:
            self.ml_variable.extend(self.s_feed.off_site_dict['variable'])
        super().__init__(geometry, reference, self.ml_variable, parameter,
                         **kwargs)

    def __call__(self, plot: bool =True, save: bool = True, **kwargs):
        """Train spline parameters with target properties."""
        super().__call__()
        self._loss = []
        for istep in range(self.steps):
            self._update_train()
            print('step: ', istep, 'loss: ', self.loss.detach())
            self._loss.append(self.loss.detach())

            break_tolerance = istep >= self.params['ml']['min_steps']
            if self.reach_convergence and break_tolerance:
                break

        if plot:
            super().__plot__(istep + 1, self.loss_list[1:])

        return self.dftb

    def _update_train(self):
        ham = hs_matrix(self.geometry, self.basis, self.h_feed)
        over = hs_matrix(self.geometry, self.basis, self.s_feed)
        self.dftb = Dftb2(self.params, self.geometry, self.shell_dict,
                          self.params['dftb']['path_to_skf'],
                          ham=ham, over=over, from_skf=True)
        # self.dftb()
        super().__loss__(self.dftb)
        self.optimizer.zero_grad()
        self.loss.backward(retain_graph=True)
        self.optimizer.step()

    def predict(self,  geometry_pred: object):
        """Predict with optimized Hamiltonian and overlap."""
        basis = Basis(geometry_pred.atomic_numbers, self.shell_dict)
        ham = hs_matrix(geometry_pred, basis, self.h_feed)
        over = hs_matrix(geometry_pred, basis, self.s_feed)
        dftb = Dftb2(self.params, geometry_pred, self.shell_dict, ham, over, from_skf=True)
        dftb()
        return dftb


class OptVcr(Optim):
    """Optimize compression radii."""

    def __init__(self, geometry: Geometry, reference, parameter,
                 compr_grid: Tensor, shell_dict: dict,
                 skf_type: Literal['h5', 'skf'] = 'h5', **kwargs):
        """Initialize parameters."""
        self.compr_grid = compr_grid
        self.global_r = kwargs.get('global_r', False)
        self.unique_atomic_numbers = geometry.unique_atomic_numbers()

        if not self.global_r:
            # self.compr = torch.ones(geometry.atomic_numbers.shape) * 3.5
            # self.compr.requires_grad_(True)
            self.compr = torch.zeros(*geometry.atomic_numbers.shape, 2)
            init_dict = {1: torch.tensor([2.5, 3.0]),
                         6: torch.tensor([7.0, 2.7]),
                         7: torch.tensor([8.0, 2.2]),
                         8: torch.tensor([8.0, 2.3])}
            for ii, iu in enumerate(self.unique_atomic_numbers):
                mask = geometry.atomic_numbers == iu
                self.compr[mask] = init_dict[iu.tolist()]

            self.compr.requires_grad_(True)
        else:
            self.compr0 = torch.tensor([3.0, 2.7, 2.2, 2.3])
            # self.compr0 = torch.ones(len(self.unique_atomic_numbers)) * 3.5
            self.compr = torch.zeros(geometry.atomic_numbers.shape)
            self.compr0.requires_grad_(True)

        self.h_compr_feed = kwargs.get('h_compr_feed', True)
        self.s_compr_feed = kwargs.get('s_compr_feed', True)

        self.shell_dict = shell_dict
        self.basis = Basis(geometry.atomic_numbers, self.shell_dict)
        if self.h_compr_feed:
            self.h_feed = VcrFeed.from_dir(
                parameter['dftb']['path_to_skf'], self.shell_dict, compr_grid,
                skf_type='h5', geometry=geometry, integral_type='H',
                interpolation='BicubInterp')
        if self.s_compr_feed:
            self.s_feed = VcrFeed.from_dir(
                parameter['dftb']['path_to_skf'], self.shell_dict, compr_grid,
                skf_type='h5', geometry=geometry, integral_type='S',
                interpolation='BicubInterp')

        if not self.global_r:
            super().__init__(
                geometry, reference, [self.compr], parameter, **kwargs)
        else:
            super().__init__(
                geometry, reference, [self.compr0], parameter, **kwargs)

        self.skparams = SkfParamFeed.from_dir(
            parameter['dftb']['path_to_skf'], self.geometry, skf_type=skf_type)
        if self.geometry.isperiodic:
            self.periodic = Periodic(self.geometry, self.geometry.cell,
                                     cutoff=self.skparams.cutoff)


    def __call__(self, plot: bool = True, save: bool = True, **kwargs):
        """Train compression radii with target properties."""
        super().__call__()
        self._compr = []
        self.ham_list,self.over_list = [], []
        for istep in range(self.steps):
            self._update_train()
            print('step: ', istep, 'loss: ', self.loss.detach())

            break_tolerance = istep >= self.params['ml']['min_steps']
            if self.reach_convergence and break_tolerance:
                break

        if plot:
            super().__plot__(istep + 1, self.loss_list[1:],
                             compression_radii=self._compr)

        return self.dftb

    def _update_train(self):
        if self.global_r:
            for ii, iu in enumerate(self.unique_atomic_numbers):
                mask = self.geometry.atomic_numbers == iu
                self.compr[mask] = self.compr0[ii]

        hs_obj = self.periodic if self.geometry.isperiodic else self.geometry
        if self.h_compr_feed:
            ham = hs_matrix(hs_obj, self.basis, self.h_feed,
                            multi_varible=self.compr)
        else:
            ham = hs_matrix(hs_obj, self.basis, self.h_feed2)

        if self.s_compr_feed:
            over = hs_matrix(hs_obj, self.basis, self.s_feed,
                             multi_varible=self.compr)
        else:
            over = hs_matrix(hs_obj, self.basis, self.s_feed2)

        self.ham_list.append(ham.detach()), self.over_list.append(over.detach())
        self.dftb = Dftb2(self.params, self.geometry, self.shell_dict,
                          self.params['dftb']['path_to_skf'],
                          H=ham, S=over, from_skf=True)
        # self.dftb()
        super().__loss__(self.dftb)
        self._compr.append(self.compr.detach().clone())
        self.optimizer.zero_grad()
        self.loss.backward(retain_graph=True)
        self.optimizer.step()
        self._check(self.params)

    def _check(self, para):
        """Check the machine learning variables each step.

        When training compression radii, sometimes the compression radii will
        be out of range of given grid points and go randomly, therefore here
        the code makes sure the compression radii is in the defined range.
        """
        # detach remove initial graph and make sure compr_ml is leaf tensor
        if not self.global_r:
            compr = self.compr.detach().clone()
            min_mask = compr[compr != 0].lt(para['ml']['compression_radii_min'])
            max_mask = compr[compr != 0].gt(para['ml']['compression_radii_max'])
        else:
            vcr = self.compr0.detach().clone()
            min_mask = vcr[vcr != 0].lt(para['ml']['compression_radii_min'])
            max_mask = vcr[vcr != 0].gt(para['ml']['compression_radii_max'])
        if True in min_mask:
            if not self.global_r:
                with torch.no_grad():
                    self.compr.clamp_(min=para['ml']['compression_radii_min'])
            else:
                with torch.no_grad():
                    self.compr0.clamp_(min=para['ml']['compression_radii_min'])
        if True in max_mask:
            if not self.global_r:
                with torch.no_grad():
                    self.compr.clamp_(max=para['ml']['compression_radii_max'])
            else:
                with torch.no_grad():
                    self.compr0.clamp_(min=para['ml']['compression_radii_min'])

    def predict(self, geometry_pred: object, split_ratio: float = 0.5, **kwargs):
        """Predict with optimized Hamiltonian and overlap."""
        basis_pred = Basis(geometry_pred.atomic_numbers, self.shell_dict)

        # predict features
        feature_type = 'acsf'
        feature = Dscribe(self.geometry, feature_type=feature_type, **kwargs).features
        feature_pred = Dscribe(geometry_pred, feature_type=feature_type, **kwargs).features

        # use scikit learn to predict
        target = self.compr.detach()[self.geometry.atomic_numbers.ne(0)]
        compr_pred2 = SciKitLearn(
            self.geometry, feature, target, system_pred=geometry_pred,
            feature_pred=feature_pred, ml_method=self.params['ml']['ml_method'],
            split=split_ratio).prediction
        compr_pred2.clamp_(min=self.params['ml']['compression_radii_min'])
        compr_pred2.clamp_(max=self.params['ml']['compression_radii_max'])

        h_feed2, s_feed2 = VcrFeed.from_dir(
            self.params['dftb']['path_to_skf'], self.compr_grid, self.shell_dict,
            geometry_pred, interpolation='BicubInterp', h_feed=True, s_feed=True)
        ham2 = hs_matrix(geometry_pred, basis_pred, h_feed2, compr_pred2)
        over2 = hs_matrix(geometry_pred, basis_pred, s_feed2, compr_pred2)
        dftb2 = Dftb2(self.params, geometry_pred, self.shell_dict, ham2, over2, from_skf=True)
        dftb2()
        return dftb2


class OptTvcr(Optim):
    """Optimize compression radii."""

    def __init__(self, geometry: Geometry, reference, parameter,
                 tvcr: Tensor, shell_dict, **kwargs):
        """Initialize parameters."""
        self.tvcr = tvcr
        interpolation = kwargs.get('interpolation', 'MultiVarInterp')
        self.global_r = kwargs.get('global_r', False)
        self.unique_atomic_numbers = geometry.unique_atomic_numbers()

        if not self.global_r:
            # self.compr = torch.ones(geometry.atomic_numbers.shape, 2) * 3.5
            # self.compr.requires_grad_(True)

            # self.compr = torch.zeros(*geometry.atomic_numbers.shape, 2)
            # init_dict = {1: torch.tensor([2.5, 3.0]),
            #              6: torch.tensor([7.0, 2.7]),
            #              7: torch.tensor([8.0, 2.2]),
            #              8: torch.tensor([8.0, 2.3])}
            # for ii, iu in enumerate(self.unique_atomic_numbers):
            #     mask = geometry.atomic_numbers == iu
            #     self.compr[mask] = init_dict[iu.tolist()]

            # self.compr.requires_grad_(True)

            raise NotImplementedError('OptTvcr only support global varibales.')
        else:
            # self.compr0 = torch.ones(len(self.unique_atomic_numbers), 2) * 3.5
            # self.compr = torch.zeros(*geometry.atomic_numbers.shape, 2)
            # self.compr0.requires_grad_(True)

            self.compr = torch.zeros(*geometry.atomic_numbers.shape, 2)
            self.compr0 = torch.tensor(
                [[2.5, 3.0], [7.0, 2.7], [8.0, 2.2], [8.0, 2.3]]).requires_grad_(True)

        self.h_compr_feed = kwargs.get('h_compr_feed', True)
        self.s_compr_feed = kwargs.get('s_compr_feed', True)

        self.shell_dict = shell_dict
        self.basis = Basis(geometry.atomic_numbers, self.shell_dict)
        if self.h_compr_feed:
            self.h_feed = TvcrFeed.from_dir(
                parameter['dftb']['path_to_skf'], self.shell_dict, tvcr,
                skf_type='h5', geometry=geometry, integral_type='H',
                interpolation=interpolation)
        if self.s_compr_feed:
            self.s_feed = TvcrFeed.from_dir(
                parameter['dftb']['path_to_skf'], self.shell_dict, tvcr,
                skf_type='h5', geometry=geometry, integral_type='S',
                interpolation=interpolation)

        if not self.global_r:
            super().__init__(
                geometry, reference, [self.compr], parameter, **kwargs)
        else:
            super().__init__(
                geometry, reference, [self.compr0], parameter, **kwargs)

    def __call__(self, plot: bool = True, save: bool = True, **kwargs):
        """Train compression radii with target properties."""
        super().__call__()
        self._compr = []
        self.ham_list,self.over_list = [], []
        for istep in range(self.steps):
            self._update_train()
            print('step: ', istep, 'loss: ', self.loss.detach())

            break_tolerance = istep >= self.params['ml']['min_steps']
            if self.reach_convergence and break_tolerance:
                break

        if plot:
            super().__plot__(istep + 1, self.loss_list[1:],
                             compression_radii=self._compr)

        return self.dftb

    def _update_train(self):
        if self.global_r:
            for ii, iu in enumerate(self.unique_atomic_numbers):
                mask = self.geometry.atomic_numbers == iu
                self.compr[mask] = self.compr0[ii]

        if self.h_compr_feed:
            ham = hs_matrix(self.geometry, self.basis, self.h_feed,
                            multi_varible=self.compr)
        else:
            ham = hs_matrix(self.geometry, self.basis, self.h_feed2)

        if self.s_compr_feed:
            over = hs_matrix(self.geometry, self.basis, self.s_feed,
                             multi_varible=self.compr)
        else:
            over = hs_matrix(self.geometry, self.basis, self.s_feed2)

        self.ham_list.append(ham.detach()), self.over_list.append(over.detach())
        self.dftb = Dftb2(self.params, self.geometry, self.shell_dict,
                          self.params['dftb']['path_to_skf'],
                          H=ham, S=over, from_skf=True)
        # self.dftb()
        super().__loss__(self.dftb)
        self._compr.append(self.compr.detach().clone())
        self.optimizer.zero_grad()
        self.loss.backward(retain_graph=True)
        self.optimizer.step()
        self._check(self.params)

    def _check(self, para):
        """Check the machine learning variables each step.

        When training compression radii, sometimes the compression radii will
        be out of range of given grid points and go randomly, therefore here
        the code makes sure the compression radii is in the defined range.
        """
        # detach remove initial graph and make sure compr_ml is leaf tensor
        if not self.global_r:
            compr = self.compr.detach().clone()
            min_mask = compr[compr != 0].lt(para['ml']['compression_radii_min'])
            max_mask = compr[compr != 0].gt(para['ml']['compression_radii_max'])
        else:
            vcr = self.compr0.detach().clone()
            min_mask = vcr[vcr != 0].lt(para['ml']['compression_radii_min'])
            max_mask = vcr[vcr != 0].gt(para['ml']['compression_radii_max'])
        if True in min_mask:
            if not self.global_r:
                with torch.no_grad():
                    self.compr.clamp_(min=para['ml']['compression_radii_min'])
            else:
                with torch.no_grad():
                    self.compr0.clamp_(min=para['ml']['compression_radii_min'])
        if True in max_mask:
            if not self.global_r:
                with torch.no_grad():
                    self.compr.clamp_(max=para['ml']['compression_radii_max'])
            else:
                with torch.no_grad():
                    self.compr0.clamp_(min=para['ml']['compression_radii_min'])
