"""Train code."""
import numpy as np
import torch
from torch.autograd import Variable
from tbmalt.common.parameter import Parameter
import tbmalt.common.maths as tb_math
from tbmalt.tb.sk import SKT
from tbmalt.common.batch import pack
from tbmalt.tb.dftb.scc import Scc
from tbmalt.tb.properties import dos
from tbmalt.io.loadskf import IntegralGenerator
import matplotlib.pyplot as plt
from tbmalt.ml.feature import Dscribe
# from tbmalt.ml.scikitlearn import SciKitLearn


class Train:
    """Train class."""

    def __init__(self, system, reference, variable, parameter=None, **kwargs):
        self.system = system
        self.reference = reference
        self.loss_list = []
        self.loss_list.append(0)
        self.loss_tol = kwargs.get('tol', 1E-8)

        if type(variable) is torch.Tensor:
            self.variable = Variable(variable, requires_grad=True)
        elif type(variable) is list:
            self.variable = variable

        self.params = Parameter if parameter is None else parameter
        self.coulomb = kwargs.get('coulomb', None)
        self.periodic = kwargs.get('periodic', None)
        self.lr = self.params.ml_params['lr']

        # get loss function type
        if self.params.ml_params['loss_function'] == 'MSELoss':
            self.criterion = torch.nn.MSELoss(reduction='sum')
        elif self.params.ml_params['loss_function'] == 'L1Loss':
            self.criterion = torch.nn.L1Loss(reduction='sum')
        elif self.params.ml_params['loss_function'] == 'KLDivLoss':
            self.criterion = torch.nn.KLDivLoss(reduction='sum')
            # self.criterion = tb_math.hellinger

        # get optimizer
        if self.params.ml_params['optimizer'] == 'SCG':
            self.optimizer = torch.optim.SGD(self.variable, lr=self.lr)
        elif self.params.ml_params['optimizer'] == 'Adam':
            self.optimizer = torch.optim.Adam(self.variable, lr=self.lr)

    def __call__(self, properties):
        """Call train class with properties."""
        self.properties = properties
        self.steps = self.params.ml_params['steps']

    def __loss__(self, results):
        """Get loss function."""
        self.loss = 0.
        if 'dipole' in self.properties:
            self.loss = self.loss + self.criterion(
                results.dipole, self.reference['dipole'])
        if 'charge' in self.properties:
            self.loss = self.loss + self.criterion(
                results.charge, self.reference['charge'])
            print("results.charge", results.charge)
        if 'gap' in self.properties:
            homolumo = results.homo_lumo
            refhl = pack(self.reference['homo_lumo'])
            gap = homolumo[:, 1] - homolumo[:, 0]
            print("gap:", gap)
            refgap = refhl[:, 1] - refhl[:, 0]
            print("refgap:", refgap)
            self.loss = self.loss + self.criterion(gap, refgap)
        if 'homo_lumo' in self.properties:
            homolumo = results.homo_lumo
            refhl = pack(self.reference['homo_lumo'])
            self.loss = self.loss + self.criterion(homolumo, refhl)
        if 'cpa' in self.properties:
            cpa = results.cpa
            refcpa = self.reference['hirshfeld_volume_ratio']
            self.loss = self.loss + self.criterion(cpa, refcpa)
        if 'dos' in self.properties:
            ref = self.reference['dos']
            refenergies = ref[..., 0]
            refdos = ref[..., 1]
            _shift = torch.stack([refenergies[ii] - self.reference['homo_lumo'][..., 0][ii]
                                  for ii in range(refenergies.size(0))])
            homo_dftb = results.homo_lumo[..., 0]
            energies_dftb = torch.stack([_shift[ii] + homo_dftb[ii]
                                         for ii in range(refenergies.size(0))])
            dos_dftb = dos((results.eigenvalue),
                           energies_dftb, results.params.dftb_params['sigma'])
            self.loss = self.loss + self.criterion(dos_dftb, refdos)

        # dos calculated from dft eigenvalues
        if 'dos_eigval' in self.properties:
            ref = self.reference['dos']
            refenergies = ref[..., 0]
            refdos = ref[..., 1]
            _shift = torch.stack([refenergies[ii] - self.reference['homo_lumo'][..., 0][ii]
                                  for ii in range(refenergies.size(0))])
            homo_dftb = results.homo_lumo[..., 0]
            energies_dftb = torch.stack([_shift[ii] + homo_dftb[ii]
                                         for ii in range(refenergies.size(0))])
            dos_dftb = dos((results.eigenvalue),
                           energies_dftb, results.params.dftb_params['sigma'])
            self.loss = self.loss + self.criterion(dos_dftb, refdos)

        self.loss_list.append(self.loss.detach())
        self.reach_convergence = abs((self.loss_list[-1] - self.loss_list[-2]) /
                                     (self.loss_list[1])) < self.loss_tol

    def __predict__(self, system):
        """Predict with training results."""
        pass

    def __plot__(self, steps, loss, **kwargs):
        """Visualize training results."""
        compression_radii = kwargs.get('compression_radii', None)

        # plot loss
        plt.plot(np.linspace(1, steps, steps), loss)
        plt.xlabel("Iteration")
        plt.ylabel("MSE Loss")
        plt.show()

        # plot compression radii
        if compression_radii is not None:
            compr = pack(compression_radii)
            for ii in range(compr.shape[1]):
                for jj in range(compr.shape[2]):
                    plt.plot(np.linspace(1, steps, steps), compr[:, ii, jj])
            plt.show()


class Integal(Train):
    """Optimize integrals."""

    def __init__(self, system, reference, parameter, **kwargs):
        """Initialize parameters."""
        self.sk = IntegralGenerator.from_dir(
            parameter.dftb_params['path_to_skf'], system, repulsive=False,
            interpolation='spline', sk_type='h5py', with_variable=True)
        self.ml_variable = self.sk.sktable_dict['variable']
        super().__init__(system, reference, self.ml_variable, parameter, **kwargs)

    def __call__(self, target, plot=True, save=True):
        """Train spline parameters with target properties."""
        super().__call__(target)
        self._loss = []
        for istep in range(self.steps):
            print('istep', istep)
            self._update_train()
            self._loss.append(self.loss.detach())

            if self.reach_convergence:
                break

        if plot:
            super().__plot__(istep + 1, self.loss_list[1:])

        if save:
            f = open('loss.dat', 'w')
            np.savetxt(f, torch.tensor(self.loss_list[1:]))
            f.close()
            f = open('charge.dat', 'w')
            np.savetxt(f, self.scc.charge.detach() - self.scc.qzero)
            f.close()

    def _update_train(self):
        skt = SKT(self.system, self.sk, self.periodic, with_variable=True,
                  fix_onsite=True, fix_U=True)
        self.scc = Scc(self.system, skt, self.params, self.coulomb, self.periodic, self.properties)
        super().__loss__(self.scc)
        self.optimizer.zero_grad()
        self.loss.backward(retain_graph=True)
        self.optimizer.step()

    def predict(self, system, coulomb, periodic):
        """Predict with optimized Hamiltonian and overlap."""
        skt = SKT(system, self.sk, periodic, with_variable=True,
                  fix_onsite=True, fix_U=True)
        return Scc(system, skt, self.params, coulomb, periodic, self.properties)


class CompressionRadii(Train):
    """Optimize compression radii."""

    def __init__(self, system, reference, parameter, **kwargs):
        """Initialize parameters."""
        self.nbatch = system.size_batch
        self.ml_variable = Variable(torch.ones(
            self.nbatch, max(system.size_system)) * 3.5, requires_grad=True)
        self.sk = IntegralGenerator.from_dir(
            parameter.dftb_params['path_to_skf'], system, repulsive=False,
            sk_type='h5py', homo=False, interpolation='bicubic_interpolation',
            compression_radii_grid=parameter.ml_params['compression_radii_grid'])
        super().__init__(system, reference, [self.ml_variable], parameter, **kwargs)

    def __call__(self, target, plot=True, save=True):
        """Train compression radii with target properties."""
        super().__call__(target)
        self._compr = []
        for istep in range(self.steps):
            self._update_train()
            print('step:', istep, self.loss)
            print('grad', self.ml_variable.grad)

            if self.reach_convergence:
                break

        if plot:
            super().__plot__(istep + 1, self.loss_list[1:], compression_radii=self._compr)

        if save:
            f = open('compr.dat', 'w')
            np.savetxt(f, self.ml_variable.detach())
            f.close()
            f = open('loss.dat', 'w')
            np.savetxt(f, torch.tensor(self.loss_list[1:]))
            f.close()
            f = open('charge.dat', 'w')
            np.savetxt(f, self.scc.charge.detach() - self.scc.qzero)
            f.close()

        return self.scc

    def _update_train(self):
        self.skt = SKT(self.system, self.sk, self.periodic,
                       compression_radii=self.ml_variable, fix_onsite=True, fix_U=True)
        self.scc = Scc(self.system, self.skt, self.params, self.coulomb,
                       self.periodic, self.properties)
        super().__loss__(self.scc)
        self._compr.append(self.ml_variable.detach().clone())

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
        compr_ml = self.ml_variable.detach().clone()
        min_mask = compr_ml[compr_ml != 0].lt(para.ml_params['compression_radii_min'])
        max_mask = compr_ml[compr_ml != 0].gt(para.ml_params['compression_radii_max'])
        if True in min_mask or True in max_mask:
            with torch.no_grad():
                self.ml_variable.clamp_(para.ml_params['compression_radii_min'],
                                        para.ml_params['compression_radii_max'])

    def predict(self, system, coulomb, periodic):
        """Predict with optimized Hamiltonian and overlap."""
        feature_type = 'acsf'
        targets = self.ml_variable
        feature = Dscribe(self.system, feature_type=feature_type).features
        feature_pred = Dscribe(system, feature_type=feature_type).features
        split_ratio = 0.5
        predict_compr = SciKitLearn(
            self.system, feature, targets.detach(), system_pred=system,
            feature_pred=feature_pred, ml_method=self.params.ml_params['ml_method'],
            split_ratio=split_ratio).prediction
        predict_compr = torch.clamp(
            predict_compr, self.params.ml_params['compression_radii_min'],
            self.params.ml_params['compression_radii_max'])

        skt = SKT(system, self.sk, compression_radii=predict_compr,
                  fix_onsite=True, fix_U=True)
        return Scc(system, skt, self.params, coulomb, periodic, self.properties)

    @staticmethod
    def predict_(params, compr_train, system_train, system_test, sk, properties=[]):
        """Predict with optimized Hamiltonian and overlap."""
        feature_type = 'acsf'
        feature = Dscribe(system_train, feature_type=feature_type).features
        feature_pred = Dscribe(system_test, feature_type=feature_type).features
        split_ratio = 0.5
        predict_compr = SciKitLearn(
            system_train, feature, compr_train.detach(), system_pred=system_test,
            feature_pred=feature_pred, ml_method=params.ml_params['ml_method'],
            split_ratio=split_ratio).prediction
        predict_compr = torch.clamp(
            predict_compr, params.ml_params['compression_radii_min'],
            params.ml_params['compression_radii_max'])

        f = open('compr.dat', 'w')
        np.savetxt(f, predict_compr)
        f.close()

        skt = SKT(system_test, sk, compression_radii=predict_compr,
                  fix_onsite=True, fix_U=True)
        return Scc(system_test, skt, params, properties)


class Charge(Train):
    """Train charge."""

    def __init__(self, system, reference, variable, parameter, sk, **kwargs):
        super().__init__(system, reference, variable, parameter, **kwargs)
        self.sk = sk

    def __call__(self, properties):
        """Call train class with properties."""
        super().__call__(properties)
        for istep in range(self.steps):
            self._update_train()

    def _update_train(self):
        skt = SKT(self.system, self.sk)
        scc = Scc(self.system, skt, self.params, properties=self.properties)
