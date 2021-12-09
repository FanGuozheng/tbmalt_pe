"""Interface to some popular ML framework."""
import torch
import pickle
from sklearn import linear_model, svm
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from tbmalt.common.batch import pack


class SciKitLearn:
    """Machine learning with optimized data.

    process data.
    perform ML prediction.
    """

    def __init__(self, system, feature, target, **kwargs):
        """Initialize."""
        system_pred = kwargs.get('system_pred', None)
        feature_pred = kwargs.get('feature_pred', None)
        self.save_model = kwargs.get('save_model', False)
        self.target = target
        # self.target = self._flatten_target(target, system.n_atoms)

        self.ml_method = kwargs.get('ml_method', 'linear')
        self.split = kwargs.get('split', 0.5)

        if feature_pred is None:
            size_sys = system.n_atoms
            self.sum_size = [sum(size_sys[: ii]) for ii in range(len(size_sys) + 1)]
            self.x_train, _, self.y_train, _ = train_test_split(
                    feature, self.target, train_size=self.split)
            self.x_pred = feature
        else:
            size_sys = system_pred.n_atoms
            self.sum_size = [sum(size_sys[: ii]) for ii in range(len(size_sys) + 1)]
            self.x_train = feature
            self.x_pred = feature_pred
            self.y_train = self.target

        self.delta = kwargs.get('delata', True)
        self.prediction, self.model = getattr(SciKitLearn, self.ml_method)(self)

        if self.save_model:
            filename = kwargs.get('model_name', 'model.pickle')
            pickle.dump(self.model, open(filename, 'wb'))

    def linear(self):
        """Use the optimization dataset for training.

        Returns:
            linear ML method predicted DFTB parameters
        shape[0] of feature_data is defined by the optimized compression R
        shape[0] of feature_test is the defined by para['n_test']

        """
        reg = linear_model.LinearRegression()
        reg.fit(self.x_train, self.y_train)
        y_pred = torch.from_numpy(reg.predict(self.x_pred))

        # sum_size = [sum(size_sys[: ii]) for ii in range(len(size_sys) + 1)]
        return pack([y_pred[isize: self.sum_size[ii + 1]] for ii, isize in
                     enumerate(self.sum_size[: -1])]), reg

    def svm(self):
        """ML process with support vector machine method."""
        reg = svm.SVR()
        reg.fit(self.x_train, self.y_train)
        y_pred = torch.from_numpy(reg.predict(self.x_pred))

        return pack([y_pred[isize: self.sum_size[ii + 1]] for ii, isize in
                     enumerate(self.sum_size[: -1])]), reg

    def random_forest(self):
        """ML process with support vector machine method."""
        reg = RandomForestRegressor(n_estimators=100)
        reg.fit(self.x_train, self.y_train)
        y_pred = torch.from_numpy(reg.predict(self.x_pred))
        return pack([y_pred[isize: self.sum_size[ii + 1]] for ii, isize in
                     enumerate(self.sum_size[: -1])]), reg

    def grad_boost(self):
        """ML process with support vector machine method."""
        reg = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1)
        size_sys = self.system.n_atoms
        reg.fit(self.x_train, self.y_train)
        y_pred = torch.from_numpy(reg.predict(self.feature))

        sum_size = [sum(size_sys[: ii]) for ii in range(len(size_sys) + 1)]
        return pack([y_pred[isize: sum_size[ii + 1]] for ii, isize in
                     enumerate(sum_size[: -1])]), reg

    def krr(self):
        """Kernel ridge regression (KRR)."""
        clf = KernelRidge(alpha=1.0)
        clf.fit(self.x_train, self.y_train)
        y_pred = torch.from_numpy(clf.predict(self.x_pred))
        return pack([y_pred[isize: self.sum_size[ii + 1]] for ii, isize in
                     enumerate(self.sum_size[: -1])]), clf

    def nn(self):
        """ML process with support vector machine method."""
        clf = MLPRegressor(solver='lbfgs', alpha=1e-5,
                           hidden_layer_sizes=(5, 2), random_state=1)
        clf.fit(self.x_train, self.y_train)
        y_pred = torch.from_numpy(clf.predict(self.x_pred))
        return pack([y_pred[isize: self.sum_size[ii + 1]] for ii, isize in
                     enumerate(self.sum_size[: -1])]), clf

    def _flatten_target(self, target, size):
        """"""
        return torch.cat([itarget[: isize] for itarget, isize in
                          zip(target, size)])
