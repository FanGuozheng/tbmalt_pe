#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""""""
from typing import Union, Dict
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, Tensor
from torch.nn.functional import normalize

from tbmalt.physics.dftb.slaterkoster import add_kpoint, hs_matrix_nn
from tbmalt.io import Dataset
from tbmalt.structures.geometry import unique_atom_pairs
from tbmalt.common.logger import get_logger
from tbmalt.common.batch import pack


class NNModel(nn.Module):
    """Multi-layer Perceptron model."""
    def __init__(
        self, n_feature: int = 4,
            out_size: int = 1,
            activation="ReLU",
            nn_type="mlp",
            size: int = 500,
    ):
        super(NNModel, self).__init__()
        self.n_feature = n_feature
        self.flatten = nn.Flatten()
        self.activation = getattr(nn, activation)()
        self.out_size = out_size
        self.size = size
        self._nn = getattr(self, nn_type)()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.flatten(x)
        x = self.dropout(x)
        return self._nn(x)

    def mlp(self):
        torch.manual_seed(1)
        return nn.Sequential(
            nn.Linear(self.n_feature, self.size),
            self.activation,
            nn.Linear(self.size, self.size),
            self.activation,
            nn.Linear(self.size, self.size),
            self.activation,
            nn.Linear(self.size, self.size),
            self.activation,
            nn.Linear(self.size, self.size),
            self.activation,
            nn.Linear(self.size, self.size),
            self.activation,
            nn.Linear(self.size, self.size),
            self.activation,
            nn.Linear(self.size, self.out_size),
        )

    def cnn(self):
        return nn.Sequential(
            nn.Conv1d(self.n_feature, self.out_size, 100),
            nn.BatchNorm2d(self.n_feature),
            self.activation,
        )

    def rnn(self):
        pass

    def transformer(self):
        pass


class Hamiltonian(nn.Module):
    """Hamiltonian training model."""

    def __init__(
        self,
        params,
        shell_dict,
        general_dict,
        pre_train,
        n_feature_hs: int,
        n_feature_onsite: int,
        out_size: int = 1,
        activation="ReLU",
        nn_type="mlp",
        optim='Adam',
        orbital_resolve=True,
        pbc=True,
        m_split=False,
        **kwargs,
    ):
        super(Hamiltonian, self).__init__()
        self.params = params
        self.shell_dict = shell_dict
        self.pre_train=pre_train
        self.general_dict = general_dict
        self.n_feature_hs = n_feature_hs
        self.n_feature_onsite = n_feature_onsite
        self.out_size = out_size
        self.activation = activation
        self.nn_type = nn_type
        self.orbital_resolve = orbital_resolve
        self.optim = optim
        self.train_h = kwargs.get("train_h", True)
        self.train_s = kwargs.get("train_s", False)
        self.train_onsite = kwargs.get("train_onsite", False)
        self.orthogonal_dftb = kwargs.get("orthogonal_dftb", False)
        self.opt_with_model = kwargs.get('opt_with_model', False)
        self.ml_method = self.params['ml_method']
        self.ham_feature = kwargs.get('ham_feature', True)
        self.logger = kwargs.get("logger", get_logger(__name__))
        self.skt = kwargs.get('skt', True)
        self.alignment = kwargs.get('alignment', 'vbm')
        self.pre_train_step = kwargs.get('pre_train_step', 500)
        self.train_step = kwargs.get('pre_train_step', 100)
        self.pre_lr = kwargs.get('pre_lr', {'lr': 1E-1, 'lr_onsite': 1E-2})
        self.train_step = kwargs.get('train_step', 100)
        self.lr = kwargs.get('lr', {'lr': 1E-2, 'lr_on': 1E-7})
        self.decay = kwargs.get('decay', None)
        self.decay_params = kwargs.get('decay_params', None)
        self.train_distance_cutoff = kwargs.get('train_distance_cutoff', None)

        # LOOP over atomic pairs, l and m
        self.unique_atoms = self.general_dict['unique_atomic_number']
        self.element_pairs = unique_atom_pairs(unique_atomic_numbers=self.unique_atoms)
        self.l_pairs = torch.tensor(
            [  # -> l1, l2
                [pair[0], pair[1], i, j]
                for pair in self.element_pairs.tolist()
                for i in range(max(shell_dict[pair[0]]) + 1)
                for j in range(max(shell_dict[pair[1]]) + 1)
                if i <= j   # Attention here!!!!!!! the i <= j
            ]
        )
        self.zlm_pairs = torch.tensor(
            [
                [pair[0], pair[1], i, j, k]  # -> l1, l2, m12
                for pair in self.element_pairs.tolist()
                for i in range(max(shell_dict[pair[0]]) + 1)
                for j in range(max(shell_dict[pair[1]]) + 1)
                for k in range(min(i, j) + 1)
                if i <= j and k <= i
            ]
        )

        if self.decay_params is None:
            self.decay_params = {tuple(ii): [10, 10, 0, 1] for ii in self.zlm_pairs}

        # Build NN model list for different orbitals
        if self.train_h and self.opt_with_model:
            self.h_models, self.h_lr, self.hs_label_dict = self._build_orb_model()

        if self.train_onsite and self.opt_with_model:
            self.onsite_feature = kwargs.get("onsite_feature", None)
            assert self.onsite_feature is not None, "train onsite without features"
            n_feature_on = self.onsite_feature.shape[-1]  # self.n_feature_onsite if not self.ham_feature else self.n_feature_onsite - 1
            self.h_onsite_models, self.onsite_lr, self.onsite_label_dict =\
                self._build_onsite_model(n_feature_on)

        if self.train_s:
            self.s_models, self.s_lr, self.hs_label_dict = self._build_orb_model()

        self.loss_fn = nn.L1Loss()
        self.pbc = pbc
        self.neig_resolve = kwargs.get("neig_resolve", True)
        self.is_pre_train = False

    def _build_orb_model(self):
        """Build models for each orbital."""
        atom_pair = self.zlm_pairs[0][:2]
        l_max = 0

        # A label to NN, so we can get to know the NN in the list is which atomic
        # pairs and which orbitals, the count is the index in NN list
        # The label will be [atom1, atom2, l1, l2, m], where m <= min(l1, l2)
        model_key_dict = {}
        count = 0

        # lr_list is a learning rate dictionary, which allows us to tune lr of
        # each orbital flexibly.
        for ii, pair in enumerate(self.zlm_pairs.tolist()):

            if ii == 0:
                assert max(pair[2:]) == 0, 'the first term in zlm_pairs should be be zero'
                models = self.ss_model()
                lr_list = [self.lr['ss0'] if 'ss0' in self.lr.keys() else self.lr['lr']]\
                          * len(list(self.ss_model().parameters()))
                model_key_dict.update({(*pair[:2], 0, 0, 0): count})
                count += 1
            elif max(pair[2:]) == 0 and (*pair[:2], 0, 0, 0) not in model_key_dict.keys():
                models, lr_list = self._build_s(models, lr_list)
                model_key_dict.update({(*pair[:2], 0, 0, 0): count})
                count += 1

            if max(pair[2:]) == 1 and (*pair[:2], 0, 1, 0) not in model_key_dict.keys():
                models, lr_list, model_key_dict, count = \
                    self._build_p(models, lr_list, pair[:2], model_key_dict, count)

            if max(pair[2:]) == 2 and (*pair[:2], 0, 2, 0) not in model_key_dict.keys():
                models, lr_list, model_key_dict, count = \
                    self._build_d(models, lr_list, pair[:2], model_key_dict, count)

            if max(pair[2:]) == 3 and l_max == 2:
                raise NotImplementedError("do not support f orbitals")

        return models, lr_list, model_key_dict

    def _build_params(self, shape):
        pass

    def _build_s(self, models, lr_list):
        """Build NN model where max orbital is s."""
        models.extend(self.ss_model())
        lr_list.extend([self.lr['ss0'] if 'ss0' in self.lr.keys() else self.lr['lr']]
                       * len(list(self.ss_model().parameters())))
        return models, lr_list

    def _build_p(self, models, lr_list, pair, model_key_dict, count):
        """Build p orbital NN for a certain atomic pair."""
        models.extend(self.sp_model())
        lr_list.extend([self.lr['sp0'] if 'sp0' in self.lr.keys() else self.lr['lr']]
                       * len(list(self.sp_model().parameters())))
        model_key_dict.update({(*pair, 0, 1, 0): count})
        count += 1

        models.extend(self.pp_model())
        lr_list.extend([self.lr['pp0'] if 'pp0' in self.lr.keys() else self.lr['lr']]
                       * len(list(self.pp_model()[0].parameters())))
        model_key_dict.update({(*pair, 1, 1, 0): count})
        count += 1
        lr_list.extend([self.lr['pp1'] if 'pp1' in self.lr.keys() else self.lr['lr']]
                       * len(list(self.pp_model()[1].parameters())))
        model_key_dict.update({(*pair, 1, 1, 1): count})
        count += 1
        return models, lr_list, model_key_dict, count

    def _build_d(self, models, lr_list, pair, model_key_dict, count):
        """Build d orbital NN for a certain atomic pair."""
        models.extend(self.sd_model())
        lr_list.extend([self.lr['sd0'] if 'sd0' in self.lr.keys() else self.lr['lr']]
                       * len(list(self.sd_model().parameters())))
        model_key_dict.update({(*pair, 0, 2, 0): count})
        count += 1

        models.extend(self.pd_model())
        lr_list.extend([self.lr['pd0'] if 'pd0' in self.lr.keys() else self.lr['lr']]
                       * len(list(self.pd_model()[0].parameters())))
        model_key_dict.update({(*pair, 1, 2, 0): count})
        count += 1
        lr_list.extend([self.lr['pd1'] if 'pd1' in self.lr.keys() else self.lr['lr']]
                       * len(list(self.pd_model()[1].parameters())))
        model_key_dict.update({(*pair, 1, 2, 1): count})
        count += 1

        models.extend(self.dd_model())
        lr_list.extend([self.lr['dd0'] if 'dd0' in self.lr.keys() else self.lr['lr']]
                       * len(list(self.dd_model()[0].parameters())))
        model_key_dict.update({(*pair, 2, 2, 0): count})
        count += 1
        lr_list.extend([self.lr['dd1'] if 'dd1' in self.lr.keys() else self.lr['lr']]
                       * len(list(self.dd_model()[1].parameters())))
        model_key_dict.update({(*pair, 2, 2, 1): count})
        count += 1
        lr_list.extend([self.lr['dd2'] if 'dd2' in self.lr.keys() else self.lr['lr']]
                       * len(list(self.dd_model()[2].parameters())))
        model_key_dict.update({(*pair, 2, 2, 2): count})
        count += 1

        return models, lr_list, model_key_dict, count

    def _build_f(self, models, lr_list):
        raise ValueError(
            "do not support f orbitals, "
            f"but get l number: {torch.max(self.l_pairs)}"
        )

    def _build_onsite_model(self, n_feature_on):
        """Build non orbital resolved onsite models."""
        lr_list = []
        model_key_dict = {}
        count = 0
        for ii, uniq_atom in enumerate(self.unique_atoms.tolist()):
            l_max = max(self.shell_dict[uniq_atom])

            if ii == 0:
                models = self.s_onsite(n_feature_on)
            else:
                models.extend(self.s_onsite(n_feature_on))
            lr_list.extend([self.lr['s'] if 's' in self.lr.keys() else self.lr['lr_onsite']]
                           * len(list(self.s_onsite().parameters())))
            model_key_dict.update({(uniq_atom, 0): count})
            count += 1

            if l_max >= 1:
                models.extend(self.p_onsite(n_feature_on))
                lr_list.extend([self.lr['p'] if 'p' in self.lr.keys() else self.lr['lr_onsite']]
                               * len(list(self.p_onsite().parameters())))
                model_key_dict.update({(uniq_atom, 1): count})
                count += 1

            if l_max >= 2:
                models.extend(self.d_onsite(n_feature_on))
                lr_list.extend([self.lr['d'] if 'd' in self.lr.keys() else self.lr['lr_onsite']]
                               * len(list(self.d_onsite().parameters())))
                model_key_dict.update({(uniq_atom, 2): count})
                count += 1

            if l_max >= 3:
                raise NotImplementedError('f orbitals are not implemented')

        return models, lr_list, model_key_dict

    def to_ham(
        self,
        hs_pred_dict,
        h_index_dict,
        dftb: object,
        type="H",
        hs_onsite={},
        h_mat_dict={},
    ):
        """Transfer predicted Hamiltonian to DFTB Hamiltonian."""

        hs_feed = dftb.h_feed if type == "H" else dftb.s_feed
        _hs_dict = {}
        if type == "S" and self.orthogonal_dftb:
            n_batch, n_mat, _ = dftb.basis.orbital_matrix_shape
            return (
                torch.eye(n_mat)
                .repeat(n_batch, torch.max(n_kpoints), 1, 1)
                .permute(0, -2, -1, 1))
        else:
            return add_kpoint(
                hs_pred_dict,
                h_index_dict,
                dftb.periodic,
                self.shell_dict,
                dftb.basis,
                hs_feed,
                train_onsite=self.train_onsite,
                hs_onsite=hs_onsite,
                hs_dict=h_mat_dict
            )

    def pre_train(self,
                  X: Tensor,
                  h_mat_dict: dict = None,
                  h_onsite_dict: dict = None,
                  n_shell=False,
                  device=torch.device("cpu")):
        self.logger.info('begin pre-training...')
        self.is_pre_train = True
        _loss = []
        optimizer = self.set_optim()
        _mask = self.dftb1_band.geometry.distances_pe.lt(6.0).flatten() * \
                self.dftb1_band.geometry.distances_pe.gt(1.0).flatten()

        if h_mat_dict is not None:
            self.h_mat_dict = h_mat_dict
            self.h_onsite_dict = h_onsite_dict
        else:
            self.h_mat_dict, self.h_onsite_dict = hs_matrix_nn(
                self.dftb1_band.geometry,
                self.dftb1_band.basis,
                self.dftb1_band.h_feed,
                train_onsite=self.train_onsite,
                pbc=self.pbc)

        self.s_mat_dict, _ = hs_matrix_nn(
            self.dftb1_band.geometry,
            self.dftb1_band.basis,
            self.dftb1_band.s_feed,
            pbc=self.pbc, n_shell=n_shell
        )

        if self.orthogonal_dftb:
            if self.train_s:
                self.logger.warning(
                    "orthogonal_dftb is True, train_s should be False, TBMaLT turn off the train_s")
                self.train_s = False

        # Compute prediction error
        for ii in range(self.pre_train_step):

            pred_h, pred_s, pred_h_on = {}, {}, {}
            atom_pair = self.zlm_pairs.tolist()[0][:2]
            for il, zlm_pair in enumerate(self.zlm_pairs.tolist()):
                if zlm_pair[0] != atom_pair[0] or zlm_pair[1] != zlm_pair[1]:
                    atom_pair = zlm_pair[:2]

                lm_pair = zlm_pair[2:]
                min_lm = int(min(lm_pair[:2]))
                key = tuple(lm_pair[:2])
                _X = X

                if lm_pair[-1] == 0:
                    pred_h.update({key: torch.zeros(_X.shape[0], min_lm + 1)})
                    if self.train_s:
                        pred_s.update({key: torch.zeros(_X.shape[0], min_lm + 1)})

                tmp = torch.zeros(_X.shape[0], 1)
                count = self.hs_label_dict[tuple(zlm_pair)]
                tmp[_mask] = self.h_models[count](_X[_mask])

                if self.decay is not None:
                    tmp[_mask] = tmp[_mask] * getattr(self, self.decay)(self.decay_params[tuple(lm_pair)])[_mask]
                pred_h[key][..., lm_pair[-1]] = tmp.squeeze(-1)

                if self.train_s:
                    tmps = torch.zeros(_X.shape[0], 1)
                    tmps[_mask] = self.s_models[il](_X[_mask])

                    if self.decay is not None:
                        tmps = tmps * getattr(self, self.decay)
                    pred_s[key][..., lm_pair[-1]] = tmps.squeeze(-1)

            for iatom in self.unique_atoms.tolist():
                for il in range(max(self.shell_dict[iatom]) + 1):
                    if self.ml_method == 'nn_scale':
                        pred_h_on.update({
                            (iatom, il): self.h_onsite_dict[il] *
                                         (self.h_onsite_models[il](self.onsite_feature) + 1)})
                    elif self.ml_method == 'nn_hs':
                        pred_h_on.update({il: self.h_onsite_models[il](self.onsite_feature)})

            # Constrain to avoid random predictions
            # self._constrain(pred_h, pred_h_on)

            loss = 0.0
            for (predk, predv), (refk, refv) in zip(pred_h.items(), self.h_mat_dict.items()):
                if self.ml_method == 'nn_scale':
                    refv = torch.ones(refv.shape)

                if self.pbc:
                    loss = loss + self.loss_fn(predv[_mask], refv.flatten(0, -2)[_mask])
                else:
                    loss = loss + self.loss_fn(predv[_mask], refv[_mask])
            if self.train_onsite and self.train_h:
                for il in range(torch.max(self.zlm_pairs).tolist() + 1):
                    if self.ml_method in ('nn', 'nn_hs'):
                        loss = loss + 0.05 * self.loss_fn(pred_h_on[il], self.h_onsite_dict[il])

            if self.train_s:
                for (predk, predv), (refk, refv) in zip(
                    pred_s.items(), self.s_mat_dict.items()):
                    if self.ml_method in ('nn', 'nn_hs'):
                        loss = loss + 5 * self.loss_fn(predv, refv.flatten(0, -2))
                    elif self.ml_method == 'nn_scale':
                        loss = loss + 5 * self.loss_fn(predv, torch.ones(predv.shape))
                    else:
                        raise ValueError(f'{self.ml_method} is not valid')

            _loss.append(loss.detach())
            self.logger.info(f"step: {ii}, loss: {loss.detach().tolist()}")

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if ii % 100 == 0:
                self._plot_hs_dict(X, pred_h, pred_h_on, pred_s, _mask)

        # plt.plot(torch.arange(len(_loss)), _loss, label="loss")
        # plt.legend()
        # plt.show()

    def forward(self, train_dict, **kwargs):
        if self.pre_train:
            pass
        self.logger.info('begin training...')

        # Set global NN parameters of each orbitals for various atomic pairs
        if self.orthogonal_dftb:
            if self.train_s:
                self.logger.warning(
                    "orthogonal_dftb is True, train_s should be False, TBMaLT turn off the train_s")
                self.train_s = False
        n_train_batch = sum(self.general_dict['n_train_batch_list'])

        for key, data in train_dict.items():
            # train_dict[key]['ksampling'] = train_dict[key]['dftb1_band'].ksampling
            train_dict[key]['n_kpoints'] = max(train_dict[key]['dftb1_band'].n_kpoints)
            # train_dict[key]['cellvec_neighbour'] = train_dict[key]['dftb1_band'].geometry.cellvec_neighbour
            # train_dict[key]['ksampling_scc'] = train_dict[key]['dftb2_scc'].ksampling
            train_dict[key]['n_kpoints_scc'] = max(train_dict[key]['dftb2_scc'].n_kpoints)
            # train_dict[key]['cellvec_neighbour_scc'] = train_dict[key]['dftb2_scc'].geometry.cellvec_neighbour

            train_dict[key]['h_mat_dict'], train_dict[key]['h_index_dict'],\
            train_dict[key]['h_onsite_dict'] = hs_matrix_nn(
                train_dict[key]['dftb1_band'].periodic,
                train_dict[key]['dftb1_band'].basis,
                train_dict[key]['dftb1_band'].h_feed,
                train_onsite=self.train_onsite,
                pbc=self.pbc)

            train_dict[key]['vec_mat_a'] = -normalize(train_dict[key]['dftb1_band'].geometry.distance_vectors, 2, -1)

            train_dict[key]['phase'] = train_dict[key]['dftb1_band'].phase if\
                train_dict[key]['dftb1_band'].geometry.is_periodic else None

            train_dict[key]['phase_scc'] = train_dict[key]['dftb2_scc'].phase if\
                train_dict[key]['dftb2_scc'].geometry.is_periodic else None
            if not self.pbc:
                train_dict[key]['phase'] = train_dict[key]['phase'].permute(1, 2, 3, 0)
                train_dict[key]['phase_scc'] = train_dict[key]['phase_scc'].permute(1, 2, 3, 0)

            (train_dict[key]['ref_vband'],
             train_dict[key]['ref_cband'],
             train_dict[key]['ref_delta_vband'],
             train_dict[key]['ref_delta_cband'],
             train_dict[key]['mask_v']
             ) = self.get_reference(train_dict[key], train_dict[key]['dftb1_band'])

            # Compute prediction error
            train_dict[key]['_mask'] = train_dict[key]['dftb1_band'].periodic.distances.lt(9.0).flatten() * \
                    train_dict[key]['dftb1_band'].periodic.distances.gt(1.0).flatten()

        if self.opt_with_model:
            self.with_nn_model(train_dict, n_train_batch)
        else:
            self.without_model(train_dict, n_train_batch)

    def with_nn_model(self, train_dict, n_train_batch):
        optimizer = self.set_optim()
        _loss = []

        n_train_batch_list = np.arange(n_train_batch)
        np.random.shuffle(n_train_batch_list)
        vband_err, cband_err = [], []

        for ii in range(self.train_step):

            this_ind = n_train_batch_list[ii % n_train_batch]
            this_train_dict = train_dict[this_ind]
            X = this_train_dict['atomic_feature']
            onsite_feature = this_train_dict['onsite_feature']
            _mask = this_train_dict['_mask']
            dftb2_scc = this_train_dict['dftb2_scc']
            dftb1_band = this_train_dict['dftb1_band']
            atom_pairs = dftb2_scc.basis.atomic_number_matrix("atomic")
            h_onsite_dict = this_train_dict['h_onsite_dict']
            h_mat_dict = this_train_dict['h_mat_dict']
            h_index_dict = this_train_dict['h_index_dict']

            u_number = torch.unique(dftb2_scc.geometry.atomic_numbers)

            pred_h, pred_s, pred_h_on = {}, {}, {}

            atom_pair = self.zlm_pairs.tolist()[0][:2]
            for il, zlm_pair in enumerate(self.zlm_pairs.tolist()):
                if zlm_pair[0] != atom_pair[0] or zlm_pair[1] != zlm_pair[1]:
                    atom_pair = zlm_pair[:2]

                lm_pair = zlm_pair[2:]
                min_l = min(lm_pair[:2])
                key = tuple(zlm_pair[:-1])

                # Skip if no such atomic pairs in current batch
                if zlm_pair[0] not in u_number or zlm_pair[1] not in u_number:
                    continue

                _mask_pair = ((atom_pairs[..., 0] == atom_pair[0]) *
                              (atom_pairs[..., 1] == atom_pair[1])).flatten()

                # To save memory, some orbitals (keys) may be not directly as
                # input, therefore these keys may be skipped
                if not key in h_index_dict.keys():
                    continue

                # To make sure when build NN model for orbitals, different m will be combined.
                # Such as pp0 and pp1 will be combined for the following SK
                if lm_pair[-1] == 0:

                    # First index select the orbitals
                    print('X', X.shape)
                    x_selec = X[[*h_index_dict[key][0]]].transpose(0, 1)
                    print('x_selec', x_selec.shape, h_index_dict[key][1].shape)
                    x_selec = x_selec[[h_index_dict[key][1]]][h_index_dict[key][2]]

                    pred_h.update({key: torch.zeros(x_selec.shape[0], min_l + 1)})
                    if self.train_s:
                        pred_s.update({key: torch.zeros(X[_mask_pair].shape[0], min_l + 1)})

                if self.ml_method == 'nn_scale':
                    if self.train_distance_cutoff is None:
                        count = self.hs_label_dict[tuple(zlm_pair)]
                        tmp = self.h_models[count](x_selec) + 1
                        pred_h[key][..., lm_pair[-1]] = tmp.squeeze()
                    else:
                        count = self.hs_label_dict[tuple(zlm_pair)]
                        tmp = torch.ones(X[_mask_pair].shape[0], 1)
                        _dist = dftb1_band.geometry.distances_pe.flatten()
                        _mask_d = _dist[_mask_pair].lt(7.0) * _dist[_mask_pair].gt(1.0)
                        tmp[_mask_d] = self.h_models[count](X[_mask_pair][_mask_d]) + 1
                        pred_h[key][..., lm_pair[-1]] = tmp.squeeze()
                else:
                    tmp = torch.zeros(X.shape[0], 1)
                    count = self.hs_label_dict[tuple(zlm_pair)]
                    tmp[_mask] = self.h_models[count](X[_mask])
                    if self.decay is not None:
                        tmp = tmp * getattr(self, self.decay)(self.decay_params[tuple(lm_pair)])

                if self.train_s:
                    tmps = torch.zeros(X.shape[0], 1)
                    tmps[_mask] = self.s_models[il](X[_mask])
                    if self.decay is not None:
                        tmps = tmps * getattr(self, self.decay)

                    if self.ml_method in ('nn', 'nn_hs'):
                        pred_s[key][..., lm_pair[-1]] = tmps.squeeze(-1)
                    elif self.ml_method == 'nn_scale':
                        pred_s[key][..., lm_pair[-1]] = \
                            tmps * h_mat_dict[key][..., lm_pair[-1]].flatten()
                    else:
                        raise ValueError(f'{self.ml_method} is not valid')

            if self.train_onsite:
                _numbers = dftb1_band.geometry.atomic_numbers
                _numbers = _numbers[_numbers.ne(0)]

                for iatom in u_number.tolist():
                    _mask_uan = _numbers == iatom
                    _onsite_feature = onsite_feature[_mask_uan]

                    for il in range(max(self.shell_dict[iatom]) + 1):
                        count = self.onsite_label_dict[(iatom, il)]
                        if self.ml_method == 'nn_scale':
                            pred_h_on.update({(iatom, il): h_onsite_dict[iatom, il] *
                                                           (self.h_onsite_models[count](_onsite_feature) + 1)})
                        elif self.ml_method == 'nn_hs':
                            pred_h_on.update({il: self.h_onsite_models[il](_onsite_feature)})

            # Check and constrain the Hamiltonian and overlap
            # self._constrain(pred_h, pred_h_on)

            # < k_i | H_ij | k_j >
            ham = self.to_ham(pred_h, h_index_dict, dftb1_band, "H",
                              pred_h_on, h_mat_dict=h_mat_dict)
            ham_scc = self.to_ham(pred_h, h_index_dict, dftb2_scc, "H",
                                  pred_h_on, h_mat_dict=h_mat_dict,)

            if self.train_s:
                over = self.to_ham(pred_s, h_index_dict, dftb1_band, "S",
                                   h_mat_dict=h_mat_dict,)
                over_scc = self.to_ham(pred_s, h_index_dict, self.dftb1_scc,
                                       "S", h_mat_dict=h_mat_dict,)
            elif self.orthogonal_dftb:
                over_scc, over = None, None
            else:
                over_scc, over = dftb2_scc.over, dftb1_band.over

            @torch.no_grad()
            def no_grad_dftb2():
                dftb2_scc(hamiltonian=ham_scc, overlap=over_scc)
                return dftb2_scc, dftb2_scc._charge

            dftb2_scc, charge = no_grad_dftb2()
            dftb1_band(charge=charge, hamiltonian=ham, overlap=over)
            pred_v0, pred_c0, delta_v, delta_c = self._alignment(
                dftb1_band, this_train_dict, dftb2_scc.E_fermi, alignment=self.alignment)

            loss = self.loss_fn(pred_v0, this_train_dict['ref_vband'])
            loss = loss + self.loss_fn(delta_v, this_train_dict['ref_delta_vband'])
            loss = loss + self.loss_fn(pred_c0, this_train_dict['ref_cband'])
            loss = loss + self.loss_fn(delta_c, this_train_dict['ref_delta_cband'])
            vband_err.append(
                [(abs(pred_v0.detach() - this_train_dict['ref_vband'])).sum(),
                 (abs(delta_v.detach() - this_train_dict['ref_delta_vband'])).sum()])
            cband_err.append(
                [(abs(pred_c0.detach() - this_train_dict['ref_cband'])).sum(),
                 (abs(delta_c.detach() - this_train_dict['ref_delta_cband'])).sum()])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _loss.append(loss.detach().tolist())
            self.logger.info(f"step: {ii}, loss: {loss.detach().tolist()}")

            # if ii in n_plot:
            eigenvalue = dftb1_band.eigenvalue.detach().clone() - \
                dftb2_scc.E_fermi.unsqueeze(-1).unsqueeze(-1)
            self._plot(ii, this_train_dict, dftb1_band, eigenvalue,
                       torch.cat([pred_v0.detach(), pred_c0.detach()], -1))
            # self._plot_hs_dict(ii, this_train_dict, X, h_mat_dict, h_onsite_dict,
            #                    h_merge_pred_dict, pred_h_on, pred_s, _mask)
            self._plot_eigenvalue(ii, pred_v0.detach(), delta_v.detach(),
                                  pred_c0.detach(), delta_c.detach(), this_train_dict)

        np.savetxt('vband_err.txt', vband_err)
        np.savetxt('cband_err.txt', cband_err)
        plt.plot(torch.arange(len(_loss)), torch.log(torch.tensor(_loss)))
        plt.savefig('train_loss')
        plt.show()

    def without_model(self, train_dict, n_train_batch):
        n_train_batch_list = np.arange(n_train_batch)
        np.random.shuffle(n_train_batch_list)
        vband_err, cband_err, _loss = [], [], []
        train_twocnt_params, train_onsite_params = [], []
        train_twocnt_dict = {}
        twocnt_label, onsite_label = [], []

        # Collect training params for all batch system
        if self.train_onsite:
            for ii in range(n_train_batch):
                this_ind = n_train_batch_list[ii]
                h_onsite_dict = train_dict[this_ind]['h_onsite_dict']
                dftb2_scc = train_dict[this_ind]['dftb2_scc']

                for key, val in h_onsite_dict.items():
                    train_onsite_params = train_onsite_params + [{
                        'params': val.requires_grad_(True),
                        'lr': self.lr['lr']}]

                    onsite_label.append((ii, key))

        if self.train_h:
            for ii in range(n_train_batch):
                this_ind = n_train_batch_list[ii]
                h_mat_dict = train_dict[this_ind]['h_mat_dict']
                h_index_dict = train_dict[this_ind]['h_index_dict']

                for key, idx in h_index_dict.items():

                    pred_param = torch.ones(h_mat_dict[key[2:]][idx[2]].shape,
                                            dtype=torch.get_default_dtype(),
                                            requires_grad=True)
                    train_twocnt_dict.update({key: pred_param})

                    # The KEY should be revised!!!
                    train_twocnt_params = train_twocnt_params + [{
                        'params': train_twocnt_dict[key],
                        'lr': self.lr['lr']
                    }]
                    twocnt_label.append((ii, key))

        optimizer = torch.optim.SGD(train_twocnt_params + train_onsite_params, lr=self.lr['lr'])

        for ii in range(self.train_step):
            this_num = ii % n_train_batch
            this_ind = n_train_batch_list[this_num]

            this_train_dict = train_dict[this_ind]
            _mask = this_train_dict['_mask']
            dftb2_scc = this_train_dict['dftb2_scc']
            dftb1_band = this_train_dict['dftb1_band']
            atom_pairs = dftb2_scc.basis.atomic_number_matrix("atomic")
            h_onsite_dict = this_train_dict['h_onsite_dict']
            h_mat_dict = this_train_dict['h_mat_dict']
            h_index_dict = this_train_dict['h_index_dict']

            pred_h, pred_s, pred_h_on = {}, {}, {}
            atom_pair = self.zlm_pairs.tolist()[0][:2]

            # Directly generate band parameters as variables
            for il, zlm_pair in enumerate(self.zlm_pairs.tolist()):

                if zlm_pair[-1] != 0:
                    continue

                if zlm_pair[0] != atom_pair[0] or zlm_pair[1] != zlm_pair[1]:
                    atom_pair = zlm_pair[:2]
                key = tuple(zlm_pair[:-1])
                lm_pair = zlm_pair[2:]
                min_l = min(lm_pair[:2])
                _mask_pair = ((atom_pairs[..., 0] == atom_pair[0]) *
                              (atom_pairs[..., 1] == atom_pair[1])).unsqueeze(-1).repeat(
                    1, 1, 1, dftb2_scc.periodic.distances.shape[-1]).flatten()

                # There is no such atomic pairs in current pairs
                if not _mask_pair.any():
                    continue

            if self.train_h:
                _numbers = dftb1_band.geometry.atomic_numbers
                _numbers = _numbers[_numbers.ne(0)]

                for key in twocnt_label:
                    if this_num == key[0]:
                        _num = twocnt_label.index(key)
                        # pred_h = train_twocnt_params[_num]['params']
                        pred_h.update({key[1]: train_twocnt_params[_num]['params']})

            if self.train_onsite:
                _numbers = dftb1_band.geometry.atomic_numbers

                for key, val in h_onsite_dict.items():
                    n_num = (_numbers == key[0]).sum()
                    _num = onsite_label.index((this_num, key))
                    onsite_val = train_onsite_params[_num]['params'][0]
                    pred_h_on.update({key: onsite_val.repeat(n_num, 1)})

            # < k_i | H_ij | k_j >
            ham = self.to_ham(pred_h, h_index_dict, dftb1_band, "H",
                              pred_h_on, h_mat_dict=h_mat_dict)

            ham_scc = self.to_ham(train_twocnt_dict, h_index_dict, dftb2_scc,
                                  "H", pred_h_on, h_mat_dict=h_mat_dict)
            over_scc, over = dftb2_scc.over, dftb1_band.over

            @torch.no_grad()
            def no_grad_dftb2():
                dftb2_scc(hamiltonian=ham_scc, overlap=over_scc)
                return dftb2_scc, dftb2_scc._charge

            dftb2_scc, charge = no_grad_dftb2()
            dftb1_band(charge=charge, hamiltonian=ham, overlap=over)
            pred_v0, pred_c0, delta_v, delta_c = self._alignment(
                dftb1_band, this_train_dict, dftb2_scc.E_fermi, alignment=self.alignment)

            loss = self.loss_fn(pred_v0, this_train_dict['ref_vband'])
            loss = loss + self.loss_fn(delta_v, this_train_dict['ref_delta_vband'])
            loss = loss + self.loss_fn(pred_c0, this_train_dict['ref_cband'])
            loss = loss + self.loss_fn(delta_c, this_train_dict['ref_delta_cband'])
            vband_err.append(
                [(abs(pred_v0.detach() - this_train_dict['ref_vband'])).sum(),
                 (abs(delta_v.detach() - this_train_dict['ref_delta_vband'])).sum()])
            cband_err.append(
                [(abs(pred_c0.detach() - this_train_dict['ref_cband'])).sum(),
                 (abs(delta_c.detach() - this_train_dict['ref_delta_cband'])).sum()])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _loss.append(loss.detach().tolist())
            self.logger.info(f"step: {ii}, loss: {loss.detach().tolist()}")

            # if ii in n_plot:
            eigenvalue = dftb1_band.eigenvalue.detach().clone() - \
                dftb2_scc.E_fermi.unsqueeze(-1).unsqueeze(-1)
            self._plot(ii, this_train_dict, dftb1_band, eigenvalue,
                       torch.cat([pred_v0.detach(), pred_c0.detach()], -1))
            # self._plot_hs_dict(ii, this_train_dict, X, h_mat_dict, h_onsite_dict,
            #                    h_merge_pred_dict, pred_h_on, pred_s, _mask)
            self._plot_eigenvalue(ii, pred_v0.detach(), delta_v.detach(),
                                  pred_c0.detach(), delta_c.detach(), this_train_dict)

        np.savetxt('vband_err.txt', vband_err)
        np.savetxt('cband_err.txt', cband_err)
        plt.plot(torch.arange(len(_loss)), torch.log(torch.tensor(_loss)))
        plt.savefig('train_loss')
        plt.show()

    def predict(self,
                X,
                ref,
                dftb2_scc_test,
                dftb1_scc_test,
                dftb1_band_test,
                onsite_feature=None,
                is_periodic=True):

        _mask = dftb1_band_test.geometry.distances_pe.lt(6.0).flatten() * \
                dftb1_band_test.geometry.distances_pe.gt(1.0).flatten()

        if dftb1_band_test.geometry.is_periodic:
            ksampling = dftb1_band_test.ksampling
            n_kpoints = max(dftb1_band_test.n_kpoints)
            cellvec_neighbour = dftb1_band_test.geometry.cellvec_neighbour
            ksampling_scc = dftb1_scc_test.ksampling
            n_kpoints_scc = max(dftb1_scc_test.n_kpoints)
            cellvec_neighbour_scc = dftb1_scc_test.geometry.cellvec_neighbour

        vec_mat_a = -normalize(dftb1_band_test.geometry.distance_vectors, 2, -1)
        phase = (
            ksampling.phase(ksampling.kpoints, cellvec_neighbour, self.pbc, vec_mat_a)
            if dftb1_band_test.geometry.is_periodic else None)
        phase_scc = (
            ksampling_scc.phase(ksampling_scc.kpoints, cellvec_neighbour_scc, self.pbc, vec_mat_a)
            if dftb1_scc_test.geometry.is_periodic else None)
        if not self.pbc:
            phase = phase.permute(1, 2, 3, 0)
            phase_scc = phase_scc.permute(1, 2, 3, 0)
        h_mat_dict, h_onsite_dict = hs_matrix_nn(dftb1_band_test.geometry,
                                                 dftb1_band_test.basis,
                                                 dftb1_band_test.h_feed,
                                                 train_onsite=self.train_onsite,
                                                 pbc=self.pbc)
        ref_vband, ref_cband, ref_delta_vband, ref_delta_cband, mask_v = self.get_reference(ref, dftb1_band_test)

        eigenvalue0 = dftb1_band_test.eigenvalue.detach().clone() -\
                      dftb1_scc_test.E_fermi.unsqueeze(-1).unsqueeze(-1)
        pred_v_init, pred_c_init, delta_v_init, delta_c_init = self._alignment(
            dftb1_band_test, ref, dftb1_scc_test.E_fermi, mask_v, alignment=self.alignment)

        _unique_atoms = torch.unique(dftb1_band_test.geometry.atomic_numbers)
        unique_atoms = _unique_atoms[_unique_atoms.ne(0)]
        pred_h, pred_s, pred_h_on = {}, {}, {}
        atom_pairs_test = dftb1_scc_test.basis.atomic_number_matrix("atomic")
        atom_pair = self.zlm_pairs.tolist()[0][:2]
        for il, zlm_pair in enumerate(self.zlm_pairs.tolist()):
            if zlm_pair[0] != atom_pair[0] or zlm_pair[1] != zlm_pair[1]:
                atom_pair = zlm_pair[:2]

            lm_pair = zlm_pair[2:]
            min_l = min(lm_pair[:2])
            key = tuple(zlm_pair[:-1])
            _mask_pair = ((atom_pairs_test[..., 0] == atom_pair[0]) *
                          (atom_pairs_test[..., 1] == atom_pair[1])).unsqueeze(-1).repeat(
                1, 1, 1, dftb1_scc_test.geometry.distances_pe.shape[-1]).flatten()

            # To make sure when build NN model for orbitals, different m will be combined.
            # Such as pp0 and pp1 will be combined for the following SKT
            if lm_pair[-1] == 0:
                pred_h.update({key: torch.zeros(X[_mask_pair].shape[0], min_l + 1)})
                if self.train_s:
                    pred_s.update({key: torch.zeros(X[_mask_pair].shape[0], min_l + 1)})

            if self.ml_method == 'nn_scale':
                if self.train_distance_cutoff is None:
                    count = self.hs_label_dict[tuple(zlm_pair)]
                    tmp = self.h_models[count](X[_mask_pair]) + 1
                    pred_h[key][..., lm_pair[-1]] = tmp.squeeze()
                else:
                    count = self.hs_label_dict[tuple(zlm_pair)]
                    tmp = torch.ones(X[_mask_pair].shape[0], 1)
                    _dist = dftb1_band_test.geometry.distances_pe.flatten()
                    _mask_d = _dist[_mask_pair].lt(7.0) * _dist[_mask_pair].gt(1.0)
                    tmp[_mask_d] = self.h_models[count](X[_mask_pair][_mask_d]) + 1
                    pred_h[key][..., lm_pair[-1]] = tmp.squeeze()
            else:
                tmp = torch.zeros(X.shape[0], 1)
                count = self.hs_label_dict[tuple(zlm_pair)]
                tmp[_mask] = self.h_models[count](X[_mask])
                if self.decay is not None:
                    tmp = tmp * getattr(self, self.decay)(self.decay_params[tuple(lm_pair)])

        if self.train_onsite:
            _numbers = dftb1_band_test.geometry.atomic_numbers
            _numbers = _numbers[_numbers.ne(0)]
            for iatom in unique_atoms.tolist():
                _mask_uan = _numbers == iatom
                _onsite_feature = onsite_feature[_mask_uan]

                for il in range(max(self.shell_dict[iatom]) + 1):
                    count = self.onsite_label_dict[(iatom, il)]
                    if self.ml_method == 'nn_scale':
                        pred_h_on.update({(iatom, il): self.h_onsite_dict[iatom, il] *
                                                       (self.h_onsite_models[count](_onsite_feature) + 1)})
                    elif self.ml_method == 'nn_hs':
                        pred_h_on.update({il: self.h_onsite_models[il](_onsite_feature)})

        # Check and constrain the Hamiltonian and overlap
        # self._constrain(pred_h, pred_h_on)

        # < k_i | H_ij | k_j >
        ham = self.to_ham(pred_h, dftb1_band_test, "H",
                          pred_h_on, h_mat_dict=h_mat_dict)
        ham_scc = self.to_ham(pred_h, dftb1_scc_test, "H", pred_h_on,
                              h_mat_dict=h_mat_dict,)

        if self.train_s:
            over, s_merge_pred_dict = self.to_ham(pred_s, dftb1_band_test, "S", h_mat_dict, n_kpoints=n_kpoints, phase=phase)
            over_scc, _ = self.to_ham(pred_s, dftb1_scc_test, "S", h_mat_dict, n_kpoints=n_kpoints_scc, phase=phase_scc)
        elif self.orthogonal_dftb:
            over_scc, over = None, None
        else:
            over_scc, over = dftb2_scc_test.over, dftb1_band_test.over

        dftb2_scc_test(hamiltonian=ham_scc, overlap=over_scc)
        charge = dftb2_scc_test.charge
        dftb1_band_test(charge=charge, hamiltonian=ham, overlap=over)
        pred_v0, pred_c0, delta_v, delta_c = self._alignment(
            dftb1_band_test, ref, dftb1_scc_test.E_fermi, mask_v, alignment=self.alignment)

        # Plot
        eigenvalue = dftb1_band_test.eigenvalue.detach().clone() - \
                     dftb1_scc_test.E_fermi.unsqueeze(-1).unsqueeze(-1)
        self._plot('predict', ref, ref_vband, ref_cband, dftb1_band_test, eigenvalue,
                   torch.cat([pred_v0.detach(), pred_c0.detach()], -1))
        self._plot('init', ref, ref_vband, ref_cband, dftb1_band_test, eigenvalue0,
                   torch.cat([pred_v_init.detach(), pred_c_init.detach()], -1))
        self._plot_hs_dict(X, h_merge_pred_dict, pred_h_on, pred_s, _mask)

        return dftb1_band_test

    def _alignment(self, dftb, ref, fermi, alignment='vbm'):
        eigenvalue = dftb.eigenvalue - fermi.unsqueeze(-1).unsqueeze(-1)

        # withgap = torch.tensor([False if gap == 0 or gap is None else True for gap in ref['gap']])
        vbm_alignment = torch.tensor([align for align in ref['vbm_alignment']])

        n_vbm = torch.round(dftb.qzero.sum(-1) / 2 - 1).long()[[vbm_alignment]]

        if vbm_alignment.any():
            eigenvalue[vbm_alignment] = eigenvalue[vbm_alignment] - torch.max(
                eigenvalue[vbm_alignment, :, n_vbm], -1)[0].unsqueeze(-1).unsqueeze(-1).detach()
        # elif withgap.any() and alignment == 'cbm':
        #     eigenvalue[withgap] = eigenvalue[withgap] - torch.min(
        #         eigenvalue[withgap, :, n_vbm + 1], -1)[0].unsqueeze(-1).unsqueeze(-1).detach()

        pred_v0, pred_c0 = Dataset.get_occ_eigenvalue(
            eigenvalue, self.params['n_band0'], ref['mask_v'],
            self.params['n_valence'], ref['n_conduction'], dftb.nelectron)
        pred_v1, pred_c1 = Dataset.get_occ_eigenvalue(
            eigenvalue, self.params['n_band1'], ref['mask_v'],
            self.params['n_valence'], ref['n_conduction'], dftb.nelectron)

        delta_v = pred_v1 - pred_v0
        delta_c = pred_c1 - pred_c0
        return pred_v0, pred_c0, delta_v, delta_c

    def get_reference(self, ref, dftb, vc_split=True):
        """Generate reference data for training and testing."""
        # train_e_low: train the bands above defined lowest energy
        occ = (dftb.nelectron / 2).long()
        ref_vband = ref['vband_tot'][:, self.params['n_band0']]
        ref_vband1 = ref['vband_tot'][:, self.params['n_band1']]
        ref_vband = [iv[..., nv - iocc: nv] for iv, nv, iocc in zip(
            ref_vband, ref['n_vband'], occ)]
        ref_vband1 = [iv[..., nv - iocc: nv] for iv, nv, iocc in zip(
            ref_vband1, ref['n_vband'], occ)]

        # Select the valence band above the lowest energy
        mask_v = [(ii.sum(0) + jj.sum(0)) / (ii.shape[0] + jj.shape[0]) > self.params['train_e_low']
                  for ii, jj in zip(ref_vband, ref_vband1)]
        ref_vband = pack([band[..., mask] for band, mask in zip(ref_vband, mask_v)])
        ref_vband1 = pack([band[..., mask] for band, mask in zip(ref_vband1, mask_v)])

        ref_cband = ref['cband_tot'][:, self.params['n_band0']]
        ref_cband1 = ref['cband_tot'][:, self.params['n_band1']]

        # Return conduction band according to energy
        # mask_c = [(ii.sum(0) + jj.sum(0)) / (ii.shape[0] + jj.shape[0]) < self.params['train_e_high']
        #           for ii, jj in zip(ref_cband, ref_cband1)]
        # ref['n_conduction'] = [ic.sum() for ic in mask_c]
        # ref_cband = pack([band[..., mask] for band, mask in zip(ref_cband, mask_c)])
        # ref_cband1 = pack([band[..., mask] for band, mask in zip(ref_cband1, mask_c)])

        # Return conduction band according to defined bands
        uan = torch.unique(dftb.geometry.atomic_numbers)
        n_conduction = torch.zeros(dftb.geometry.atomic_numbers.shape[0], dtype=torch.int16)
        for ii in uan[uan.ne(0)]:
            mask = dftb.geometry.atomic_numbers == ii
            n_conduction = n_conduction + self.params['n_conduction'][ii.tolist()] * mask.sum(-1)
        ref['n_conduction'] = n_conduction.tolist()
        ref_cband = pack([band[..., :nc] for band, nc in zip(ref_cband, n_conduction.tolist())])
        ref_cband1 = pack([band[..., :nc] for band, nc in zip(ref_cband1, n_conduction.tolist())])

        # ref_cband = ref['cband_tot'][:, self.params['n_band0'], :self.params['n_conduction']]
        # ref_cband1 = ref['cband_tot'][:, self.params['n_band1'], :self.params['n_conduction']]

        if vc_split:
            return ref_vband, ref_cband, ref_vband1 - ref_vband, ref_cband1 - ref_cband, mask_v
        else:
            return torch.cat([ref_vband, ref_cband], -1), \
                    torch.cat([ref_vband1, ref_cband1], -1) -\
                    torch.cat([ref_vband, ref_cband], -1)

    def _constrain(self, pred_h, pred_h_on):
        # Make constrains for each orbital based on DFTB H & S
        error_hs = {
            (0, 0): 0.8,
            (0, 1): 0.8,
            (0, 2): 0.8,
            (1, 1): 1.0,
            (1, 2): 1.0,
            (2, 2): 1.0}

        for il, lm_pair in enumerate(self.zlm_pairs.tolist()):
            # the 3rd parameter in the lm_pair equals to min(l), in this
            # case, if min(l) >= 1, like pp0, pp1, here will be gathered
            key = tuple(lm_pair[:2])
            if min(key) == lm_pair[-1]:
                error = self.h_mat_dict[key].flatten(0, -2) - pred_h[key]
                clamp_min = error.lt(-self.exp_decay() * error_hs[key]).any()
                clamp_max = error.gt(self.exp_decay() * error_hs[key]).any()
                if clamp_min:
                    self.logger.info(f'{key} prediction exceed the min value')
                    pred_h[key] = torch.clamp(
                        pred_h[key],
                        min=self.h_mat_dict[key].flatten(0, -2) - self.exp_decay() * error_hs[key]
                    )
                if clamp_max:
                    self.logger.info(f'{key} prediction exceed the max value')
                    pred_h[key] = torch.clamp(
                        pred_h[key],
                        max=self.h_mat_dict[key].flatten(0, -2) + self.exp_decay() * error_hs[key],
                    )

        if self.train_onsite:
            error_on = {0: 0.0075, 1: 0.015, 2: 0.05}
            onsite_key = torch.max(self.zlm_pairs).tolist()
            for il in range(onsite_key + 1):
                ih_on = pred_h_on[il]
                error = ih_on - self.h_onsite_dict[il]
                clamp_min = error.lt(-error_on[il]).any()
                clamp_max = error.gt(error_on[il]).any()
                if clamp_min:
                    self.logger.info(f'onsite {str(il)} prediction exceed the min value')
                    ih_on = torch.clamp(ih_on, min=self.h_onsite_dict[il] - error_on[il])
                if clamp_max:
                    self.logger.info(f'onsite {str(il)} prediction exceed the max value')
                    ih_on = torch.clamp(ih_on, max=self.h_onsite_dict[il] + error_on[il])
                pred_h_on.update({il: ih_on})

    def _plot(self, label, ref, dftb, eigenvalue, pred_band0):
        for ii, (ir, idftb) in enumerate(zip(ref["band_tot"], eigenvalue)):
            plt.plot(torch.arange(len(ir)), ir, color="r")
            plt.plot(torch.arange(len(idftb)), idftb, color="g", linestyle="--")
            plt.plot([0], [-10], color="r", label="ref" + str(ii))
            plt.plot([0], [-10], color="g", label=label)
            plt.title(torch.unique(dftb.geometry.atomic_numbers[ii]))
            plt.ylim(-20, 20)
            plt.legend()
            plt.savefig(str(label) + '_' + ref['labels'][ii])
            plt.close()
        for ii, (ir, idftb) in enumerate(zip(torch.cat([ref['ref_vband'], ref['ref_cband']], -1), pred_band0)):
            plt.plot(torch.arange(len(ir)), ir, color="r")
            plt.plot(torch.arange(len(idftb)), idftb, color="g", linestyle="--")
            plt.plot([0], [-10], color="r", label="ref" + str(ii))
            plt.plot([0], [-10], color="g", label=label)
            plt.title(torch.unique(dftb.geometry.atomic_numbers[ii]))
            plt.ylim(-20, 20)
            plt.legend()
            plt.savefig(str(label) + '_' + ref['labels'][ii])
            plt.close()

        if self.train_s:
            pass

    def _plot_hs_dict(self, label, ref, X, h_mat_dict, h_onsite_dict,
                      pred_h={}, pred_h_on={}, pred_s={}, mask=None):
        for key in h_mat_dict.keys():
            plt.plot(h_mat_dict[key].flatten(0, -2)[mask],
                     pred_h[key].detach().flatten(0, -2)[mask], ".", label="H" + str(key))
            plt.plot([-0.5, 0.5], [-0.5, 0.5], "k")
        plt.xlabel("DFTB H")
        plt.ylabel("predicted H")
        plt.legend()
        plt.savefig(str(label) + '_' + str(ref['labels']))
        for key in h_mat_dict.keys():
            plt.plot(h_mat_dict[key].flatten(0, -2)[mask],
                     h_mat_dict[key].flatten(0, -2)[mask] - pred_h[key].detach().flatten(0, -2)[mask], ".", label="H" + str(key))
            plt.plot([-0.2, 0.2], [0., 0.], "k")
        plt.xlabel("DFTB H")
        plt.ylabel("predicted H")
        plt.legend()
        plt.savefig(str(label) + '_' + str(ref['labels']))
        plt.close()

        if self.train_onsite:
            for key in pred_h_on.keys():
                _pred = pred_h_on[key].detach()
                plt.plot(h_onsite_dict[key].repeat(len(_pred), 1), _pred, "x", label="onsite" + str(key))
                plt.plot([-0.8, 0.4], [-0.8, 0.4], "k")
            plt.title("DFTB onsite vs predicted onsite")
            plt.legend()
            plt.savefig(str(label) + '_' + str(ref['labels']))
            plt.close()
        for key in h_mat_dict.keys():
            plt.plot(X.sum(-1), pred_h[key].detach().flatten(0, -2), ".", label="pred H" + str(key))
            plt.plot(X.sum(-1), h_mat_dict[key].flatten(0, -2), "x", label="DFTB H" + str(key))
        plt.xlabel("X sum")
        plt.ylabel("predicted H")
        plt.legend()
        plt.savefig(str(label) + '_' + str(ref['labels']))
        plt.close()

        if self.train_s:
            for key in s_mat_dict.keys():
                plt.plot(s_mat_dict[key].flatten(0, -2),
                         pred_s[key].detach().flatten(0, -2), ".", label="S" + str(key))
                plt.plot([-0.5, 0.5], [-0.5, 0.5], "k")
            plt.xlabel("DFTB s")
            plt.ylabel("predicted s")
            plt.legend()
            plt.show()
            for key in s_mat_dict.keys():
                plt.plot(X[..., 0], pred_s[key].detach(), ".", label="S" + str(key))
            plt.xlabel("X0")
            plt.ylabel("predicted S")
            plt.legend()
            plt.show()

    def _plot_eigenvalue(self, label, pred_v0, delta_v, pred_c0, delta_c, ref_dict):
        plt.plot(ref_dict['ref_vband'].flatten(), pred_v0.flatten(), 'rx')
        loss = sum(abs(ref_dict['ref_vband'].flatten() - pred_v0.flatten())) / len(pred_v0.flatten())
        plt.plot([-12, 2], [-12, 2], 'k')
        plt.title('occupied eigenvalues, loss=' + str(loss))
        plt.savefig(str(label) + '_' + 'vband' + str(ref_dict['labels']))
        plt.close()
        plt.plot(ref_dict['ref_cband'].flatten(), pred_c0.flatten(), 'rx')
        loss = sum(abs(ref_dict['ref_cband'].flatten() - pred_c0.flatten())) / len(pred_c0.flatten())
        plt.plot([-2, 12], [-2, 12], 'k')
        plt.title('unoccupied eigenvalues, loss=' + str(loss))
        plt.savefig(str(label) + '_' + 'cband' + str(ref_dict['labels']))
        plt.close()
        plt.plot(ref_dict['ref_delta_vband'].flatten(), delta_v.flatten(), 'rx')
        loss = sum(abs(ref_dict['ref_delta_vband'].flatten() - delta_v.flatten())) / len(delta_v.flatten())
        plt.title('occupied delta eigenvalues, loss=' + str(loss))
        plt.plot([-4, 4], [-4, 4], 'k')
        plt.savefig(str(label) + '_' + 'delta_vband' + str(ref_dict['labels']))
        plt.close()
        plt.plot(ref_dict['ref_delta_cband'].flatten(), delta_c.flatten(), 'rx')
        loss = sum(abs(ref_dict['ref_delta_cband'].flatten() - delta_c.flatten())) / len(delta_c.flatten())
        plt.title('unoccupied delta eigenvalues, loss=' + str(loss))
        plt.plot([-4, 4], [-4, 4], 'k')
        plt.savefig(str(label) + '_' + 'delta_cband' + str(ref_dict['labels']))
        plt.close()

    def exp_decay(self, params=None):
        r_cut = self.dftb1_band.geometry.cutoff
        r_min = 1.0
        distances = self.dftb1_band.geometry.distances_pe if \
            self.is_periodic and self.pbc else self.dftb1_band.geometry.distances

        fc = torch.exp((2 - distances.flatten()) * 0.1).unsqueeze(-1)

        fc[distances.flatten() > r_cut] = 0.0
        fc[distances.flatten() < r_min] = 0.0
        return fc

    def cos_decay(self, params=None):
        r_cut = self.dftb1_band.geometry.cutoff
        r_min = 1.0
        distances = self.dftb1_band.geometry.distances_pe if \
            self.is_periodic and self.pbc else self.dftb1_band.geometry.distances

        fc = 0.5 * (torch.cos(np.pi * distances.flatten()
                              / r_cut) + 1.0).unsqueeze(-1)

        fc[distances.flatten() > r_cut] = 0.0
        fc[distances.flatten() < r_min] = 0.0
        return fc

    def cosexp_decay(self, params):
        cut1, cut2, a1, a2 = params
        r_cut = self.dftb1_band.geometry.cutoff
        r_min = 1.0
        distances = self.dftb1_band.geometry.distances_pe.flatten() if \
            self.is_periodic and self.pbc else self.dftb1_band.geometry.distances.flatten()

        _dist = distances.clone()
        # _dist[_dist.lt(cut1)] = 0
        # _dist[_dist.gt(cut2)] = cut2
        fc = 0.5 * (torch.cos(np.pi * distances / r_cut) + 1.0).unsqueeze(-1)
        fc = fc * torch.exp(-_dist * 0.12).unsqueeze(-1)

        fc[distances.flatten() > r_cut] = 0.0
        fc[distances.flatten() < r_min] = 0.0

        return fc

    def coscos_decay(self, params):
        cut11, cut12, a1, a2, extend = params
        r_cut = self.dftb1_band.geometry.cutoff
        r_min = 1.0
        distances = self.dftb1_band.geometry.distances_pe.flatten() if \
            self.is_periodic and self.pbc else self.dftb1_band.geometry.distances.flatten()

        fc = 0.5 * (torch.cos(np.pi * distances.flatten()
                              / (r_cut + extend)) + 1.0).unsqueeze(-1)
        _dist = distances.clone()
        # _dist[_dist.lt(cut11)] = 0
        # _dist[_dist.gt(cut12)] = cut12
        # fc = fc * 0.5 * (torch.cos(np.pi * (_dist - cut11) / cut12) + 1.0).unsqueeze(-1)

        _dist[_dist.gt(cut11)] = cut11
        fc = fc * (torch.cos(np.pi * _dist / cut12) + 1.0).unsqueeze(-1)

        fc[distances.flatten() > r_cut] = 0.0
        fc[distances.flatten() < r_min] = 0.0

        return fc

    @property
    def is_periodic(self):
        return self.dftb1_band.geometry.is_periodic

    def set_optim(self):
        if self.train_h:
            assert len(list(self.h_models.parameters())) == len(self.h_lr)
            train_params = [
                {'params': params, 'lr': lr} for params, lr in zip(list(self.h_models.parameters()), self.h_lr)]

        if self.train_s:
            assert len(list(self.s_models.parameters())) == len(self.s_lr)
            train_params = train_params + [
                {'params': params, 'lr': lr} for params, lr in zip(self.s_models.parameters(), self.s_lr)]

        if self.train_onsite:
            assert len(list(self.h_onsite_models.parameters())) == len(self.onsite_lr)
            train_params = train_params + [
                {'params': params, 'lr': lr} for params, lr in zip(self.h_onsite_models.parameters(), self.onsite_lr)]

        return torch.optim.SGD(train_params, lr=self.lr['lr'])

    def s_onsite(self, n_feature_onsite=None):
        """Return ss0 model."""
        n_feature_onsite = self.n_feature_onsite if n_feature_onsite is None else n_feature_onsite
        return nn.ModuleList(
            [
                NNModel(
                    n_feature=n_feature_onsite,
                    out_size=1,
                    activation=self.activation,
                    nn_type=self.nn_type,
                )
            ]
        )

    def p_onsite(self, n_feature_onsite=None):
        """Return ss0 model."""
        n_feature_onsite = self.n_feature_onsite if n_feature_onsite is None else n_feature_onsite
        out_size = 3 if self.orbital_resolve else 1
        return nn.ModuleList(
            [
                NNModel(
                    n_feature=n_feature_onsite,
                    out_size=out_size,
                    activation=self.activation,
                    nn_type=self.nn_type,
                )
            ]
        )

    def d_onsite(self, n_feature_onsite=None):
        """Return ss0 model."""
        n_feature_onsite = self.n_feature_onsite if n_feature_onsite is None else n_feature_onsite
        out_size = 5 if self.orbital_resolve else 1
        return nn.ModuleList(
            [
                NNModel(
                    n_feature=n_feature_onsite,
                    out_size=out_size,
                    activation=self.activation,
                    nn_type=self.nn_type,
                )
            ]
        )

    def ss_model(self):
        """Return ss0 model."""
        return nn.ModuleList(
            [
                NNModel(
                    n_feature=self.n_feature_hs,
                    out_size=self.out_size,
                    activation=self.activation,
                    nn_type=self.nn_type,
                )
            ]
        )

    def sp_model(self):
        """Return sp0 model."""
        return nn.ModuleList(
            [
                NNModel(
                    n_feature=self.n_feature_hs,
                    out_size=self.out_size,
                    activation=self.activation,
                    nn_type=self.nn_type,
                )
            ]
        )

    def sd_model(self):
        """Return sd0 model."""
        return nn.ModuleList(
            [
                NNModel(
                    n_feature=self.n_feature_hs,
                    out_size=self.out_size,
                    activation=self.activation,
                    nn_type=self.nn_type,
                )
            ]
        )

    def pp_model(self):
        """Return pp0, pp1 models."""
        return nn.ModuleList(
            [
                NNModel(
                    n_feature=self.n_feature_hs,
                    out_size=self.out_size,
                    activation=self.activation,
                    nn_type=self.nn_type,
                ),
                NNModel(
                    n_feature=self.n_feature_hs,
                    out_size=self.out_size,
                    activation=self.activation,
                    nn_type=self.nn_type,
                ),
            ]
        )

    def pd_model(self):
        """Return pd0, pd1 models."""
        return nn.ModuleList(
            [
                NNModel(
                    n_feature=self.n_feature_hs,
                    out_size=self.out_size,
                    activation=self.activation,
                    nn_type=self.nn_type,
                ),
                NNModel(
                    n_feature=self.n_feature_hs,
                    out_size=self.out_size,
                    activation=self.activation,
                    nn_type=self.nn_type,
                ),
            ]
        )

    def dd_model(self):
        """Return dd0, dd1, dd2 models."""
        return nn.ModuleList(
            [
                NNModel(
                    n_feature=self.n_feature_hs,
                    out_size=self.out_size,
                    activation=self.activation,
                    nn_type=self.nn_type,
                ),
                NNModel(
                    n_feature=self.n_feature_hs,
                    out_size=self.out_size,
                    activation=self.activation,
                    nn_type=self.nn_type,
                ),
                NNModel(
                    n_feature=self.n_feature_hs,
                    out_size=self.out_size,
                    activation=self.activation,
                    nn_type=self.nn_type,
                ),
            ]
        )


class Test(nn.Module):
    def __init__(
        self, n_feature_hs: int = 4, out_size: int = 1, activation="ReLU", nn_type="linear"
    ):
        super(Test, self).__init__()
        self.n_feature_hs = n_feature_hs
        self.out_size = out_size
        self.n_orbs = torch.range(1, 5)

        # Build NN model list for different orbitals
        self.models = nn.ModuleList(
            [
                NNModel(
                    n_feature=self.n_feature_hs,
                    out_size=self.out_size,
                    activation=activation,
                    nn_type=nn_type,
                )
            ]
            * len(self.n_orbs)
        )
        self.loss_fn = nn.L1Loss()

    def forward(self, X, ref=None):
        optimizer = torch.optim.SGD(self.models.parameters(), lr=5e-6)
        # self.model.train()
        self.ref = ref
        _loss = []

        # Compute prediction error
        for ii in range(5):
            if ii <= 30:
                optimizer = torch.optim.SGD(self.models.parameters(), lr=6e-5)
            elif ii <= 60:
                optimizer = torch.optim.SGD(self.models.parameters(), lr=4e-5)
            elif ii <= 90:
                optimizer = torch.optim.SGD(self.models.parameters(), lr=3e-5)
            elif ii <= 120:
                optimizer = torch.optim.SGD(self.models.parameters(), lr=2e-5)

            pred = {}
            for lm_pair, model in zip(self.n_orbs.tolist(), self.models):
                pred.update({int(lm_pair): model(X)})

            for ii, key in enumerate(pred.keys()):
                if ii == 0:
                    eigen = pred[key]
                else:
                    eigen = eigen + pred[key] ** ii + pred[key] * ii
            # # Consider the shift
            loss = self.loss_fn(eigen, ref)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _loss.append(loss.detach())


# if __name__ == '__main__':
#     test = Test(3, 1)
#     test(torch.randn(10, 3), ref=torch.randn(10, 1))
