"""Train code."""
import re
import time
from typing import Literal
import numpy as np
import torch
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
from ase.atoms import Atoms

from tbmalt import Geometry, SkfParamFeed
from tbmalt.common.maths import hellinger
from tbmalt.common.batch import pack
from tbmalt.physics.dftb.dftb import Dftb2, Dftb1
from tbmalt.ml.skfeeds import SkfFeed, VcrFeed, TvcrFeed
from tbmalt.structures.basis import Basis
from tbmalt.physics.dftb.slaterkoster import hs_matrix
from tbmalt.ml.feature import Dscribe
from tbmalt.ml.scikitlearn import SciKitLearn
from tbmalt.structures.periodic import Periodic
from tbmalt.io import Dataset
from tbmalt.data.units import _Bohr__AA

Tensor = torch.Tensor


class Optim:
    """Optimizer template for DFTB parameters training.

    Arguments:
        geometry: TBMaLT geometry object.
        reference: A dictionary for reference data.
        variables: A list of variables with gradients.
        params: Dictionary which stores ML parameters.
        tolerance: Accuracy for machine learning loss convergence.

    """

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
        plt.ylabel('loss')
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


class OptHsPe(Optim):

    def __init__(self,
                 elements,
                 parameter,
                 shell_dict,
                 skf_type: Literal['h5', 'skf'] = 'h5', **kwargs):
        kpoints = kwargs.get('kpoints', None)
        self.params = parameter
        self.lr = self.params['ml']['lr']
        self.shell_dict = shell_dict
        self.skf_type = skf_type
        self.alignment = kwargs.get('alignment', 'vbm')
        build_abcd_h = kwargs.get('build_abcd_h', True)
        build_abcd_s = kwargs.get('build_abcd_s', True)
        self.h_feed = SkfFeed.from_dir(
            parameter['dftb']['path_to_skf'], shell_dict, elements=elements,
            interpolation='Spline1d', integral_type='H', skf_type=skf_type,
            build_abcd=build_abcd_h)
        self.s_feed = SkfFeed.from_dir(
            parameter['dftb']['path_to_skf'], shell_dict, elements=elements,
            interpolation='Spline1d', integral_type='S', skf_type=skf_type,
            build_abcd=build_abcd_s)

        self.ml_variable = []
        if build_abcd_h:
            self.ml_variable.extend(self.h_feed.off_site_dict['variable'])
        if build_abcd_s:
            self.ml_variable.extend(self.s_feed.off_site_dict['variable'])

        self.skparams = SkfParamFeed.from_dir(
            parameter['dftb']['path_to_skf'], elements=elements, skf_type=skf_type)
        self.loss_fn = getattr(torch.nn, 'L1Loss')()
        self.optimizer = getattr(
            torch.optim, self.params['ml']['optimizer'])(self.ml_variable, lr=self.lr)
        self._loss = []

    def __call__(self, this_train_dict, ii, plot: bool =True, save: bool = True, **kwargs):
        """Train spline parameters with target properties."""
        # super().__call__()
        dftb1_band = this_train_dict['dftb1_band']
        dftb2_scc = this_train_dict['dftb2_scc']
        (this_train_dict['ref_vband'],
         this_train_dict['ref_cband'],
         this_train_dict['ref_delta_vband'],
         this_train_dict['ref_delta_cband'],
         this_train_dict['mask_v']
         ) = self.get_reference(this_train_dict, this_train_dict['dftb1_band'])

        @torch.no_grad()
        def no_grad_dftb2():
            dftb2_scc()
            return dftb2_scc, dftb2_scc._charge

        dftb2_scc, charge = no_grad_dftb2()

        periodic = dftb1_band.periodic
        ham = hs_matrix(periodic, dftb1_band.basis, self.h_feed)
        over = hs_matrix(periodic, dftb1_band.basis, self.s_feed)
        dftb1_band(charge=charge, hamiltonian=ham, overlap=over)
        pred_e, pred_v0, pred_c0, delta_v, delta_c = self._alignment(
            dftb1_band, this_train_dict, dftb2_scc.E_fermi, alignment=self.alignment)

        loss = 0
        loss = loss + self.loss_fn(pred_v0, this_train_dict['ref_vband'])
        loss = loss + self.loss_fn(delta_v, this_train_dict['ref_delta_vband'])
        loss = loss + self.loss_fn(pred_c0, this_train_dict['ref_cband'])
        loss = loss + self.loss_fn(delta_c, this_train_dict['ref_delta_cband'])

        self._loss.append(loss.detach())
        print(f" loss: {loss.detach().tolist()}")

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._plot(ii, this_train_dict, dftb1_band, dftb1_band.eigenvalue,
                   torch.cat([pred_v0.detach(), pred_c0.detach()], -1))
        self._plot_eigenvalue(ii, pred_v0.detach(), delta_v.detach(),
                              pred_c0.detach(), delta_c.detach(), this_train_dict)

        # return self.dftb

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

        # Return conduction band according to defined bands
        uan = torch.unique(dftb.geometry.atomic_numbers)
        n_conduction = torch.zeros(dftb.geometry.atomic_numbers.shape[0], dtype=torch.int16)
        for ii in uan[uan.ne(0)]:
            mask = dftb.geometry.atomic_numbers == ii
            n_conduction = n_conduction + self.params['n_conduction'][ii.tolist()] * mask.sum(-1)
        ref['n_conduction'] = n_conduction.tolist()
        ref_cband = pack([band[..., :nc] for band, nc in zip(ref_cband, n_conduction.tolist())])
        ref_cband1 = pack([band[..., :nc] for band, nc in zip(ref_cband1, n_conduction.tolist())])

        if vc_split:
            return ref_vband, ref_cband, ref_vband1 - ref_vband, ref_cband1 - ref_cband, mask_v
        else:
            return torch.cat([ref_vband, ref_cband], -1), \
                    torch.cat([ref_vband1, ref_cband1], -1) -\
                    torch.cat([ref_vband, ref_cband], -1)


class Scale(Optim):

    def __init__(self,
                 elements,
                 parameter,
                 shell_dict,
                 train_dict,
                 ml_variable,
                 train_onsite,  # local, global
                 loss='L1Loss',
                 orbital_resolved=False,
                 scale_ham=True,
                 train_1der: bool = True,
                 skf_type: Literal['h5', 'skf'] = 'h5', **kwargs):
        """
        Use scaling parameters to tune Hamiltonian in DFTB.

        Arguments:
            elements:
            parameter:
            shell_dict:

        """
        self.params = parameter
        self.lr = self.params['ml']['lr']
        self.shell_dict = shell_dict
        self.train_dict = train_dict
        self.orbital_resolved = orbital_resolved
        self.scale_ham = scale_ham
        self.skf_type = skf_type
        self.train_1der = train_1der
        self.train_onsite = train_onsite

        self.alignment = kwargs.get('alignment', 'vbm')
        self.tolerance = kwargs.get('tolerance', 1E-4)
        self.h_feed = SkfFeed.from_dir(
            parameter['dftb']['path_to_skf'], shell_dict, elements=elements,
            interpolation='Spline1d', integral_type='H', skf_type=skf_type)
        self.s_feed = SkfFeed.from_dir(
            parameter['dftb']['path_to_skf'], shell_dict, elements=elements,
            interpolation='Spline1d', integral_type='S', skf_type=skf_type)

        self.ml_variable = ml_variable

        self.skparams = SkfParamFeed.from_dir(
            parameter['dftb']['path_to_skf'], elements=elements, skf_type=skf_type)
        self.loss_fn = getattr(torch.nn, loss)()
        self.optimizer = getattr(
            torch.optim, self.params['ml']['optimizer'])(self.ml_variable, lr=self.lr)
        self.lr_scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.1, patience=10, verbose=True)

        self._loss = []

    def __call__(self, this_train_dict, ii, plot: bool = True,
                 save: bool = True, save_step: int = 1, **kwargs):
        """Train spline parameters with target properties."""
        if 'reach_convergence' in this_train_dict.keys():
            first_loop = False
            if this_train_dict['reach_convergence'].all():
                return
        else:
            this_train_dict['loss'] = [0]
            first_loop = True

        dftb1_band = this_train_dict['dftb1_band']
        dftb2_scc = this_train_dict['dftb2_scc']
        (this_train_dict['ref_vband'],
         this_train_dict['ref_cband'],
         this_train_dict['ref_delta_vband'],
         this_train_dict['ref_delta_cband'],
         this_train_dict['mask_v']
         ) = self.get_reference(this_train_dict, this_train_dict['dftb1_band'])

        @torch.no_grad()
        def no_grad_dftb2():
            """SCC-DFTB without gradient to get Mulliken charges."""
            scale_dict = this_train_dict if self.scale_ham else None
            ham = hs_matrix(dftb2_scc.periodic, dftb2_scc.basis, self.h_feed,
                            scale_dict=scale_dict,
                            ml_onsite=this_train_dict['ml_onsite'],
                            orbital_resolved=self.orbital_resolved,
                            train_onsite=self.train_onsite
                            )
            over = hs_matrix(dftb2_scc.periodic, dftb2_scc.basis, self.s_feed)
            dftb2_scc(hamiltonian=ham, overlap=over)
            return dftb2_scc, dftb2_scc._charge

        dftb2_scc, charge = no_grad_dftb2()

        periodic = dftb1_band.periodic
        scale_dict = this_train_dict if self.scale_ham else None
        ham = hs_matrix(periodic, dftb1_band.basis, self.h_feed,
                        scale_dict=scale_dict,
                        ml_onsite=this_train_dict['ml_onsite'],
                        orbital_resolved=self.orbital_resolved,
                        train_onsite=self.train_onsite
                        )
        over = hs_matrix(periodic, dftb1_band.basis, self.s_feed)
        dftb1_band(charge=charge, hamiltonian=ham, overlap=over)
        pred_e, pred_v0, pred_c0, delta_v, delta_c = self._alignment(
            dftb1_band, this_train_dict, dftb2_scc.E_fermi, alignment=self.alignment)

        lv0 = self.loss_fn(pred_v0, this_train_dict['ref_vband'])
        lc0 = self.loss_fn(pred_c0, this_train_dict['ref_cband'])
        # lv1 = self.loss_fn(delta_v, this_train_dict['ref_delta_vband'])
        # lc1 = self.loss_fn(delta_c, this_train_dict['ref_delta_cband'])
        # loss = lv0 * lv0.clone().detach() / min_l + lc0 * lc0.clone().detach() / min_l
        loss = lv0 + lc0

        # calculate average error of each geometry
        loss_v0 = torch.abs(pred_v0.clone().detach() - this_train_dict['ref_vband']).flatten(start_dim=1)
        loss_c0 = torch.abs(pred_c0.clone().detach() - this_train_dict['ref_cband']).flatten(start_dim=1)

        loss_v1 = torch.abs(delta_v.clone().detach() - this_train_dict['ref_delta_vband']).flatten(start_dim=1)
        loss_c1 = torch.abs(delta_c.clone().detach() - this_train_dict['ref_delta_cband']).flatten(start_dim=1)
        loss_mae0 = torch.cat([loss_v0, loss_c0], dim=1)
        loss_mae1 = torch.cat([loss_v1, loss_c1], dim=1)

        if 'loss_mae0' in this_train_dict.keys():
            this_train_dict['loss_mae0'].append(loss_mae0.detach())
            this_train_dict['loss_mae1'].append(loss_mae1.detach())
            this_train_dict['delta_loss'].append(
                torch.abs(this_train_dict['loss_mae0'][-2] - loss_mae0) +
                torch.abs(this_train_dict['loss_mae1'][-2] - loss_mae1))

        else:
            this_train_dict['loss_mae0'] = [loss_mae0.detach()]
            this_train_dict['loss_mae1'] = [loss_mae1.detach()]
            this_train_dict['delta_loss'] = [torch.abs(loss_mae0.detach() + loss_mae1.detach())]

        self._loss.append(loss.detach())
        this_train_dict['loss'].append(loss.detach())
        print(f"step: {ii}, loss: {loss.detach().tolist()}",
              f"average loss0: {torch.mean(loss_mae0)}",
              f"average loss1: {torch.mean(loss_mae1)}",
              f"delta loss: {torch.mean(this_train_dict['delta_loss'][-1])}")

        this_train_dict['reach_convergence'] = torch.mean(this_train_dict['delta_loss'][-1], 1).lt(self.tolerance)
        if this_train_dict['reach_convergence'].all():
            print(f"reach convergence, delta loss: {this_train_dict['delta_loss'][-1]}")

        if ii % save_step == 0:
            with open('onsite.dat', 'a') as f:
                    np.savetxt(f, [ii])
                    f.write("\n")
                    np.savetxt(f, this_train_dict['ml_onsite'].detach().tolist())
                    f.write("\n")
                    f.write("\n")

            with open('scale.dat', 'a') as f:
                for i in range(3):
                    for j in range(i, 3):
                        if (i, j) in this_train_dict.keys():
                            np.savetxt(f, [ii])
                            f.write("\n")
                            np.savetxt(f, [[i, j]])
                            f.write("\n")
                            np.savetxt(f, [this_train_dict[(i, j)].detach().flatten().tolist()])
                            f.write("\n")
                            f.write("\n")

        # Backpropagation
        if this_train_dict['loss'][-2] - this_train_dict['loss'][-1] < 0 and not first_loop:
            for var in self.ml_variable:
                var.update({'lr': var['lr'] * 0.01})
            print('stop improving')

            self.optimizer = getattr(
            torch.optim, self.params['ml']['optimizer'])(self.ml_variable, lr=self.lr)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # undo the ratio revision
            for var in self.ml_variable:
                var.update({'lr': var['lr'] * 98})
            self.optimizer = getattr(
                torch.optim, self.params['ml']['optimizer'])(self.ml_variable, lr=self.lr)
        else:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if ii % save_step == 0:
            self._plot(ii, this_train_dict, dftb1_band, pred_e.detach(),
                       torch.cat([pred_v0.detach(), pred_c0.detach()], -1))
            self._plot_eigenvalue(ii, pred_v0.detach(), delta_v.detach(),
                                  pred_c0.detach(), delta_c.detach(), this_train_dict)

    @classmethod
    def pred(cls, this_train_dict, path_to_skf, params, shell_dict, elements, skf_type,
             alignment, train_onsite, orbital_resolved, scale_ham, plot: dict = None,
             skf_list: list = None, shell_dict_std=None, plot_band=True,
             plot_slope=False, plot_std=True, **kwargs):
            """Train spline parameters with target properties."""
            # plot_fermi is used to plot band near Fermi level
            plot_fermi = kwargs.get('plot_fermi', False)
            fermi_band = kwargs.get('fermi_band', [0])
            cls.params = params
            dftb1_band = this_train_dict['dftb1_band']
            dftb2_scc = this_train_dict['dftb2_scc']
            h_feed = SkfFeed.from_dir(
                path_to_skf, shell_dict, elements=elements,
                interpolation='Spline1d', integral_type='H', skf_type=skf_type)
            s_feed = SkfFeed.from_dir(
                path_to_skf, shell_dict, elements=elements,
                interpolation='Spline1d', integral_type='S', skf_type=skf_type)

            (this_train_dict['ref_vband'],
             this_train_dict['ref_cband'],
             this_train_dict['ref_delta_vband'],
             this_train_dict['ref_delta_cband'],
             this_train_dict['mask_v']
             ) = Scale.get_reference(cls, this_train_dict, this_train_dict['dftb1_band'])

            scale_dict = this_train_dict if scale_ham else None

            # SCC-DFTB for charges
            ham = hs_matrix(dftb2_scc.periodic, dftb2_scc.basis, h_feed,
                            scale_dict=scale_dict,
                            ml_onsite=this_train_dict['ml_onsite'],
                            orbital_resolved=orbital_resolved,
                            train_onsite=train_onsite
                            )
            over = hs_matrix(dftb2_scc.periodic, dftb2_scc.basis, s_feed)
            dftb2_scc(hamiltonian=ham, overlap=over)
            charge = dftb2_scc._charge

            # one SCC loop DFTB with charges
            periodic = dftb1_band.periodic
            ham = hs_matrix(periodic, dftb1_band.basis, h_feed,
                            scale_dict=scale_dict,
                            ml_onsite=this_train_dict['ml_onsite'],
                            orbital_resolved=orbital_resolved,
                            train_onsite=train_onsite
                            )
            over = hs_matrix(periodic, dftb1_band.basis, s_feed)
            dftb1_band(charge=charge, hamiltonian=ham, overlap=over)
            pred_e, pred_v0, pred_c0, delta_v, delta_c = Scale._alignment(cls,
                dftb1_band, this_train_dict, dftb2_scc.E_fermi, alignment=alignment)

            loss_v0 = torch.abs(pred_v0.clone().detach() - this_train_dict['ref_vband']).flatten(start_dim=1)
            loss_c0 = torch.abs(pred_c0.clone().detach() - this_train_dict['ref_cband']).flatten(start_dim=1)
            loss_v1 = torch.abs(delta_v.clone().detach() - this_train_dict['ref_delta_vband']).flatten(start_dim=1)
            loss_c1 = torch.abs(delta_c.clone().detach() - this_train_dict['ref_delta_cband']).flatten(start_dim=1)
            print("this_train_dict['ref_vband']", this_train_dict['ref_vband'].shape,
                  pred_v0.clone().detach()- this_train_dict['ref_vband'],
                  torch.mean(torch.abs(pred_v0.clone().detach() - this_train_dict['ref_vband']).flatten(start_dim=1)))
            loss_mae0 = torch.cat([loss_v0, loss_c0], dim=1)
            with open('ham.dat', 'a') as f:
                for i in ham[..., 0]:
                    np.savetxt(f, i.real)
                    f.write("\n")

            loss_mae1 = torch.cat([loss_v1, loss_c1], dim=1)

            # STD SCC-DFTB
            print('skf_list', skf_list)
            std_err, std_bands, std_vs, std_cs = [], [], [], []
            if skf_list is not None:
                for ii, skf in enumerate(skf_list):
                    dftb2 = Dftb2(
                        dftb2_scc.geometry, shell_dict=shell_dict_std[ii],
                        path_to_skf=skf, skf_type='skf', periodic=dftb2_scc.periodic  # dftb2_scc.kpoints
                    )
                    dftb2()
                    dftb1 = Dftb1(
                        dftb1_band.geometry, shell_dict=shell_dict_std[ii],
                        path_to_skf=skf, skf_type='skf', klines=this_train_dict['klines'])
                    dftb1(charge=dftb2.charge,)

                    pred_es, pred_v0s, pred_c0s, delta_vs, delta_cs = Scale._alignment(
                        cls, dftb1, this_train_dict, dftb2.E_fermi, alignment=alignment)
                    std_err.append({
                        'mae_v0': torch.abs(pred_v0s.clone().detach() - this_train_dict['ref_vband']).flatten(start_dim=1),
                        'mae_c0': torch.abs(pred_c0s.clone().detach() - this_train_dict['ref_cband']).flatten(start_dim=1)})
                    std_err[-1].update({'mae0': torch.cat([
                        std_err[-1]['mae_v0'], std_err[-1]['mae_c0']], dim=1)})
                    std_bands.append(pred_es)
                    std_vs.append(pred_v0s)
                    std_cs.append(pred_c0s)

                    print("std", skf, this_train_dict['ref_vband'].shape,
                          pred_v0s.clone().detach()- this_train_dict['ref_vband'],
                          torch.mean(torch.abs(pred_v0s.clone().detach() - this_train_dict['ref_vband']).flatten(start_dim=1)))

            for ii, (ir, idftb, istd) in enumerate(zip(this_train_dict["band_tot"], pred_e, pred_es)):
                if plot_slope + plot_band + plot_fermi == 2:
                    fig, axs = plt.subplots(1, 2, figsize=(9, 6), width_ratios=[2, 1])
                    axs0 = axs[0]
                    axs1 = axs[1]
                elif plot_band:
                    fig, axs = plt.subplots(1, 1, figsize=(6, 6))
                    axs0 = axs
                elif plot_slope:
                    fig, axs = plt.subplots(1, 1, figsize=(6, 6))
                    axs1 = axs
                if plot_band:
                    axs0.plot(torch.arange(len(ir)), ir, color="r")
                    #axs0.plot(torch.arange(len(idftb)), idftb, color="g", linestyle="--")
                    axs0.plot(torch.arange(len(idftb)), idftb, '.', markersize=15, color="tab:green", alpha=0.3)
                    # axs0.plot(torch.arange(len(istd)), istd, color="c")
                    # axs0.plot([0], [-20], "r", label='HSE')
                    # axs0.plot([0], [-20], "g--", label='DFTB-ML')

                    if plot_std:
                        axs0.plot(torch.arange(len(std_bands[0][ii])), std_bands[0][ii], "c,", alpha=0.4)
                        axs0.plot(torch.arange(len(std_bands[1][ii])), std_bands[1][ii], "b,", alpha=0.4)
                        axs0.plot([0], [-20], "c,", label='pbc', alpha=0.4)
                        axs0.plot([0], [-20], "b,", label='global', alpha=0.4)
                    axs0.set_xlim(0, min(len(idftb), len(ir)) - 1)

                    if plot is not None:
                        axs0.set_ylim(plot['min_e'], plot['max_e'])
                    else:
                        axs0.set_ylim(-15, 10)
                    axs0.set_ylabel(r'$E - E_{VBM}$ (eV)', fontsize='large')
                    axs0.set_xlabel('K-path')

                    k_list = []
                    geo = Atoms(dftb1_band.geometry.atomic_numbers.numpy()[ii],
                                dftb1_band.geometry.positions.numpy()[ii] * _Bohr__AA,
                                cell=dftb1_band.geometry.cell.numpy()[ii] * _Bohr__AA)
                    kpt = geo.cell.bandpath(npoints=10)
                    path = re.sub(r'[,]', '', kpt.path)
                    for i in range(len(path)):
                        if i + 1 < len(path) and path[i + 1].isdigit():
                            k_list.append(path[i] + path[i + 1])
                        elif not path[i].isdigit():
                            k_list.append(path[i])
                    xticks = np.arange(len(ir) + 1)[::10]
                    xticks[-1] -= 1
                    axs0.set_xticks(xticks)
                    axs0.set_xticklabels(k_list[: (len(ir) // 10 + 1)], fontsize='large')
                    axs0.legend(fontsize="large")

                if plot_slope:
                    axs1.plot(this_train_dict['ref_vband'].flatten(), pred_v0.flatten(), 'r.', alpha=0.4)
                    axs1.plot(this_train_dict['ref_cband'].flatten(), pred_c0.flatten(), 'r.', alpha=0.4)

                    axs1.plot(ir.flatten(), ir.flatten(), 'k')
                    axs1.plot([torch.min(pred_v0) - 1, torch.max(pred_c0) + 1],
                                [torch.min(pred_v0) - 1, torch.max(pred_c0) + 1], 'k')
                    axs1.plot([0], [-20], "r.", label='DFTB-ML')

                    if plot_std:
                        axs1.plot(this_train_dict['ref_vband'].flatten(),
                                    std_vs[0].flatten(), "c.", alpha=0.4)
                        axs1.plot(this_train_dict['ref_cband'].flatten(),
                                    std_cs[0].flatten(), "c.", alpha=0.4)
                        axs1.plot(this_train_dict['ref_vband'].flatten(),
                                    std_vs[1].flatten(), "b.", alpha=0.4)
                        axs1.plot(this_train_dict['ref_cband'].flatten(),
                                    std_cs[1].flatten(), "b.", alpha=0.4)
                        axs1.plot([0], [-20], "c.", label='pbc-0-3', alpha=0.4)
                        axs1.plot([0], [-20], "b.", label='global', alpha=0.4)

                    # axs[1].set_title('MAE: {:.2f}'.format(mae))
                    if plot is not None:
                        axs1.set_ylim(torch.min(pred_v0) - 1, torch.max(pred_c0) + 1)
                        axs1.set_xlim(torch.min(pred_v0) - 1, torch.max(pred_c0) + 1)
                    else:
                        axs1.set_xlim(-15, 15)
                        axs1.set_ylim(-15, 15)

                    axs1.legend(fontsize="large")
                    axs1.set_ylabel(r'$E^{DFTB-ML} - E_{VBM}^{DFTB-ML}$ (eV)', fontsize="large")
                    axs1.set_xlabel(r'$E^{DFT-HSE} - E_{VBM}^{DFT-HSE}$ (eV)', fontsize="large")
                    if plot_slope + plot_band == 2:
                        axs1.yaxis.set_label_position("right")

                if plot_fermi:
                    ind = np.arange(2)  # -> xx
                    width = 0.45
                    # error_s = torch.cat([std_err[-1]['mae_v0'], std_err[-1]['mae_c0']], dim=1).numpy()
                    error = np.mean(torch.cat([loss_v0, loss_c0], dim=1).numpy(), 1)
                    error_f = np.mean(torch.stack([loss_v0[:, -1], loss_c0[:, 0]]).numpy(), 0)
                    axs1.bar(ind + 3 * width,
                               (#np.mean(error_s),
                                np.mean(error), np.mean(error_f)), width, alpha=0.8)
                    axs1.errorbar(
                        ind + 3 * width, (#np.mean(error_s),
                                          np.mean(error), np.mean(error_f)),
                        yerr=[[#np.mean(error_s) - min(error_s),
                               np.mean(error) - min(error), np.mean(error_f) - min(error_f)],
                              [#max(error_s) - np.mean(error_s),
                               max(error) - np.mean(error), max(error_f) - np.mean(error_f)]],
                        fmt="k.", capsize=4)
                    axs1.set_ylabel("band structure MAEs (eV)")#, fontsize="large")
                    axs1.yaxis.tick_right()
                    axs1.set_xticks([1.4, 2.4], ["Total", "VBM+CBM"])#, fontsize="large")

                plt.savefig(str(ii) + '_' + this_train_dict['labels'][ii], dpi=300)
                plt.close()

            return loss_mae0, loss_mae1, std_err

    def _plot(self, label, ref, dftb, eigenvalue, pred_band0):
        for ii, (ir, idftb) in enumerate(zip(ref["band_tot"], eigenvalue)):
            plt.plot(torch.arange(len(ir)), ir, color="r")
            plt.plot(torch.arange(len(idftb)), idftb, color="g", linestyle="--")
            plt.plot([0], [-10], color="r", label="ref" + str(ii))
            plt.plot([0], [-10], color="g", label=label)
            plt.title(torch.unique(dftb.geometry.atomic_numbers[ii]))
            plt.xlim(0, min(len(idftb), len(ir)))
            plt.ylim(-15, 10)
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
            plt.savefig(str(label) + '_opt' + ref['labels'][ii])
            plt.close()

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

    def _alignment(self, dftb, ref, fermi, alignment='vbm'):
        eigenvalue = dftb.eigenvalue - fermi.unsqueeze(-1).unsqueeze(-1)

        # withgap = torch.tensor([False if gap == 0 or gap is None else True for gap in ref['gap']])
        vbm_alignment = torch.tensor([align for align in ref['vbm_alignment']])
        n_vbm = torch.round(dftb.qzero.sum(-1) / 2 - 1).long()[[vbm_alignment]]

        if vbm_alignment.any():
            eigenvalue[vbm_alignment] = eigenvalue[vbm_alignment] - torch.max(
                eigenvalue[vbm_alignment, :, n_vbm], -1)[0].unsqueeze(-1).unsqueeze(-1).detach()

        pred_v0, pred_c0 = Dataset.get_occ_eigenvalue(
            eigenvalue, self.params['n_band0'], ref['mask_v'],
            self.params['n_valence'], ref['n_conduction'], dftb.nelectron)
        pred_v1, pred_c1 = Dataset.get_occ_eigenvalue(
            eigenvalue, self.params['n_band1'], ref['mask_v'],
            self.params['n_valence'], ref['n_conduction'], dftb.nelectron)

        delta_v = pred_v1 - pred_v0
        delta_c = pred_c1 - pred_c0
        return eigenvalue, pred_v0, pred_c0, delta_v, delta_c

    def get_reference(self, ref, dftb, vc_split=True):
        """Generate reference data for training and testing."""
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


class VcrPe(Scale):

    def __init__(self,
                 elements,
                 parameter,
                 shell_dict,
                 train_dict,
                 ml_variable,
                 vcr,
                 # weight: list = [2.0, 2.0, 1.0, 1.0],
                 loss='L1Loss',
                 h_compr_feed: bool = True,
                 s_compr_feed: bool = True,
                 skf_type: Literal['h5', 'skf'] = 'h5', **kwargs):
        self.params = parameter
        self.lr = self.params['ml']['lr']
        self.shell_dict = shell_dict
        self.train_dict = train_dict
        # self.weight = weight
        self.skf_type = skf_type
        self.alignment = kwargs.get('alignment', 'vbm')
        self.h_compr_feed = h_compr_feed
        self.s_compr_feed = s_compr_feed

        self.h_feed = VcrFeed.from_dir(
            self.params['dftb']['path_to_skf'], self.shell_dict, vcr,
            skf_type='h5', elements=['C', 'Si'], integral_type='H',
            interpolation='BicubInterp')
        self.s_feed = VcrFeed.from_dir(
            self.params['dftb']['path_to_skf'], self.shell_dict, vcr,
            skf_type='h5', elements=['C', 'Si'], integral_type='S',
            interpolation='BicubInterp')

        self.ml_variable = ml_variable

        self.skparams = SkfParamFeed.from_dir(
            parameter['dftb']['path_to_skf'], elements=elements, skf_type=skf_type,
            repulsive=False)
        self.loss_fn = getattr(torch.nn, loss)()
        self.optimizer = getattr(
            torch.optim, self.params['ml']['optimizer'])(self.ml_variable, lr=self.lr)
        self._loss = []

    def __call__(self, this_train_dict, ii, *args, **kwargs):
        dftb1_band = this_train_dict['dftb1_band']
        dftb2_scc = this_train_dict['dftb2_scc']
        (this_train_dict['ref_vband'],
         this_train_dict['ref_cband'],
         this_train_dict['ref_delta_vband'],
         this_train_dict['ref_delta_cband'],
         this_train_dict['mask_v']
         ) = self.get_reference(this_train_dict, this_train_dict['dftb1_band'])
        compr = torch.zeros(*dftb1_band.geometry.distances.shape, 2)
        compr[..., 0] = this_train_dict['compr0'].unsqueeze(-1)
        compr[..., 1] = this_train_dict['compr0'].unsqueeze(-2)

        @torch.no_grad()
        def no_grad_dftb2():
            ham = hs_matrix(dftb2_scc.periodic, dftb2_scc.basis,
                            this_train_dict['h_feed'], multi_varible=compr)

            over = hs_matrix(dftb2_scc.periodic, dftb2_scc.basis,
                             this_train_dict['s_feed'], multi_varible=compr)
            dftb2_scc(hamiltonian=ham, overlap=over)
            return dftb2_scc, dftb2_scc._charge

        dftb2_scc, charge = no_grad_dftb2()

        ml_onsite = dftb1_band.h_feed.on_site_dict["ml_onsite"]
        # ham = hs_matrix(periodic, dftb1_band.basis, self.h_feed, scale_dict=this_train_dict,
        #                 ml_onsite=ml_onsite, train_onsite=self.params['ml']['train_onsite'])
        # over = hs_matrix(periodic, dftb1_band.basis, self.s_feed)

        ham = hs_matrix(dftb1_band.periodic, dftb1_band.basis,
                        this_train_dict['h_feed'], multi_varible=compr)

        over = hs_matrix(dftb1_band.periodic, dftb1_band.basis,
                         this_train_dict['s_feed'], multi_varible=compr)

        dftb1_band(charge=charge, hamiltonian=ham, overlap=over)
        pred_e, pred_v0, pred_c0, delta_v, delta_c = self._alignment(
            dftb1_band, this_train_dict, dftb2_scc.E_fermi, alignment=self.alignment)

        loss = 0
        lv0 = self.loss_fn(pred_v0, this_train_dict['ref_vband'])
        lv1 = self.loss_fn(delta_v, this_train_dict['ref_delta_vband'])
        lc0 = self.loss_fn(pred_c0, this_train_dict['ref_cband'])
        lc1 = self.loss_fn(delta_c, this_train_dict['ref_delta_cband'])
        min_l = torch.min(pack([lv0, lv1, lc0, lc1])).clone().detach()
        loss = lv0 * lv0.clone().detach() / min_l + lv1 * lv1.clone().detach() / min_l + \
               lc0 * lc0.clone().detach() / min_l + lc1 * lc1.clone().detach() / min_l

        self._loss.append(loss.detach())
        print(f"step: {ii}, loss: {loss.detach().tolist()}")
        with open('onsite.dat', 'a') as f:
            np.savetxt(f, [ii])
            f.write("\n")
            np.savetxt(f, [ml_onsite.detach().tolist()])
            f.write("\n")
            f.write("\n")

        with open('scale.dat', 'a') as f:
            for i in range(3):
                for j in range(i, 3):
                    if (i, j) in this_train_dict.keys():
                        np.savetxt(f, [ii])
                        f.write("\n")
                        np.savetxt(f, [[i, j]])
                        f.write("\n")
                        np.savetxt(f, [this_train_dict[(i, j)].detach().flatten().tolist()])
                        f.write("\n")
                        f.write("\n")

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._plot(ii, this_train_dict, dftb1_band, dftb1_band.eigenvalue.detach(),
                   torch.cat([pred_v0.detach(), pred_c0.detach()], -1))
        self._plot_eigenvalue(ii, pred_v0.detach(), delta_v.detach(),
                              pred_c0.detach(), delta_c.detach(), this_train_dict)


class OptHs(Optim):
    """Optimize integrals with spline interpolation."""

    def __init__(self, geometry: Geometry, reference, parameter, shell_dict,
                 skf_type: Literal['h5', 'skf'] = 'h5', **kwargs):
        kpoints = kwargs.get('kpoints', None)
        self.basis = Basis(geometry.atomic_numbers, shell_dict)
        self.shell_dict = shell_dict
        self.skf_type = skf_type
        build_abcd_h = kwargs.get('build_abcd_h', True)
        build_abcd_s = kwargs.get('build_abcd_s', True)
        self.h_feed = SkfFeed.from_dir(
            parameter['dftb']['path_to_skf'], shell_dict, geometry=geometry,
            interpolation='Spline1d', integral_type='H', skf_type=skf_type,
            build_abcd=build_abcd_h)
        self.s_feed = SkfFeed.from_dir(
            parameter['dftb']['path_to_skf'], shell_dict, geometry=geometry,
            interpolation='Spline1d', integral_type='S', skf_type=skf_type,
            build_abcd=build_abcd_s)

        self.ml_variable = []
        if build_abcd_h:
            self.ml_variable.extend(self.h_feed.off_site_dict['variable'])
        if build_abcd_s:
            self.ml_variable.extend(self.s_feed.off_site_dict['variable'])
        super().__init__(geometry, reference, self.ml_variable, parameter,
                         **kwargs)

        self.skparams = SkfParamFeed.from_dir(
            parameter['dftb']['path_to_skf'], self.geometry, skf_type=skf_type)
        if self.geometry.is_periodic:
            self.periodic = Periodic(self.geometry, self.geometry.cell,
                                     cutoff=self.skparams.cutoff, **kwargs)

    def __call__(self, plot: bool =True, save: bool = True, no_grad=False, **kwargs):
        """Train spline parameters with target properties."""
        super().__call__()
        self._loss = []
        self.grad_q = {}
        self.grad_h, self.grad_s = [], []
        for istep in range(self.steps):
            self._update_train(istep, no_grad)
            print('step: ', istep, 'loss: ', self.loss.detach())
            self._loss.append(self.loss.detach())

            break_tolerance = istep >= self.params['ml']['min_steps']
            if self.reach_convergence and break_tolerance:
                break

        if plot:
            super().__plot__(istep + 1, self.loss_list[1:])

        return self.dftb

    def _update_train(self, istep, no_grad):
        if self.geometry.is_periodic:
            ham = hs_matrix(self.periodic, self.basis, self.h_feed)
            over = hs_matrix(self.periodic, self.basis, self.s_feed)

        else:
            ham = hs_matrix(self.geometry, self.basis, self.h_feed)
            over = hs_matrix(self.geometry, self.basis, self.s_feed)
        if no_grad:
            with torch.no_grad():
                dftb = Dftb2(self.geometry, self.shell_dict,
                                  self.params['dftb']['path_to_skf'], from_skf=True)
                dftb(hamiltonian=ham, overlap=over)
            self.dftb = Dftb1(self.geometry, self.shell_dict,
                              self.params['dftb']['path_to_skf'], from_skf=True, maxiter=200)
            self.dftb(charge=dftb.charge, hamiltonian=ham, overlap=over)
            self.dftb.charge.register_hook(lambda grad: self.grad_q.update({istep: grad}))

            for var in self.h_feed.off_site_dict['variable']:
                var.register_hook(lambda grad: self.grad_h.append(grad))
            for var in self.s_feed.off_site_dict['variable']:
                var.register_hook(lambda grad: self.grad_s.append(grad))
        else:
            self.dftb = Dftb2(self.geometry, self.shell_dict,
                              self.params['dftb']['path_to_skf'], skf_type=self.skf_type)
            self.dftb(hamiltonian=ham, overlap=over)
            self.dftb.charge.register_hook(lambda grad: self.grad_q.update({istep: grad}))
            for var in self.h_feed.off_site_dict['variable']:
                var.register_hook(lambda grad: self.grad_h.append(grad))
            for var in self.s_feed.off_site_dict['variable']:
                var.register_hook(lambda grad: self.grad_s.append(grad))

        # self.dftb.charge.register_hook(lambda grad: print('grad', grad))
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
            self.compr0 = torch.ones(geometry.atomic_numbers.shape)
            self.compr = torch.zeros(*geometry.distances.shape, 2)
            init_dict = {6: torch.tensor([3.0]),
                         14: torch.tensor([5.0])}
            for ii, iu in enumerate(self.unique_atomic_numbers):
                mask = geometry.atomic_numbers == iu
                self.compr0[mask] = init_dict[iu.tolist()]
            self.compr0.requires_grad_(True)

            self.compr[..., 0] = self.compr0.unsqueeze(-1)
            self.compr[..., 1] = self.compr0.unsqueeze(-2)

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
                geometry, reference, [self.compr0], parameter, **kwargs)
        else:
            super().__init__(
                geometry, reference, [self.compr0], parameter, **kwargs)

        self.skparams = SkfParamFeed.from_dir(
            parameter['dftb']['path_to_skf'], self.geometry,
            skf_type=skf_type, repulsive=False)
        if self.geometry.is_periodic:
            self.periodic = Periodic(self.geometry, self.geometry.cell,
                                     cutoff=self.skparams.cutoff, **kwargs)
        else:
            self.periodic = None

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

        hs_obj = self.periodic if self.geometry.is_periodic else self.geometry
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
        self.dftb = Dftb2(self.geometry, self.shell_dict,
                          self.params['dftb']['path_to_skf'], repulsive=False)
        self.dftb(hamiltonian=ham, overlap=over)
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
