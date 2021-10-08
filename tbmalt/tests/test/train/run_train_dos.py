"""Example to run training."""
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from tbmalt.common.structures.system import System
from tbmalt.io.loadskf import IntegralGenerator
from tbmalt.ml.train import Integal, CompressionRadii
from tbmalt.common.parameter import Parameter
from tbmalt.io.loadhdf import LoadHdf
from tbmalt.tb.sk import SKT
from tbmalt.tb.dftb.scc import Scc
from tbmalt.common.structures.periodic import Periodic
from tbmalt.tb.coulomb import Coulomb
from tbmalt.tb.properties import dos
from tbmalt.tb.properties import band_pass_state_filter
import time

torch.set_printoptions(6)
torch.set_default_dtype(torch.float64)
size_train, size_test = 1, 1
params = Parameter(ml_params=True)
params.ml_params['task'] = 'mlIntegral'
params.ml_params['steps'] = 100
params.ml_params['target'] = ['dos_eigval']  # charge, dipole, gap, cpa, homo_lumo, dos
params.ml_params['ml_method'] = 'forest'  # nn, linear, forest
# params.ml_params['loss_function'] = 'KLDivLoss'
# params.dftb_params['maxiter'] = 5
params.dftb_params['sigma'] = 0.09
params.dftb_params['with_periodic'] = True
# params.dftb_params['mix'] = 'simple'
dataset = '/home/wbsun/DFTBMaLT/preiodic_train_clean/tbmalt/tbmalt/tests/test/train/dataset/test_aims_si44.hdf'
dataset_dftb = '/home/wbsun/DFTBMaLT/preiodic_train_clean/tbmalt/tbmalt/tests/test/train/dataset/test_dftb_si44.hdf'


def train():
    """Initialize parameters."""
    if params.ml_params['task'] == 'mlIntegral':
        params.dftb_params['path_to_skf'] = '/home/wbsun/DFTBMaLT/preiodic_train_clean/tbmalt/tbmalt/tests/test/train/skf_pbc.hdf'
        params.ml_params['lr'] = 0.000001
    elif params.ml_params['task'] == 'mlCompressionR':
        params.dftb_params['path_to_skf'] = '/home/wbsun/DFTBMaLT/preiodic_train_clean/tbmalt/tbmalt/tests/test/train/skf.hdf.comprwav'
        params.ml_params['lr'] = 0.05
        params.ml_params['compression_radii_grid'] = torch.tensor([
            01.00, 01.50, 02.00, 02.50, 03.00, 03.50, 04.00,
            04.50, 05.00, 06.00, 08.00, 10.00])

    # build a random index
    rand_index = torch.randperm(1)
    train_index = rand_index[:size_train].tolist()
    test_index = rand_index[:size_test].tolist()
    f = open('rand_index.dat', 'w')
    np.savetxt(f, rand_index.detach())
    f.close()

    numbers_train, positions_train, latvecs_train, data_train = LoadHdf.load_reference_si(
        dataset, train_index, ['charge', 'energy', 'homo_lumo', 'dos', 'dos_raw',
                               'dos_raw_gamma', 'eigenvalue'], periodic=True)
    if 'dos' in params.ml_params['target']:
        energies = data_train['dos_raw'][..., 0]
        dos_values = data_train['dos_raw'][..., 1]
        mask_mid = torch.stack([abs(energies[ii] -
                                    data_train['homo_lumo'][..., 0][ii]) < 2E-2
                                for ii in range(size_train)])
        energy_mid = energies[mask_mid]
        mask_vb = torch.stack([abs(energies[ii] - energy_mid[ii]) < 0.6
                               for ii in range(size_train)])
        dos_train = torch.stack([dos_values[ii][mask_vb[ii]]
                                 for ii in range(size_train)])
        energies_train = torch.stack([energies[ii][mask_vb[ii]]
                                      for ii in range(size_train)])

        data_train['dos'] = torch.cat((energies_train.unsqueeze(-1),
                                       dos_train.unsqueeze(-1)), -1)

    if 'dos_eigval' in params.ml_params['target']:
        eigval_ref = data_train['eigenvalue']
        print("eigval_ref", eigval_ref)
        energies = torch.linspace(-18, 5, 500).repeat(size_train, 1)
        dos_ref = dos((eigval_ref), energies,
                      params.dftb_params['sigma'])
        plt.plot(energies[0], dos_ref[0])
        plt.xlim((-18.2, 5.2))
        plt.ylim((-1, 30))
        plt.title('dft')
        plt.xlabel("Energy [eV]")
        plt.ylabel("DOS")
        plt.savefig('ref_dos_cal.png', dpi=300)
        plt.show()
        mask_mid = torch.stack([abs(energies[ii] -
                                    data_train['homo_lumo'][..., 0][ii]) < 3E-2
                                for ii in range(size_train)])
        energy_mid = energies[mask_mid]

        mask_vb = torch.stack([abs(energies[ii] - energy_mid[ii]) < 0.6
                               for ii in range(size_train)])
        dos_train = torch.stack([dos_ref[ii][mask_vb[ii]]
                                 for ii in range(size_train)])
        energies_train = torch.stack([energies[ii][mask_vb[ii]]
                                      for ii in range(size_train)])
        data_train['dos'] = torch.cat((energies_train.unsqueeze(-1),
                                       dos_train.unsqueeze(-1)), -1)

    if params.dftb_params['with_periodic']:
        sys_train = System(numbers_train, positions_train, latvecs_train)
        periodic_train = Periodic(sys_train, sys_train.cell, cutoff=9.98)
        coulomb_train = Coulomb(sys_train, periodic_train)
    else:
        sys_train = System(numbers_train, positions_train)
        periodic_train, coulomb_train = None, None

    numbers_test, positions_test, latvecs_test, data_test = LoadHdf.load_reference_si(
        dataset, test_index, ['charge', 'energy', 'homo_lumo', 'dos',
                              'dos_raw', 'dos_raw_gamma', 'eigenvalue'], periodic=True)
    numbers_dftb_train, positions_dftb_train, latvecs_dftb_train, data_dftb_train = LoadHdf.load_reference_si(
        dataset_dftb, train_index, ['charge', 'energy', 'homo_lumo'], periodic=True)
    numbers_dftb, positions_dftb, latvecs_dftb, data_dftb = LoadHdf.load_reference_si(
        dataset_dftb, test_index, ['charge', 'energy', 'homo_lumo'], periodic=True)
    if params.dftb_params['with_periodic']:
        sys_test = System(numbers_test, positions_test, latvecs_test)
        periodic_test = Periodic(sys_test, sys_test.cell, cutoff=9.98)
        coulomb_test = Coulomb(sys_test, periodic_test)
    else:
        sys_test = System(numbers_test, positions_test)
        periodic_test, coulomb_test = None, None

    # optimize integrals directly
    time1 = time.time()
    if params.ml_params['task'] == 'mlIntegral':
        integral = Integal(sys_train, data_train, params,
                           coulomb=coulomb_train, periodic=periodic_train)
        integral(params.ml_params['target'])
        print('numbers_test', len(numbers_test))
        time2 = time.time()
        scc_pred = integral.predict(sys_test, coulomb_test, periodic_test)

        if 'cpa' in params.ml_params['target']:
            sk = IntegralGenerator.from_dir(
                './slko/skf.hdf.mio', sys_test, repulsive=False, sk_type='h5py')
            skt = SKT(sys_test, sk, periodic_test)
            scc_dftb = Scc(sys_test, skt, params, coulomb_test, periodic_test,
                           params.ml_params['target'])
            f = open('cpaPred.dat', 'w')
            np.savetxt(f, scc_pred.cpa.detach())
            f.close()
            f = open('cpaDftb.dat', 'w')
            np.savetxt(f, scc_dftb.cpa)
            f.close()
            f = open('cpaRef.dat', 'w')
            np.savetxt(f, data_test['hirshfeld_volume_ratio'])
            f.close()
        if 'dos' in params.ml_params['target']:
            sk = IntegralGenerator.from_dir(
                params.dftb_params['path_to_skf'], sys_test, repulsive=False, sk_type='h5py')
            skt = SKT(sys_test, sk, periodic_test)
            scc_dftb = Scc(sys_test, skt, params, coulomb_test, periodic_test,
                           params.ml_params['target'])
        if 'dos_eigval' in params.ml_params['target']:
            sk = IntegralGenerator.from_dir(
                params.dftb_params['path_to_skf'], sys_test, repulsive=False, sk_type='h5py')
            skt = SKT(sys_test, sk, periodic_test)
            scc_dftb = Scc(sys_test, skt, params, coulomb_test, periodic_test,
                           params.ml_params['target'])

    elif params.ml_params['task'] == 'mlCompressionR':
        compr = CompressionRadii(sys_train, data_train, params,
                                 coulomb=coulomb_train, periodic=periodic_train)
        _train = compr(params.ml_params['target'])
        print('property:', 'dipole train')
        print(abs((_train.dipole - data_train['dipole'])).sum())
        print(abs((data_dftb_train['dipole'] - data_train['dipole'])).sum())

        scc_pred = compr.predict(sys_test, coulomb_test, periodic_test)

    if 'charge' in params.ml_params['target']:
        print('time:', time2 - time1)
        print('property:', 'charge')
        print(abs((scc_pred.charge - data_test['charge'])).sum())
        print(abs((data_dftb['charge'] - data_test['charge'])).sum())
        print('predict', scc_pred.charge)
        print('ref', data_test['charge'])
        f = open('netchargePred.dat', 'w')
        np.savetxt(f, scc_pred.charge.detach() - scc_pred.qzero)
        f.close()
        f = open('netchargeDftb.dat', 'w')
        np.savetxt(f, data_dftb['charge'] - scc_pred.qzero)
        f.close()
        f = open('netchargeRef.dat', 'w')
        np.savetxt(f, data_test['charge'] - scc_pred.qzero)
        f.close()
        f = open('Predcharge.dat', 'w')
        np.savetxt(f, scc_pred.charge.detach())
        f.close()
        f = open('DFTBcharge.dat', 'w')
        np.savetxt(f, data_dftb['charge'])
        f.close()
        f = open('Refcharge.dat', 'w')
        np.savetxt(f, data_test['charge'])
        f.close()
        heavy_mask = data_test['charge'].gt(1.5)
        heavy_qzero = torch.floor(data_test['charge'][heavy_mask])
        xx = torch.linspace(1, heavy_qzero.shape[0], heavy_qzero.shape[0])
        plt.plot(xx, scc_pred.charge.detach()[heavy_mask] - heavy_qzero, 'xr', label='predict')
        plt.legend()
        plt.show()

    if 'dipole' in params.ml_params['target']:
        print('property:', 'dipole')
        print(abs((scc_pred.dipole - data_test['dipole'])).sum())
        print(abs((data_dftb['dipole'] - data_test['dipole'])).sum())
        plt.plot(data_test['dipole'], scc_pred.dipole.detach(), 'xr', label='predict')
        plt.plot(data_test['dipole'], data_dftb['dipole'], 'vb', label='previous parameters')
        plt.plot(torch.linspace(-0.8, 0.8, 10), torch.linspace(-0.8, 0.8, 10), color='k')
        plt.legend()
        plt.show()
        print('scc_pred.dipole', len(scc_pred.dipole))
        f = open('dipolePred.dat', 'w')
        np.savetxt(f, scc_pred.dipole.detach())
        f.close()
        f = open('dipoleDftb.dat', 'w')
        np.savetxt(f, data_dftb['dipole'])
        f.close()
        f = open('dipoleRef.dat', 'w')
        np.savetxt(f, data_test['dipole'])
        f.close()

    if 'homo_lumo' in params.ml_params['target']:
        print('time:', time2 - time1)
        print('property:', 'homo_lumo')
        hl_pred = scc_pred.homo_lumo
        hl_ref = data_test['homo_lumo']
        hl_dftb = data_dftb['homo_lumo']
        print(abs((hl_pred - hl_ref)).sum())
        print(abs((hl_dftb - hl_ref)).sum())
        f = open('Pred_homo_lumo.dat', 'w')
        np.savetxt(f, hl_pred.detach())
        f.close()
        f = open('DFTBcharge.dat', 'w')
        np.savetxt(f, data_dftb['homo_lumo'])
        f.close()
        f = open('Refcharge.dat', 'w')
        np.savetxt(f, data_test['homo_lumo'])
        f.close()
        plt.plot(hl_ref, hl_pred.detach(), 'xr', label='predict')
        plt.plot(hl_ref, hl_dftb, 'vb', label='previous parameters')
        plt.plot(torch.linspace(-0.8, 0.8, 10), torch.linspace(-0.8, 0.8, 10), color='k')
        plt.legend()
        plt.show()

    if 'gap' in params.ml_params['target']:
        print('property:', 'gap')
        gap_pred = scc_pred.homo_lumo[:, 1] - scc_pred.homo_lumo[:, 0]
        gap_ref = data_test['homo_lumo'][:, 1] - data_test['homo_lumo'][:, 0]
        gap_dftb = data_dftb['homo_lumo'][:, 1] - data_dftb['homo_lumo'][:, 0]
        print(abs((gap_pred - gap_ref)).sum())
        print(abs((gap_dftb - gap_ref)).sum())
        f = open('Pred_gap.dat', 'w')
        np.savetxt(f, gap_pred.detach())
        f.close()
        f = open('DFTB_gap.dat', 'w')
        np.savetxt(f, gap_dftb.detach())
        f.close()
        f = open('Ref_gap.dat', 'w')
        np.savetxt(f, gap_ref.detach())
        f.close()
        f = open('Pred_homo_lumo.dat', 'w')
        np.savetxt(f, scc_pred.homo_lumo.detach())
        f.close()
        f = open('DFTB_hl.dat', 'w')
        np.savetxt(f, data_dftb['homo_lumo'])
        f.close()
        f = open('Ref_hl.dat', 'w')
        np.savetxt(f, data_test['homo_lumo'])
        plt.plot(gap_ref, gap_pred.detach(), 'xr', label='predict')
        plt.plot(gap_ref, gap_dftb, 'vb', label='previous parameters')
        plt.plot(torch.linspace(-0.8, 0.8, 10), torch.linspace(-0.8, 0.8, 10), color='k')
        plt.legend()
        plt.show()

    if 'cpa' in params.ml_params['target']:
        sys_dftb = System(numbers_dftb, positions_dftb)
        sk = IntegralGenerator.from_dir('./slko/skf.hdf.init', sys_dftb, sk_type='h5py')
        skt = SKT(sys_dftb, sk)
        scc = Scc(sys_dftb, skt, params, ['cpa'])
        print('property:', 'cpa')
        print(abs((scc_pred.cpa - data_test['hirshfeld_volume_ratio'])).sum())
        print(abs((scc.cpa - data_test['hirshfeld_volume_ratio'])).sum())

    if 'dos' in params.ml_params['target']:
        print('property:', 'dos')
        # Plot DFT dos
        plt.plot(data_test['dos_raw'][..., 0][0], data_test['dos_raw'][..., 1][0])
        plt.xlim((-18.2, 5.2))
        plt.title('dft')
        plt.xlabel("Energy [eV]")
        plt.ylabel("DOS")
        plt.savefig('ref_dos_kpoints.png', dpi=300)
        plt.show()

        plt.plot(data_test['dos_raw_gamma'][..., 0][0], data_test['dos_raw_gamma'][..., 1][0])
        plt.title('dft')
        plt.xlabel("Energy [eV]")
        plt.ylabel("DOS")
        plt.savefig('ref_dos_gamma.png', dpi=300)
        plt.show()

        # Plot DFTB dos
        plt.plot(scc_dftb.dos_energy, scc_dftb.dos[0])
        plt.xlim((-17, 5.2))
        plt.ylim((-1, 30))
        plt.title('before training')
        plt.xlabel("Energy [eV]")
        plt.ylabel("DOS")
        plt.savefig('before_dos.png', dpi=300)
        plt.show()

        plt.plot(scc_pred.dos_energy.detach(), scc_pred.dos[0].detach())
        plt.xlim((-17, 5.2))
        plt.ylim((-1, 30))
        plt.title('after training')
        plt.xlabel("Energy [eV]")
        plt.ylabel("DOS")
        plt.savefig('after_dos.png', dpi=300)
        plt.show()

    if 'dos_eigval' in params.ml_params['target']:
        print('property:', 'dos')
        # Plot DFT dos
        plt.plot(data_test['dos_raw'][..., 0][0], data_test['dos_raw'][..., 1][0])
        plt.xlim((-18.2, 5.2))
        plt.title('dft')
        plt.xlabel("Energy [eV]")
        plt.ylabel("DOS")
        plt.savefig('ref_dos_kpoints.png', dpi=300)
        plt.show()

        plt.plot(data_test['dos_raw_gamma'][..., 0][0], data_test['dos_raw_gamma'][..., 1][0])
        plt.title('dft')
        plt.xlabel("Energy [eV]")
        plt.ylabel("DOS")
        plt.savefig('ref_dos_gamma.png', dpi=300)
        plt.show()

        # Plot DFTB dos
        plt.plot(scc_dftb.dos_energy, scc_dftb.dos[0])
        plt.xlim((-17, 5.2))
        plt.ylim((-1, 31))
        plt.title('before training')
        plt.xlabel("Energy [eV]")
        plt.ylabel("DOS")
        plt.savefig('before_dos.png', dpi=300)
        plt.show()

        plt.plot(scc_pred.dos_energy.detach(), scc_pred.dos[0].detach())
        plt.xlim((-17, 5.2))
        plt.ylim((-1, 31))
        plt.title('after training')
        plt.xlabel("Energy [eV]")
        plt.ylabel("DOS")
        plt.savefig('after_dos.png', dpi=300)
        plt.show()


def test():
    """Initialize parameters."""
    params.dftb_params['path_to_skf'] = './slko/skf.hdf.comprwav'
    size_train, size_test = 600, 1000
    compr_train = 'data/comprwav_ani1_3dipCha_size600.dat'
    dataset = './dataset/aims_6000_01.hdf'
    dataset_dftb = './dataset/scc_6000_01.hdf'
    properties = ['dipole', 'charge']
    params.ml_params['ml_method'] = 'forest'  # nn, linear, forest
    params.ml_params['compression_radii_grid'] = torch.tensor([
        01.00, 01.50, 02.00, 02.50, 03.00, 03.50, 04.00,
        04.50, 05.00, 06.00, 08.00, 10.00])

    numbers_train, positions_train, data_train = LoadHdf.load_reference(
        dataset, size_train, ['charge', 'dipole', 'homo_lumo', 'hirshfeld_volume_ratio'])
    numbers_test, positions_test, data_test = LoadHdf.load_reference(
        dataset, size_test, ['charge', 'dipole', 'homo_lumo', 'hirshfeld_volume_ratio'])
    numbers_dftb_train, positions_dftb_train, data_dftb_train = LoadHdf.load_reference(
        dataset_dftb, size_train, ['charge', 'dipole', 'homo_lumo'])
    numbers_dftb, positions_dftb, data_dftb = LoadHdf.load_reference(
        dataset_dftb, size_test, ['charge', 'dipole', 'homo_lumo'])

    compr_train = torch.from_numpy(np.loadtxt(compr_train))
    sys_train = System(numbers_train, positions_train)
    sys_test = System(numbers_test, positions_test)

    sk = IntegralGenerator.from_dir(
        params.dftb_params['path_to_skf'], sys_test, repulsive=False,
        sk_type='h5py', homo=False, interpolation='bicubic_interpolation',
        compression_radii_grid=params.ml_params['compression_radii_grid'])

    skt = SKT(sys_train, sk, compression_radii=compr_train,
              fix_onsite=True, fix_U=True)
    scc_opt = Scc(sys_train, skt, params, properties)
    if 'charge' in properties:
        print('opt property:', 'charge')
        print(abs((scc_opt.charge - data_train['charge'])).sum())
        print(abs((data_dftb_train['charge'] - data_train['charge'])).sum())
    if 'dipole' in properties:
        print('opt property:', 'dipole')
        print(abs((scc_opt.dipole - data_train['dipole'])).sum())
        print(abs((data_dftb_train['dipole'] - data_train['dipole'])).sum())

    compr = CompressionRadii(sys_test, data_test, params)
    scc_pred = compr.predict_(params, compr_train, sys_train, sys_test, sk, properties)
    if 'charge' in properties:
        print('property:', 'charge')
        print(abs((scc_pred.charge - data_test['charge'])).sum())
        print(abs((data_dftb['charge'] - data_test['charge'])).sum())
        f = open('netChargePred.dat', 'w')
        np.savetxt(f, scc_pred.charge - scc_pred.qzero)
        f.close()
        f = open('netChargeDftb.dat', 'w')
        np.savetxt(f, data_dftb['charge'] - scc_pred.qzero)
        f.close()
        f = open('netChargeRef.dat', 'w')
        np.savetxt(f, data_test['charge'] - scc_pred.qzero)
        f.close()
    if 'dipole' in properties:
        print('property:', 'dipole')
        print(abs((scc_pred.dipole - data_test['dipole'])).sum())
        print(abs((data_dftb['dipole'] - data_test['dipole'])).sum())
        f = open('dipolePred.dat', 'w')
        np.savetxt(f, scc_pred.dipole)
        f.close()
        f = open('dipoleDftb.dat', 'w')
        np.savetxt(f, data_dftb['dipole'])
        f.close()
        f = open('dipoleRef.dat', 'w')
        np.savetxt(f, data_test['dipole'])

        f.close()
    if 'cpa' in properties:
        sk = IntegralGenerator.from_dir(
            './slko/skf.hdf.mio', sys_test, repulsive=False, sk_type='h5py')
        skt = SKT(sys_test, sk)
        scc_dftb = Scc(sys_test, skt, params, properties)
        print('property:', 'cpa')
        print(abs((scc_pred.cpa - data_test['hirshfeld_volume_ratio'])).sum())
        print(abs((scc_dftb.cpa - data_test['hirshfeld_volume_ratio'])).sum())
        f = open('cpaPred.dat', 'w')
        np.savetxt(f, scc_pred.cpa)
        f.close()
        f = open('cpaDftb.dat', 'w')
        np.savetxt(f, scc_dftb.cpa)
        f.close()
        f = open('cpaRef.dat', 'w')
        np.savetxt(f, data_test['hirshfeld_volume_ratio'])
        f.close()


train()
