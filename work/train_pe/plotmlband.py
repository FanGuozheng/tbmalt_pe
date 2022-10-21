import pickle
import h5py
import torch
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from tbmalt import Geometry, Basis, Dftb2, Dftb1
from tbmalt.common.batch import pack, merge
from tbmalt.ml.cacsf_pair import g_pe_pair, g_pe
from tbmalt.io import Dataset
from tbmalt.ml.optim import Scale
from tbmalt.common.logger import get_logger
from train_pe import train_function, pred

torch.set_default_dtype(torch.float64)
logger = get_logger(__name__)

shell_dict = {6: [0, 1, 2], 14: [0, 1, 2]}
shell_dict_std = {6: [0, 1], 14: [0, 1]}
n_interval = [5, 12]  # interval difference, interval number
dataset = 'band_c_si.h5'

ml_method = 'rf'
param_dict = {'ml': {'batch_size': 1, 'train_onsite': 'local'}}
param_dict.update({"n_band0": torch.tensor([int(ii * n_interval[0]) for ii in range(n_interval[1])])})
param_dict.update({"n_band1": torch.tensor([int(ii * n_interval[0] + 1) for ii in range(n_interval[1])])})
param_dict.update({"n_valence": 'all', "n_conduction": {6: 1, 14: 1}, "train_e_low": -30, "train_e_high": 20})
cutoff = 10.0
elements = ['Si', 'C']
skf_type = 'skf'
kpoint_level = 'high'
path_sk = './slko'
param_dict.update({'dftb': {'path_to_skf': path_sk}})
alignment = 'vbm'
train_1der = False
loss_fn = 'MSELoss'
inverse = True
shell_dict_list = [shell_dict_std, shell_dict]


def plot_size(params):
    """"""
    gen_dia_c_data = False
    gen_hex_c_data = False
    gen_dia_sic_data = False

    # Use gen_dia_c_data == True to generate the following data
    c_dia1 = [0.5015, 0.3252, 0.3103, 0.3844, 0.4557, 0.3510, 0.2219, 0.2597, 0.2899]
    c_dia2 = [0.2716, 0.4471, 0.5151, 0.5411, 0.2479, 0.4835, 0.2240, 0.3928, 0.3207]
    c_dia3 = [0.3921, 0.2819, 0.3597, 0.2606, 0.2498, 0.4078, 0.3175, 0.3334, 0.3950]
    c_dia4 = [0.2681, 0.3057, 0.3027, 0.2684, 0.2456, 0.3087, 0.3466, 0.2422, 0.3953]
    c_dia5 = [0.2584, 0.3185, 0.2900, 0.2851, 0.2877, 0.2917, 0.2970, 0.2950, 0.4386]
    c_dia6 = [0.2776, 0.3040, 0.3101, 0.3292, 0.3103, 0.3252, 0.3363, 0.3236, 0.3305]
    c_dimand = np.array([c_dia1, c_dia2, c_dia3, c_dia4, c_dia5, c_dia6])
    error_c_dia = [c_dimand.mean(1) - c_dimand.min(1), c_dimand.max(1) - c_dimand.mean(1)]

    if gen_dia_c_data:
        file_dia = [
            '../data/opt_c_bulk_diamond2_0.05.pkl',
            '../data/opt_c_bulk_diamond2_0.1.pkl',
            '../data/opt_c_bulk_diamond2_0.2.pkl',
            '../data/opt_c_bulk_diamond2_0.3.pkl',
            '../data/opt_c_bulk_diamond2_0.4.pkl',
            '../data/opt_c_bulk_diamond2_0.5.pkl',
        ]
        dia_pred = {'c_bulk_diamond2': 0.2,}
        for file in file_dia:
            mae0, mae1, error_dict, _ = _load_train_pkl([file], dia_pred, params, [0.35] * 6)
            # print('mae0', torch.mean(mae0), torch.mean(mae0, 1), torch.mean(mae1, 1))

    c_hex1 = [0.4677, 0.3984, 0.4262, 0.5040, 0.4571, 0.5010, 0.4998, 0.5123, 0.4745]
    c_hex2 = [0.4452, 0.4889, 0.4153, 0.4437, 0.4706, 0.4686, 0.5201, 0.4490, 0.4028]
    c_hex3 = [0.4609, 0.5204, 0.5096, 0.3942, 0.4968, 0.5692, 0.3663, 0.2929, 0.4687]
    c_hex4 = [0.5786, 0.4439, 0.4018, 0.3303, 0.4304, 0.4417, 0.3261, 0.3139, 0.4658]
    c_hex5 = [0.5945, 0.4536, 0.3677, 0.3084, 0.3960, 0.4816, 0.3203, 0.3131, 0.5370]
    c_hex6 = [0.6073, 0.4662, 0.3823, 0.3297, 0.4083, 0.3709, 0.3109, 0.3005, 0.4779]
    c_hex = np.array([c_hex1, c_hex2, c_hex3, c_hex4, c_hex5, c_hex6])
    error_c_hex = [c_hex.mean(1) - c_hex.min(1), c_hex.max(1) - c_hex.mean(1)]
    if gen_hex_c_data:
        file_hex = [
            '../data/opt_c_bulk_hex2_0.05.pkl',
            '../data/opt_c_bulk_hex2_0.1.pkl',
            '../data/opt_c_bulk_hex2_0.2.pkl',
            '../data/opt_c_bulk_hex2_0.3.pkl',
            '../data/opt_c_bulk_hex2_0.4.pkl',
            '../data/opt_c_bulk_hex2_0.5.pkl',
        ]
        hex_pred = {'c_bulk_hex2': 0.2,}
        for file in file_hex:
            mae0, mae1, error_dict, _ = _load_train_pkl([file], hex_pred, params, [0.4] * 6)
            # print('mae0', torch.mean(mae0), torch.mean(mae0, 1), torch.mean(mae1, 1))

    sic_dia1 = [0.3051, 0.2828, 0.5916, 0.3384, 0.3234, 0.3218, 0.2350, 0.3407, 0.3175]
    sic_dia2 = [0.5003, 0.4983, 0.6348, 0.4156, 0.5394, 0.5526, 0.5498, 0.4636, 0.8236]
    sic_dia3 = [0.4583, 0.4810, 0.6007, 0.4717, 0.5646, 0.4597, 0.4525, 0.4562, 0.5737]
    sic_dia4 = [0.2877, 0.3410, 0.5627, 0.2608, 0.5611, 0.3851, 0.4503, 0.4920, 0.6449]
    sic_dia5 = [0.2201, 0.3080, 0.3619, 0.2470, 0.3390, 0.3118, 0.3247, 0.2901, 0.3464]
    sic_dia6 = [0.3048, 0.3250, 0.3646, 0.2492, 0.2515, 0.2779, 0.3157, 0.3017, 0.3821]
    sic_dia = np.array([sic_dia1, sic_dia2, sic_dia3, sic_dia4, sic_dia5, sic_dia6])
    error_sic_dia = [sic_dia.mean(1) - sic_dia.min(1), sic_dia.max(1) - sic_dia.mean(1)]
    if gen_dia_sic_data:
        file_dia = [
            '../data/opt_sic_bulk_diamond2_0.05.pkl',
            '../data/opt_sic_bulk_diamond2_0.1.pkl',
            '../data/opt_sic_bulk_diamond2_0.2.pkl',
            '../data/opt_sic_bulk_diamond2_0.3.pkl',
            '../data/opt_sic_bulk_diamond2_0.4.pkl',
            '../data/opt_sic_bulk_diamond2_0.5.pkl',
        ]
        dia_pred = {'sic_bulk_diamond2': 0.2,}
        for file in file_dia:
            mae0, mae1, error_dict, _ = _load_train_pkl([file], dia_pred, params, [0.4] * 6)
            # print('mae0', torch.mean(mae0), torch.mean(mae0, 1), torch.mean(mae1, 1))

    # plt.errorbar([0.05, 0.1, 0.2, 0.3, 0.4, 0.5], c_dimand.mean(1), yerr=error_c_dia, fmt='o:', elinewidth=2, ms=5, capsize=4)
    # plt.errorbar([0.05, 0.1, 0.2, 0.3, 0.4, 0.5], c_hex.mean(1), yerr=error_c_hex, fmt='o:', elinewidth=2, ms=5, capsize=4)
    # plt.errorbar([0.05, 0.1, 0.2, 0.3, 0.4, 0.5], sic_dia.mean(1), yerr=error_sic_dia, fmt='o:', elinewidth=2, ms=5, capsize=4)

    opt = [0.9998, 1.0074, 1.0152, 1.0176, 1.0060, 1.0301, 1.0337, 1.0184, 1.0104]
    localopt = [0.2521, 0.2148, 0.1854, 0.2789, 0.2283, 0.2031, 0.2087, 0.1689, 0.1699, 0.2259, 0.2351, 0.1461, 0.2527, 0.2161, 0.2037, 0.2758, 0.1962, 0.2377, 0.1983, 0.1802, 0.1945]
    globalopt = [0.3149, 0.2241, 0.2222, 0.2637, 0.3271, 0.2570, 0.2488, 0.2097, 0.2221, 0.2326, 0.2792, 0.2458, 0.2692, 0.2585, 0.2463, 0.3380, 0.2530, 0.3104, 0.2507, 0.2406, 0.2861]
    scaleoly = [0.3149, 0.2241, 0.2222, 0.2637, 0.3271, 0.2570, 0.2488, 0.2097, 0.2221, 0.2326, 0.2792, 0.2458, 0.2692, 0.2585, 0.2463, 0.3380, 0.2530, 0.3104, 0.2507, 0.2406, 0.2861]
    fig, ax = plt.subplots(2, 1, figsize=(6, 8))  # sharex=True, sharey=True
    ind = np.arange(4)  # -> xx
    width = 0.35

    # plt.plot([0.1, 0.2, 0.3, 0.4, 0.5], c_dimand.mean(1)[1:], 'o-', label='C: diamond')
    # plt.plot([0.1, 0.2, 0.3, 0.4, 0.5], c_hex.mean(1)[1:], 'o-', label='C: hexagonal')
    # plt.plot([0.1, 0.2, 0.3, 0.4, 0.5], sic_dia.mean(1)[1:], 'o-', label='SiC: diamond')
    # plt.ylim(0.2, 0.6)
    # plt.legend()
    # plt.xlabel('Data set ratios')
    # plt.ylabel('Average predicting errors (eV)')

    ax[0].plot([0.1, 0.2, 0.3, 0.4, 0.5], c_dimand.mean(1)[1:], 'o-', label='C: diamond')
    ax[0].plot([0.1, 0.2, 0.3, 0.4, 0.5], c_hex.mean(1)[1:], 'o-', label='C: hexagonal')
    ax[0].plot([0.1, 0.2, 0.3, 0.4, 0.5], sic_dia.mean(1)[1:], 'o-', label='SiC: diamond')
    # plt.ylim(0.2, 0.6)
    ax[0].legend()
    ax[0].set_xlabel('Data set ratios')
    ax[0].set_ylabel('Average predicting errors (eV)')
    ax[0].set_ylim(0.25, 0.625)
    ax[0].text(0.1, 0.59, '(a)', fontsize="x-large")

    ax[1].yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax[1].bar(ind + 1 * width,
              (np.mean(opt), np.mean(localopt), np.mean(globalopt) , np.mean(scaleoly)),
               width, label="Carbon diamond", edgecolor='red', alpha=0.8)
    # ax[1].bar(ind + 2 * width,
    #            (np.mean(c_hex_pbc), np.mean(c_hex_glo), np.mean(c_hex_ml)),
    #            width, label="C: hexagonal", edgecolor='g', alpha=0.8)
    ax[1].set_ylim(0, 1.2)
    ax[1].set_xticks([])
    ax[1].set_ylabel("band structure MAEs (eV)",)# fontsize="large")
    ax[1].legend()#fontsize="large")
    ax[1].set_xticks([0.3, 1.3, 2.3, 3.3], [r"Optimized", r"local onsite", "global onsite", "no onsite"],)# fontsize="x-large")
    ax[1].errorbar(  # error bar
        ind + 1 * width,
               (np.mean(opt), np.mean(localopt), np.mean(globalopt) , np.mean(scaleoly)),
        yerr=[
            [np.mean(opt) - min(opt), np.mean(localopt) - min(localopt), np.mean(globalopt) - min(globalopt), np.mean(scaleoly) - min(scaleoly)],
            [max(opt) - np.mean(opt), max(localopt) - np.mean(localopt), max(globalopt) - np.mean(globalopt), max(scaleoly) - np.mean(scaleoly)]],
        fmt="k.", capsize=4
    )
    ax[1].text(0.2, 1.1, '(b)', fontsize="x-large")

    plt.savefig('effectSizeBand.png', dpi=100)
    plt.show()


def plot_bulk(params):
    """Plot bulk MAE errors."""
    gen_data = False
    if gen_data:
        files_train = [
            # '../data/opt_c_bulk_diamond2_0.4.pkl',
            # '../data/opt_c_bulk_hex2_0.4.pkl',
            # '../data/opt_c_bulk_hex4_0.4.pkl',  # 0.6
            '../data/opt_si_bulk_diamond2_0.4.pkl',
            # '../data/opt_si_bulk_hex4_0.4.pkl',
            # '../data/opt_si_bulk_tetrag4_0.4.pkl',
            # '../data/opt_sic_bulk_diamond2_0.4.pkl',
            ]
        dia_pred = {
            # 'c_bulk_diamond2': 0.2,
            # 'c_bulk_hex2': 0.2,
            # 'sic_bulk_diamond2': 0.2,
            # 'sic_bulk_cubic2': 0.2
            'si_bulk_diamond2': 0.2
            # 'si_bulk_hex4': 0.2,
            # 'si_bulk_tetrag4': 0.2,
        }

        skf_list = ['./slko/pbc/', './slko/']
        # for file in file_dia:
        mae0, mae1, err_list = _load_train_pkl(
            files_train, dia_pred, params, [0.35, 0.4, 0.6, 0.35, 0.4, 0.4, 0.4], skf_list=skf_list)

        print(torch.cat([ii for ii in mae0], dim=0).mean(1))
        print('PBC', [ii[0]['mae0'].shape for ii in err_list])
        print('PBC', torch.cat([ii[0]['mae0'] for ii in err_list], dim=0).mean(1))
        print('global', torch.cat([ii[1]['mae0'] for ii in err_list], dim=0).mean(1))

    # C diamond
    c_dia_pbc = [3.7426, 3.6328, 3.5213, 3.7523, 3.6956, 3.8512, 3.9452, 3.7589, 3.6724]
    c_dia_glo = [0.9998, 1.0074, 1.0152, 1.0176, 1.0060, 1.0301, 1.0337, 1.0184, 1.0104]
    c_dia_ml = [0.2815, 0.3137, 0.2853, 0.2804, 0.3089, 0.2696, 0.3050, 0.2937, 0.4335]

    # C hex
    c_hex_pbc = [3.5323, 3.6292, 3.3012, 3.5573, 3.4595, 3.5070, 3.5497, 3.5223, 3.4598]
    c_hex_glo = [1.1312, 1.1332, 1.1244, 1.1478, 1.1630, 1.1494, 1.1438, 1.1439, 1.1320]
    c_hex_ml = [0.6153, 0.4469, 0.3731, 0.2976, 0.4015, 0.4902, 0.3365, 0.3209, 0.5329]

    # Si diamond
    si_dia_pbc = [1.6879, 1.7937, 1.6319, 1.5444, 1.7898, 1.6911, 1.6136, 1.7323, 1.6129, 1.6860, 1.5697, 1.7829]
    si_dia_glo = [0.2938, 0.2938, 0.2983, 0.3043, 0.2939, 0.2985, 0.2960, 0.2929, 0.2998, 0.2938, 0.3032, 0.2946]
    si_dia_ml = [0.2004, 0.2319, 0.2379, 0.1847, 0.2999, 0.1550, 0.2193, 0.1763, 0.2279, 0.1863, 0.1541, 0.2635]

    # Si hex
    si_hex_pbc = [1.6702, 1.6091, 1.7474, 1.5162, 1.7866, 1.6251, 1.6589, 1.6387, 1.7198]
    si_hex_glo = [0.3128, 0.3057, 0.3200, 0.3096, 0.3145, 0.3079, 0.3073, 0.3144, 0.3110]
    si_hex_ml = [0.2128, 0.1723, 0.2798, 0.1683, 0.2210, 0.2098, 0.1675, 0.2088, 0.2366]

    # Si tet
    si_tet_pbc = [1.4773, 1.4063, 1.3627, 1.4115, 1.4149, 1.4607, 1.4983, 1.4034, 1.1381, 1.3927, 1.4485, 1.4814]
    si_tet_glo = [0.6982, 0.7880, 0.6435, 0.7428, 0.7457, 0.7027, 0.7265, 0.7288, 0.5139, 0.6657, 0.7342, 0.7849]
    si_tet_ml = [0.3530, 0.3541, 0.3445, 0.3697, 0.3725, 0.3576, 0.3436, 0.3486, 0.3395, 0.3693, 0.3905, 0.3466]

    # SiC diamond
    sic_dia_pbc = [2.1484, 2.1594, 2.2200, 2.3075, 2.2620, 2.2834, 2.1865, 2.2216, 2.1570]
    sic_dia_glo = [0.7602, 0.7539, 0.7556, 0.7691, 0.7576, 0.7679, 0.7485, 0.7609, 0.7608]
    sic_dia_ml = [0.3660, 0.4635, 0.4705, 0.3298, 0.3776, 0.4205, 0.4206, 0.3956, 0.4081]

    # SiC cubic
    sic_cub_pbc = [2.1484, 2.1594, 2.2200, 2.3075, 2.2620, 2.2834, 2.1865, 2.2216, 2.1570]
    sic_cub_glo = [0.7602, 0.7539, 0.7556, 0.7691, 0.7576, 0.7679, 0.7485, 0.7609, 0.7608]
    sic_cub_ml = [0.3660, 0.4635, 0.4705, 0.3298, 0.3776, 0.4205, 0.4206, 0.3956, 0.4081]

    fig, axs = plt.subplots(2, 1, figsize=(6, 6))  # sharex=True, sharey=True
    ind = np.arange(3)  # -> xx
    width = 0.15
    axs[0].yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axs[0].bar(ind + 1 * width,
               (np.mean(c_dia_pbc), np.mean(c_dia_glo), np.mean(c_dia_ml)),
               width, label="C: diamond", edgecolor='red', alpha=0.8)
    axs[0].bar(ind + 2 * width,
               (np.mean(c_hex_pbc), np.mean(c_hex_glo), np.mean(c_hex_ml)),
               width, label="C: hexagonal", edgecolor='g', alpha=0.8)
    axs[0].bar(ind + 3 * width,
               (np.mean(sic_dia_pbc), np.mean(sic_dia_glo), np.mean(sic_dia_ml)),
               width, label="SiC: diamond", edgecolor='b', alpha=0.8)
    axs[0].bar(ind + 4 * width,
               (np.mean(sic_cub_pbc), np.mean(sic_cub_glo), np.mean(sic_cub_ml)),
               width, label="SiC: cubic", edgecolor='b', alpha=0.8)

    axs[0].errorbar(  # Carbon
        ind + 1 * width,
        (np.mean(c_dia_pbc), np.mean(c_dia_glo), np.mean(c_dia_ml)),
        yerr=[
            [np.mean(c_dia_pbc) - min(c_dia_pbc), np.mean(c_dia_glo) - min(c_dia_glo), np.mean(c_dia_ml) - min(c_dia_ml)],
            [max(c_dia_pbc) - np.mean(c_dia_pbc), max(c_dia_glo) - np.mean(c_dia_glo), max(c_dia_ml) - np.mean(c_dia_ml)]],
        fmt="k.", capsize=4
    )
    axs[0].errorbar(  # Carbon-hex
        ind + 2 * width,
        (np.mean(c_hex_pbc), np.mean(c_hex_glo), np.mean(c_hex_ml)),
        yerr=[
            [np.mean(c_hex_pbc) - min(c_hex_pbc), np.mean(c_hex_glo) - min(c_hex_glo), np.mean(c_hex_ml) - min(c_hex_ml)],
            [max(c_hex_pbc) - np.mean(c_hex_pbc), max(c_hex_glo) - np.mean(c_hex_glo), max(c_hex_ml) - np.mean(c_hex_ml)]],
        fmt="k.", capsize=4
    )
    axs[0].errorbar(  # SiC
        ind + 3 * width,
        (np.mean(sic_dia_pbc), np.mean(sic_dia_glo), np.mean(sic_dia_ml)),
        yerr=[
            [np.mean(sic_dia_pbc) - min(sic_dia_pbc), np.mean(sic_dia_glo) - min(sic_dia_glo), np.mean(sic_dia_ml) - min(sic_dia_ml)],
            [max(sic_dia_pbc) - np.mean(sic_dia_pbc), max(sic_dia_glo) - np.mean(sic_dia_glo), max(sic_dia_ml) - np.mean(sic_dia_ml)]],
        fmt="k.", capsize=4
    )
    axs[0].errorbar(  # SiC
        ind + 4 * width,
        (np.mean(sic_cub_pbc), np.mean(sic_cub_glo), np.mean(sic_cub_ml)),
        yerr=[
            [np.mean(sic_cub_pbc) - min(sic_cub_pbc), np.mean(sic_cub_glo) - min(sic_cub_glo), np.mean(sic_cub_ml) - min(sic_cub_ml)],
            [max(sic_cub_pbc) - np.mean(sic_cub_pbc), max(sic_cub_glo) - np.mean(sic_cub_glo), max(sic_cub_ml) - np.mean(sic_cub_ml)]],
        fmt="k.", capsize=4
    )
    axs[0].set_xticks([])
    axs[0].set_ylabel("band structure MAEs (eV)",)# fontsize="large")
    axs[0].legend()#fontsize="large")
    axs[0].set_xticks([0.3, 1.3, 2.3], [r"pbc", r"global", "DFTB-ML"],)# fontsize="x-large")
    axs[0].text(0.1, 4.1, '(a)', fontsize="x-large")
    axs[0].set_ylim(0, 4.5)

    axs[1].yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axs[1].bar(ind + 1 * width,
               (np.mean(si_dia_pbc), np.mean(si_dia_glo), np.mean(si_dia_ml)),
               width, label="Si: diamond", edgecolor='red', alpha=0.8)
    axs[1].bar(ind + 2 * width,
               (np.mean(si_hex_pbc), np.mean(si_hex_glo), np.mean(si_hex_ml)),
               width, label="Si: hexagonal", edgecolor='g', alpha=0.8)
    axs[1].bar(ind + 3 * width,
               (np.mean(si_tet_pbc), np.mean(si_tet_glo), np.mean(si_tet_ml)),
               width, label="Si: tetragonal", edgecolor='b', alpha=0.8)
    axs[1].errorbar(  # error dipole mio1, mio3
        ind + 1 * width,
        (np.mean(si_dia_pbc), np.mean(si_dia_glo), np.mean(si_dia_ml)),
        yerr=[
            [np.mean(si_dia_pbc) - min(si_dia_pbc), np.mean(si_dia_glo) - min(si_dia_glo), np.mean(si_dia_ml) - min(si_dia_ml)],
            [max(si_dia_pbc) - np.mean(si_dia_pbc), max(si_dia_glo) - np.mean(si_dia_glo), max(si_dia_ml) - np.mean(si_dia_ml)]],
        fmt="k.", capsize=4
    )
    axs[1].errorbar(  # error dipole_compr_global1, dipole_compr_global3
        ind + 2 * width,
        (np.mean(si_hex_pbc), np.mean(si_hex_glo), np.mean(si_hex_ml)),
        yerr=[
            [np.mean(si_hex_pbc) - min(si_hex_pbc), np.mean(si_hex_glo) - min(si_hex_glo), np.mean(si_hex_ml) - min(si_hex_ml)],
            [max(si_hex_pbc) - np.mean(si_hex_pbc), max(si_hex_glo) - np.mean(si_hex_glo), max(si_hex_ml) - np.mean(si_hex_ml)]],
        fmt="k.", capsize=4
    )
    axs[1].errorbar(  # error dipole_compr_local1, dipole_compr_local3
        ind + 3 * width,
        (np.mean(si_tet_pbc), np.mean(si_tet_glo), np.mean(si_tet_ml)),
        yerr=[
            [np.mean(si_tet_pbc) - min(si_tet_pbc), np.mean(si_tet_glo) - min(si_tet_glo), np.mean(si_tet_ml) - min(si_tet_ml)],
            [max(si_tet_pbc) - np.mean(si_tet_pbc), max(si_tet_glo) - np.mean(si_tet_glo), max(si_tet_ml) - np.mean(si_tet_ml)]],
        fmt="k.", capsize=4
    )
    axs[1].set_xticks([])
    axs[1].set_ylabel("band structure MAEs (eV)",)# fontsize="large")
    axs[1].legend()#fontsize="large")
    axs[1].set_xticks([0.3, 1.3, 2.3], [r"pbc", r"global", "DFTB-ML"],)# fontsize="x-large")
    axs[1].set_ylim(0, 2.2)
    axs[1].text(0.1, 1.9, '(b)', fontsize="x-large")

    plt.savefig('bulkError.png', dpi=300)
    plt.show()


def plot_bulk_band():
    batch_size = 3
    lr = 3E-3
    onsite_lr = 1E-3
    skf_list = ['./slko/']
    err0, err1, _ = pred([f'../data/opt-c_bulk_diamond2-{lr}-{onsite_lr}.pkl'],
                         dataset, ['c_bulk_diamond2'], [0.2], batch_size,
                         plot_fermi=True, skf_list=skf_list, fermi_band=[])


def plot_vac_band():
    batch_size = 3
    lr = 3E-3
    onsite_lr = 1E-3
    skf_list = ['./slko/']
    err0, err1, _ = pred([f'../data/opt-c_vac_diamond-{lr}-{onsite_lr}.pkl'],
                         dataset, ['c_vac_diamond'], [0.2], batch_size,
                         plot_fermi=True, skf_list=skf_list, fermi_band=[])


def plot_slabdefect(params):
    gen_data = False
    if gen_data:
        files_train = [
            '../data/opt_c_slab_diamond_0.2.pkl',
            '../data/opt_c_vac_diamond63_0.3.pkl',
            '../data/opt_si_slab_diamond_0.2.pkl',
            '../data/opt_si_vac_diamond2_0.3.pkl',
            '../data/opt_sic_slab_diamond_0.2.pkl',
            '../data/opt_sic_vac_diamond2_0.3.pkl',
            ]
        dia_pred = {
            'c_slab_diamond': 0.0,
            'c_vac_diamond': 0.2,
            'si_slab_diamond': 0.0,
            'si_vac_diamond': 0.0,
            'sic_slab': 0.0,
            'sic_vac_diamond': 0.0,
        }

        skf_list = ['./slko/pbc/', './slko/']
        # for file in file_dia:
        mae0, mae1, error_dict, std_err = _load_train_pkl(
            files_train, dia_pred, params, [0.35, 0.3, 0.3, 0.3, 0.35, 0.3], skf_list=skf_list)
        print(torch.cat([ii for ii in mae0], dim=0).mean(1))
        print(torch.cat([ii[0]['mae0'] for ii in std_err], dim=0).mean(1))
        print(torch.cat([ii[1]['mae0'] for ii in std_err], dim=0).mean(1))

    # C diamond slab
    c_slab_pbc = [3.0409, 3.0257, 3.0752, 3.0713, 3.0267, 2.9727, 3.0694, 3.1580, 3.1378,
        3.1753, 3.1456, 3.1160, 3.2512, 3.2018, 3.0902, 3.2143, 3.1332, 3.1778,
        3.1343, 3.0863, 3.1683, 3.1360, 3.2025, 3.0577, 3.1953, 3.1663, 3.1978]
    c_slab_glo = [1.2530, 1.1473, 1.2471, 1.1469, 1.1797, 1.2184, 1.1370, 1.2764, 1.2493,
        1.0485, 1.0211, 1.0741, 1.0757, 1.0554, 1.0656, 1.0764, 1.0403, 1.0783,
        1.1276, 1.1396, 1.0985, 1.1094, 1.0919, 1.1327, 1.0950, 1.0968, 1.1307]
    c_slab_ml = [0.3307, 0.2992, 0.2921, 0.2830, 0.2655, 0.4141, 0.2701, 0.3016, 0.2605,
        0.5094, 0.7678, 0.9006, 0.5337, 0.6999, 0.5382, 0.4980, 0.9929, 0.5845,
        0.7668, 0.5745, 0.8708, 0.5174, 0.4532, 0.5519, 0.8733, 0.9980, 0.5281]

    # C diamond vac
    c_vac_pbc = [4.2852, 3.4955, 3.4894, 3.5045, 3.5559, 3.4729, 3.4862, 3.5995, 3.5411, 3.4819]
    c_vac_glo = [1.2468, 1.0794, 1.0725, 1.0690, 1.1608, 1.0767, 1.0373, 1.1577, 1.0570, 1.0490]
    c_vac_ml = [0.2160, 0.1640, 0.1368, 0.1526, 0.2318, 0.1411, 0.1406, 0.1575, 0.1550, 0.2237]

    # Si diamond slab
    si_slab_pbc = [1.3535, 1.2666, 1.2732, 1.3427, 1.3558, 1.3896, 1.3220, 1.2728, 1.4132,
                   1.3490, 0.9837, 1.2211, 0.9346, 0.9278, 1.0391, 1.0332, 1.0025, 1.0523,
                   1.1302, 1.2030, 1.0999, 1.1485, 1.1678, 1.1482, 1.1359, 1.2078, 1.1378,
                   1.1771, 1.1413]
    si_slab_glo = [0.3195, 0.3025, 0.3077, 0.3150, 0.3058, 0.3226, 0.3017, 0.3106, 0.3395,
                   0.2840, 0.5064, 0.4249, 0.5299, 0.4886, 0.4530, 0.5151, 0.5503, 0.5560,
                   0.5254, 0.3048, 0.3014, 0.3174, 0.3090, 0.3145, 0.3376, 0.3084, 0.3120,
                   0.3314, 0.3225]
    si_slab_ml = [0.1196, 0.1605, 0.1581, 0.1499, 0.1585, 0.1211, 0.1521, 0.1378, 0.1333,
                  0.1296, 0.0966, 0.1046, 0.1166, 0.0806, 0.1467, 0.1153, 0.1231, 0.1164,
                  0.1391, 0.2114, 0.2047, 0.2378, 0.2242, 0.2135, 0.2371, 0.2106, 0.2244,
                  0.2460, 0.2318]

    # Si diamond vac
    si_vac_pbc = [2.4539, 2.4139, 2.4077, 2.5108, 2.4099, 2.4816]
    si_vac_glo = [1.3605, 1.3130, 1.3306, 1.4358, 1.2564, 1.4173]
    si_vac_ml = [0.6495, 0.4132, 0.3213, 0.3029, 0.4583, 0.9989]

    # SiC diamond slab
    sic_slab_pbc = [2.2112, 2.0548, 2.1090, 2.1668, 2.0869, 2.1634, 2.1167, 2.0557, 2.1429,
        2.1208, 1.9885, 2.1517, 2.0312, 2.0565, 1.9569, 1.9919, 2.1037, 2.0955,
        2.0956, 1.9860, 2.0367, 1.9292, 2.2735, 2.1028, 2.0092, 2.0884, 2.1538,
        2.0546, 2.0224, 2.4555]
    sic_slab_glo = [0.8886, 0.8798, 0.8576, 0.9118, 0.8675, 0.8538, 0.8888, 0.9716, 0.8795,
        0.8649, 1.1512, 0.9534, 1.1397, 1.1651, 1.0914, 1.1377, 1.1375, 1.0056,
        0.9709, 1.1091, 0.7757, 0.8290, 0.8202, 0.8158, 0.4723, 0.7603, 0.7566,
        0.7683, 0.8322, 0.7917]
    sic_slab_ml = [0.4653, 0.4086, 0.3743, 0.2009, 0.2710, 0.2190, 0.5757, 0.2657, 0.4627,
        0.4471, 0.2478, 0.2758, 0.4003, 0.2119, 0.3920, 0.2600, 0.2580, 0.3977,
        0.2054, 0.2904, 0.2928, 0.2707, 0.3633, 0.4180, 0.3078, 0.4916, 0.3923,
        0.3046, 0.7755, 0.3851]

    # SiC diamond vac
    sic_vac_pbc = [1.6702, 1.6091, 1.7474, 1.5162, 1.7866, 1.6251, 1.6589, 1.6387, 1.7198]
    sic_vac_glo = [0.3128, 0.3057, 0.3200, 0.3096, 0.3145, 0.3079, 0.3073, 0.3144, 0.3110]
    sic_vac_ml = [0.2128, 0.1723, 0.2798, 0.1683, 0.2210, 0.2098, 0.1675, 0.2088, 0.2366]

    fig, axs = plt.subplots(2, 1, figsize=(6, 6))  # sharex=True, sharey=True
    ind = np.arange(3)  # -> xx
    width = 0.15
    axs[0].yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axs[0].bar(ind + 1 * width,
               (np.mean(c_slab_pbc), np.mean(c_slab_glo), np.mean(c_slab_ml)),
               width, label="C: slab", edgecolor='red', alpha=0.8)
    axs[0].bar(ind + 2 * width,
               (np.mean(c_vac_pbc), np.mean(c_vac_glo), np.mean(c_vac_ml)),
               width, label="C: vacancy", edgecolor='g', alpha=0.8)
    axs[0].bar(ind + 3 * width,
               (np.mean(si_slab_pbc), np.mean(si_slab_glo), np.mean(si_slab_ml)),
               width, label="Si: slab", edgecolor='b', alpha=0.8)
    axs[0].errorbar(  # error dipole mio1, mio3
        ind + 1 * width,
        (np.mean(c_slab_pbc), np.mean(c_slab_glo), np.mean(c_slab_ml)),
        yerr=[
            [np.mean(c_slab_pbc) - min(c_slab_pbc), np.mean(c_slab_glo) - min(c_slab_glo), np.mean(c_slab_ml) - min(c_slab_ml)],
            [max(c_slab_pbc) - np.mean(c_slab_pbc), max(c_slab_glo) - np.mean(c_slab_glo), max(c_slab_ml) - np.mean(c_slab_ml)]],
        fmt="k.", capsize=4
    )
    axs[0].errorbar(  # error dipole mio1, mio3
        ind + 2 * width,
        (np.mean(c_vac_pbc), np.mean(c_vac_glo), np.mean(c_vac_ml)),
        yerr=[
            [np.mean(c_vac_pbc) - min(c_vac_pbc), np.mean(c_vac_glo) - min(c_vac_glo), np.mean(c_vac_ml) - min(c_vac_ml)],
            [max(c_vac_pbc) - np.mean(c_vac_pbc), max(c_vac_glo) - np.mean(c_vac_glo), max(c_vac_ml) - np.mean(c_vac_ml)]],
        fmt="k.", capsize=4
    )
    axs[0].errorbar(
        ind + 3 * width,
        (np.mean(si_slab_pbc), np.mean(si_slab_glo), np.mean(si_slab_ml)),
        yerr=[
            [np.mean(si_slab_pbc) - min(si_slab_pbc), np.mean(si_slab_glo) - min(si_slab_glo), np.mean(si_slab_ml) - min(si_slab_ml)],
            [max(si_slab_pbc) - np.mean(si_slab_pbc), max(si_slab_glo) - np.mean(si_slab_glo), max(si_slab_ml) - np.mean(si_slab_ml)]],
        fmt="k.", capsize=4
    )
    axs[0].set_xticks([])
    axs[0].set_ylabel("band structure MAEs (eV)")#, fontsize="large")
    axs[0].legend()#fontsize="large")
    axs[0].set_xticks([0.3, 1.3, 2.3], [r"pbc", r"global", "DFTB-ML"])#, fontsize="large")
    axs[0].text(0.1, 4.1, '(a)', fontsize="x-large")
    axs[0].set_ylim(0, 4.5)

    axs[1].yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axs[1].bar(ind + 1 * width,
               (np.mean(si_vac_pbc), np.mean(si_vac_glo), np.mean(si_vac_ml)),
               width, label="Si: vacancy", edgecolor='red', alpha=0.8)
    axs[1].bar(ind + 2 * width,
               (np.mean(sic_slab_pbc), np.mean(sic_slab_glo), np.mean(sic_slab_ml)),
               width, label="SiC: slab", edgecolor='b', alpha=0.8)
    axs[1].bar(ind + 3 * width,
               (np.mean(sic_vac_pbc), np.mean(sic_vac_glo), np.mean(sic_vac_ml)),
               width, label="SiC: vacancy", edgecolor='g', alpha=0.8)
    axs[1].errorbar(
        ind + 1 * width,
        (np.mean(si_vac_pbc), np.mean(si_vac_glo), np.mean(si_vac_ml)),
        yerr=[
            [np.mean(si_vac_pbc) - min(si_vac_pbc), np.mean(si_vac_glo) - min(si_vac_glo), np.mean(si_vac_ml) - min(si_vac_ml)],
            [max(si_vac_pbc) - np.mean(si_vac_pbc), max(si_vac_glo) - np.mean(si_vac_glo), max(si_vac_ml) - np.mean(si_vac_ml)]],
        fmt="k.", capsize=4
    )
    axs[1].errorbar(
        ind + 2 * width,
        (np.mean(sic_slab_pbc), np.mean(sic_slab_glo), np.mean(sic_slab_ml)),
        yerr=[
            [np.mean(sic_slab_pbc) - min(sic_slab_pbc), np.mean(sic_slab_glo) - min(sic_slab_glo), np.mean(sic_slab_ml) - min(sic_slab_ml)],
            [max(sic_slab_pbc) - np.mean(sic_slab_pbc), max(sic_slab_glo) - np.mean(sic_slab_glo), max(sic_slab_ml) - np.mean(sic_slab_ml)]],
        fmt="k.", capsize=4
    )
    axs[1].errorbar(  # error dipole mio1, mio3
        ind + 3 * width,
        (np.mean(sic_vac_pbc), np.mean(sic_vac_glo), np.mean(sic_vac_ml)),
        yerr=[
            [np.mean(sic_vac_pbc) - min(sic_vac_pbc), np.mean(sic_vac_glo) - min(sic_vac_glo), np.mean(sic_vac_ml) - min(sic_vac_ml)],
            [max(sic_vac_pbc) - np.mean(sic_vac_pbc), max(sic_vac_glo) - np.mean(sic_vac_glo), max(sic_vac_ml) - np.mean(sic_vac_ml)]],
        fmt="k.", capsize=4
    )
    axs[1].set_xticks([])
    axs[1].set_ylabel("band structure MAEs (eV)")#, fontsize="large")
    axs[1].legend()#fontsize="large")
    axs[1].set_xticks([0.3, 1.3, 2.3], [r"pbc", r"global", "DFTB-ML"])#, fontsize="large")
    axs[1].text(0.1, 2.65, '(b)', fontsize="x-large")
    axs[1].set_ylim(0., 2.9)

    plt.savefig('slabDefect.png', dpi=300)
    plt.show()


def plot_trans(params):
    gen_data = False
    if gen_data:
        files_train = [
            '../data/opt_c_bulk_diamond2_0.4.pkl',
            '../data/opt_c_bulk_hex2_0.4.pkl',
            '../data/opt_c_bulk_hex4_0.4.pkl',  # 0.6
            # '../data/opt_si_bulk_diamond2_0.4.pkl',
            # '../data/opt_si_bulk_hex4_0.4.pkl',
            # '../data/opt_si_bulk_tetrag4_0.4.pkl',
            # '../data/opt_sic_bulk_diamond2_0.4.pkl',
            ]
        dia_pred = {
            # 'c_bulk_diamond2': 0.2,
            'c_bulk_diamond64': 0.2,
            # 'c_vac_diamond': 0.2,
            # 'c_bulk_hex2': 0.2,
            # 'sic_bulk_diamond2': 0.2,
            # 'sic_bulk_cubic2': 0.2
            # 'si_bulk_diamond2': 0.2
            # 'si_bulk_hex4': 0.2,
            # 'si_bulk_tetrag4': 0.2,
        }
        skf_list = ['./slko/pbc/', './slko/']
        # for file in file_dia:
        mae0, mae1, error_list = _load_train_pkl(
            files_train, dia_pred, params, [0.35, 0.4, 0.6, 0.35, 0.4, 0.4, 0.4], skf_list=skf_list, plot_std=True)

        print(torch.cat([ii for ii in mae0], dim=0).mean(1))
        print('PBC', torch.cat([ii[0]['mae0'] for ii in error_list], dim=0).mean(1))
        print('global', torch.cat([ii[1]['mae0'] for ii in error_list], dim=0).mean(1))

        # 64 atoms of C diamond
        pbc = [3.6746, 3.6734, 3.7166, 3.6917, 3.7274, 3.7370, 3.7428, 3.7355, 3.7507, 3.7216]
        glob = [0.9995, 0.9990, 1.0018, 1.0017, 1.0049, 1.0024, 1.0063, 1.0069, 1.0099, 1.0048]
        ml = [0.3338, 0.3216, 0.3015, 0.3210, 0.3030, 0.2765, 0.2881, 0.2916, 0.3021, 0.3267]

        # defect errors
        pbc = [4.2852, 3.4955, 3.4894, 3.5045, 3.5559, 3.4729, 3.4862, 3.5995, 3.5411, 3.4819]
        glob = [1.2468, 1.0794, 1.0725, 1.0690, 1.1608, 1.0767, 1.0373, 1.1577, 1.0570, 1.0490]
        ml = [0.8441, 0.4755, 0.5927, 0.5607, 0.7026, 0.6492, 0.5326, 0.6977, 0.5743, 0.5936]

        # 64 atoms




def _load_train_pkl(pickle_files, pred_group, params: dict = {}, max_error=None,
                    skf_list: list = None, plot_std=False):
    idx_dict = {inum: i for i, inum in enumerate(shell_dict.keys())}
    idx_dict.update({(inum, jnum): i + j
                     for i, inum in enumerate(shell_dict.keys())
                     for j, jnum in enumerate(shell_dict.keys())})
    params.update({"n_band0": torch.tensor([int(ii * n_interval[0]) for ii in range(n_interval[1])])})
    params.update({"n_band1": torch.tensor([int(ii * n_interval[0] + 1) for ii in range(n_interval[1])])})
    params.update({"n_valence": 'all', "n_conduction": {6: 1, 14: 1},
                   "train_e_low": -30, "train_e_high": 20})

    # load optimized object
    scale, onsite, num_opt, pos_pe_opt, dist_pe_opt = {}, [], [], [], []
    geometry_opt = None

    error_dict = {}
    loaded_model = [pickle.load(open(file, 'rb')) for file in pickle_files]

    # for ii, model in enumerate(loaded_model):
    #     m_err = None if max_error is None else max_error[ii]
    #     for key, data in model.train_dict.items():

            # remove the largest error
            # err0 = model.train_dict[key]['loss_mae0'][-1].mean(1)
            # mask_err = err0.lt(m_err) if m_err is not None else torch.tensor([True]).repeat(len(err0))

            # dftb1_band = model.train_dict[key]['dftb1_band']
            # num_opt.append(dftb1_band.geometry.atomic_numbers[mask_err])
            # pos_pe_opt.append(dftb1_band.periodic.neighbour_pos[mask_err])
            # dist_pe_opt.append(dftb1_band.periodic.distances[mask_err])
            #
            # print('train model:', ii, geometry_opt, data.keys())
            # if geometry_opt is None and mask_err.any():
            #     geometry_opt = dftb1_band.geometry[mask_err]
            #     for i in range(0, 3):
            #         for j in range(i, 3):
            #             scale.update({(i, j): [data[(i, j)][mask_err]]})
            # elif mask_err.any():
            #     geometry_opt = geometry_opt + dftb1_band.geometry[mask_err]
            #     for i in range(0, 3):
            #         for j in range(i, 3):
            #             scale[(i, j)].append(data[(i, j)][mask_err])
            #
            # n_repeat = int(3 * num_opt[-1].shape[-1])  # 3 means s, p, d
            # onsite.append(dftb1_band.h_feed.on_site_dict["ml_onsite"][mask_err.repeat_interleave(n_repeat)])
    # basis_opt = Basis(geometry_opt.atomic_numbers, shell_dict)
    # num_opt = merge(num_opt)
    # pos_pe_opt = merge(pos_pe_opt)
    # dist_pe_opt = merge(dist_pe_opt)
    # onsite_opt = torch.cat(onsite).detach()
    # shells_per_atom_opt = basis_opt.shells_per_atom[basis_opt.shells_per_atom.ne(0)]
    # onsite_opt = pack(onsite_opt.split(tuple(shells_per_atom_opt)))

    for model in loaded_model:
        num_opt.extend(model.num_opt)
        pos_pe_opt.extend(model.pos_pe_opt)
        dist_pe_opt.extend(model.dist_pe_opt)
        onsite.extend(model.onsite)
        if geometry_opt is None:
            geometry_opt = model.geometry_opt
        else:
            geometry_opt = geometry_opt + model.geometry_opt

        for i in range(0, 3):
            for j in range(i, 3):
                if (i, j) in model.scale.keys():
                    scale[(i, j)] = model.scale[(i, j)]
                    print('dist_pe_opt', dist_pe_opt[-1].shape, scale[(i, j)][-1].shape)
                else:
                    scale.update({(i, j): scale[(i, j)].extend(model.scale[(i, j)])})

    basis_opt = Basis(geometry_opt.atomic_numbers, shell_dict)
    num_opt = merge(num_opt)
    pos_pe_opt = merge(pos_pe_opt)
    dist_pe_opt = merge(dist_pe_opt)
    onsite_opt = torch.cat(onsite).detach()
    shells_per_atom_opt = basis_opt.shells_per_atom[basis_opt.shells_per_atom.ne(0)]
    onsite_opt = pack(onsite_opt.split(tuple(shells_per_atom_opt)))


    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # build optimized features, one-body
    n_idx = len(shell_dict.keys())
    out_o11 = np.zeros((*num_opt.shape, n_idx))
    out_o12 = np.zeros((*num_opt.shape, n_idx))
    out_o14 = np.zeros((*num_opt.shape, len(idx_dict.keys()) - n_idx))
    out_o11, out_o12, out_o14 = g_pe(
        out_o11, out_o12, out_o14, geometry_opt.atomic_numbers.numpy(),
        geometry_opt.positions.numpy(), pos_pe_opt.numpy(),
        geometry_opt.n_atoms.numpy(),
        cutoff=10.0, eta=0.02, lamda=-1.0, zeta=1.0, idx_dict=idx_dict)

    # build optimized features, two-body
    out_o21 = np.zeros((*dist_pe_opt.shape, 1))
    out_o22 = np.zeros((*dist_pe_opt.shape, len(shell_dict.keys())))
    out_o23 = np.zeros((*dist_pe_opt.shape, len(shell_dict.keys())))
    out_o24 = np.zeros((*dist_pe_opt.shape, len(shell_dict.keys())))
    out_o21, out_o22, out_o23, out_o24 = g_pe_pair(
        out_o21, out_o22, out_o23, out_o24, geometry_opt.atomic_numbers.numpy(),
        geometry_opt.positions.numpy(), pos_pe_opt.numpy(),
        geometry_opt.n_atoms.numpy(),
        cutoff=10.0, eta=0.02, lamda=-1.0, zeta=1.0, idx_dict=idx_dict)

    x_train_1 = np.concatenate([out_o11, out_o12, out_o14], axis=-1)
    x_train = np.concatenate([out_o21, out_o22, out_o23, out_o24], axis=-1)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # update training params in optimized object, inverse is TRUE to avoid get
    # the same dataset in training
    test_dict, general_dict = Dataset.band(
        dataset, shell_dict, batch_size=params['ml']['batch_size'],
        cutoff=cutoff, groups=pred_group, logger=logger, inverse=True)

    mae0_list, mae1_list, err_list = [], [], []
    for ii, (key, data) in enumerate(test_dict.items()):
        geometry_pred = data['geometry']
        num_pred = geometry_pred.atomic_numbers
        kpoints = pack([Dataset.get_kpoints(cell, kpoint_level) for cell in geometry_pred.cell])
        dftb2_scc = Dftb2(geometry_pred, shell_dict, path_sk, skf_type="skf", kpoints=kpoints,
                          h_basis_type='normal', s_basis_type='normal', repulsive=False)
        data['dftb2_scc'] = dftb2_scc
        pe_pred = dftb2_scc.periodic
        dftb1_band = Dftb1(geometry_pred, shell_dict, path_to_skf=path_sk, skf_type="skf", klines=data['klines'],
                           h_basis_type='normal', s_basis_type='normal', repulsive=False, )
        dftb1_band.h_feed.gen_onsite(dftb1_band.geometry, dftb1_band.basis)
        data['dftb1_band'] = dftb1_band

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # build pred features, one-body
        n_idx = len(shell_dict.keys())
        out_p11 = np.zeros((*num_pred.shape, n_idx))
        out_p12 = np.zeros((*num_pred.shape, n_idx))
        out_p14 = np.zeros((*num_pred.shape, len(idx_dict.keys()) - n_idx))
        out_p11, out_p12, out_p14 = g_pe(
            out_p11, out_p12, out_p14, geometry_pred.atomic_numbers.numpy(),
            geometry_pred.positions.numpy(), pe_pred.neighbour_pos.numpy(),
            geometry_pred.n_atoms.numpy(),
            cutoff=10.0, eta=0.02, lamda=-1.0, zeta=1.0, idx_dict=idx_dict)

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # build pred features, two-body
        basis_pred = Basis(geometry_pred.atomic_numbers, shell_dict)
        out21 = np.zeros((*pe_pred.distances.shape, 1))
        out22 = np.zeros((*pe_pred.distances.shape, len(shell_dict.keys())))
        out23 = np.zeros((*pe_pred.distances.shape, len(shell_dict.keys())))
        out24 = np.zeros((*pe_pred.distances.shape, len(shell_dict.keys())))
        out21, out22, out23, out24 = g_pe_pair(
            out21, out22, out23, out24, geometry_pred.atomic_numbers.numpy(),
            geometry_pred.positions.numpy(), pe_pred.neighbour_pos.numpy(),
            geometry_pred.n_atoms.numpy(),
            cutoff=10.0, eta=0.02, lamda=-1.0, zeta=1.0, idx_dict=idx_dict)

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # train and pred scale and onsite
        x_pred_1 = np.concatenate([out_p11, out_p12, out_p14], axis=-1)
        x_pred = np.concatenate([out21, out22, out23, out24], axis=-1)
        anp_train = basis_opt.atomic_number_matrix("atomic")
        anp_pred = basis_pred.atomic_number_matrix("atomic")

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # One-center targets to be predicted
        if ml_method == 'rf':
            reg = RandomForestRegressor(n_estimators=100)
        elif ml_method == 'svm':
            reg = SVR()
        elif ml_method == 'nn':
            reg = MLPRegressor()

        onsite_pred = np.zeros((*num_pred.shape, 3))
        for ia in shell_dict.keys():
            mask_train_1 = num_opt == ia
            mask_pred_1 = num_pred == ia
            for i in range(3):
                y_train = onsite_opt[num_opt[num_opt.ne(0)] == ia][..., i]

                # predict onsite, make sure there is such atom and orbital in testing data
                if x_pred_1[mask_pred_1].shape[0] > 0 and x_pred_1[mask_pred_1].shape[0] > 0:
                    reg.fit(x_train_1[mask_train_1], y_train)

                    pred = reg.predict(x_pred_1[mask_pred_1])
                    onsite_pred[mask_pred_1, i] = pred

        data['ml_onsite'] = torch.from_numpy(onsite_pred).flatten()

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # Two-body targets predictions
        for i in range(3):
            for j in range(i, 3):

                y_pred_scale = np.zeros((x_pred.shape[:-1]))

                for ia in shell_dict.keys():
                    for ja in shell_dict.keys():
                        mask_train = (anp_train[..., 0] == ia) * (anp_train[..., 1] == ja)
                        mask_pred = (anp_pred[..., 0] == ia) * (anp_pred[..., 1] == ja)

                        y_train = merge(scale[(i, j)]).detach()
                        mask_opt = mask_train.unsqueeze(-1) * dist_pe_opt.lt(cutoff)
                        # print('mask_opt', mask_opt.shape, 'y_train', y_train.shape, )
                        mask_opt = mask_opt * y_train.ne(0.0) * y_train.ne(1.0)

                        mask_p = mask_pred.unsqueeze(-1) * pe_pred.distances.lt(cutoff)

                        if x_train[mask_opt].shape[0] == 0:
                            print(f'training data do not include {ia}-{ja}-{i}-{j}')
                        elif x_pred[mask_p].shape[0] == 0:
                            print(f'testing data do not include {ia}-{ja}-{i}-{j}')
                        else:
                            print('x_train[mask_opt], y_train[mask_opt]',
                                  x_train[mask_opt].shape, y_train[mask_opt].shape)
                            reg.fit(x_train[mask_opt], y_train[mask_opt])
                            y_pred = torch.from_numpy(reg.predict(x_pred[mask_p]))
                            y_pred_scale[mask_p] = y_pred

                data[(i, j)] = torch.from_numpy(y_pred_scale)

        mae0, mae1, err_listdict = Scale.pred(
            data, path_sk, params, shell_dict, elements, skf_type, alignment, param_dict['ml']['train_onsite'],
            train_1der, loss_fn, skf_list=skf_list, shell_dict_std=shell_dict_list,
            plot_std=plot_std
        )
        print(ii, mae0,)
        mae0_list.append(mae0)
        mae1_list.append(mae1)
        err_list.append(err_listdict)

    return mae0_list, mae1_list, err_list


if __name__ == '__main__':
    # plot_size(param_dict)
    # plot_bulk(param_dict)
    plot_vac_band()
    # plot_trans(param_dict)
    # plot_slabdefect(param_dict)
