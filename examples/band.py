import re
import torch
import ase.io as io

from tbmalt import Geometry, Dftb1, Dftb2
from tbmalt.io.dataset import GeometryTo
from tbmalt.utils.plot.band import BandPlot

torch.set_default_dtype(torch.float64)



if __name__ == '__main__':
    # build_geo()
    geo = io.read('./geo.gen',)
    shell_dict = {8: [0, 1], 22: [0, 1, 2]}

    geometry = Geometry.from_ase_atoms([geo])
    kpoints2 = torch.tensor([5, 5, 5])
    kpt = geo.cell.bandpath(npoints=10)
    path = re.sub(r'[,]', '', kpt.path)
    k_list = []
    for i in range(len(path)):
        if i + 1 < len(path) and path[i + 1].isdigit():
            k_list.append(path[i] + path[i + 1])
        elif not path[i].isdigit():
            k_list.append(path[i])

    klines = torch.tensor([[0., 0., 0., -0.5,  0.5,  0., 10],
     [-0.5, 0.5, 0., -0.5, 0.5, -0.07654977, 10],
     [-0.5, 0.5, -0.07654977, -0.28827489, 0.28827489, -0.28827489, 10],
     [-0.28827489, 0.28827489, -0.28827489, 0., 0., 0., 10],
     [0., 0., 0., 0.5, 0.5, -0.5, 10],
     [0.5, 0.5, -0.5, 0.28827489,  0.71172511, -0.71172511, 10],
     [0.28827489, 0.71172511, -0.71172511, 0., 0.5, -0.5, 10],
     [0., 0.5, -0.5, -0.25, 0.75, -0.25, 10],
     [-0.25, 0.75, -0.25, 0.07654977, 0.92345023, -0.5, 10],
     [0.07654977, 0.92345023, -0.5, 0.5, 0.5, -0.5, 10],
     [0.5, 0.5, -0.5, -0.5, 0.5, 0., 10],
     [-0.5, 0.5, 0., -0.25, 0.75, -0.25, 10]])

    path_to_skf = '/Users/gz_fan/Downloads/software/dftbplus/dftbplus/work/dissertation/band/slko/'

    dftb_scc = Dftb2(geometry, shell_dict=shell_dict, path_to_skf=path_to_skf,
                     skf_type='skf', kpoints=kpoints2)
    dftb_scc()

    dftb_band = Dftb1(geometry, shell_dict=shell_dict, path_to_skf=path_to_skf,
                      skf_type='skf', klines=klines)
    dftb_band(charge=dftb_scc.charge)

    band = BandPlot(e_min=-6, e_max=4, alignment='fermi')
    print('dftb_band.kpoints', dftb_band.kpoints.shape, len(k_list))
    # xlabel = [k_list[i // 10] if i % 10 == 0 else '' for i, ik in enumerate(dftb_band.kpoints[0])]

    data1 = {
        'path': '/Users/gz_fan/Downloads/software/dftbplus/dftbplus/work/dissertation/band',
        'form': 'dftbplus',
        'properties': ['band'],
        'label': 'DFTB+',
        'band_xlabel': k_list,
    }
    data2 = {'path': {'band': dftb_band, 'scc': dftb_scc},
             'form': 'tbmalt', 'properties': ['band'], 'label': 'TBMaLT'}
    band(data1, data2)
         # form1='dftbplus', properties=['band', 'dos'],
         # path2={'band': dftb_band, 'scc': dftb_scc}, form2='tbmalt')
    print(dftb_band.eigenvalue[:, 0])
