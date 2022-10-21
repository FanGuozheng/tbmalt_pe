import re
from typing import Union

import ase
import ase.io as io
import numpy as np
import matplotlib.pyplot as plt


class ToolKit:

    def __init__(self):
        pass

    def read_geometry(self):
        pass

    def write(self):
        pass

    @staticmethod
    def get_kpath(file: Union[object, str],
                  n_kpath: int = 10,
                  npoints: int = 10,
                  write_KPOINTS: bool = True):
        """"""
        if isinstance(file, ase.Atoms):
            _band = file.cell.bandpath(npoints=n_kpath)
        else:
            _obj = io.read(file)
            _band = _obj.cell.bandpath(npoints=n_kpath)

        _path = [i for i in _band.path.replace(',', '')]  # merge separate string
        path = []
        for ii, (item1, item2) in enumerate(zip(_path[:-1], _path[1:])):
            if not item1.isdigit() and item2.isdigit():
                path.append(item1 + item2)
            elif not item1.isdigit() and not item2.isdigit():
                path.append(item1)

        if not _path[-1].isdigit():
            path.append(_path[-1])
        else:
            path[-1] += _path[-1]

        if len(_band.kpts) == len(path):
            kpts = _band.kpts
            path = path

        return kpts, path

    @staticmethod
    def plot_band(data, xlabels, ymin=None, ymax=None, color='r', title=None):
        if ymin is None:
            ymin = min(data)
        if ymax is None:
            ymax = max(data)

        plt.plot(np.arange(len(data)), data, color=color)
        plt.ylim(ymin, ymax)

        if title is not None:
            plt.title(title)
        # plt.xticks(np.arange(xlabels + 1)[::10], xlabels)
        plt.show()
