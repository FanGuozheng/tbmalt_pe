from typing import Union, List
from itertools import islice
import xml.etree.ElementTree as ET

import numpy as np
import ase.io as io

from pymatgen.io.vasp.outputs import Vasprun
import matplotlib.pyplot as plt

from tbmalt import Geometry
from tbmalt.utils import ToolKit


class Vasp(ToolKit):
    """Python interface to Vienna Ab initio Simulation Package (VASP).

    This code aims to realize automatic and efficient interface to VASP.

    """

    def __init__(self, geometry: Geometry, format: str):
        self.ase_geometry = io.read(geometry, format=format)

    def write_input(self):
        pass

    def read_POSCAR(self, files: Union[str, List[str]]):
        pass

    def write_KPOINTS(self,
                      format='G',
                      n_kpath: int = None,
                      npoints: int = 10,
                      output='KPOINTS'
        ):
        if format == 'line':
            kpts, path = self.get_kpath(self.ase_geometry)
            with open(output, 'w') as fp:
                fp.write('k points along high symmetry lines\n')
                fp.write(str(npoints) + '\n')
                fp.write('line\n')
                fp.write('reciprocal\n')
                for k0, k1, p0, p1 in zip(kpts[:-1], kpts[1:], path[:-1], path[1:]):
                    fp.write(np.array2string(k0).replace('[', '').replace(']', '') + ' ' + p0 + '\n')
                    fp.write(np.array2string(k1).replace('[', '').replace(']', '') + ' ' + p1 + '\n\n')

    def write_POTCAR(self):
        pass

    def read_energy(self, plot=False):
        pass

    def read_band(self, files: Union[str, List[str]], plot=False, **kwargs):
        # vasprun = Vasprun(file, )
        if isinstance(files, str):
            if files.endswith('xml'):
                data = Vasp._single_read_band_xml(files)
            elif files.endswith('EIGENVAL'):
                data = Vasp._single_read_band_eigenvalue(files)
        elif isinstance(files, list):
            data = []
            for file in files:
                if files.endswith('xml'):
                    data.append(Vasp._single_read_band_xml(file))
                elif files.endswith('EIGENVAL'):
                    data.append(Vasp._single_read_band_eigenvalue(file))

        if plot:
            _, path = self.get_kpath(self.ase_geometry, n_kpath=10)
            self.plot_band(data, path, **kwargs)

        return data

    @staticmethod
    def _single_read_band_eigenvalue(file):
        with open(file, 'r') as fp:
            lines = fp.readlines()

            _, n_kpt, n_states = map(int, lines[5].split())

            data = []
            for ik in range(n_kpt):
                start = 8 + ik * (n_states + 2)
                data.append([float(i.split()[1]) for i in lines[start: start + n_states]])

            return np.array(data)
    @staticmethod
    def _single_read_band_xml(file):

        tree = ET.parse(file)
        root = tree.getroot()
        print(root)
        for item in root.findall('kpointlist'):
            name = item.get('name')
            quantity = item.text
            print(f"{name}: {quantity}")

        # Get the eigenvalues and k-points from the vasprun object
        eigenvalues = vasprun.eigenvalues
        kpoints = vasprun.actual_kpoints

        keys = eigenvalues.keys()
        print(eigenvalues[list(keys)[0]][..., 1].sum())
        # Flatten the eigenvalues array and convert to eV
        bands = eigenvalues[list(keys)[0]][..., 0]
        # bands = np.array([eigenvalues[spin][band_index][0] for spin in range(len(eigenvalues))
        #                   for band_index in
        #                   range(len(eigenvalues[list(keys)[0]][0]))])
        # bands *= 13.6058  # Convert to eV from Hartree
        num_bands = len(bands[0])
        num_kpoints = len(kpoints)
        x_ticks_labels = [f"{int(num_kpoints * i / 10)}" for i in range(11)]

        # Plot the band structure
        plt.figure(figsize=(8, 6))
        for i in range(num_bands):
            plt.plot(range(num_kpoints), bands[:, i], color='b')

        # Add vertical lines at high-symmetry k-points
        # high_symmetry_kpoints = vasprun.actual_kpoints.kpts
        # for k in high_symmetry_kpoints:
        #     plt.axvline(x=k[0] * num_kpoints, color='gray', linestyle='--')

        # Set x-axis ticks and labels
        plt.xticks(np.arange(0, num_kpoints, num_kpoints // 10), x_ticks_labels)
        plt.xlabel('K-points')
        plt.ylabel('Energy (eV)')
        plt.ylim(0, 8)
        plt.title('Band Structure')
        plt.tight_layout()
        plt.show()

    def read_band_dos(self):
        pass
