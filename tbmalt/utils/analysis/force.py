"""Analysis of forces.."""
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import ase.io as io

from tbmalt import Geometry
from tbmalt.data.units import force_units


class Force:

    def __init__(self, path: str, form: str, path_geo: str = None):
        self.path = path
        self.form = form
        self.force, self.geometry = getattr(Force, form)(self, path, path_geo)

    @staticmethod
    def dftbplus_batch(files, n_atoms):
        forces, mask = [], []
        for file, n_atom in zip(files, n_atoms):
            # print(file, n_atom)
            force = Force.dftbplus(file, n_atom)
            if len(force) == n_atom:
                forces.append(force)
                mask.append(True)
            else:
                mask.append(False)

        return torch.stack(forces), torch.tensor(mask)


    @staticmethod
    def dftbplus(file, n_atom):
        """"""
        force = []

        with open(file) as f:

            for line in f:

                # Starting line of force
                if "Total Forces" in line:
                    is_force = True

                    for ii in range(n_atom):
                        next_line = next(f, None)
                        force.append([float(ii) for ii in next_line.split()[1:]])

        return torch.tensor(force)

    @staticmethod
    def aims(file: str, geometry: Geometry):
        """"""
        force = []

        # read force from output
        with open(file, 'r') as f:
            lines = f.readlines()
            for il, line in enumerate(lines):

                # Starting line of force
                if "Total atomic forces" in line:
                    force.append([[float(ii) * force_units['ev_angstrom']
                                   for ii in this_line.split()[-3:]]
                                  for this_line in lines[il + 1: il + 1 + geometry.n_atoms]])

        # if len(force) > len(geometry.atomic_numbers):
        #     force = force[-len(geometry.atomic_numbers):]

        return torch.tensor(force)

    def tbmalt(self):
        pass


if __name__ == '__main__':
    path_dftb = '/Users/gz_fan/Downloads/software/dftbplus/test/work/battery/lipscl/Li6PSCl/neb/dftb_elect/detailed.out.01'
    form = 'dftbplus'
    force_dftb = Force(path_dftb, form)

    path_aims = '/Users/gz_fan/Downloads/software/dftbplus/test/work/battery/lipscl/Li6PSCl/neb/dftb_elect/aims.out'
    form = 'aims'
    force_dftb = Force(path_aims, form)

