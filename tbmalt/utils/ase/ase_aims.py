"""Interface to ASE."""
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import os
import torch
import numpy as np
import torch as t
from ase import Atoms
import subprocess
from ase.calculators.aims import Aims
DFTB_ENERGY = {"H": -0.238600544, "C": -1.398493891, "N": -2.0621839400,
               "O": -3.0861916005}
AIMS_ENERGY = {"H": -0.45891649, "C": -37.77330663, "N": -54.46973501,
               "O": -75.03140052}
core_charge = {"H": 0, "C": 2, "N": 2, "O": 2, "Si": 10}
HIRSH_VOL = {"H": 10.31539447, "C": 38.37861207, "N": 29.90025370,
             "O": 23.60491416}


class AseAims:
    """Run ASEAims will return FHI-aims with both batch or single calculations.

    Arguments:
        path_to_aims: Joint path and executable binary FHI-aims.
        aims_specie: path to species_defaults.
    """

    def __init__(self, path_to_aims, aims_specie, **kwargs):
        self.aims = path_to_aims
        self.aims_specie = aims_specie
        self.periodic = kwargs.get('periodic', False)
        if self.periodic:
            self.pbc = kwargs.get('pbc', [1, 1, 1])
            self.kpoint = kwargs.get('kpoint', '1 1 1')

        self.set_env()  # set environment before calculations

    def set_env(self):
        """Set the environment before DFTB calculations with ase."""
        # check if aims exists
        if not os.path.isfile(self.aims):
            raise FileNotFoundError('%s not found' % self.aims)

        # copy binary to current path
        os.system('cp ' + self.aims + ' ./aims.x')

        # get the current binary path and name
        path_bin = os.path.join(os.getcwd(), 'aims.x')
        self.aimsout = os.path.join(os.getcwd(), 'aims.out')

        # set ase environemt
        os.environ['ASE_AIMS_COMMAND'] = path_bin + ' > PREFIX.out'
        os.environ['AIMS_SPECIES_DIR'] = self.aims_specie

    def run_aims(self, positions, symbols, latvecs, properties):
        """Run batch systems with ASE-DFTB."""
        self.symbols = symbols
        self.latvecs = latvecs
        results = {}
        for iproperty in properties:
            results[iproperty] = []

        for iposition, isymbol, ilatvec in zip(positions, symbols, latvecs):
            # run each molecule in batches
            self.symbol, self.position,\
                self.latvec = isymbol, iposition, ilatvec
            self.nat = len(iposition)

            self.ase_iaims()

            # process each property: energy, homo lumo, eigenvect, dipole...
            for iproperty in properties:
                func = getattr(AseAims, iproperty)
                results[iproperty].append(func(self))

        self.remove()  # remove all FHI-aims related files

        return results

    def ase_iaims(self):
        """Build Aims input by ASE."""
        # set Atoms with molecule specie and coordinates
        if not self.periodic:
            mol = Atoms(self.symbol, positions=self.position)
        elif self.periodic:
            mol = Atoms(self.symbol, positions=self.position, cell=self.latvec,
                        pbc=self.pbc)

        cal = Aims(xc='PBE',
                   output=['mulliken', 'dos -18.  5.  500  0.1'],
                   dos_kgrid_factors='8 8 8',
                   sc_accuracy_etot=1e-6,
                   sc_accuracy_eev=1e-3,
                   sc_accuracy_rho=1e-6,
                   sc_accuracy_forces=1e-4,
                   many_body_dispersion=' ',
                   k_grid=self.kpoint,
                   command="ulimit -s unlimited; mpirun -np 4 ./aims.x > aims.out")

        # get calculators
        mol.calc = cal
        try:
            mol.get_potential_energy()
        except UnboundLocalError:
            mol.calc.__dict__["parameters"]['Options_WriteHS'] = 'No'
            mol.get_potential_energy()

    def homo_lumo(self):
        """Read homo and lumo."""
        commh = "grep 'Highest occupied state (VBM) at' " + \
            self.aimsout + " | tail -n 1 | awk '{print $6}'"
        self.homo = subprocess.check_output(commh, shell=True).decode('utf-8')
        comml = "grep 'Lowest unoccupied state (CBM) at' " + \
            self.aimsout + " | tail -n 1 | awk '{print $6}'"
        self.lumo = subprocess.check_output(comml, shell=True).decode('utf-8')
        return torch.from_numpy(np.asarray(
            [float(self.homo), float(self.lumo)]))

    def dipole(self):
        """Read dipole."""
        commdip = "grep 'Total dipole moment' "
        cdx = commdip + self.aimsout + " | awk '{print $7}'"
        cdy = commdip + self.aimsout + " | awk '{print $8}'"
        cdz = commdip + self.aimsout + " | awk '{print $9}'"
        idipx = float(subprocess.check_output(cdx, shell=True).decode('utf-8'))
        idipy = float(subprocess.check_output(cdy, shell=True).decode('utf-8'))
        idipz = float(subprocess.check_output(cdz, shell=True).decode('utf-8'))
        return torch.from_numpy(np.asarray([idipx, idipy, idipz]))

    def energy(self):
        """Get total energy."""
        comme = "grep 'Total energy                  :' " + self.aimsout + \
            " | tail -n 1 | awk '{print $5}'"
        return float(
            subprocess.check_output(comme, shell=True).decode('utf-8'))

    def formation_energy(self):
        """Return formation energy."""
        energy = self.energy()
        return self.get_formation_energy(energy, self.symbol)

    def alpha_mbd(self):
        """Read polarizability."""
        commp = "grep -A " + str(self.nat + 1) + \
            " 'C6 coefficients and polarizabilities' " + self.aimsout + \
                " | tail -n " + str(self.nat) + " | awk '{print $6}'"
        ipol = subprocess.check_output(commp, shell=True).decode('utf-8')
        return torch.from_numpy(np.asarray([float(i)
                                            for i in ipol.split('\n')[:-1]]))

    def charge(self):
        """Read charges."""
        commc = "grep -A " + str(self.nat) + \
            " 'atom       electrons          charge' " + self.aimsout + \
                " | tail -n " + str(self.nat) + " | awk '{print $3}'"
        icharge = subprocess.check_output(commc, shell=True).decode('utf-8')
        charge = torch.from_numpy(
            np.asarray([float(i) for i in icharge.split('\n')[:-1]]))
        return self.remove_core_charge(charge, self.symbol)

    def dos(self):
        """Read DOS."""
        lmf = lambda xx: list(map(float, xx.split()))
        file = open('./KS_DOS_total.dat', 'r').read()
        lines = file.split('\n')
        dos = lines[3:]
        return torch.tensor([lmf(dos[ii]) for ii in range(len(dos)-1)])

    def dos_gamma(self):
        """Read DOS."""
        lmf = lambda xx: list(map(float, xx.split()))
        file = open('./KS_DOS_total_gamma.dat', 'r').read()
        lines = file.split('\n')
        dos = lines[3:]
        return torch.tensor([lmf(dos[ii]) for ii in range(len(dos)-1)])

    def dos_raw(self):
        """Read DOS_raw."""
        lmf = lambda xx: list(map(float, xx.split()))
        file = open('./KS_DOS_total_raw.dat', 'r').read()
        lines = file.split('\n')
        dos = lines[3:]
        return torch.tensor([lmf(dos[ii]) for ii in range(len(dos)-1)])

    def dos_raw_gamma(self):
        """Read DOS_raw."""
        lmf = lambda xx: list(map(float, xx.split()))
        file = open('./KS_DOS_total_raw_gamma.dat', 'r').read()
        lines = file.split('\n')
        dos = lines[3:]
        return torch.tensor([lmf(dos[ii]) for ii in range(len(dos)-1)])

    def eigenvalue(self):
        """Read eigenvalues."""
        commh = "grep -A 642 'Writing Kohn-Sham eigenvalues.' " + \
            self.aimsout + " | tail -n 640 | awk '{print $4}'"
        eigval = subprocess.check_output(commh, shell=True).decode('utf-8').split()
        return torch.tensor([float(eigval[ii]) for ii in range(len(eigval))])

    def hirshfeld_volume(self):
        """Read Hirshfeld volume."""
        cvol = "grep 'Hirshfeld volume        :' " + self.aimsout + \
            " | awk '{print $5}'"
        ivol = subprocess.check_output(cvol, shell=True).decode('utf-8')
        return torch.from_numpy(
            np.asarray([float(i) for i in ivol.split('\n')[:-1]]))

    def hirshfeld_volume_ratio(self):
        """Read Hirshfeld volume ratio."""
        hirshfeld_volume = self.hirshfeld_volume()
        return torch.from_numpy(np.array(
            [hirshfeld_volume[ia] / HIRSH_VOL[self.symbol[ia]]
             for ia in range(self.nat)]))

    def get_formation_energy(self, energy, symbol):
        """Calculate formation energy."""
        return energy - sum([AIMS_ENERGY[symbol[ia]] for ia in range(self.nat)])

    def remove_core_charge(self, charge, symbol):
        """Calculate formation energy."""
        return torch.from_numpy(np.array([charge[ia] - core_charge[symbol[ia]]
                                          for ia in range(self.nat)]))

    def remove(self):
        """Remove all DFTB data after calculations."""
        os.system('rm aims.x aims.out control.in geometry.in')
        os.system('rm Mulliken.out parameters.ase')
        os.system('rm KS_DOS_total.dat KS_DOS_total_raw.dat')
