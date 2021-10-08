"""Interface to ASE."""
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import os
import numpy as np
import torch
from ase import Atoms
from ase.calculators.dftb import Dftb
ATOMNUM = {'H': 1, 'C': 6, 'N': 7, 'O': 8}
DFTB_ENERGY = {"H": -0.238600544, "C": -1.398493891, "N": -2.0621839400,
               "O": -3.0861916005}
AIMS_ENERGY = {"H": -0.45891649, "C": -37.77330663, "N": -54.46973501,
               "O": -75.03140052}
_AUEV = 27.2113845


class AseDftb:
    """Run DFTB+, return results, write into hdf5."""

    def __init__(self, path_to_dftbplus, path_to_skf, properties, **kwargs):
        """Initialize parameters.

        Args:
            dftbplus: binary executable DFTB+
            directory_sk: path to SKF files

        """
        # path to dftb+
        self.dftb = path_to_dftbplus
        self.dftb_type = kwargs.get('dftb_type', 'scc')

        # SKF path
        self.slko_path = path_to_skf

        # set environment before calculations
        self.set_env()

    def set_env(self):
        """Set the environment before DFTB calculations with ase."""
        # check if skf dataset exists
        if not os.path.isfile(self.dftb):
            raise FileNotFoundError('%s not found' % self.dftb)

        # copy binary to current path
        os.system('cp ' + self.dftb + ' ./dftb+')

        # get the current binary path and name
        path_bin = os.path.join(os.getcwd(), 'dftb+')

        # set ase environemt
        os.environ['ASE_DFTB_COMMAND'] = path_bin + ' > PREFIX.out'
        os.environ['DFTB_PREFIX'] = self.slko_path

    def run_dftb(self, positions, symbols, latvecs, properties):
        """Run batch systems with ASE-DFTB."""
        self.symbols = symbols
        self.latvecs = latvecs
        results = {}
        for iproperty in properties:
            results[iproperty] = []

        if self.latvecs is None:
            for iposition, isymbol in zip(positions, symbols):
                # run each molecule in batches
                self.position, self.symbol = iposition, isymbol
                self.nat = len(iposition)

                fun_dftb = getattr(AseDftb, self.dftb_type)
                fun_dftb(self)

                # process each result (overmat, eigenvalue, eigenvect, dipole)
                for iproperty in properties:
                    func = getattr(AseDftb, iproperty)
                    self.symbol = isymbol
                    results[iproperty].append(func(self))
        else:
            for iposition, isymbol, ilatvec in zip(positions, symbols, latvecs):
                # run each molecule in batches
                self.position, self.symbol,\
                    self.latvec = iposition, isymbol, ilatvec
                self.nat = len(iposition)

                fun_dftb = getattr(AseDftb, self.dftb_type)
                fun_dftb(self)

                # process each result (overmat, eigenvalue, eigenvect, dipole)
                for iproperty in properties:
                    func = getattr(AseDftb, iproperty)
                    self.symbol = isymbol
                    results[iproperty].append(func(self))

        self.remove()  # remove all the DFTB+ related files

        return results

    def nonscc(self):
        """Build DFTB input by ASE."""
        # set Atoms with molecule specie and coordinates
        mol = Atoms(self.symbol, positions=self.position)

        # set DFTB caulation parameters
        cal = Dftb(Hamiltonian_='DFTB',
                   Hamiltonian_SCC='no',
                   Hamiltonian_MaxAngularMomentum_='',
                   Hamiltonian_MaxAngularMomentum_H='s',
                   Hamiltonian_MaxAngularMomentum_C='p',
                   Hamiltonian_MaxAngularMomentum_N='p',
                   Hamiltonian_MaxAngularMomentum_O='p',
                   Options_='',
                   Analysis_='',
                   Analysis_MullikenAnalysis='Yes',
                   Analysis_WriteEigenvectors='Yes',
                   Analysis_EigenvectorsAsText='Yes',
                   ParserOptions_='',
                   ParserOptions_IgnoreUnprocessedNodes='Yes')

        # get calculators
        mol.calc = cal
        mol.get_potential_energy()

    def scc(self):
        """Build DFTB input by ASE."""
        # set Atoms with molecule specie and coordinates
        mol = Atoms(self.symbol, positions=self.position)

        # set DFTB caulation parameters
        cal = Dftb(Hamiltonian_='DFTB',
                   Hamiltonian_SCC='yes',
                   Hamiltonian_SCCTolerance=1e-8,
                   Hamiltonian_MaxAngularMomentum_='',
                   Hamiltonian_MaxAngularMomentum_H='s',
                   Hamiltonian_MaxAngularMomentum_C='p',
                   Hamiltonian_MaxAngularMomentum_N='p',
                   Hamiltonian_MaxAngularMomentum_O='p',
                   Hamiltonian_MaxAngularMomentum_Si='p',
                   Options_='',
                   Analysis_='',
                   Analysis_MullikenAnalysis='Yes',
                   Analysis_WriteEigenvectors='Yes',
                   Analysis_EigenvectorsAsText='Yes',
                   ParserOptions_='',
                   ParserOptions_IgnoreUnprocessedNodes='Yes')

        # get calculators
        mol.calc = cal
        mol.get_potential_energy()

    def scc_pbc(self):
        """Build DFTB input by ASE."""
        # set Atoms with molecule specie and coordinates
        mol = Atoms(self.symbol, positions=self.position, pbc=True,
                    cell=[5.0, 5.0, 5.0])

        # set DFTB caulation parameters
        cal = Dftb(Hamiltonian_='DFTB',
                   Hamiltonian_SCC='yes',
                   Hamiltonian_SCCTolerance=1e-8,
                   Hamiltonian_MaxAngularMomentum_='',
                   Hamiltonian_MaxAngularMomentum_H='s',
                   Hamiltonian_MaxAngularMomentum_C='p',
                   Hamiltonian_MaxAngularMomentum_N='p',
                   Hamiltonian_MaxAngularMomentum_O='p',
                   Hamiltonian_MaxAngularMomentum_Si='p',
                   Options_='',
                   Analysis_='',
                   Analysis_MullikenAnalysis='Yes',
                   Analysis_WriteEigenvectors='Yes',
                   Analysis_EigenvectorsAsText='Yes',
                   ParserOptions_='',
                   ParserOptions_IgnoreUnprocessedNodes='Yes',
                   kpts=(1, 1, 1))

        # get calculators
        mol.calc = cal
        mol.get_potential_energy()

    def scc_si_pbc(self):
        """Build DFTB input by ASE."""
        # set Atoms with molecule specie and coordinates
        mol = Atoms(self.symbol, positions=self.position, pbc=True,
                    cell=self.latvec)

        # set DFTB caulation parameters
        cal = Dftb(Hamiltonian_='DFTB',
                   Hamiltonian_SCC='yes',
                   Hamiltonian_SCCTolerance=1e-8,
                   Hamiltonian_MaxAngularMomentum_='',
                   Hamiltonian_MaxAngularMomentum_H='s',
                   Hamiltonian_MaxAngularMomentum_C='p',
                   Hamiltonian_MaxAngularMomentum_N='p',
                   Hamiltonian_MaxAngularMomentum_O='p',
                   Hamiltonian_MaxAngularMomentum_Si='p',
                   Options_='',
                   Analysis_='',
                   Analysis_MullikenAnalysis='Yes',
                   Analysis_WriteEigenvectors='Yes',
                   Analysis_EigenvectorsAsText='Yes',
                   ParserOptions_='',
                   ParserOptions_IgnoreUnprocessedNodes='Yes',
                   kpts=(1, 1, 1))

        # get calculators
        mol.calc = cal
        mol.get_potential_energy()

    def mbd_scc(self):
        """Build DFTB input by ASE."""
        # set Atoms with molecule specie and coordinates
        mol = Atoms(self.symbol, positions=self.positions)

        # set DFTB caulation parameters
        cal = Dftb(Hamiltonian_='DFTB',
                   Hamiltonian_SCC='yes',
                   Hamiltonian_SCCTolerance=1e-8,
                   Hamiltonian_MaxAngularMomentum_='',
                   Hamiltonian_MaxAngularMomentum_H='s',
                   Hamiltonian_MaxAngularMomentum_C='p',
                   Hamiltonian_MaxAngularMomentum_N='p',
                   Hamiltonian_MaxAngularMomentum_O='p',
                   Hamiltonian_Dispersion_ReferenceSet='ts',
                   Hamiltonian_Dispersion_NOmegaGrid=15,
                   Hamiltonian_Dispersion_Beta=1.05,
                   Options_='',
                   Options_WriteHS='Yes',
                   Analysis_='',
                   Analysis_MullikenAnalysis='Yes',
                   Analysis_WriteEigenvectors='Yes',
                   Analysis_EigenvectorsAsText='Yes',
                   ParserOptions_='',
                   ParserOptions_IgnoreUnprocessedNodes='Yes')

        # get calculators
        mol.calc = cal
        try:
            mol.get_potential_energy()
        except UnboundLocalError:
            mol.calc.__dict__["parameters"]['Options_WriteHS'] = 'No'
            mol.get_potential_energy()

    def eigenvalue(self, band_file='band.out'):
        """Read and return eigenvalue."""
        return get_eigenvalue(band_file)[0]

    def occupancy(self, band_file='band.out'):
        """Read and return homo and lumo."""
        return get_eigenvalue(band_file)[1]

    def homo_lumo(self, band_file='band.out'):
        """Read and return homo and lumo."""
        return get_eigenvalue(band_file)[2]

    def energy(self):
        """Read and return total energy."""
        return read_detailed_out(self.nat)[0]

    def formation_energy(self):
        """Read and return formation energy."""
        energy = self.energy()
        return self.get_formation_energy(energy)

    def charge(self):
        """Read charges."""
        return read_detailed_out(self.nat)[1]

    # def dipole(self):
    #     """Read dipole."""
    #     return read_detailed_out(self.nat)[2]

    def hamiltonian(self, hamiltonian_name='hamsqr1.dat'):
        """Read and return Hamiltonian."""
        return get_matrix(hamiltonian_name)

    def overlap(self, overlap_name='oversqr.dat'):
        """Reand and return overlap."""
        return get_matrix(overlap_name)

    def eigenvec(self, eigenvec_name='eigenvec.out'):
        """Reand and return eigenvector."""
        return get_matrix(eigenvec_name)

    def get_formation_energy(self, energy):
        """Calculate formation energy."""
        return energy - sum([DFTB_ENERGY[self.symbol[iat]]
                             for iat in range(self.nat)])

    def remove(self):
        """Remove all DFTB data after calculations."""
        os.system('rm dftb+ band.out charges.bin detailed.out')
        os.system('rm dftb_in.hsd dftb.out dftb_pin.hsd eigenvec.bin')
        os.system('rm eigenvec.out geo_end.gen hamsqr1.dat oversqr.dat')


def get_matrix(filename):
    """Read DFTB+ hamsqr1.dat and oversqr.dat."""
    text = ''.join(open(filename, 'r').readlines())
    string = re.search('(?<=MATRIX\n).+(?=\n)', text, flags=re.DOTALL).group(0)
    out = np.array([[float(i) for i in row.split()] for row in string.split('\n')])
    return torch.from_numpy(out)


def get_eigenvec(filename):
    """Read DFTB+ eigenvec.out."""
    string = []
    text = ''.join(open(filename, 'r').readlines())

    # only read float
    string_ = re.findall(r"[-+]?\d*\.\d+", text)

    # delete even column
    del string_[1::2]
    [string.append(float(ii)) for ii in string_]
    nstr = int(np.sqrt(len(string)))

    # transfer list to ==> numpy(float64) ==> torch
    eigenvec = np.asarray(string).reshape(nstr, nstr)
    return torch.from_numpy(eigenvec).T


def get_eigenvalue(filename):
    """Read DFTB+ band.out."""
    text = ''.join(open(filename, 'r').readlines())

    # only read float
    string = re.findall(r"[-+]?\d*\.\d+", text)

    # delete even column
    eigenval_ = [float(ii) for ii in string[1::2]]
    occ_ = [float(ii) for ii in string[0::2]][1:]  # remove the first value

    # transfer list to ==> numpy(float64) ==> torch
    eigenval = torch.from_numpy(np.asarray(eigenval_))
    occ = torch.from_numpy(np.asarray(occ_))
    humo_lumo = np.asarray([eigenval[np.where(occ != 0)[0]][-1],
                           eigenval[np.where(occ == 0)[0]][0]])
    return eigenval, occ, humo_lumo


def read_detailed_out(natom):
    """Read DFTB+ output file detailed.out."""
    charge, dipole = [], []
    text = ''.join(open('detailed.out', 'r').readlines())
    E_tot_ = re.search('(?<=Total energy:).+(?=\n)',
                       text, flags=re.DOTALL | re.MULTILINE).group(0)
    E_tot = re.findall(r"[-+]?\d*\.\d+", E_tot_)[0]

    # read charge
    text2 = re.search('(?<=Atom       Population\n).+(?=\n)',
                      text, flags=re.DOTALL | re.MULTILINE).group(0)
    qatom_ = re.findall(r"[-+]?\d*\.\d+", text2)[:natom]
    [charge.append(float(ii)) for ii in qatom_]

    return float(E_tot), torch.from_numpy(np.asarray(charge))

    # read dipole (Debye)
    # text3 = re.search('(?<=Dipole moment:).+(?=\n)',
    #                   text, flags=re.DOTALL | re.MULTILINE).group(0)
    # # if tail is [-3::], read Debye dipole, [:3] will read au dipole
    # dip_ = re.findall(r"[-+]?\d*\.\d+", text3)[:3]
    # [dipole.append(float(ii)) for ii in dip_]

    # return float(E_tot), torch.from_numpy(np.asarray(charge)), \
    #     torch.from_numpy(np.asarray(dipole))
