import torch
from ase import Atoms as AtomsAse
from dscribe.descriptors import CoulombMatrix, ACSF
from tbmalt.common.structures.system import System
from tbmalt.common.batch import pack
_val = [1, 2, 1, 2, 3, 4, 5, 6]
_U = [4.196174261214E-01, 0, 0, 0, 0, 3.646664973641E-01,
      4.308879578818E-01, 4.954041702122E-01]
# J. Chem. Phys. 41, 3199 (1964)
_atom_r_emp = [25, "He", "Li", "Be", "B", 70, 65, 60]
# J. Chem. Phys. 47, 1300 (1967)
_atom_r_cal = [53, "He", "Li", "Be", "B", 67, 56, 48]
# https://en.wikipedia.org/wiki/Molar_ionization_energies_of_the_elements
_ionization_energy = [1312.0, "He", "Li", "Be", "B", 1086.5, 1402.3, 1313.9]
# https://en.wikipedia.org/wiki/Electronegativity
_electronegativity = [2.20, "He", "Li", "Be", "B", 2.55, 3.04, 3.44]
# https://en.wikipedia.org/wiki/Electron_affinity
_electron_affinity = [73, "He", "Li", "Be", "B", 122, -7, 141]
_l_number = [0, "He", "Li", "Be", "B", 1, 1, 1]


class Dscribe:
    """Interface to Dscribe.

    Returns:
        features for machine learning

    """

    def __init__(self, atoms, **kwargs):
        self.atoms = atoms
        self.global_specie = self.atoms.get_global_species()[0]
        self.feature_type = kwargs.get('feature_type', 'acsf')

        self.features = self._get_features(**kwargs)
        self.features = self._staic_params(**kwargs)

    def _get_features(self, **kwargs):
        if self.feature_type == 'cm':
            return self._cm(**kwargs)
        elif self.feature_type == 'acsf':
            return self._acsf(**kwargs)
        elif self.feature_type == 'sine':
            return self._sine(**kwargs)
        elif self.feature_type == 'ewald':
            return self._ewald(**kwargs)
        elif self.feature_type == 'soap':
            return self._soap(**kwargs)
        elif self.feature_type == 'kernels':
            return self._kernels(**kwargs)

    def _staic_params(self, **kwargs):
        static_params = kwargs.get('static_parameter', [])
        if 'U' in static_params:
            _u = torch.cat([
                torch.tensor([_U[ii - 1] for ii in isys[isys.ne(0)]])
                for isys in self.atoms.numbers])
            self.features = torch.cat((self.features, _u.unsqueeze(1)), 1)
        if 'valence' in static_params:
            _v = torch.cat([
                torch.tensor([_val[ii - 1] for ii in isys[isys.ne(0)]])
                for isys in self.atoms.numbers])
            self.features = torch.cat((self.features, _v.unsqueeze(1)), 1)
        if 'atom_radii_emp' in static_params:
            _v = torch.cat([
                torch.tensor([_atom_r_emp[ii - 1] for ii in isys[isys.ne(0)]])
                for isys in self.atoms.numbers])
            self.features = torch.cat((self.features, _v.unsqueeze(1)), 1)
        if 'atom_radii_cal' in static_params:
            _v = torch.cat([
                torch.tensor([_atom_r_cal[ii - 1] for ii in isys[isys.ne(0)]])
                for isys in self.atoms.numbers])
            self.features = torch.cat((self.features, _v.unsqueeze(1)), 1)
        if 'ionization_energy' in static_params:
            _v = torch.cat([
                torch.tensor([_ionization_energy[ii - 1] for ii in isys[isys.ne(0)]])
                for isys in self.atoms.numbers])
            self.features = torch.cat((self.features, _v.unsqueeze(1)), 1)
        if 'electronegativity' in static_params:
            _v = torch.cat([
                torch.tensor([_electronegativity[ii - 1] for ii in isys[isys.ne(0)]])
                for isys in self.atoms.numbers])
            self.features = torch.cat((self.features, _v.unsqueeze(1)), 1)
        if 'electron_affinity' in static_params:
            _v = torch.cat([
                torch.tensor([_electron_affinity[ii - 1] for ii in isys[isys.ne(0)]])
                for isys in self.atoms.numbers])
            self.features = torch.cat((self.features, _v.unsqueeze(1)), 1)
        if 'l_number' in static_params:
            _v = torch.cat([
                torch.tensor([_l_number[ii - 1] for ii in isys[isys.ne(0)]])
                for isys in self.atoms.numbers])
            self.features = torch.cat((self.features, _v.unsqueeze(1)), 1)

        return self.features

    def _cm(self, rcut=6.0, nmax=8, lmax=6, n_atoms_max_=20):
        """Coulomb method for atomic environment.

        Phys. Rev. Lett., 108:058301, Jan 2012.
        """
        cm = CoulombMatrix(n_atoms_max=n_atoms_max_)
        positions = self.atoms.positions
        atom = Atoms(self.global_specie, positions=positions)
        cm_test = cm.create(atom)
        return torch.from_numpy(cm_test)

    def _sine(self):
        pass

    def _ewald(self):
        pass

    def _acsf(self, **kwargs):
        """Atom-centered Symmetry Functions method for atomic environment.

        J. chem. phys., 134.7 (2011): 074106.
        You should define all the atom species to fix the feature dimension!
        """
        species = Atoms.to_element(self.atoms.numbers)
        dtype = torch.get_default_dtype()

        g1 = kwargs.get('G1', 8)  # 6
        g2 = kwargs.get('G2', [[2., 1.5]])  # 1, 1
        g4 = kwargs.get('G4', [[0.02, 1., 1.]])  # 0.02, 1., -1.
        acsf = ACSF(species=self.global_specie,
                    rcut=g1,
                    g2_params=g2,
                    g4_params=g4)

        return torch.cat([
            torch.tensor(acsf.create(AtomsAse(ispe, ipos[inum.ne(0)])), dtype=dtype)
            for ispe, inum, ipos in
            zip(species, self.atoms.numbers, self.atoms.positions.numpy())])

    def _soap(self):
        pass

    def _manybody(self):
        pass

    def _kernels(self):
        pass


def _get_acsf_dim(specie_global, **kwargs):
    """Get the dimension (column) of ACSF method."""
    g2 = kwargs.get('G2', [1., 1.])
    g4 = kwargs.get('G4', [0.02, 1., -1.])

    nspecie = len(specie_global)
    col = 0
    if nspecie == 1:
        n_types, n_type_pairs = 1, 1
    elif nspecie == 2:
        n_types, n_type_pairs = 2, 3
    elif nspecie == 3:
        n_types, n_type_pairs = 3, 6
    elif nspecie == 4:
        n_types, n_type_pairs = 4, 10
    elif nspecie == 5:
        n_types, n_type_pairs = 5, 15
    col += n_types  # G0
    if g2 is not None:
        col += len(g2) * n_types  # G2
    if g4 is not None:
        col += (len(g4)) * n_type_pairs  # G4
    return col
