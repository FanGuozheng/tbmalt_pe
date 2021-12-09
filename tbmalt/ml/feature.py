import torch
import numpy as np
from ase import Atoms as Atoms
from dscribe.descriptors import CoulombMatrix, ACSF, SOAP
from tbmalt.structures.geometry import Geometry
from tbmalt.data import chemical_symbols
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
# https://en.wikipedia.org/wiki/electronnegativity
_electronnegativity = [2.20, "He", "Li", "Be", "B", 2.55, 3.04, 3.44]
# https://en.wikipedia.org/wiki/Electron_affinity
_electron_affinity = [73, "He", "Li", "Be", "B", 122, -7, 141]
_l_number = [0, "He", "Li", "Be", "B", 1, 1, 1]
_homo = [-6.493, 'He', 'Li', 'Be', 'B', -5.289, -7.095, -9.038]


class Dscribe:
    """Interface to Dscribe.

    Returns:
        features for machine learning

    """

    def __init__(self, geometry, **kwargs):
        self.geometry = geometry
        self.global_specie = self.geometry.unique_atomic_numbers()
        self.feature_type = kwargs.get('feature_type', 'acsf')
        self.periodic = self.geometry.isperiodic
        geo_feature = kwargs.get('geo_feature', False)

        self._num = self.geometry.atomic_numbers.repeat_interleave(
            self.geometry.atomic_numbers.shape[1], 0)[self.geometry.atomic_numbers.ne(0).flatten()]
        self.features = getattr(Dscribe, self.feature_type)(self, **kwargs)
        self.features = self._staic_params(**kwargs)

    def _staic_params(self, **kwargs):
        static_params = kwargs.get('static_parameter', [])
        dist_decay = kwargs.get('dist_decay', [False] * len(static_params))
        self.specie_res = kwargs.get('specie_res', False)
        assert len(static_params) == len(dist_decay)
        for ipara, idecay in zip(static_params, dist_decay):
            getattr(Dscribe, ipara)(self, idecay)
        return self.features

    def U(self, idecay=False):
        if not idecay:
            _u = torch.cat([torch.tensor([_U[ii - 1] for ii in isys[isys.ne(0)]])
                            for isys in self.geometry.atomic_numbers])
            self.features = torch.cat((self.features, _u.unsqueeze(1)), 1)
        else:
            _v = self._distanceU()
            self.features = torch.cat([self.features, _v], 1)

    def valence(self, idecay=False):
        if not idecay:
            _v = torch.cat([torch.tensor([_val[ii - 1] for ii in isys[isys.ne(0)]])
                            for isys in self.geometry.atomic_numbers])
            self.features = torch.cat((self.features, _v.unsqueeze(1)), 1)
        else:
            _v = self._distancevalence()
            self.features = torch.cat([self.features, _v], 1)

    def atom_radii_emp(self, idecay=False):
        if not idecay:
            _v = torch.cat([torch.tensor([_atom_r_emp[ii - 1] for ii in isys[isys.ne(0)]])
                            for isys in self.geometry.atomic_numbers])
            self.features = torch.cat((self.features, _v.unsqueeze(1)), 1)
        else:
            _v = self._distanceremp()
            self.features = torch.cat([self.features, _v], 1)

    def atom_radii_cal(self, idecay=False):
        if not idecay:
            _v = torch.cat([torch.tensor([_atom_r_cal[ii - 1] for ii in isys[isys.ne(0)]])
                            for isys in self.geometry.atomic_numbers])
            self.features = torch.cat((self.features, _v.unsqueeze(1)), 1)
        else:
            _v = self._distancercal()
            self.features = torch.cat([self.features, _v], 1)

    def ionization_energy(self, idecay=False):
        if not idecay:
            _v = torch.cat([torch.tensor([_ionization_energy[ii - 1] for ii in isys[isys.ne(0)]])
                            for isys in self.geometry.atomic_numbers])
            self.features = torch.cat((self.features, _v.unsqueeze(1)), 1)
        else:
            _v = self._distanceion()
            self.features = torch.cat([self.features, _v], 1)

    def electronnegativity(self, idecay=False):
        if not idecay:
            _v = torch.cat([torch.tensor([_electronnegativity[ii - 1] for ii in isys[isys.ne(0)]])
                            for isys in self.geometry.atomic_numbers])
            self.features = torch.cat((self.features, _v.unsqueeze(1)), 1)
        else:
            _v = self._distanceelectronnegativity()
            self.features = torch.cat([self.features, _v], 1)

    def electron_affinity(self, idecay=False):
        if not idecay:
            _v = torch.cat([torch.tensor([_electron_affinity[ii - 1] for ii in isys[isys.ne(0)]])
                            for isys in self.geometry.atomic_numbers])
            self.features = torch.cat((self.features, _v.unsqueeze(1)), 1)
        else:
            _v = self._distanceaff()
            self.features = torch.cat([self.features, _v], 1)

    def l_number(self, idecay=False):
        if not idecay:
            _v = torch.cat([torch.tensor([_l_number[ii - 1] for ii in isys[isys.ne(0)]])
                            for isys in self.geometry.atomic_numbers])
            self.features = torch.cat((self.features, _v.unsqueeze(1)), 1)
        else:
            _v = self._distanceln()
            self.features = torch.cat([self.features, _v], 1)

    def homo(self, idecay=False):
        if not idecay:
            _v = torch.cat([torch.tensor([_homo[ii - 1] for ii in isys[isys.ne(0)]])
                            for isys in self.geometry.atomic_numbers])
            self.features = torch.cat((self.features, _v.unsqueeze(1)), 1)
        else:
            _v = self._distancehomo()
            self.features = torch.cat([self.features, _v], 1)


    def cm(self, **kwargs):
        """Coulomb method for atomic environment.

        Phys. Rev. Lett., 108:058301, Jan 2012.
        """
        rcut, nmax, lmax, n_atoms_max_ = 6.0, 8, 6, 20
        species = Geometry.to_element(self.geometry.numbers)
        dtype = torch.get_default_dtype()
        cm = CoulombMatrix(n_atoms_max=n_atoms_max_)
        # positions = self.geometry.positions
        # atom = Atoms(self.global_specie, positions=positions)
        # cm_test = cm.create(atom)
        _cm = torch.cat([
            torch.tensor(cm.create(Atoms(ispe, ipos[inum.ne(0)])),
                         dtype=dtype).reshape(n_atoms_max_, n_atoms_max_)[:len(ispe)]
            for ispe, inum, ipos in
            zip(species, self.geometry.numbers, self.geometry.positions.numpy())])
        return _cm

    def sine(self):
        pass

    def ewald(self):
        pass

    def acsf(self, **kwargs):
        """Atom-centered Symmetry Functions method for atomic environment.

        J. chem. phys., 134.7 (2011): 074106.
        You should define all the atom species to fix the feature dimension!
        """
        # species = Geometry.to_element(self.geometry.numbers)
        uat = self.geometry.unique_atomic_numbers()
        species = [chemical_symbols[iat] for iat in uat]
        dtype = torch.get_default_dtype()
        g1 = kwargs.get('G1', 6)
        g2 = kwargs.get('G2', [[1.0, 1.0]])
        g4 = kwargs.get('G4', [[0.02, 1.0, -1.0]])
        acsf = ACSF(species=species,
                    rcut=g1,
                    g2_params=g2,
                    g4_params=g4,
                    periodic=self.periodic
                    )

        _acsf = torch.cat([
            torch.tensor(acsf.create(Atoms(ispe, ipos[inum.ne(0)])), dtype=dtype)
            for ispe, inum, ipos in
            zip(self.geometry.chemical_symbols, self.geometry.atomic_numbers,
                self.geometry.positions.numpy())])

        # return torch.sum(_acsf, -1).reshape(-1, 1)
        return _acsf

    def soap(self, **kwargs):
        """Smooth overlap of atomic positions (SOAP)."""
        uat = self.geometry.unique_atomic_numbers()
        species = [chemical_symbols[iat] for iat in uat]
        dtype = torch.get_default_dtype()
        soap = SOAP(species=species,
                    periodic=False,
                    rcut=5,
                    nmax=8,
                    lmax=8,
                    # average=True,
                    sparse=False)

        _soap = torch.cat([
            torch.tensor(soap.create(Atoms(ispe, ipos[inum.ne(0)])), dtype=dtype)
            for ispe, inum, ipos in
            zip(self.geometry.chemical_symbols, self.geometry.atomic_numbers,
                self.geometry.positions.numpy())])

        # _soap = torch.cat([
        #     torch.tensor(soap.create(Atoms(ispe, ipos[inum.ne(0)])), dtype=dtype)
        #     for ispe, inum, ipos in
        #     zip(species, self.geometry.numbers, self.geometry.positions.numpy())])
        return _soap

    def manybody(self):
        pass

    def kernels(self):
        pass

    def _distanceU(self, cutoff=6.0, **kwargs):
        """Build electron negativity features."""
        _feature = torch.zeros(*self.geometry.distances.shape, 4)
        feature = torch.zeros(*self.geometry.distances.shape)

        hubu = []
        for inum, ipos in zip(self.geometry.numbers, self.geometry.distances):
            _neg = torch.tensor([_U[ii - 1] for ii in inum[inum.ne(0)]])
            hubu.append(_neg.unsqueeze(1) - _neg.unsqueeze(0))
        hubu = pack(hubu)
        mask_dist = self.geometry.distances.lt(cutoff)
        feature[mask_dist] = 0.5 * (torch.cos(
            np.pi * self.geometry.distances[mask_dist] / cutoff) + 1) * hubu[mask_dist]

        if self.specie_res:
            _fea = feature.flatten(end_dim=1)[self.geometry.numbers.ne(0).flatten()]
            _feature = torch.zeros(_fea.shape[0], 4)
            _feature[..., 0] = _fea.masked_fill(self._num != 1 , 0).sum(-1)
            _feature[..., 1] = _fea.masked_fill(self._num != 6 , 0).sum(-1)
            _feature[..., 2] = _fea.masked_fill(self._num != 7 , 0).sum(-1)
            _feature[..., 3] = _fea.masked_fill(self._num != 8 , 0).sum(-1)

            return _feature  #_feature.sum(-2)[mask0]
        else:
            return feature.flatten(end_dim=1)[
                self.geometry.numbers.ne(0).flatten()].sum(-1).unsqueeze(1)


    def _distancevalence(self, cutoff=6.0, **kwargs):
        """Build electron negativity features."""
        _feature = torch.zeros(*self.geometry.distances.shape, 4)
        feature = torch.zeros(*self.geometry.distances.shape)

        val = []
        for inum, ipos in zip(self.geometry.atomic_numbers, self.geometry.distances):
            _neg = torch.tensor([_val[ii - 1] for ii in inum[inum.ne(0)]])
            val.append(_neg.unsqueeze(1) - _neg.unsqueeze(0))
        val = pack(val)
        mask_dist = self.geometry.distances.lt(cutoff)
        feature[mask_dist] = 0.5 * (torch.cos(
            np.pi * self.geometry.distances[mask_dist] / cutoff) + 1) * val[mask_dist]

        if self.specie_res:
            _fea = feature.flatten(end_dim=1)[self.geometry.numbers.ne(0).flatten()]
            _feature = torch.zeros(_fea.shape[0], 4)
            _feature[..., 0] = _fea.masked_fill(self._num != 1 , 0).sum(-1)
            _feature[..., 1] = _fea.masked_fill(self._num != 6 , 0).sum(-1)
            _feature[..., 2] = _fea.masked_fill(self._num != 7 , 0).sum(-1)
            _feature[..., 3] = _fea.masked_fill(self._num != 8 , 0).sum(-1)

            return _feature  # _feature.sum(-2)[mask0]
        else:
            return feature.flatten(end_dim=1)[
                self.geometry.atomic_numbers.ne(0).flatten()].sum(-1).unsqueeze(1)

    def _distanceremp(self, cutoff=6.0, **kwargs):
        _feature = torch.zeros(*self.geometry.distances.shape, 4)
        feature = torch.zeros(*self.geometry.distances.shape)

        remp = []
        for inum, ipos in zip(self.geometry.numbers, self.geometry.distances):
            _neg = torch.tensor([_atom_r_emp[ii - 1] for ii in inum[inum.ne(0)]])
            remp.append(_neg.unsqueeze(1) - _neg.unsqueeze(0))
        remp = pack(remp)
        mask_dist = self.geometry.distances.lt(cutoff)
        feature[mask_dist] = 0.5 * (torch.cos(
            np.pi * self.geometry.distances[mask_dist] / cutoff) + 1) * remp[mask_dist]

        if self.specie_res:
            _fea = feature.flatten(end_dim=1)[self.geometry.numbers.ne(0).flatten()]
            _feature = torch.zeros(_fea.shape[0], 4)
            _feature[..., 0] = _fea.masked_fill(self._num != 1 , 0).sum(-1)
            _feature[..., 1] = _fea.masked_fill(self._num != 6 , 0).sum(-1)
            _feature[..., 2] = _fea.masked_fill(self._num != 7 , 0).sum(-1)
            _feature[..., 3] = _fea.masked_fill(self._num != 8 , 0).sum(-1)

            return _feature  # _feature.sum(-2)[mask0]
        else:
            return feature.flatten(end_dim=1)[
                self.geometry.numbers.ne(0).flatten()].sum(-1).unsqueeze(1)

    def _distancercal(self, cutoff=6.0, **kwargs):
        _feature = torch.zeros(*self.geometry.distances.shape, 4)
        feature = torch.zeros(*self.geometry.distances.shape)

        remp = []
        for inum, ipos in zip(self.geometry.numbers, self.geometry.distances):
            _neg = torch.tensor([_atom_r_cal[ii - 1] for ii in inum[inum.ne(0)]])
            remp.append(_neg.unsqueeze(1) - _neg.unsqueeze(0))
        remp = pack(remp)
        mask_dist = self.geometry.distances.lt(cutoff)
        feature[mask_dist] = 0.5 * (torch.cos(
            np.pi * self.geometry.distances[mask_dist] / cutoff) + 1) * remp[mask_dist]

        if self.specie_res:
            _fea = feature.flatten(end_dim=1)[self.geometry.numbers.ne(0).flatten()]
            _feature = torch.zeros(_fea.shape[0], 4)
            _feature[..., 0] = _fea.masked_fill(self._num != 1 , 0).sum(-1)
            _feature[..., 1] = _fea.masked_fill(self._num != 6 , 0).sum(-1)
            _feature[..., 2] = _fea.masked_fill(self._num != 7 , 0).sum(-1)
            _feature[..., 3] = _fea.masked_fill(self._num != 8 , 0).sum(-1)

            return _feature  # _feature.sum(-2)[mask0]
        else:
            return feature.flatten(end_dim=1)[
                self.geometry.numbers.ne(0).flatten()].sum(-1).unsqueeze(1)

    def _distanceion(self, cutoff=6.0, **kwargs):
        """Build electron negativity features."""
        _feature = torch.zeros(*self.geometry.distances.shape, 4)
        feature = torch.zeros(*self.geometry.distances.shape)

        neg = []
        for inum, ipos in zip(self.geometry.numbers, self.geometry.distances):
            _neg = torch.tensor([_ionization_energy[ii - 1] for ii in inum[inum.ne(0)]])
            neg.append(_neg.unsqueeze(1) - _neg.unsqueeze(0))
        neg = pack(neg)
        mask_dist = self.geometry.distances.lt(cutoff)
        feature[mask_dist] = 0.5 * (torch.cos(
            np.pi * self.geometry.distances[mask_dist] / cutoff) + 1) * neg[mask_dist]

        if self.specie_res:
            _fea = feature.flatten(end_dim=1)[self.geometry.numbers.ne(0).flatten()]
            _feature = torch.zeros(_fea.shape[0], 4)
            _feature[..., 0] = _fea.masked_fill(self._num != 1 , 0).sum(-1)
            _feature[..., 1] = _fea.masked_fill(self._num != 6 , 0).sum(-1)
            _feature[..., 2] = _fea.masked_fill(self._num != 7 , 0).sum(-1)
            _feature[..., 3] = _fea.masked_fill(self._num != 8 , 0).sum(-1)

            return _feature  # _feature.sum(-2)[mask0]
        else:
            return feature.flatten(end_dim=1)[
                self.geometry.numbers.ne(0).flatten()].sum(-1).unsqueeze(1)

    def _distanceelectronnegativity(self, cutoff=6.0, **kwargs):
        """Build electron negativity features."""
        # _feature = torch.zeros(*self.geometry.distances.shape, 4)
        feature = torch.zeros(*self.geometry.distances.shape)

        neg = []
        for inum, ipos in zip(self.geometry.atomic_numbers, self.geometry.distances):
            _neg = torch.tensor([_electronnegativity[ii - 1] for ii in inum[inum.ne(0)]])
            neg.append(_neg.unsqueeze(1) - _neg.unsqueeze(0))
        neg = pack(neg)

        mask_dist = self.geometry.distances.lt(cutoff)
        feature[mask_dist] = 0.5 * (torch.cos(
            np.pi * self.geometry.distances[mask_dist] / cutoff) + 1) * neg[mask_dist]

        if self.specie_res:
            _fea = feature.flatten(end_dim=1)[self.geometry.numbers.ne(0).flatten()]
            _feature = torch.zeros(_fea.shape[0], 4)
            _feature[..., 0] = _fea.masked_fill(self._num != 1 , 0).sum(-1)
            _feature[..., 1] = _fea.masked_fill(self._num != 6 , 0).sum(-1)
            _feature[..., 2] = _fea.masked_fill(self._num != 7 , 0).sum(-1)
            _feature[..., 3] = _fea.masked_fill(self._num != 8 , 0).sum(-1)

            return _feature  # _feature.sum(-2)[mask0]
        else:
            return feature.flatten(end_dim=1)[
                self.geometry.atomic_numbers.ne(0).flatten()].sum(-1).unsqueeze(1)

    def _distanceaff(self, cutoff=6.0, **kwargs):
        """Build electron negativity features."""
        _feature = torch.zeros(*self.geometry.distances.shape, 4)
        feature = torch.zeros(*self.geometry.distances.shape)

        neg = []
        for inum, ipos in zip(self.geometry.numbers, self.geometry.distances):
            _neg = torch.tensor([_electron_affinity[ii - 1] for ii in inum[inum.ne(0)]])
            neg.append(_neg.unsqueeze(1) - _neg.unsqueeze(0))
        neg = pack(neg)
        mask_dist = self.geometry.distances.lt(cutoff)
        feature[mask_dist] = 0.5 * (torch.cos(
            np.pi * self.geometry.distances[mask_dist] / cutoff) + 1) * neg[mask_dist]

        if self.specie_res:
            _fea = feature.flatten(end_dim=1)[self.geometry.numbers.ne(0).flatten()]
            _feature = torch.zeros(_fea.shape[0], 4)
            _feature[..., 0] = _fea.masked_fill(self._num != 1 , 0).sum(-1)
            _feature[..., 1] = _fea.masked_fill(self._num != 6 , 0).sum(-1)
            _feature[..., 2] = _fea.masked_fill(self._num != 7 , 0).sum(-1)
            _feature[..., 3] = _fea.masked_fill(self._num != 8 , 0).sum(-1)

            return _feature  # _feature.sum(-2)[mask0]
        else:
            return feature.flatten(end_dim=1)[
                self.geometry.numbers.ne(0).flatten()].sum(-1).unsqueeze(1)

    def _distanceln(self, cutoff=6.0, **kwargs):
        """Build electron negativity features."""
        _feature = torch.zeros(*self.geometry.distances.shape, 4)
        feature = torch.zeros(*self.geometry.distances.shape)

        neg = []
        for inum, ipos in zip(self.geometry.numbers, self.geometry.distances):
            _neg = torch.tensor([_l_number[ii - 1] for ii in inum[inum.ne(0)]])
            neg.append(_neg.unsqueeze(1) - _neg.unsqueeze(0))
        neg = pack(neg)
        mask_dist = self.geometry.distances.lt(cutoff)
        feature[mask_dist] = 0.5 * (torch.cos(
            np.pi * self.geometry.distances[mask_dist] / cutoff) + 1) * neg[mask_dist]

        if self.specie_res:
            _fea = feature.flatten(end_dim=1)[self.geometry.numbers.ne(0).flatten()]
            _feature = torch.zeros(_fea.shape[0], 4)
            _feature[..., 0] = _fea.masked_fill(self._num != 1 , 0).sum(-1)
            _feature[..., 1] = _fea.masked_fill(self._num != 6 , 0).sum(-1)
            _feature[..., 2] = _fea.masked_fill(self._num != 7 , 0).sum(-1)
            _feature[..., 3] = _fea.masked_fill(self._num != 8 , 0).sum(-1)

            return _feature  # _feature.sum(-2)[mask0]
        else:
            return feature.flatten(end_dim=1)[
                self.geometry.numbers.ne(0).flatten()].sum(-1).unsqueeze(1)


    def _distancehomo(self, cutoff=6.0, **kwargs):
        """Build electron negativity features."""
        _feature = torch.zeros(*self.geometry.distances.shape, 4)
        feature = torch.zeros(*self.geometry.distances.shape)

        neg = []
        for inum, ipos in zip(self.geometry.numbers, self.geometry.distances):
            _neg = torch.tensor([_homo[ii - 1] for ii in inum[inum.ne(0)]])
            neg.append(_neg.unsqueeze(1) - _neg.unsqueeze(0))
        neg = pack(neg)
        mask_dist = self.geometry.distances.lt(cutoff)
        feature[mask_dist] = 0.5 * (torch.cos(
            np.pi * self.geometry.distances[mask_dist] / cutoff) + 1) * neg[mask_dist]

        if self.specie_res:
            _fea = feature.flatten(end_dim=1)[self.geometry.numbers.ne(0).flatten()]
            _feature = torch.zeros(_fea.shape[0], 4)
            _feature[..., 0] = _fea.masked_fill(self._num != 1 , 0).sum(-1)
            _feature[..., 1] = _fea.masked_fill(self._num != 6 , 0).sum(-1)
            _feature[..., 2] = _fea.masked_fill(self._num != 7 , 0).sum(-1)
            _feature[..., 3] = _fea.masked_fill(self._num != 8 , 0).sum(-1)

            return _feature  # _feature.sum(-2)[mask0]
        else:
            return feature.flatten(end_dim=1)[
                self.geometry.numbers.ne(0).flatten()].sum(-1).unsqueeze(1)


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
