"""Performs tests on functions present in the tbmalt.common.maths module"""
import torch
from ase.atoms import Atoms
import h5py
from tbmalt.common.structures.geometry import Geometry
from tbmalt.common.batch import pack
torch.set_default_dtype(torch.float64)
torch.set_printoptions(15)
_bohr = 0.529177249


def _calculate_distance(numbers, positions):
    """The traditional way to test the distance calculation."""
    positions = positions.unsqueeze(0) if positions.dim() == 2 else positions
    numbers = numbers.unsqueeze(0) if numbers.dim() == 1 else numbers
    distance = torch.zeros(positions.shape[0],
                           positions.shape[1], positions.shape[1])
    for ib, i_position in enumerate(positions):
        mask = numbers[ib].ne(0)  # get non zero positions
        for ii, ipos in enumerate(i_position[mask]):
            for jj, jpos in enumerate(i_position[mask]):
                distance[ib, ii, jj] = torch.sqrt(sum((ipos - jpos) ** 2))
    return distance


def _calculate_vector_position(positions):
    """Calculate vector of positions between atoms."""
    positions = positions.unsqueeze(0) if positions.dim() == 2 else positions
    vector = torch.zeros(positions.shape[0], positions.shape[1],
                         positions.shape[1], 3)
    for ib in range(positions.shape[0]):
        for iat in range(positions.shape[1]):
            for jat in range(positions.shape[1]):
                vector[ib, iat, jat] = positions[ib, iat] - positions[ib, jat]
    return vector


def test_geometry_single(device):
    """Test single geometry input."""
    numbers = torch.tensor([6, 1, 1, 1, 1])
    positions = torch.tensor([[0.0000000000, 0.0000000000, 0.0000000000],
                              [0.6287614522, 0.6287614522, 0.6287614522],
                              [-0.6287614522, -0.6287614522, 0.6287614522],
                              [-0.6287614522, 0.6287614522, -0.6287614522],
                              [0.6287614522, -0.6287614522, -0.6287614522]])
    sys = Geometry(numbers, positions)
    distance_ref = _calculate_distance(sys.numbers, sys.positions)

    assert torch.max(abs(sys.distances - distance_ref)) < 1E-14
    assert sys.symbols == [['C', 'H', 'H', 'H', 'H']]
    assert sys.size_batch == 1
    assert sys.size_geometry == [5]
    assert (sys.l_max == torch.tensor([[1, 0, 0, 0, 0]])).all()
    assert sys.hs_shape == torch.Size([1, 8, 8])
    assert torch.max(abs(sys.get_positions_vec() - _calculate_vector_position(
        sys.positions))) < 1E-14


def test_system_ase_byhand_single(device):
    """Test single input, deal with ase input by hand."""
    ch4 = Atoms('CH4', positions=[
        [0., 0., 0.], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5],
        [0.5, -0.5, 0.5], [0.5, 0.5, -0.5]])
    # deal with input by hand
    positions = torch.from_numpy(ch4.positions)
    numbers = torch.from_numpy(ch4.numbers)

    sys = Geometry(numbers, positions.clone())
    distance_ref = _calculate_distance(numbers, positions / _bohr)

    assert torch.max(abs(sys.distances - distance_ref)) < 1E-14
    assert sys.symbols == [['C', 'H', 'H', 'H', 'H']]
    assert sys.size_batch == 1
    assert sys.size_geometry == [5]
    assert (sys.l_max == torch.tensor([[1, 0, 0, 0, 0]])).all()
    assert sys.hs_shape == torch.Size([1, 8, 8])
    assert torch.max(abs(sys.get_positions_vec() - _calculate_vector_position(
        sys.positions))) < 1E-14


def test_geometry_ase_byhand_batch(device):
    """Batch evaluation of geometry class, positions, numbers input by hand."""
    ch4 = Atoms('CH4', positions=[
        [0., 0., 0.], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5],
        [0.5, -0.5, 0.5], [0.5, 0.5, -0.5]])
    h2 = Atoms('H2', positions=[[0., 0., 0.], [0.5, 0.5, 0.5]])

    # deal with input by hand
    pos_ = [torch.from_numpy(ch4.positions), torch.from_numpy(h2.positions)]
    num_ = [torch.from_numpy(ch4.numbers), torch.from_numpy(h2.numbers)]
    sys = Geometry(num_, pos_)

    # get positions, numbers by hand
    distance_ref = _calculate_distance(pack(num_), pack(pos_) / _bohr)

    assert torch.max(abs(sys.distances[0] - distance_ref[0])) < 1E-14
    assert torch.max(abs(sys.distances[1] - distance_ref[1])) < 1E-14
    assert sys.symbols[0] == ['C', 'H', 'H', 'H', 'H']
    assert sys.symbols[1] == ['H', 'H']
    assert sys.size_batch == 2
    assert sys.size_geometry == [5, 2]
    assert (sys.l_max == torch.tensor([[1, 0, 0, 0, 0],
                                       [0, 0, -1, -1, -1]])).all()
    assert sys.hs_shape == torch.Size([2, 8, 8])
    assert torch.max(abs(sys.get_positions_vec() - _calculate_vector_position(
        sys.positions))) < 1E-14


def test_from_ase_single(device):
    """Single evaluation of geometry class from ase input."""
    # ase Atoms as input for single geometry
    ch4 = Atoms('CH4', positions=[
        [0., 0., 0.], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5],
        [0.5, -0.5, 0.5], [0.5, 0.5, -0.5]])
    sys = Geometry.from_ase_atoms(ch4)
    distance_ref = _calculate_distance(sys.numbers, sys.positions)

    assert torch.max(abs(sys.distances - distance_ref)) < 1E-14
    assert torch.max(abs(sys.distances - distance_ref)) < 1E-14
    assert sys.symbols == [['C', 'H', 'H', 'H', 'H']]
    assert sys.size_batch == 1
    assert sys.size_geometry == [5]
    assert (sys.l_max == torch.tensor([[1, 0, 0, 0, 0]])).all()
    assert sys.hs_shape == torch.Size([1, 8, 8])
    assert (sys.get_valence_electrons() == torch.tensor([4, 1, 1, 1, 1])).all()
    assert torch.max(abs(sys.get_positions_vec() - _calculate_vector_position(
        sys.positions))) < 1E-14


def test_from_ase_batch(device):
    """Batch evaluation of geometry class from ase input."""
    # ase Atoms as input for batch geometry
    ch4 = Atoms('CH4', positions=[
        [0., 0., 0.], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5],
        [0.5, -0.5, 0.5], [0.5, 0.5, -0.5]])
    h2 = Atoms('H2', positions=[[0., 0., 0.], [0.5, 0.5, 0.5]])
    sys = Geometry.from_ase_atoms([ch4, h2])
    distance_ref = _calculate_distance(sys.numbers, sys.positions)

    assert torch.max(abs(sys.distances - distance_ref)) < 1E-14
    assert sys.symbols == [['C', 'H', 'H', 'H', 'H'], ['H', 'H']]
    assert sys.size_batch == 2
    assert sys.size_geometry == [5, 2]
    assert (sys.l_max == torch.tensor([[1, 0, 0, 0, 0],
                                       [0, 0, -1, -1, -1]])).all()
    assert sys.hs_shape == torch.Size([2, 8, 8])
    assert (sys.get_valence_electrons() == torch.tensor(
        [[4, 1, 1, 1, 1], [1, 1, 0, 0, 0]])).all()
    assert sys.get_global_species()[0] == ['H', 'C']
    assert sys.get_global_species()[1] == [1, 6]
    assert sys.get_global_species()[2] == [['C', 'H'], ['C', 'C'],
                                           ['H', 'H'], ['H', 'C']]
    assert sys.get_global_species()[3] == [[6, 1], [6, 6], [1, 1], [1, 6]]
    # assert (sys.get_resolved_orbital() == torch.tensor(
    #     [[0, 1, 1, 1, 0, 0, 0, 0], [0, 0, -1, -1, -1, -1, -1, -1]])).all()
    orb_res = [[torch.tensor([0, 1, 1, 1]), torch.tensor([0]),
                torch.tensor([0]), torch.tensor([0]), torch.tensor([0])],
               [torch.tensor([0]), torch.tensor([0]), torch.tensor([]),
                torch.tensor([]), torch.tensor([])]]
    sys_orb = sys.get_resolved_orbital()
    assert pack([pack([(ii == jj[jj.ge(0)]).all() for ii, jj in zip(ior, jor)])
                 for ior, jor in zip(orb_res, sys_orb)]).all()


def test_au_batch(device):
    """Batch evaluation of geometry class from ase input with d orbitals."""
    # ase Atoms as input for batch geometry
    auo = Atoms('AuO', positions=[[0., 0., 0.], [0.8, 0.8, 0.8]])
    auau = Atoms('AuAu', positions=[[0., 0., 0.], [0.5, 0.5, 0.5]])
    sys = Geometry.from_ase_atoms([auo, auau])
    distance_ref = _calculate_distance(sys.numbers, sys.positions)

    assert torch.max(abs(sys.distances - distance_ref)) < 1E-14
    assert sys.symbols == [['Au', 'O'], ['Au', 'Au']]
    assert sys.size_batch == 2
    assert sys.size_geometry == [2, 2]
    assert (sys.l_max == torch.tensor([[2, 1], [2, 2]])).all()
    assert sys.hs_shape == torch.Size([2, 18, 18])
    assert (sys.get_valence_electrons() == torch.tensor(
        [[11, 6], [11, 11]])).all()
    assert sys.get_global_species()[0] == ['O', 'Au']
    assert sys.get_global_species()[1] == [8, 79]
    assert sys.get_global_species()[2] == [['Au', 'O'], ['Au', 'Au'],
                                           ['O', 'O'], ['O', 'Au']]
    assert sys.get_global_species()[3] == [[79, 8], [79, 79], [8, 8], [8, 79]]
    orb_res = [[torch.tensor([0, 1, 1, 1, 2, 2, 2, 2, 2]),
                torch.tensor([0, 1, 1, 1])],
               [torch.tensor([0, 1, 1, 1, 2, 2, 2, 2, 2]),
                torch.tensor([0, 1, 1, 1, 2, 2, 2, 2, 2])]]
    sys_orb = sys.get_resolved_orbital()
    assert pack([pack([
        (ii == jj[jj.ge(0)]).all() for ii, jj in zip(ior, jor)])
        for ior, jor in zip(orb_res, sys_orb)]).all()


def test_classmethod(device):
    """Test class methods in geometry."""
    element_single = ['C', 'H']
    element_batch = [['C', 'H'], ['N', 'O', 'Au']]
    assert (Geometry.to_element_number(element_single) == torch.tensor(
        [6, 1])).all()
    assert (Geometry.to_element_number(element_batch) == torch.tensor([
        [6, 1, 0], [7,  8, 79]])).all()
    number_single = torch.tensor([6, 1])
    number_batch = torch.tensor([[6, 1, 0], [79, 1, 6]])
    assert Geometry.to_element(number_single) == [['C', 'H']]
    assert Geometry.to_element(number_batch) == [['C', 'H'], ['Au', 'H', 'C']]


def test_hdf5(device):
    """Test h5py input/output."""
    ch4 = Atoms('CH4', positions=[
        [0., 0., 0.], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5],
        [0.5, -0.5, 0.5], [0.5, 0.5, -0.5]])
    positions = torch.from_numpy(ch4.positions)
    numbers = torch.from_numpy(ch4.numbers)

    with h5py.File('test.hdf5', 'w') as f:
        Geometry(numbers, positions.clone()).to_hd5(f)

    with h5py.File('test.hdf5', 'r') as f:
        sys = Geometry.from_hd5(f, unit='bohr')
        assert torch.max(abs(sys.positions - positions / _bohr)) < 1E-14
