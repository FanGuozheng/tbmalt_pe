"""Perform tests on functions which read SKF or SK transformations.

Reference is from DFTB+."""
import torch
from ase.build import molecule as molecule_database
from tbmalt.common.structures.system import System
from tbmalt.tb.sk import SKT
from tbmalt.io.loadskf import IntegralGenerator
torch.set_default_dtype(torch.float64)
torch.set_printoptions(15)


def test_read_skf_auorg(device):
    """Read auorg type SKF files."""
    sk = IntegralGenerator.from_dir('../slko/auorg-1-1', elements=['C', 'H'])
    # Hpp0 Hpp1, Hsp0, Hss0, Spp0 Spp1, Ssp0, Sss0 at distance 2.0
    atom_pair = torch.tensor([6, 6])
    distance = torch.tensor([2.0])
    c_c_ref = torch.tensor([
        3.293893775138E-01, -2.631898290831E-01, 4.210227871585E-01,
        -4.705514912464E-01, -3.151402994035E-01, 3.193776711119E-01,
        -4.531014049627E-01, 4.667288655632E-01])
    c_c_sktable = torch.cat([
        sk(distance, atom_pair, torch.tensor([1, 1]), hs_type='H').squeeze(0),
        sk(distance, atom_pair, torch.tensor([0, 1]), hs_type='H').squeeze(0),
        sk(distance, atom_pair, torch.tensor([0, 0]), hs_type='H').squeeze(0),
        sk(distance, atom_pair, torch.tensor([1, 1]), hs_type='S').squeeze(0),
        sk(distance, atom_pair, torch.tensor([0, 1]), hs_type='S').squeeze(0),
        sk(distance, atom_pair, torch.tensor([0, 0]), hs_type='S').squeeze(0)])
    assert torch.max(abs(c_c_ref - c_c_sktable)) < 1E-14, 'Tolerance check'


def test_read_skf_h5py(device):
    """Read auorg type SKF files."""
    molecule = molecule_database('CH4')
    path = './skf.hdf'
    system = System.from_ase_atoms(molecule)
    sk = IntegralGenerator.from_dir(path, system, sk_type='h5py')
    c_c_ref = torch.tensor([
        3.293893775138E-01, -2.631898290831E-01, 4.210227871585E-01,
        -4.705514912464E-01, -3.151402994035E-01, 3.193776711119E-01,
        -4.531014049627E-01, 4.667288655632E-01])
    atom_pair = torch.tensor([6, 6])
    distance = torch.tensor([2.0])
    c_c_sktable = torch.cat([
        sk(distance, atom_pair, torch.tensor([1, 1]), hs_type='H').squeeze(0),
        sk(distance, atom_pair, torch.tensor([0, 1]), hs_type='H').squeeze(0),
        sk(distance, atom_pair, torch.tensor([0, 0]), hs_type='H').squeeze(0),
        sk(distance, atom_pair, torch.tensor([1, 1]), hs_type='S').squeeze(0),
        sk(distance, atom_pair, torch.tensor([0, 1]), hs_type='S').squeeze(0),
        sk(distance, atom_pair, torch.tensor([0, 0]), hs_type='S').squeeze(0)])
    assert torch.max(abs(c_c_ref - c_c_sktable)) < 1E-14, 'Tolerance check'


def test_sk_single(device):
    """Test SK transformation values of single molecule."""
    numbers = torch.tensor([6, 1, 1, 1, 1])
    positions = torch.tensor([
        [0., 0., 0.], [0.629118, 0.629118, 0.629118],
        [-0.629118, -0.629118, 0.629118], [0.629118, -0.629118, -0.629118],
        [-0.629118, 0.629118, -0.629118]])
    molecule = System(numbers, positions)
    sktable = IntegralGenerator.from_dir('../slko/auorg-1-1', molecule)
    skt = SKT(molecule, sktable)
    assert torch.max(abs(skt.H - h_ch4)) < 1E-14, 'Tolerance check'
    assert torch.max(abs(skt.S - s_ch4)) < 1E-14, 'Tolerance check'


def test_sk_single_d_orb(device):
    """Test SK transformation values of single molecule with d orbitals."""
    numbers = torch.tensor([79, 8])
    positions = torch.tensor([[0., 0., 0.], [1., 1., 0.]])
    molecule = System(numbers, positions)

    sktable = IntegralGenerator.from_dir('../slko/auorg-1-1/', molecule)
    skt = SKT(molecule, sktable)
    assert torch.max(abs(skt.H - h_auo)) < 1E-14, 'Tolerance check'
    assert torch.max(abs(skt.S - s_auo)) < 1E-14, 'Tolerance check'


def test_sk_batch(device):
    """Test SK transformation values of batch molecules.

    Test p-d, s-p, d-d orbitals."""
    numbers = [torch.tensor([79, 8]), torch.tensor([79, 79]),
               torch.tensor([6, 1, 1, 1, 1])]
    positions = [torch.tensor([[0., 0., 0.], [1., 1., 0.]]),
                 torch.tensor([[0., 0., 0.], [1., 1., 0.]]),
                 torch.tensor([[0., 0., 0.], [0.629118, 0.629118, 0.629118],
                               [-0.629118, -0.629118, 0.629118],
                               [0.629118, -0.629118, -0.629118],
                               [-0.629118, 0.629118, -0.629118]])]
    molecule = System(numbers, positions)
    sktable = IntegralGenerator.from_dir('../slko/auorg-1-1/', molecule)
    skt = SKT(molecule, sktable)
    assert torch.max(abs(skt.H[0][:h_auo.shape[0], :h_auo.shape[1]] - h_auo)
                     ) < 1E-14, 'Tolerance check'
    assert torch.max(abs(skt.S[0][:s_auo.shape[0], :s_auo.shape[1]] - s_auo)
                     ) < 1E-14, 'Tolerance check'
    assert torch.max(abs(skt.H[1][:h_auau.shape[0], :h_auau.shape[1]] - h_auau)
                     ) < 1E-14, 'Tolerance check'
    assert torch.max(abs(skt.S[1][:s_auau.shape[0], :s_auau.shape[1]] - s_auau)
                     ) < 1E-14, 'Tolerance check'
    assert torch.max(abs(skt.H[2][:h_ch4.shape[0], :h_ch4.shape[1]] - h_ch4)
                     ) < 1E-14, 'Tolerance check'
    assert torch.max(abs(skt.S[2][:s_ch4.shape[0], :s_ch4.shape[1]] - s_ch4)
                     ) < 1E-14, 'Tolerance check'


def test_sk_ase_single(device):
    """Test SK transformation values of single ASE molecule."""
    molecule = molecule_database('CH4')
    molecule = System.from_ase_atoms(molecule)
    sktable = IntegralGenerator.from_dir('../slko/auorg-1-1/', molecule)
    skt = SKT(molecule, sktable)
    assert torch.max(abs(skt.H - h_ch4)) < 1E-14, 'Tolerance check'
    assert torch.max(abs(skt.S - s_ch4)) < 1E-14, 'Tolerance check'


def test_sk_ase_batch(device):
    """Test SK transformation values of batch ASE molecules."""
    molecule = System.from_ase_atoms([
        molecule_database('H2'), molecule_database('N2'),
        molecule_database('CH4'), molecule_database('NH3'),
        molecule_database('H2O'), molecule_database('CO2'),
        molecule_database('CH3CHO')])
    sktable = IntegralGenerator.from_dir('../slko/auorg-1-1/', molecule)
    skt = SKT(molecule, sktable)
    assert torch.max(abs(skt.H[0][:h_h2.shape[0], :h_h2.shape[1]] - h_h2)
                     ) < 1E-14, 'Tolerance check'
    assert torch.max(abs(skt.S[0][:s_h2.shape[0], :s_h2.shape[1]] - s_h2)
                     ) < 1E-14, 'Tolerance check'
    assert torch.max(abs(skt.H[1][:h_n2.shape[0], :h_n2.shape[1]] - h_n2)
                     ) < 1E-14, 'Tolerance check'
    assert torch.max(abs(skt.S[1][:s_n2.shape[0], :s_n2.shape[1]] - s_n2)
                     ) < 1E-14, 'Tolerance check'
    assert torch.max(abs(skt.H[2][:h_ch4.shape[0], :h_ch4.shape[1]] - h_ch4)
                     ) < 1E-14, 'Tolerance check'
    assert torch.max(abs(skt.S[2][:s_ch4.shape[0], :s_ch4.shape[1]] - s_ch4)
                     ) < 1E-14, 'Tolerance check'
    assert torch.max(abs(skt.H[3][:h_nh3.shape[0], :h_nh3.shape[1]] - h_nh3)
                     ) < 1E-14, 'Tolerance check'
    assert torch.max(abs(skt.S[3][:s_nh3.shape[0], :s_nh3.shape[1]] - s_nh3)
                     ) < 1E-14, 'Tolerance check'
    assert torch.max(abs(skt.H[4][:h_h2o.shape[0], :h_h2o.shape[1]] - h_h2o)
                     ) < 1E-14, 'Tolerance check'
    assert torch.max(abs(skt.S[4][:s_h2o.shape[0], :s_h2o.shape[1]] - s_h2o)
                     ) < 1E-14, 'Tolerance check'
    assert torch.max(abs(skt.H[5][:h_co2.shape[0], :h_co2.shape[1]] - h_co2)
                     ) < 1E-14, 'Tolerance check'
    assert torch.max(abs(skt.S[5][:s_co2.shape[0], :s_co2.shape[1]] - s_co2)
                     ) < 1E-14, 'Tolerance check'
    assert torch.max(abs(skt.H[6][:h_ch3cho.shape[0], :h_ch3cho.shape[1]]
                         - h_ch3cho)) < 1E-14, 'Tolerance check'
    assert torch.max(abs(skt.S[6][:s_ch3cho.shape[0], :s_ch3cho.shape[1]]
                         - s_ch3cho)) < 1E-14, 'Tolerance check'


def test_sk_ase_batch_cubic(device):
    """Test batch molecule SK transformtion value.

    The interpolation of integral is cubic interpolation, which is different
    from DFTB+."""
    molecule = System.from_ase_atoms([
        molecule_database('H2'), molecule_database('N2'),
        molecule_database('CH4'), molecule_database('NH3'),
        molecule_database('H2O'), molecule_database('CN'),
        molecule_database('CO2'), molecule_database('CH3CHO')])
    sktable = IntegralGenerator.from_dir(
        '../slko/auorg-1-1/', molecule, sk_interpolation='cubic_interpolation')
    skt = SKT(molecule, sktable)
    assert torch.max(abs(skt.H[0][:h_h2.shape[0], :h_h2.shape[1]] - h_h2)
                     ) < 1E-9, 'Tolerance check'
    assert torch.max(abs(skt.H[4][:h_h2o.shape[0], :h_h2o.shape[1]] - h_h2o)
                     ) < 1E-9, 'Tolerance check'


h_h2 = torch.tensor([[-2.386005440482000E-01, -3.210767605238942E-01],
                     [-3.210767605238942E-01, -2.386005440482000E-01]])

s_h2 = torch.tensor([[1, 6.432524199588737E-01],
                     [6.432524199588737E-01, 1]])

h_n2 = torch.tensor([
    [-0.64, 0, 0, 0, -4.872910016436111E-01, 0, -4.892318909251249E-01, 0],
    [0, -2.607280834228000E-01, 0, 0, 0, -2.204556833617439E-01, 0, 0],
    [0, 0, -2.607280834228000E-01, 0, 4.892318909251249E-01, 0,
     4.206528423702421E-01, 0],
    [0, 0, 0, -2.607280834228000E-01, 0, 0, 0, -2.204556833617439E-01],
    [-4.872910016436111E-01, 0, 4.892318909251249E-01, 0, -0.64, 0, 0, 0],
    [0, -2.204556833617439E-01, 0, 0, 0, -2.607280834228000E-01, 0, 0],
    [-4.892318909251249E-01, 0, 4.206528423702421E-01, 0, 0, 0,
     -2.607280834228000E-01, 0],
    [0, 0, 0, -2.204556833617439E-01, 0, 0, 0, -2.607280834228000E-01]])

s_n2 = torch.tensor([
    [1, 0, 0, 0, 3.322354266612609E-01, 0, 3.719872995527780E-01, 0],
    [0, 1, 0, 0, 0, 1.975525624129341E-01, 0, 0],
    [0, 0, 1, 0, -3.719872995527780E-01, 0, -3.400288645138444E-01, 0],
    [0, 0, 0, 1, 0, 0, 0, 1.975525624129341E-01],
    [3.322354266612609E-01, 0, -3.719872995527780E-01, 0, 1, 0, 0, 0],
    [0, 1.975525624129341E-01, 0, 0, 0, 1, 0, 0],
    [3.719872995527780E-01, 0, -3.400288645138444E-01, 0, 0, 0, 1, 0],
    [0, 0, 0, 1.975525624129341E-01, 0, 0, 0, 1.]])

h_nh3 = torch.tensor([
    [-6.400000000000000E-01, 0, 0, 0, -4.268000227088792E-01,
     -4.268001305058769E-01, -4.268001305058769E-01],
    [0, -2.607280834228000E-01, 0, 0, -3.005989818584597E-01,
     1.502993849522482E-01, 1.502993849522482E-01],
    [0, 0, -2.607280834228000E-01, 0, 1.242075475414713E-01,
     1.242075921356201E-01, 1.242075921356201E-01],
    [0, 0, 0, -2.607280834228000E-01, 0, -2.603264741044196E-01,
     2.603264741044196E-01],
    [-4.268000227088792E-01, -3.005989818584597E-01, 1.242075475414713E-01, 0,
     -2.386005440482000E-01, -9.524111460017705E-02, -9.524111460017705E-02],
    [-4.268001305058769E-01, 1.502993849522482E-01, 1.242075921356201E-01,
     -2.603264741044196E-01, -9.524111460017705E-02, -2.386005440482000E-01,
     -9.524102770150955E-02],
    [-4.268001305058769E-01, 1.502993849522482E-01, 1.242075921356201E-01,
     2.603264741044196E-01, -9.524111460017705E-02, -9.524102770150955E-02,
     -2.386005440482000E-01]])

s_nh3 = torch.tensor([
    [1, 0, 0, 0, 4.248797716933945E-01, 4.248798803810287E-01,
     4.248798803810287E-01],
    [0, 1, 0, 0, 3.765626589741126E-01, -1.882811896933286E-01,
     -1.882811896933286E-01],
    [0, 0, 1, 0, -1.555957511156608E-01, -1.555958011649100E-01,
     -1.555958011649100E-01],
    [0, 0, 0, 1, 0, 3.261129662547994E-01, -3.261129662547994E-01],
    [4.248797716933945E-01, 3.765626589741126E-01, -1.555957511156608E-01,
     0, 1, 1.551427148615352E-01, 1.551427148615352E-01],
    [4.248798803810287E-01, -1.882811896933286E-01, -1.555958011649100E-01,
     3.261129662547994E-01, 1.551427148615352E-01, 1, 1.551425395257848E-01],
    [4.248798803810287E-01, -1.882811896933286E-01, -1.555958011649100E-01,
     -3.261129662547994E-01, 1.551427148615352E-01, 1.551425395257848E-01, 1]])


h_ch4 = torch.tensor([
    [-5.048917654780000E-01, 0, 0, 0, -3.310339110747502E-01,
     -3.310339110747502E-01, -3.310339110747502E-01, -3.310339110747502E-01],
    [0, -1.943551799163000E-01, 0, 0, -1.621329527608377E-01,
     1.621329527608377E-01, 1.621329527608377E-01, -1.621329527608377E-01],
    [0, 0, -1.943551799163000E-01, 0, -1.621329527608377E-01,
     -1.621329527608377E-01, 1.621329527608377E-01, 1.621329527608377E-01],
    [0, 0, 0, -1.943551799163000E-01, -1.621329527608377E-01,
     1.621329527608377E-01, -1.621329527608377E-01, 1.621329527608377E-01],
    [-3.310339110747502E-01, -1.621329527608377E-01, -1.621329527608377E-01,
     -1.621329527608377E-01, -2.386005440482000E-01, -7.364358423231271E-02,
     -7.364358423231271E-02, -7.364358423231271E-02],
    [-3.310339110747502E-01, 1.621329527608377E-01, -1.621329527608377E-01,
     1.621329527608377E-01, -7.364358423231271E-02, -2.386005440482000E-01,
     -7.364358423231271E-02, -7.364358423231271E-02],
    [-3.310339110747502E-01, 1.621329527608377E-01, 1.621329527608377E-01,
     -1.621329527608377E-01, -7.364358423231271E-02, -7.364358423231271E-02,
     -2.386005440482000E-01, -7.364358423231271E-02],
    [-3.310339110747502E-01, -1.621329527608377E-01, 1.621329527608377E-01,
     1.621329527608377E-01, -7.364358423231271E-02, -7.364358423231271E-02,
     -7.364358423231271E-02, -2.386005440482000E-01]])

s_ch4 = torch.tensor([
    [1, 0, 0, 0, 4.286998612221351E-01,
     4.286998612221351E-01, 4.286998612221351E-01, 4.286998612221351E-01],
    [0, 1, 0, 0, 2.580853601954335E-01,
     -2.580853601954335E-01, -2.580853601954335E-01, 2.580853601954335E-01],
    [0, 0, 1, 0, 2.580853601954335E-01,
     2.580853601954335E-01, -2.580853601954335E-01, -2.580853601954335E-01],
    [0, 0, 0, 1, 2.580853601954335E-01,
     -2.580853601954335E-01, 2.580853601954335E-01, -2.580853601954335E-01],
    [4.286998612221351E-01, 2.580853601954335E-01, 2.580853601954335E-01,
     2.580853601954335E-01, 1, 1.126933785177702E-01,
     1.126933785177702E-01, 1.126933785177702E-01],
    [4.286998612221351E-01, -2.580853601954335E-01, 2.580853601954335E-01,
     -2.580853601954335E-01, 1.126933785177702E-01, 1,
     1.126933785177702E-01, 1.126933785177702E-01],
    [4.286998612221351E-01, -2.580853601954335E-01, -2.580853601954335E-01,
     2.580853601954335E-01, 1.126933785177702E-01, 1.126933785177702E-01,
     1, 1.126933785177702E-01],
    [4.286998612221351E-01, 2.580853601954335E-01, -2.580853601954335E-01,
     -2.580853601954335E-01, 1.126933785177702E-01, 1.126933785177702E-01,
     1.126933785177702E-01, 1]])

h_h2o = torch.tensor([
    [-8.788325840774993E-01, 0, 0, 0, -5.009850265252284E-01,
     -5.009850265252284E-01],
    [0, -3.321317735293993E-01, 0, 0, -2.704289178531905E-01,
     2.704289178531905E-01],
    [0, 0, -3.321317735293993E-01, 0, 2.112827011933590E-01,
     2.112827011933590E-01],
    [0, 0, 0, -3.321317735293993E-01, 0, 0],
    [-5.009850265252284E-01, -2.704289178531905E-01, 2.112827011933590E-01,
     0, -2.386005440481994E-01, -1.120988652530896E-01],
    [-5.009850265252284E-01, 2.704289178531905E-01, 2.112827011933590E-01,
     0, -1.120988652530896E-01, -2.386005440481994E-01]])

s_h2o = torch.tensor([
    [1, 0, 0, 0, 4.253702443409864E-01, 4.253702443409864E-01],
    [0, 1, 0, 0, 3.098372121187390E-01, -3.098372121187390E-01],
    [0, 0, 1, 0, -2.420719042414147E-01, -2.420719042414147E-01],
    [0, 0, 0, 1, 0, 0],
    [4.253702443409864E-01, 3.098372121187390E-01, -2.420719042414147E-01,
     0, 1, 1.897382512654583E-01],
    [4.253702443409864E-01, -3.098372121187390E-01, -2.420719042414147E-01,
     0, 1.897382512654583E-01, 1]])

h_co2 = torch.tensor([
    [-5.048917654780000E-01, 0, 0, 0, -4.540609750654194E-01, 0,
     3.698605199232547E-01, 0, -4.540609750654194E-01, 0,
     -3.698605199232547E-01, 0],
    [0, -1.943551799163000E-01, 0, 0, 0, -1.952284028405140E-01, 0, 0, 0,
     -1.952284028405140E-01, 0, 0],
    [0, 0, -1.943551799163000E-01, 0, -5.329863053378449E-01, 0,
     3.650885342628561E-01, 0, 5.329863053378449E-01, 0,
     3.650885342628561E-01, 0],
    [0, 0, 0, -1.943551799163000E-01, 0, 0, 0, -1.952284028405140E-01, 0, 0, 0,
     -1.952284028405140E-01],
    [-4.540609750654194E-01, 0, -5.329863053378449E-01,
     0, -8.788325840775000E-01, 0, 0, 0, -1.475176180368041E-02,
     0, -2.427418864314294E-02, 0],
    [0, -1.952284028405140E-01, 0, 0, 0, -3.321317735294000E-01, 0, 0, 0,
     -4.950424046334406E-03, 0, 0],
    [3.698605199232547E-01, 0, 3.650885342628561E-01, 0, 0, 0,
     -3.321317735294000E-01, 0, 2.427418864314294E-02, 0,
     3.464000505898970E-02, 0],
    [0, 0, 0, -1.952284028405140E-01, 0, 0, 0, -3.321317735294000E-01, 0, 0,
     0, -4.950424046334406E-03],
    [-4.540609750654194E-01, 0, 5.329863053378449E-01, 0,
     -1.475176180368041E-02, 0, 2.427418864314294E-02, 0,
     -8.788325840775000E-01, 0, 0, 0],
    [0, -1.952284028405140E-01, 0, 0, 0, -4.950424046334406E-03, 0, 0, 0,
     -3.321317735294000E-01, 0, 0],
    [-3.698605199232547E-01, 0, 3.650885342628561E-01, 0,
     -2.427418864314294E-02, 0, 3.464000505898970E-02, 0, 0, 0,
     -3.321317735294000E-01, 0],
    [0, 0, 0, -1.952284028405140E-01, 0, 0, 0, -4.950424046334406E-03, 0, 0, 0,
     -3.321317735294000E-01]])

s_co2 = torch.tensor([
    [1, 0, 0, 0, 3.217029316809677E-01, 0, -3.243052784350625E-01, 0,
     3.217029316809677E-01, 0, 3.243052784350625E-01, 0],
    [0, 1, 0, 0, 0, 1.924055787809134E-01, 0, 0, 0, 1.924055787809134E-01,
     0, 0],
    [0, 0, 1, 0, 4.001984259030480E-01, 0, -3.246226044670189E-01, 0,
     -4.001984259030480E-01, 0, -3.246226044670189E-01, 0],
    [0, 0, 0, 1, 0, 0, 0, 1.924055787809134E-01, 0, 0, 0,
     1.924055787809134E-01],
    [3.217029316809677E-01, 0, 4.001984259030480E-01, 0, 1, 0, 0, 0,
     7.450838474212128E-03, 0, 1.435398445073693E-02, 0],
    [0, 1.924055787809134E-01, 0, 0, 0, 1, 0, 0, 0, 3.201684271578112E-03,
     0, 0],
    [-3.243052784350625E-01, 0, -3.246226044670189E-01, 0, 0, 0, 1, 0,
     -1.435398445073693E-02, 0, -2.596177903406008E-02, 0],
    [0, 0, 0, 1.924055787809134E-01, 0, 0, 0, 1, 0, 0, 0,
     3.201684271578112E-03],
    [3.217029316809677E-01, 0, -4.001984259030480E-01, 0,
     7.450838474212128E-03,
     0, -1.435398445073693E-02, 0, 1, 0, 0, 0],
    [0, 1.924055787809134E-01, 0, 0, 0, 3.201684271578112E-03, 0, 0, 0, 1,
     0, 0],
    [3.243052784350625E-01, 0, -3.246226044670189E-01, 0,
     1.435398445073693E-02, 0, -2.596177903406008E-02, 0, 0, 0, 1, 0],
    [0, 0, 0, 1.924055787809134E-01, 0, 0, 0, 3.201684271578112E-03,
     0, 0, 0, 1.]])

h_ch3cho = torch.tensor([
    [-8.788325840774983E-01, 0, 0, 0, -4.192763320840661E-01,
     4.189893765125811E-02, 0, -4.960027455784472E-01, -5.624299239798478E-02,
     -2.278125430539886E-02, -1.589788938569719E-02, 0, -3.244586224516966E-02,
     -1.301006984047251E-02, -2.252367046802404E-03, -2.252367046802404E-03],
    [0, -3.321317735293983E-01, 0, 0, -2.941035683728336E-02,
     -1.716965451445849E-01, 0, -4.393089967416337E-02, -3.318566345118520E-02,
     1.390180743964336E-02, 3.042020168985295E-03, 0, 2.045604266512174E-02,
     1.370217075880276E-02, 1.182757058830421E-03, 1.182757058830421E-03],
    [0, 0, -3.321317735293983E-01, 0, 0, 0, -1.754075286881634E-01, 0, 0, 0, 0,
     -6.981073102565004E-03, 0, 0, -1.027848162401093E-03,
     1.027848162401093E-03],
    [0, 0, 0, -3.321317735293983E-01, 3.481619954461156E-01,
     -4.393089967416337E-02, 0, 3.446497347079183E-01, 5.095717378766497E-02,
     2.837207620474096E-02, 2.045604266512174E-02, 0, 3.476748397262602E-02,
     1.101406160700023E-02, 3.283687308763237E-03, 3.283687308763237E-03],
    [-4.192763320840661E-01, -2.941035683728336E-02, 0, 3.481619954461156E-01,
     -5.048917654779984E-01, 0, 0, 0, -3.224397125892385E-01,
     -2.441670989431892E-01, -1.998840467923470E-01, 0, -1.627717812536066E-01,
     -5.247636208142074E-02, -5.197146960521135E-02, -5.197146960521135E-02],
    [4.189893765125811E-02, -1.716965451445849E-01, 0, -4.393089967416337E-02,
     0, -1.943551799162985E-01, 0, 0, -2.482961177393543E-01,
     1.998840467923470E-01, 1.010691070387094E-01, 0, 1.639961300399785E-01,
     6.108666911933189E-02, 3.217050854600069E-02, 3.217050854600069E-02],
    [0, 0, -1.754075286881634E-01, 0, 0, 0, -1.943551799162985E-01, 0, 0, 0,
     0, -1.003184423154486E-01, 0, 0, -2.538086850673211E-02,
     2.538086850673211E-02],
    [-4.960027455784472E-01, -4.393089967416337E-02, 0, 3.446497347079183E-01,
     0, 0, 0, -1.943551799162985E-01, 1.183595537246191E-01,
     1.627717812536066E-01, 1.639961300399785E-01, 0, 3.322869480568869E-02,
     1.123546697455484E-02, 4.599148346837957E-02, 4.599148346837957E-02],
    [-5.624299239798478E-02, -3.318566345118520E-02, 0, 5.095717378766497E-02,
     -3.224397125892385E-01, -2.482961177393543E-01, 0, 1.183595537246191E-01,
     -2.386005440481987E-01, -4.376130300192231E-02, -5.190745040916926E-02,
     0, -1.128697771166868E-02, -3.843009631329070E-03, -1.550042798647350E-02,
     -1.550042798647350E-02],
    [-2.278125430539886E-02, 1.390180743964336E-02, 0, 2.837207620474096E-02,
     -2.441670989431892E-01, 1.998840467923470E-01, 0, 1.627717812536066E-01,
     -4.376130300192231E-02, -5.048917654779986E-01, 0, 0, 0,
     -3.307908188491402E-01, -3.287631796388840E-01, -3.287631796388840E-01],
    [-1.589788938569719E-02, 3.042020168985295E-03, 0, 2.045604266512174E-02,
     -1.998840467923470E-01, 1.010691070387094E-01, 0, 1.639961300399785E-01,
     -5.190745040916926E-02, 0, -1.943551799162986E-01, 0, 0,
     2.404710303603413E-01, -1.216003482919111E-02, -1.216003482919111E-02],
    [0, 0, -6.981073102565004E-03, 0, 0, 0, -1.003184423154486E-01, 0, 0, 0,
     0, -1.943551799162986E-01, 0, 0, -2.247515691969993E-01,
     2.247515691969993E-01],
    [-3.244586224516966E-02, 2.045604266512174E-02, 0, 3.476748397262602E-02,
     -1.627717812536066E-01, 1.639961300399785E-01, 0, 3.322869480568869E-02,
     -1.128697771166868E-02, 0, 0, 0, -1.943551799162986E-01,
     -1.447195396449281E-01, 1.653770349525507E-01, 1.653770349525507E-01],
    [-1.301006984047251E-02, 1.370217075880276E-02, 0, 1.101406160700023E-02,
     -5.247636208142074E-02, 6.108666911933189E-02, 0, 1.123546697455484E-02,
     -3.843009631329070E-03, -3.307908188491402E-01, 2.404710303603413E-01, 0,
     -1.447195396449281E-01, -2.386005440481988E-01, -7.226351092589751E-02,
     -7.226351092589751E-02],
    [-2.252367046802404E-03, 1.182757058830421E-03, -1.027848162401093E-03,
     3.283687308763237E-03, -5.197146960521135E-02, 3.217050854600069E-02,
     -2.538086850673211E-02, 4.599148346837957E-02, -1.550042798647350E-02,
     -3.287631796388840E-01, -1.216003482919111E-02, -2.247515691969993E-01,
     1.653770349525507E-01, -7.226351092589751E-02, -2.386005440481988E-01,
     -7.592272676034512E-02],
    [-2.252367046802404E-03, 1.182757058830421E-03, 1.027848162401093E-03,
     3.283687308763237E-03, -5.197146960521135E-02, 3.217050854600069E-02,
     2.538086850673211E-02, 4.599148346837957E-02, -1.550042798647350E-02,
     -3.287631796388840E-01, -1.216003482919111E-02, 2.247515691969993E-01,
     1.653770349525507E-01, -7.226351092589751E-02, -7.592272676034512E-02,
     -2.386005440481988E-01]])

s_ch3cho = torch.tensor([
    [1, 0, 0, 0, 2.977034225162787E-01, -3.175513752220835E-02, 0,
     3.759196838911634E-01, 4.195850123218638E-02, 1.379580868152652E-02,
     1.062371989861518E-02, 0, 2.168185625142462E-02, 8.265848402591982E-03,
     1.232813929666194E-03, 1.232813929666194E-03],
    [0, 1, 0, 0, 2.604069891529716E-02, 1.704513956117439E-01, 0,
     4.153379720819477E-02, 3.403590333343955E-02, -1.051693155294893E-02,
     -3.412273253544209E-03, 0, -1.776934866568812E-02, -1.117427730455904E-02,
     -7.211582036825674E-04, -7.211582036825674E-04],
    [0, 0, 1, 0, 0, 0, 1.739598882464302E-01, 0, 0, 0, 0,
     5.294388243332855E-03, 0, 0, 6.267061599180667E-04,
     -6.267061599180667E-04],
    [0, 0, 0, 1, -3.082717339106865E-01, 4.153379720819477E-02, 0,
     -3.177202975623971E-01, -5.226273218052246E-02, -2.146391285851148E-02,
     -1.776934866568812E-02, 0, -3.097091874213414E-02, -8.982093480848686E-03,
     -2.002150841851338E-03, -2.002150841851338E-03],
    [2.977034225162787E-01, 2.604069891529716E-02, 0, -3.082717339106865E-01,
     1, 0, 0, 0, 4.169130272484295E-01, 2.346463618796522E-01,
     2.242675256917606E-01, 0, 1.826280046856871E-01, 5.316704246650546E-02,
     5.257137161832435E-02, 5.257137161832435E-02],
    [-3.175513752220835E-02, 1.704513956117439E-01, 0, 4.153379720819477E-02,
     0, 1, 0, 0, 3.963343969187328E-01, -2.242675256917606E-01,
     -1.399150065563199E-01, 0, -2.156253200058745E-01, -7.757039406316023E-02,
     -4.077271808736145E-02, -4.077271808736145E-02],
    [0, 0, 1.739598882464302E-01, 0, 0, 0, 1, 0, 0, 0, 0,
     1.248732830460221E-01, 0, 0, 3.216756722877565E-02,
     -3.216756722877565E-02],
    [3.759196838911634E-01, 4.153379720819477E-02, 0, -3.177202975623971E-01,
     0, 0, 0, 1, -1.889274901763081E-01, -1.826280046856871E-01,
     -2.156253200058745E-01, 0, -5.071710540152109E-02, -1.426726343512530E-02,
     -5.828934246390400E-02, -5.828934246390400E-02],
    [4.195850123218638E-02, 3.403590333343955E-02, 0, -5.226273218052246E-02,
     4.169130272484295E-01, 3.963343969187328E-01, 0, -1.889274901763081E-01,
     1, 4.301808039780808E-02, 6.354805692984511E-02, 0, 1.381816091010149E-02,
     3.113332181064658E-03, 1.610097920365655E-02, 1.610097920365655E-02],
    [1.379580868152652E-02, -1.051693155294893E-02, 0, -2.146391285851148E-02,
     2.346463618796522E-01, -2.242675256917606E-01, 0, -1.826280046856871E-01,
     4.301808039780808E-02, 1, 0, 0, 0, 4.283668579251758E-01,
     4.255883596178757E-01, 4.255883596178757E-01],
    [1.062371989861518E-02, -3.412273253544209E-03, 0, -1.776934866568812E-02,
     2.242675256917606E-01, -1.399150065563199E-01, 0, -2.156253200058745E-01,
     6.354805692984511E-02, 0, 1, 0, 0, -3.828169304579084E-01,
     1.937138096796834E-02, 1.937138096796834E-02],
    [0, 0, 5.294388243332855E-03, 0, 0, 0, 1.248732830460221E-01, 0, 0, 0, 0,
     1, 0, 0, 3.580374835450522E-01, -3.580374835450522E-01],
    [2.168185625142462E-02, -1.776934866568812E-02, 0, -3.097091874213414E-02,
     1.826280046856871E-01, -2.156253200058745E-01, 0, -5.071710540152109E-02,
     1.381816091010149E-02, 0, 0, 0, 1, 2.303857136601256E-01,
     -2.634516752968856E-01, -2.634516752968856E-01],
    [8.265848402591982E-03, -1.117427730455904E-02, 0, -8.982093480848686E-03,
     5.316704246650546E-02, -7.757039406316023E-02, 0, -1.426726343512530E-02,
     3.113332181064658E-03, 4.283668579251758E-01, -3.828169304579084E-01,
     0, 2.303857136601256E-01, 1, 1.100658058928675E-01,
     1.100658058928675E-01],
    [1.232813929666194E-03, -7.211582036825674E-04, 6.267061599180667E-04,
     -2.002150841851338E-03, 5.257137161832435E-02, -4.077271808736145E-02,
     3.216756722877565E-02, -5.828934246390400E-02, 1.610097920365655E-02,
     4.255883596178757E-01, 1.937138096796834E-02, 3.580374835450522E-01,
     -2.634516752968856E-01, 1.100658058928675E-01, 1, 1.170570541308637E-01],
    [1.232813929666194E-03, -7.211582036825674E-04, -6.267061599180667E-04,
     -2.002150841851338E-03, 5.257137161832435E-02, -4.077271808736145E-02,
     -3.216756722877565E-02, -5.828934246390400E-02, 1.610097920365655E-02,
     4.255883596178757E-01, 1.937138096796834E-02, -3.580374835450522E-01,
     -2.634516752968856E-01, 1.100658058928675E-01, 1.170570541308637E-01,
     1.000000000000000E+00]])

h_auo = torch.tensor([
    [-2.107700668744000E-01, 0, 0, 0, 0, 0, 0, 0, 0, -3.766727030358486E-01,
     8.942413868442971E-02, 0, 8.942413868442971E-02],
    [0, -2.785941987392000E-02, 0, 0, 0, 0, 0, 0, 0, -3.410451340387461E-01,
     -4.557024300257728E-02, 0., 9.111984787675188E-02],
    [0, 0, -2.785941987392000E-02, 0, 0, 0, 0, 0, 0, 0, 0,
     -1.366900908793291E-01, 0],
    [0, 0, 0, -2.785941987392000E-02, 0, 0, 0, 0, 0, -3.410451340387461E-01,
     9.111984787675188E-02, 0., -4.557024300257724E-02],
    [0, 0, 0, 0, -2.531805351853000E-01, 0, 0, 0, 0, -3.713579659355442E-01,
     1.802385078011458E-01, 0., 1.802385078011459E-01],
    [0, 0, 0, 0, 0, -2.531805351853000E-01, 0, 0, 0, 0, 0,
     -1.607929453863335E-01, 0.],
    [0, 0, 0, 0, 0, 0, -2.531805351853000E-01, 0., 0., 2.144036215985983E-01,
     -1.040607509973280E-01, 0, -1.040607509973280E-01],
    [0, 0, 0, 0, 0, 0, 0., -2.531805351853000E-01, 0., 0., 0.,
     -1.607929453863335E-01, 0.],
    [0, 0, 0, 0, 0, 0, 0, 0, -2.531805351853000E-01, -8.245803283192112E-17,
     1.607929453863335E-01, 0, -1.607929453863334E-01],
    [-3.766727030358486E-01, -3.410451340387461E-01, 0, -3.410451340387461E-01,
     -3.713579659355442E-01, 0,
     2.144036215985983E-01, 0, -8.245803283192112E-17, -8.788325840775000E-01,
     0., 0., 0.],
    [8.942413868442971E-02, -4.557024300257728E-02, 0,
     9.111984787675188E-02, 1.802385078011458E-01, 0., -1.040607509973280E-01,
     0., 1.607929453863335E-01, 0., -3.321317735294000E-01, 0., 0.],
    [0, 0, -1.366900908793291E-01,
     0, 0, -1.607929453863335E-01,
     0., -1.607929453863335E-01, 0., 0., 0., -3.321317735294000E-01, 0.],
    [8.942413868442971E-02, 9.111984787675188E-02, 0,
     -4.557024300257724E-02, 1.802385078011459E-01, 0., -1.040607509973280E-01,
     0., -1.607929453863334E-01, 0., 0., 0., -3.321317735294000E-01]])

s_auo = torch.tensor([
    [1., 0, 0, 0, 0, 0, 0, 0, 0, 3.302274274717535E-01, -1.035018733501897E-01,
     0, -1.035018733501897E-01],
    [0, 1., 0, 0, 0, 0, 0, 0, 0, 3.440028787868190E-01, 4.564650679801431E-02,
     0, -1.518470471490936E-01],
    [0, 0, 1., 0, 0, 0, 0, 0, 0, 0, 0, 1.974935539471079E-01, 0],
    [0, 0, 0, 1., 0, 0, 0, 0, 0, 3.440028787868190E-01, -1.518470471490936E-01,
     0, 4.564650679801423E-02],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 2.635539939454993E-01, -1.229114393448020E-01,
     0, -1.229114393448021E-01],
    [0, 0, 0, 0, 0, 1., 0, 0, 0, 0, 0, 1.511294855619158E-01, 0],
    [0, 0, 0, 0, 0, 0, 1., 0, 0, -1.521629693504350E-01,
     7.096295259220584E-02, 0, 7.096295259220584E-02],
    [0, 0, 0, 0, 0, 0, 0, 1., 0, 0, 0, 1.511294855619158E-01, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1., 5.852074246204247E-17, -1.511294855619159E-01,
     0, 1.511294855619157E-01],
    [3.302274274717535E-01, 3.440028787868190E-01, 0, 3.440028787868190E-01,
     2.635539939454993E-01, 0, -1.521629693504350E-01, 0,
     5.852074246204247E-17, 1., 0, 0, 0],
    [-1.035018733501897E-01, 4.564650679801431E-02, 0, -1.518470471490936E-01,
     -1.229114393448020E-01, 0, 7.096295259220584E-02, 0,
     -1.511294855619159E-01, 0, 1., 0, 0],
    [0, 0, 1.974935539471079E-01, 0, 0, 1.511294855619158E-01, 0,
     1.511294855619158E-01, 0, 0, 0, 1, 0],
    [-1.035018733501897E-01, -1.518470471490936E-01, 0, 4.564650679801423E-02,
     -1.229114393448021E-01, 0, 7.096295259220584E-02, 0,
     1.511294855619157E-01, 0, 0, 0, 1.]])

h_auau = torch.tensor([
    [-2.107700668744000E-01, 0, 0, 0, 0, 0, 0, 0, 0, -1.386975525794172E-01,
     1.116190059076135E-02, 0, 1.116190059076135E-02, 4.319236066977346E-02, 0,
     -2.493712105962911E-02, 0, 9.590630660699307E-18],
    [0, -2.785941987392000E-02, 0, 0, 0, 0, 0, 0, 0, -1.116190059076135E-02,
     -6.379458173124930E-02, 0, -4.339795005639185E-02, 9.306222383971831E-02,
     0, -5.372949998524657E-02, 0, -5.592893341553306E-02],
    [0, 0, -2.785941987392000E-02, 0, 0, 0, 0, 0, 0, 0, 0,
     -2.039663167485748E-02, 0, 0, 5.592893341553307E-02, 0,
     5.592893341553307E-02, 0],
    [0, 0, 0, -2.785941987392000E-02, 0, 0, 0, 0, 0, -1.116190059076135E-02,
     -4.339795005639185E-02, 0, -6.379458173124933E-02, 9.306222383971829E-02,
     0, -5.372949998524657E-02, 0, 5.592893341553308E-02],
    [0, 0, 0, 0, -2.531805351853000E-01, 0, 0, 0, 0, 4.319236066977346E-02,
     -9.306222383971831E-02, 0, -9.306222383971829E-02, -2.537050998719814E-01,
     0, 6.834552503631700E-02, 0, -1.395042252142131E-16],
    [0, 0, 0, 0, 0, -2.531805351853000E-01, 0, 0, 0, 0, 0,
     -5.592893341553307E-02, 0, 0, 1.196194493901244E-01, 0,
     2.549466274292341E-01, 0],
    [0, 0, 0, 0, 0, 0, -2.531805351853000E-01, 0, 0, -2.493712105962911E-02,
     5.372949998524657E-02, 0, 5.372949998524657E-02, 6.834552503631700E-02,
     0, -1.747864853167335E-01, 0, 1.517575510508284E-17],
    [0, 0, 0, 0, 0, 0, 0, -2.531805351853000E-01, 0, 0, 0,
     -5.592893341553307E-02, 0, 0, 2.549466274292341E-01, 0,
     1.196194493901245E-01, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, -2.531805351853000E-01, 9.590630660699307E-18,
     5.592893341553306E-02, 0, -5.592893341553308E-02, -1.395042252142131E-16,
     0, 1.517575510508284E-17, 0, 3.745660768193586E-01],
    [-1.386975525794172E-01, -1.116190059076135E-02, 0, -1.116190059076135E-02,
     4.319236066977346E-02, 0, -2.493712105962911E-02, 0,
     9.590630660699307E-18, -2.107700668744000E-01, 0, 0, 0, 0, 0, 0, 0, 0],
    [1.116190059076135E-02, -6.379458173124930E-02, 0, -4.339795005639185E-02,
     -9.306222383971831E-02, 0, 5.372949998524657E-02, 0,
     5.592893341553306E-02, 0, -2.785941987392000E-02, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, -2.039663167485748E-02, 0, 0, -5.592893341553307E-02, 0,
     -5.592893341553307E-02, 0, 0, 0, -2.785941987392000E-02,
     0, 0, 0, 0, 0, 0],
    [1.116190059076135E-02, -4.339795005639185E-02, 0, -6.379458173124933E-02,
     -9.306222383971829E-02, 0, 5.372949998524657E-02, 0,
     -5.592893341553308E-02, 0, 0, 0, -2.785941987392000E-02, 0, 0, 0, 0, 0],
    [4.319236066977346E-02, 9.306222383971831E-02, 0, 9.306222383971829E-02,
     -2.537050998719814E-01, 0, 6.834552503631700E-02, 0,
     -1.395042252142131E-16, 0, 0, 0, 0, -2.531805351853000E-01, 0, 0, 0, 0],
    [0, 0, 5.592893341553307E-02, 0, 0, 1.196194493901244E-01, 0,
     2.549466274292341E-01, 0, 0, 0, 0, 0, 0, -2.531805351853000E-01, 0, 0, 0],
    [-2.493712105962911E-02, -5.372949998524657E-02, 0, -5.372949998524657E-02,
     6.834552503631700E-02, 0, -1.747864853167335E-01, 0,
     1.517575510508284E-17, 0, 0, 0, 0, 0, 0, -2.531805351853000E-01, 0, 0],
    [0, 0, 5.592893341553307E-02, 0, 0, 2.549466274292341E-01, 0,
     1.196194493901245E-01, 0, 0, 0, 0, 0, 0, 0, 0, -2.531805351853000E-01, 0],
    [9.590630660699307E-18, -5.592893341553306E-02, 0, 5.592893341553308E-02,
     -1.395042252142131E-16, 0, 1.517575510508284E-17, 0,
     3.745660768193586E-01, 0, 0, 0, 0, 0, 0, 0, 0, -2.531805351853000E-01]])

s_auau = torch.tensor([
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 5.138504034141882E-01, -2.678271637115430E-01,
     0, -2.678271637115430E-01, -1.360446021310750E-02, 0,
     7.854538766217169E-03, 0, -3.020796993237762E-18],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 2.678271637115430E-01, 2.346359993015822E-01,
     0, -2.547584418029646E-01, -1.064092686555959E-01, 0,
     6.143541990257951E-02, 0, 1.639168193472070E-01],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 4.893944411045468E-01, 0, 0,
     -1.639168193472070E-01, 0, -1.639168193472070E-01, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 2.678271637115430E-01, -2.547584418029646E-01,
     0, 2.346359993015821E-01, -1.064092686555959E-01, 0,
     6.143541990257951E-02, 0, -1.639168193472071E-01],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, -1.360446021310750E-02, 1.064092686555959E-01,
     0, 1.064092686555959E-01, 1.083304386420483E-01, 0, 2.359277379526532E-02,
     0, 8.763646984815479E-17],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1.639168193472070E-01, 0, 0,
     -6.857740638161991E-02, 0, -2.177717279285474E-01, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 7.854538766217169E-03, -6.143541990257951E-02,
     0, -6.143541990257951E-02, 2.359277379526532E-02, 0,
     1.355730272453010E-01, 0, 5.238648136455314E-18],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1.639168193472070E-01, 0, 0,
     -2.177717279285474E-01, 0, -6.857740638161999E-02, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, -3.020796993237762E-18, -1.639168193472070E-01,
     0, 1.639168193472071E-01, 8.763646984815479E-17, 0, 5.238648136455314E-18,
     0, -2.863491343101673E-01],
    [5.138504034141882E-01, 2.678271637115430E-01, 0, 2.678271637115430E-01,
     -1.360446021310750E-02, 0, 7.854538766217169E-03,
     0, -3.020796993237762E-18, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [-2.678271637115430E-01, 2.346359993015822E-01, 0, -2.547584418029646E-01,
     1.064092686555959E-01, 0, -6.143541990257951E-02, 0,
     -1.639168193472070E-01, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 4.893944411045468E-01, 0, 0, 1.639168193472070E-01, 0,
     1.639168193472070E-01, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [-2.678271637115430E-01, -2.547584418029646E-01, 0, 2.346359993015821E-01,
     1.064092686555959E-01, 0, -6.143541990257951E-02, 0,
     1.639168193472071E-01, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [-1.360446021310750E-02, -1.064092686555959E-01, 0, -1.064092686555959E-01,
     1.083304386420483E-01, 0, 2.359277379526532E-02, 0, 8.763646984815479E-17,
     0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, -1.639168193472070E-01, 0, 0, -6.857740638161991E-02, 0,
     -2.177717279285474E-01, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [7.854538766217169E-03, 6.143541990257951E-02, 0, 6.143541990257951E-02,
     2.359277379526532E-02, 0, 1.355730272453010E-01, 0, 5.238648136455314E-18,
     0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, -1.639168193472070E-01, 0, 0, -2.177717279285474E-01, 0,
     -6.857740638161999E-02, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [-3.020796993237762E-18, 1.639168193472070E-01, 0, -1.639168193472071E-01,
     8.763646984815479E-17, 0, 5.238648136455314E-18, 0,
     -2.863491343101673E-01, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
