import torch
import pytest
from tbmalt.tests.test_utils import *
from torch.autograd import gradcheck
from tbmalt.common.maths.mixer import Simple, Anderson
torch.set_default_dtype(torch.float64)
torch.set_printoptions(15)


def _simple(qnew, qold, mix_param=0.2):
    """Example simple mixing for testing."""
    return qold + (qnew - qold) * mix_param


def test_simple_single(device):
    """Test single molecule SK integral value."""
    qzero = torch.tensor([4., 1., 1., 1., 1.])
    qnew = torch.tensor([
        [4.430184080858728, 0.884728369453420, 0.895065983990741,
         0.881608314535792, 0.908413251161311],
        [4.410630195445064, 0.890028160886006, 0.899293372122927,
         0.887168618521978, 0.912879653024026],
        [4.354309097711402, 0.905285555553202, 0.911574975058711,
         0.903151734339127, 0.925678637337559],
        [4.354276884239578, 0.905274734260467, 0.911730435030402,
         0.903107467323003, 0.925610479146557]])
    mixer = Simple(qzero, return_convergence=False)
    qmix_ref = []
    qmix_ref.append(qzero)
    for ii in range(4):
        _qmix_ref = _simple(qnew[ii], qmix_ref[0], 0.2)
        qmix_ref.insert(0, _qmix_ref)
        qmix = mixer(qnew[ii])
        assert torch.max(abs(qmix - _qmix_ref)) < 1E-14


def test_simple_batch_mask(device):
    """Test single molecule SK integral value, mainly test mask."""
    qzero = torch.tensor([[4., 1., 1., 1., 1.],
                          [6., 1., 1., 0., 0.],
                          [4., 1., 1., 1., 1.]])
    qnew = torch.tensor([[4.5, 0.8, 0.9, 0.95, 0.85],
                         [6.5, 0.8, 0.7, 0., 0.],
                         [4.4, 0.85, 0.9, 0.95, 0.9]])
    mixer = Simple(qzero, mix_param=0.2, return_convergence=True)
    mask = torch.tensor([True, True, False])
    qmix_ref = _simple(qnew, qzero, 0.2)
    qmix = mixer(qnew[mask], mask=mask)[0]
    assert torch.max(abs(qmix - qmix_ref[:2])) < 1E-14


def test_simple_batch(device):
    """Test single molecule SK integral value, mainly test mask."""
    qzero = torch.tensor([[4., 1., 1., 1., 1.],
                          [6., 1., 1., 0., 0.],
                          [4., 1., 1., 1., 1.]])
    qnew = torch.tensor([[4.5, 0.8, 0.9, 0.95, 0.85],
                         [6.5, 0.8, 0.7, 0., 0.],
                         [4.4, 0.85, 0.9, 0.95, 0.9]])
    mixer = Simple(qzero, mix_param=0.2)
    qmix_ref = _simple(qnew, qzero, 0.2)
    qmix = mixer(qnew)
    assert torch.max(abs(qmix - qmix_ref)) < 1E-14


@pytest.mark.grad
def test_sym_grad(device):
    """Gradient evaluation of maths.sym function."""
    qzero = torch.tensor([4., 1., 1., 1., 1.])
    qnew = torch.tensor([4.5, 0.8, 0.9, 0.95, 0.85],
                        requires_grad=True, device=device)
    q_ref = torch.tensor([[4.3, 0.87, 0.95, 0.98, 0.9]])
    mixer = Simple(qzero, mix_param=0.2)
    optimizer = torch.optim.Adam([qnew], lr=0.01)
    criterion = torch.nn.MSELoss(reduction='sum')
    qmix = mixer(qnew)
    loss = criterion(qmix, q_ref)
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()
    grad_is_safe = gradcheck(mixer, (qnew), raise_exception=False)
    # assert grad_is_safe, 'Gradient stability test'


def test_anderson_single(device):
    """Test single molecule SK integral value, test tolerance."""
    qzero = torch.tensor([4., 1., 1., 1., 1.])
    mixer = Anderson(qzero, mix_param=0.2)
    for ii in range(5):
        qmix = mixer(qdiff[ii] + qmix_ref[ii])
        assert torch.max(abs(qmix - qmix_ref[ii + 1])) < 1E-14


def test_anderson_single_qold(device):
    """Test single molecule SK integral value, test with q_old input."""
    qzero = torch.tensor([4., 1., 1., 1., 1.])
    mixer = Anderson(qzero, mix_param=0.2)
    qold = []
    qold.append(qzero)
    for ii in range(5):
        qmix = mixer(qdiff[ii] + qmix_ref[ii], q_old=qold[-1])
        qold.append(qmix)
        assert torch.max(abs(qmix - qmix_ref[ii + 1])) < 1E-14


def test_anderson_batch(device):
    """Test single molecule SK integral value."""
    qzero = torch.tensor([[4., 1., 1., 1., 1.],
                          [1., 1., 0., 0., 0.]])
    mixer = Anderson(qzero, mix_param=0.2, return_convergence=True)
    for ii in range(5):
        qnew = torch.zeros(*qzero.shape)
        qnew[0], qnew[1, :2] = qdiff[ii] + qmix_ref[ii], torch.tensor([1., 1.])
        qmix = mixer(qnew)[0]
        assert torch.max(abs(qmix[0] - qmix_ref[ii + 1])) < 1E-14


qdiff = torch.tensor([
    [0.355674578080263, -9.313540467944026E-002, -7.372362289040546E-002,
     -0.101674861511676, -8.714068899873795E-002],
    [0.271174405657918, -7.114043461418307E-002, -5.575857331110201E-002,
     -7.804264734223987E-002, -6.623275039039367E-002],
    [-4.659614061708339E-005, -3.807750010893507E-004, 1.403622903409785E-003,
     -1.620151062899589E-003, 6.438993012007899E-004],
    [-2.960195929091469E-005, -2.693973553786755E-004, 1.030490367038506E-003,
     -1.199859284013671E-003, 4.683682316486415E-004],
    [-9.189995253677807E-008, 3.364955236939693E-005, -3.755624609191877E-006,
     -1.788599732155394E-005, -1.191603049000012E-005]])

qmix_ref = torch.tensor([
    [4., 1., 1., 1., 1.],
    [4.07113491561605, 0.981372919064112, 0.985255275421919, 0.979665027697665,
     0.982571862200252],
    [4.29947455731876, 0.921469841887761, 0.938304337626661, 0.913950012711562,
     0.926801250455258],
    [4.29949281671084, 0.921386000174856, 0.938580993725389, 0.913616168021637,
     0.926924021367275],
    [4.29955862763525, 0.921157597155631, 0.939347646476122, 0.912676962967435,
     0.927259165765570],
    [4.29955857460661, 0.921164731062883, 0.939347116135461, 0.912672833168578,
     0.927256745026475]])

# @pytest.mark.grad
# def test_sym_grad(device):
#     """Gradient evaluation of maths.sym function."""
#     qzero = torch.tensor([4., 1., 1., 1., 1.])
#     qnew = torch.tensor([4.5, 0.8, 0.9, 0.95, 0.85],
#                         requires_grad=True, device=device)
#     mixer = Anderson(qzero, mix_param=0.2)
#     grad_is_safe = gradcheck(mixer, (qnew), raise_exception=False)
#     assert grad_is_safe, 'Gradient stability test'
