import numpy as np
import etorch
import etorch.nn as nn
import etorch.functional as F
import etorch.optim as optim


def test_creation_and_add():
    a = etorch.tensor([1.0, 2.0], requires_grad=True)
    b = etorch.tensor([3.0, 4.0], requires_grad=True)
    c = a + b
    assert isinstance(c, etorch.Tensor)
    assert np.allclose(c.numpy(), [4, 6])
    c.backward(etorch.tensor([1.0, 1.0]))
    assert np.allclose(a.grad, [1, 1])


def test_matmul_backward():
    etorch.manual_seed(0)
    a = etorch.randn(4, 5, requires_grad=True)
    b = etorch.randn(5, 3, requires_grad=True)
    c = a @ b
    assert c.shape == (4, 3)
    c.sum().backward()
    assert a.grad.shape == (4, 5) and b.grad.shape == (5, 3)


def test_broadcast_backward():
    a = etorch.ones((3, 4), requires_grad=True)
    b = etorch.tensor(2.0, requires_grad=True)
    (a * b).sum().backward()
    assert np.allclose(b.grad, 12.0)


def test_mlp_trains_xor():
    etorch.manual_seed(1)
    X = etorch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    y = etorch.tensor([0, 1, 1, 0])
    model = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 2))
    opt = optim.Adam(model.parameters(), lr=0.05)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(400):
        opt.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        opt.step()
    with etorch.no_grad():
        pred = model(X).numpy().argmax(1)
    assert (pred == np.array([0, 1, 1, 0])).all(), pred


def test_save_load_and_dataloader():
    import tempfile, os
    etorch.manual_seed(0)
    X = etorch.randn(20, 4)
    y = etorch.randint(0, 2, size=20)
    ds = etorch.TensorDataset(X, y)
    dl = etorch.DataLoader(ds, batch_size=5, shuffle=True)
    n = sum(1 for _ in dl)
    assert n == 4
    m = nn.Linear(4, 2)
    sd = m.state_dict()
    m2 = nn.Linear(4, 2)
    m2.load_state_dict(sd)
    assert etorch.allclose(m.weight.numpy(), m2.weight.numpy())


def test_conv2d_forward_backward():
    etorch.manual_seed(0)
    conv = nn.Conv2d(1, 2, 3, padding=1)
    x = etorch.randn(2, 1, 8, 8, requires_grad=True)
    y = conv(x)
    assert y.shape == (2, 2, 8, 8)
    y.sum().backward()
    assert x.grad.shape == (2, 1, 8, 8)
