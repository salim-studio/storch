import numpy as np
import storch
import storch.nn as nn
import storch.functional as F
import storch.optim as optim
import storch.preprocessing as pp
import storch.metrics as metrics
import storch.models as models
import storch.datasets as datasets
import storch.db as db
import storch.io as eio


def test_indexing_and_where():
    a = storch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    assert np.allclose(a[0].numpy(), [1, 2])
    assert np.allclose(a[:, 1].numpy(), [2, 4])
    a[0, 0] = 10.0
    assert a.numpy()[0, 0] == 10.0
    b = storch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b[0].sum().backward()
    assert np.allclose(b.grad, [[1, 1], [0, 0]])
    w = storch.where(storch.tensor([True, False]), storch.tensor([1.0, 2.0]), 0.0)
    assert np.allclose(w.numpy(), [1, 0])


def test_ops_and_linalg():
    t = storch.tensor([3.0, 1.0, 2.0])
    vals, idx = storch.topk(t, 2)
    assert np.allclose(vals.numpy(), [3, 2]) and idx.numpy().tolist() == [0, 2]
    assert storch.argmax(t).item() == 0
    assert storch.norm(storch.tensor([3.0, 4.0])).item() == np.float32(5.0)
    assert np.allclose(storch.linalg.inv(storch.eye(2)).numpy(), np.eye(2))
    assert storch.linalg.det(storch.eye(3)).item() == 1.0
    assert storch.einsum("ij,jk->ik", storch.ones(2, 3), storch.ones(3, 4)).shape == (2, 4)


def test_db_and_io(tmp_path):
    con = db.connect(":memory:")
    db.create_table(con, "t", {"a": "REAL", "b": "INTEGER"})
    db.insert_rows(con, "t", ["a", "b"], [(1.0, 0), (2.0, 1), (3.0, 1)])
    tbl = db.read_sql(con, "SELECT * FROM t WHERE a > 1")
    assert tbl.shape == (2, 2)
    X, y = tbl.to_tensors(["a"], "b")
    assert X.shape == (2, 1) and y.numpy().tolist() == [1, 1]
    p = str(tmp_path / "t.csv")
    eio.save_csv(p, tbl)
    t2 = eio.load_csv(p)
    assert t2.shape == (2, 2)
    d = db.DatabaseDataset(con, "SELECT * FROM t", ["a"], "b")
    assert len(d) == 3


def test_preprocessing_metrics_datasets_models():
    import numpy as _np
    Xa = _np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    sc = pp.StandardScaler()
    Xt = sc.fit_transform(Xa)
    assert abs(Xt.mean()) < 1e-6
    Xtr, Xte = pp.train_test_split(Xa, test_size=0.25, seed=0)
    assert len(Xtr) == 3 and len(Xte) == 1
    assert metrics.accuracy([0, 1, 1], [0, 1, 0]) == 2 / 3
    assert metrics.r2_score([1.0, 2.0], [1.0, 2.0]) == 1.0
    assert metrics.f1_score([0, 1, 1, 1], [0, 1, 0, 1]) > 0.7
    Xc, yc = datasets.make_classification(n_samples=60, seed=0)
    assert Xc.shape == (60, 4)
    lr = models.LogisticRegression(epochs=60).fit(Xc, yc)
    assert lr.score(Xc, yc) > 0.8
    km = models.KMeans(n_clusters=2, seed=0).fit(Xc)
    assert len(km.labels_) == 60
    assert models.KNNClassifier(3).fit(Xc, yc).score(Xc, yc) > 0.8
    assert models.PCA(2).fit_transform(Xc).shape == (60, 2)
    assert models.LinearRegression().fit(*datasets.make_regression(seed=0)).coef_.shape == (4,)


def test_new_nn_blocks():
    storch.manual_seed(0)
    x = storch.randn(8, 4, requires_grad=True)
    y = nn.BatchNorm1d(4)(x)
    assert y.shape == (8, 4)
    y.sum().backward()
    assert x.grad.shape == (8, 4)
    xp = storch.randn(2, 2, 4, 4, requires_grad=True)
    assert nn.MaxPool2d(2)(xp).shape == (2, 2, 2, 2)
    assert nn.AvgPool2d(2)(xp).shape == (2, 2, 2, 2)
    assert nn.AdaptiveAvgPool2d(1)(xp).shape == (2, 2, 1, 1)
    c1 = nn.Conv1d(2, 4, 3, padding=1)
    x1 = storch.randn(2, 2, 8, requires_grad=True)
    assert c1(x1).shape == (2, 4, 8)
    xr = storch.randn(2, 5, 4)
    o, h = nn.RNN(4, 8)(xr)
    assert o.shape == (2, 5, 8)
    o, _ = nn.LSTM(4, 8)(xr)
    assert o.shape == (2, 5, 8)
    o, _ = nn.GRU(4, 8)(xr)
    assert o.shape == (2, 5, 8)
    q = storch.randn(2, 5, 16)
    assert nn.MultiheadAttention(16, 4)(q, q, q).shape == (2, 5, 16)
    assert nn.TransformerEncoderLayer(16, 4)(q).shape == (2, 5, 16)
    assert nn.HuberLoss()(storch.tensor([1.0]), storch.tensor([2.0])).numpy().size == 1
    # eval-mode batchnorm stays differentiable
    bn = nn.BatchNorm1d(4).eval()
    xe = storch.randn(4, 4, requires_grad=True)
    bn(xe).sum().backward()
    assert xe.grad.shape == (4, 4)


def test_dataloader_utils():
    ds = storch.TensorDataset(storch.randn(10, 3), storch.randint(0, 2, size=10))
    tr, te = storch.random_split(ds, [8, 2], seed=0)
    assert len(tr) == 8 and len(te) == 2
    sub = storch.Subset(ds, [0, 1, 2])
    assert len(sub) == 3
    dl = storch.DataLoader(tr, batch_size=4)
    assert len(dl) == 2


def test_etorch_compat_shim():
    import sys
    import warnings
    sys.modules.pop("etorch", None)
    sys.modules.pop("etorch.nn", None)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        import etorch
        import etorch.nn  # noqa: F401
    assert any(isinstance(w.message, DeprecationWarning) for w in rec)
    assert etorch.__version__ == storch.__version__
    assert etorch.tensor([1.0]).numpy().tolist() == [1.0]
