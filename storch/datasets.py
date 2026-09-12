"""storch.datasets: ready-made datasets + synthetic generators for learning and testing.

- make_classification / make_regression / make_blobs / make_moons / make_circles
- load_iris / load_wine (bundled, no internet needed)
- CSVDataset: wraps a CSV file directly as a Dataset
"""
from __future__ import annotations

import math

import numpy as _np

from ._core import Tensor
from .utils.data import Dataset

__all__ = [
    "make_classification", "make_regression", "make_blobs",
    "make_moons", "make_circles", "load_iris", "load_wine",
    "CSVDataset", "ToyDataset",
]


def _rng(seed):
    return _np.random.default_rng(seed)


def make_classification(n_samples=200, n_features=4, n_classes=2, n_informative=None,
                        sep=2.0, seed=0):
    rng = _rng(seed)
    n_informative = n_informative or min(n_features, 2)
    X = rng.standard_normal((n_samples, n_features)).astype(_np.float32)
    W = rng.standard_normal((n_informative, n_classes)).astype(_np.float32) * sep
    logits = X[:, :n_informative] @ W
    y = logits.argmax(axis=1).astype(_np.int64)
    return Tensor(X), Tensor(y)


def make_regression(n_samples=200, n_features=4, noise=0.5, seed=0):
    rng = _rng(seed)
    X = rng.standard_normal((n_samples, n_features)).astype(_np.float32)
    w = rng.standard_normal((n_features, 1)).astype(_np.float32)
    y = (X @ w).reshape(-1) + rng.standard_normal(n_samples).astype(_np.float32) * noise
    return Tensor(X), Tensor(y)


def make_blobs(n_samples=200, n_features=2, centers=3, cluster_std=1.0, seed=0):
    rng = _rng(seed)
    centers = rng.standard_normal((centers, n_features)).astype(_np.float32) * 5
    per = n_samples // len(centers)
    Xs, ys = [], []
    for i, c in enumerate(centers):
        n = per if i < len(centers) - 1 else n_samples - per * (len(centers) - 1)
        Xs.append(c + rng.standard_normal((n, n_features)).astype(_np.float32) * cluster_std)
        ys.append(_np.full(n, i, dtype=_np.int64))
    return Tensor(_np.concatenate(Xs).astype(_np.float32)), Tensor(_np.concatenate(ys))


def make_moons(n_samples=200, noise=0.1, seed=0):
    rng = _rng(seed)
    n1, n2 = n_samples // 2, n_samples - n_samples // 2
    t1 = _np.linspace(0, math.pi, n1)
    t2 = _np.linspace(0, math.pi, n2)
    outer = _np.stack([_np.cos(t1), _np.sin(t1)], 1)
    inner = _np.stack([1 - _np.cos(t2), 1 - _np.sin(t2) - 0.5], 1)
    X = _np.concatenate([outer, inner]).astype(_np.float32)
    X += rng.standard_normal(X.shape).astype(_np.float32) * noise
    y = _np.concatenate([_np.zeros(n1), _np.ones(n2)]).astype(_np.int64)
    return Tensor(X), Tensor(y)


def make_circles(n_samples=200, noise=0.05, factor=0.5, seed=0):
    rng = _rng(seed)
    n1, n2 = n_samples // 2, n_samples - n_samples // 2
    t1 = rng.uniform(0, 2 * math.pi, n1)
    t2 = rng.uniform(0, 2 * math.pi, n2)
    outer = _np.stack([_np.cos(t1), _np.sin(t1)], 1)
    inner = _np.stack([_np.cos(t2) * factor, _np.sin(t2) * factor], 1)
    X = _np.concatenate([outer, inner]).astype(_np.float32)
    X += rng.standard_normal(X.shape).astype(_np.float32) * noise
    y = _np.concatenate([_np.zeros(n1), _np.ones(n2)]).astype(_np.int64)
    return Tensor(X), Tensor(y)


def load_iris():
    """Bundled Iris-like dataset (150 samples) with a fixed seed — no download needed."""
    rng = _rng(42)
    means = _np.array([[5.0, 3.4, 1.5, 0.2], [5.9, 2.8, 4.2, 1.3], [6.5, 3.0, 5.5, 2.0]])
    cov = _np.array([[0.12, 0.05, 0.02, 0.01], [0.05, 0.10, 0.02, 0.01],
                     [0.02, 0.02, 0.18, 0.03], [0.01, 0.01, 0.03, 0.03]])
    Xs = [rng.multivariate_normal(m, cov, 50) for m in means]
    X = _np.concatenate(Xs).astype(_np.float32)
    y = _np.concatenate([_np.full(50, i) for i in range(3)]).astype(_np.int64)
    return Tensor(X), Tensor(y)


def load_wine():
    rng = _rng(7)
    means = _np.array([[13.5, 2.0, 2.4, 18.0, 105.0], [12.3, 2.5, 2.2, 20.0, 95.0], [13.0, 3.0, 2.8, 22.0, 110.0]])
    std = _np.array([0.8, 0.4, 0.3, 2.0, 10.0])
    Xs = [m + rng.standard_normal((60, 5)) * std for m in means]
    X = _np.concatenate(Xs).astype(_np.float32)
    y = _np.concatenate([_np.full(60, i) for i in range(3)]).astype(_np.int64)
    return Tensor(X), Tensor(y)


class ToyDataset(Dataset):
    def __init__(self, X, y=None):
        from ._core import _as_t
        self.X = _as_t(X)
        self.y = _as_t(y) if y is not None else None

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        if self.y is None:
            return (Tensor(self.X._data[i]),)
        return Tensor(self.X._data[i]), Tensor(_np.asanyarray(self.y._data[i]))


class CSVDataset(Dataset):
    """A Dataset that reads a CSV once and serves Tensors (beginner friendly)."""

    def __init__(self, path, target=None, features=None, dtype=_np.float32, **csv_kwargs):
        from .io import load_csv as _lc
        t = _lc(path, **csv_kwargs)
        feats = features or ([c for c in t.columns if c != target] if target else t.columns)
        self.tensors = t.to_tensors(feats, target, dtype)
        self.has_target = target is not None

    def __len__(self):
        X = self.tensors[0] if self.has_target else self.tensors
        return X.shape[0]

    def __getitem__(self, i):
        if not self.has_target:
            return (Tensor(self.tensors._data[i]),)
        X, y = self.tensors
        return Tensor(X._data[i]), Tensor(_np.asanyarray(y._data[i]))
