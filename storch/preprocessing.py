"""storch.preprocessing: data analyst / scientist toolkit — sklearn-compatible API.

StandardScaler / MinMaxScaler / OneHotEncoder / SimpleImputer / LabelEncoder /
train_test_split / shuffle — work on both numpy arrays and storch.Tensors.
"""
from __future__ import annotations

import numpy as _np

from ._core import Tensor, _as_t

__all__ = [
    "train_test_split", "shuffle", "StandardScaler", "MinMaxScaler",
    "MaxAbsScaler", "RobustScaler", "OneHotEncoder", "LabelEncoder",
    "SimpleImputer", "PolynomialFeatures",
]


def _to_arr(x):
    return x._data if isinstance(x, Tensor) else _np.asanyarray(x)


def _from_arr(a, ref):
    return Tensor(a) if isinstance(ref, Tensor) else _np.ascontiguousarray(a)


def shuffle(*arrays, seed=None):
    rng = _np.random.default_rng(seed)
    n = len(_to_arr(arrays[0]))
    idx = rng.permutation(n)
    out = []
    for a in arrays:
        arr = _to_arr(a)
        out.append(_from_arr(arr[idx], a))
    return out[0] if len(out) == 1 else tuple(out)


def train_test_split(*arrays, test_size=0.2, seed=None, shuffle_data=True, stratify=None):
    """Torch/sklearn-style split. stratify: class labels for stratified splitting."""
    rng = _np.random.default_rng(seed)
    n = len(_to_arr(arrays[0]))
    for a in arrays:
        assert len(_to_arr(a)) == n, "all arrays must have same first dim"
    if stratify is not None:
        y = _np.asanyarray(_to_arr(stratify)).reshape(-1)
        idx_tr, idx_te = [], []
        for cls in _np.unique(y):
            ci = _np.where(y == cls)[0]
            if shuffle_data:
                ci = rng.permutation(ci)
            k = int(round(len(ci) * test_size))
            idx_te.extend(ci[:k].tolist())
            idx_tr.extend(ci[k:].tolist())
        idx_tr, idx_te = _np.array(idx_tr), _np.array(idx_te)
        if shuffle_data:
            idx_tr, idx_te = rng.permutation(idx_tr), rng.permutation(idx_te)
    else:
        idx = _np.arange(n)
        if shuffle_data:
            idx = rng.permutation(idx)
        import math as _m
        k = int(_m.ceil(test_size * n)) if isinstance(test_size, float) else int(test_size)
        idx_te, idx_tr = idx[:k], idx[k:]
    out = []
    for a in arrays:
        arr = _to_arr(a)
        out += [_from_arr(arr[idx_tr], a), _from_arr(arr[idx_te], a)]
    return tuple(out)


class StandardScaler:
    """z = (x - mean) / scale_"""

    def __init__(self, with_mean=True, with_std=True):
        self.with_mean, self.with_std = with_mean, with_std

    def fit(self, X):
        a = _to_arr(X).astype(float)
        self.mean_ = a.mean(axis=0) if self.with_mean else _np.zeros(a.shape[1])
        s = a.std(axis=0) if self.with_std else _np.ones(a.shape[1])
        self.scale_ = _np.where(s == 0, 1.0, s)
        return self

    def transform(self, X):
        a = _to_arr(X).astype(float)
        out = ((a - self.mean_) / self.scale_).astype(_np.float32)
        return _from_arr(out, X)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        a = _to_arr(X).astype(float)
        out = (a * self.scale_ + self.mean_).astype(_np.float32)
        return _from_arr(out, X)


class MinMaxScaler:
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range

    def fit(self, X):
        a = _to_arr(X).astype(float)
        self.min_, self.max_ = a.min(axis=0), a.max(axis=0)
        return self

    def transform(self, X):
        a = _to_arr(X).astype(float)
        rng = _np.where(self.max_ == self.min_, 1.0, self.max_ - self.min_)
        lo, hi = self.feature_range
        out = ((a - self.min_) / rng * (hi - lo) + lo).astype(_np.float32)
        return _from_arr(out, X)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        a = _to_arr(X).astype(float)
        lo, hi = self.feature_range
        rng = _np.where(self.max_ == self.min_, 1.0, self.max_ - self.min_)
        return _from_arr((((a - lo) / (hi - lo)) * rng + self.min_).astype(_np.float32), X)


class MaxAbsScaler:
    def fit(self, X):
        a = _to_arr(X).astype(float)
        self.max_abs_ = _np.where((_np.abs(a).max(axis=0)) == 0, 1.0, _np.abs(a).max(axis=0))
        return self

    def transform(self, X):
        return _from_arr((_to_arr(X).astype(float) / self.max_abs_).astype(_np.float32), X)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class RobustScaler:
    def fit(self, X):
        a = _to_arr(X).astype(float)
        self.median_ = _np.median(a, axis=0)
        q75, q25 = _np.percentile(a, 75, axis=0), _np.percentile(a, 25, axis=0)
        self.iqr_ = _np.where((q75 - q25) == 0, 1.0, q75 - q25)
        return self

    def transform(self, X):
        return _from_arr(((_to_arr(X).astype(float) - self.median_) / self.iqr_).astype(_np.float32), X)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class OneHotEncoder:
    def __init__(self, dtype=_np.float32):
        self.dtype = dtype

    def fit(self, y):
        a = _np.asanyarray(_to_arr(y)).reshape(-1)
        self.classes_ = _np.unique(a)
        self._map = {v: i for i, v in enumerate(self.classes_.tolist())}
        return self

    def transform(self, y):
        a = _np.asanyarray(_to_arr(y)).reshape(-1)
        idx = _np.array([self._map.get(v, -1) for v in a.tolist()])
        out = _np.zeros((len(a), len(self.classes_)), dtype=self.dtype)
        valid = idx >= 0
        out[valid, idx[valid]] = 1
        return Tensor(out) if isinstance(y, Tensor) else out

    def fit_transform(self, y):
        return self.fit(y).transform(y)


class LabelEncoder:
    def fit(self, y):
        a = _np.asanyarray(_to_arr(y)).reshape(-1)
        self.classes_ = _np.unique(a)
        self._map = {v: i for i, v in enumerate(self.classes_.tolist())}
        return self

    def transform(self, y):
        a = _np.asanyarray(_to_arr(y)).reshape(-1)
        return _np.array([self._map[v] for v in a.tolist()], dtype=_np.int64)

    def fit_transform(self, y):
        return self.fit(y).transform(y)

    def inverse_transform(self, idx):
        return _np.array([self.classes_[i] for i in _np.asanyarray(idx).reshape(-1)])


class SimpleImputer:
    def __init__(self, strategy="mean", fill_value=0.0):
        assert strategy in ("mean", "median", "most_frequent", "constant")
        self.strategy, self.fill_value = strategy, fill_value

    def fit(self, X):
        a = _to_arr(X).astype(float)
        if self.strategy == "mean":
            self.stat_ = _np.nanmean(a, axis=0)
        elif self.strategy == "median":
            self.stat_ = _np.nanmedian(a, axis=0)
        elif self.strategy == "most_frequent":
            self.stat_ = _np.array([_np.bincount(_np.nan_to_num(c, nan=-999999).astype(int)).argmax()
                                    for c in a.T], dtype=float)
        else:
            self.stat_ = _np.full(a.shape[1], self.fill_value)
        self.stat_ = _np.where(_np.isnan(self.stat_), self.fill_value, self.stat_)
        return self

    def transform(self, X):
        a = _to_arr(X).astype(float)
        out = _np.where(_np.isnan(a), self.stat_, a).astype(_np.float32)
        return _from_arr(out, X)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class PolynomialFeatures:
    def __init__(self, degree=2, include_bias=True):
        self.degree, self.include_bias = degree, include_bias

    def fit(self, X):
        return self

    def transform(self, X):
        a = _to_arr(X).astype(float)
        n, d = a.shape
        cols = [ _np.ones((n, 1)) ] if self.include_bias else []
        cols.append(a)
        cur = a
        for _ in range(2, self.degree + 1):
            nxt = []
            for j in range(d):
                nxt.append((cur * a[:, j:j + 1]))
            # keep degree-wise products compact: use elementwise powers + interactions of raw features
            cur = _np.concatenate(nxt, axis=1) if nxt else cur
            cols.append(cur)
        out = _np.concatenate(cols, axis=1).astype(_np.float32)
        return _from_arr(out, X)

    def fit_transform(self, X):
        return self.fit(X).transform(X)
