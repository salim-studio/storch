"""storch.models: classical machine-learning algorithms — sklearn-compatible API.

LinearRegression / Ridge / LogisticRegression / KNNClassifier / KNNRegressor /
KMeans / GaussianNB / PCA — numpy-powered (fast on CPU), accept Tensors and return Tensors.
"""
from __future__ import annotations

import numpy as _np

from ._core import Tensor

__all__ = [
    "LinearRegression", "Ridge", "LogisticRegression",
    "KNNClassifier", "KNNRegressor", "KMeans", "GaussianNB", "PCA",
]


def _X(x):
    return x._data if isinstance(x, Tensor) else _np.asanyarray(x, dtype=float)


def _y(x):
    return (x._data if isinstance(x, Tensor) else _np.asanyarray(x)).reshape(-1)


class LinearRegression:
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        A = _X(X).astype(float)
        b = _y(y).astype(float)
        if self.fit_intercept:
            A = _np.concatenate([A, _np.ones((len(A), 1))], axis=1)
        sol, *_ = _np.linalg.lstsq(A, b, rcond=None)
        if self.fit_intercept:
            self.coef_, self.intercept_ = sol[:-1].astype(_np.float32), float(sol[-1])
        else:
            self.coef_, self.intercept_ = sol.astype(_np.float32), 0.0
        return self

    def predict(self, X):
        A = _X(X).astype(float)
        out = A @ self.coef_.astype(float) + self.intercept_
        return Tensor(out.astype(_np.float32)) if isinstance(X, Tensor) else out.astype(_np.float32)

    def score(self, X, y):
        from .metrics import r2_score
        return r2_score(self.predict(X), y)


class Ridge(LinearRegression):
    def __init__(self, alpha=1.0, fit_intercept=True):
        super().__init__(fit_intercept)
        self.alpha = alpha

    def fit(self, X, y):
        A = _X(X).astype(float)
        b = _y(y).astype(float)
        n = A.shape[1]
        if self.fit_intercept:
            m = A.mean(axis=0)
            A = A - m
            bc = b.mean()
            b = b - bc
            coef = _np.linalg.solve(A.T @ A + self.alpha * _np.eye(n), A.T @ b)
            self.coef_ = coef.astype(_np.float32)
            self.intercept_ = float(bc - m @ coef)
        else:
            self.coef_ = _np.linalg.solve(A.T @ A + self.alpha * _np.eye(n), A.T @ b).astype(_np.float32)
            self.intercept_ = 0.0
        return self


class LogisticRegression:
    """Multiclass logistic regression (softmax, trained with storch autograd + Adam)."""

    def __init__(self, lr=0.1, epochs=500, l2=1e-4, seed=0, verbose=False):
        self.lr, self.epochs, self.l2 = lr, epochs, l2
        self.seed, self.verbose = seed, verbose

    def fit(self, X, y):
        import storch as _E
        _E.manual_seed(self.seed)
        A = _X(X).astype(_np.float32)
        b = _y(y).astype(_np.int64)
        n, d = A.shape
        C = int(b.max()) + 1
        Xt = Tensor(A)
        yt = Tensor(b)
        W = Tensor(_np.zeros((d, C), dtype=_np.float32), requires_grad=True)
        Bi = Tensor(_np.zeros(C, dtype=_np.float32), requires_grad=True)
        from . import optim as _O
        opt = _O.Adam([W, Bi], lr=self.lr)
        from . import functional as _F
        for ep in range(self.epochs):
            opt.zero_grad()
            loss = _F.cross_entropy(Xt @ W + Bi, yt)
            if self.l2:
                loss = loss + (W * W).sum() * (self.l2 / 2)
            loss.backward()
            opt.step()
        self.coef_ = W._data.copy()
        self.intercept_ = Bi._data.copy()
        self.classes_ = _np.arange(C)
        return self

    def predict_proba(self, X):
        A = _X(X).astype(_np.float32)
        z = A @ self.coef_ + self.intercept_
        e = _np.exp(z - z.max(axis=1, keepdims=True))
        return (e / e.sum(axis=1, keepdims=True)).astype(_np.float32)

    def predict(self, X):
        p = self.predict_proba(X).argmax(axis=1).astype(_np.int64)
        return Tensor(p) if isinstance(X, Tensor) else p

    def score(self, X, y):
        from .metrics import accuracy
        return accuracy(self.predict(X), y)


class KNNClassifier:
    def __init__(self, n_neighbors=5, weights="uniform"):
        self.k, self.weights = n_neighbors, weights

    def fit(self, X, y):
        self.X_ = _X(X).astype(float)
        self.y_ = _y(y)
        self.classes_ = _np.unique(self.y_)
        return self

    def predict(self, X):
        A = _X(X).astype(float)
        D = ((A[:, None, :] - self.X_[None, :, :]) ** 2).sum(axis=2)
        k = min(self.k, len(self.X_))
        idx = _np.argpartition(D, k - 1, axis=1)[:, :k]
        out = []
        for i in range(len(A)):
            lab, cnt = _np.unique(self.y_[idx[i]], return_counts=True)
            if self.weights == "distance":
                w = 1 / (_np.sqrt(D[i][idx[i]]) + 1e-8)
                s = {}
                for l, c in zip(self.y_[idx[i]], w):
                    s[l] = s.get(l, 0) + c
                out.append(max(s, key=s.get))
            else:
                out.append(lab[_np.argmax(cnt)])
        out = _np.array(out)
        return Tensor(out.astype(_np.int64)) if isinstance(X, Tensor) else out

    def score(self, X, y):
        from .metrics import accuracy
        return accuracy(self.predict(X), y)


class KNNRegressor:
    def __init__(self, n_neighbors=5):
        self.k = n_neighbors

    def fit(self, X, y):
        self.X_ = _X(X).astype(float)
        self.y_ = _y(y).astype(float)
        return self

    def predict(self, X):
        A = _X(X).astype(float)
        D = ((A[:, None, :] - self.X_[None, :, :]) ** 2).sum(axis=2)
        k = min(self.k, len(self.X_))
        idx = _np.argpartition(D, k - 1, axis=1)[:, :k]
        out = self.y_[idx].mean(axis=1).astype(_np.float32)
        return Tensor(out) if isinstance(X, Tensor) else out


class KMeans:
    def __init__(self, n_clusters=3, max_iter=100, seed=0, n_init=3):
        self.K, self.max_iter, self.seed, self.n_init = n_clusters, max_iter, seed, n_init

    def fit(self, X):
        A = _X(X).astype(float)
        best = None
        rng = _np.random.default_rng(self.seed)
        for _ in range(self.n_init):
            ci = rng.choice(len(A), self.K, replace=False)
            C = A[ci].copy()
            for _ in range(self.max_iter):
                D = ((A[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
                lab = D.argmin(axis=1)
                Cn = _np.array([A[lab == k].mean(axis=0) if (lab == k).any() else C[k] for k in range(self.K)])
                if _np.allclose(C, Cn):
                    C = Cn
                    break
                C = Cn
            inertia = float(((A - C[lab]) ** 2).sum())
            if best is None or inertia < best[0]:
                best = (inertia, C, lab)
        self.inertia_, self.cluster_centers_, self.labels_ = best[0], best[1].astype(_np.float32), best[2]
        return self

    def predict(self, X):
        A = _X(X).astype(float)
        D = ((A[:, None, :] - self.cluster_centers_[None, :, :]) ** 2).sum(axis=2)
        out = D.argmin(axis=1).astype(_np.int64)
        return Tensor(out) if isinstance(X, Tensor) else out

    def fit_predict(self, X):
        return self.fit(X).labels_

    def score(self, X):
        from .metrics import silhouette_score
        return silhouette_score(X, self.predict(X))


class GaussianNB:
    def fit(self, X, y):
        A = _X(X).astype(float)
        b = _y(y)
        self.classes_ = _np.unique(b)
        self.theta_ = _np.array([A[b == c].mean(axis=0) for c in self.classes_])
        self.var_ = _np.array([A[b == c].var(axis=0) + 1e-9 for c in self.classes_])
        self.prior_ = _np.array([(b == c).mean() for c in self.classes_])
        return self

    def predict(self, X):
        A = _X(X).astype(float)
        ll = -0.5 * (_np.log(2 * _np.pi * self.var_) + ((A[:, None, :] - self.theta_) ** 2) / self.var_).sum(axis=2)
        out = self.classes_[_np.argmax(ll + _np.log(self.prior_), axis=1)]
        return Tensor(out.astype(_np.int64)) if isinstance(X, Tensor) else out

    def score(self, X, y):
        from .metrics import accuracy
        return accuracy(self.predict(X), y)


class PCA:
    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X):
        A = _X(X).astype(float)
        self.mean_ = A.mean(axis=0)
        U, S, Vt = _np.linalg.svd(A - self.mean_, full_matrices=False)
        self.components_ = Vt[:self.n_components].astype(_np.float32)
        ev = (S ** 2) / (len(A) - 1)
        self.explained_variance_ = ev[:self.n_components]
        self.explained_variance_ratio_ = ev / ev.sum()
        return self

    def transform(self, X):
        out = ((_X(X).astype(float) - self.mean_) @ self.components_.T).astype(_np.float32)
        return Tensor(out) if isinstance(X, Tensor) else out

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, Z):
        out = (_X(Z).astype(float) @ self.components_.astype(float) + self.mean_).astype(_np.float32)
        return Tensor(out) if isinstance(Z, Tensor) else out
