"""storch.linalg: torch.linalg-compatible helpers (NumPy-backed)."""
from __future__ import annotations

import numpy as _np

from ._core import _wrap, _ensure_contig, _as_t

__all__ = ["norm", "inv", "det", "solve", "svd", "qr", "eigh", "matrix_rank", "pinv"]


def norm(t, ord=None, dim=None, keepdim=False):
    d = _as_t(t)._data
    out = _np.linalg.norm(d, ord=ord, axis=dim, keepdims=keepdim)
    return _wrap(_ensure_contig(_np.asanyarray(out)), requires_grad=False)


def inv(t):
    return _wrap(_ensure_contig(_np.linalg.inv(_as_t(t)._data)), requires_grad=False)


def det(t):
    return _wrap(_ensure_contig(_np.asanyarray(_np.linalg.det(_as_t(t)._data))), requires_grad=False)


def solve(A, B):
    out = _np.linalg.solve(_as_t(A)._data, _as_t(B)._data)
    return _wrap(_ensure_contig(out), requires_grad=False)


def svd(t, full_matrices=False):
    U, S, Vh = _np.linalg.svd(_as_t(t)._data, full_matrices=full_matrices)
    return (_wrap(_ensure_contig(U), requires_grad=False),
            _wrap(_ensure_contig(S), requires_grad=False),
            _wrap(_ensure_contig(Vh), requires_grad=False))


def qr(t):
    Q, R = _np.linalg.qr(_as_t(t)._data)
    return _wrap(_ensure_contig(Q), requires_grad=False), _wrap(_ensure_contig(R), requires_grad=False)


def eigh(t):
    w, v = _np.linalg.eigh(_as_t(t)._data)
    return _wrap(_ensure_contig(w), requires_grad=False), _wrap(_ensure_contig(v), requires_grad=False)


def matrix_rank(t, tol=None):
    return int(_np.linalg.matrix_rank(_as_t(t)._data, tol=tol))


def pinv(t):
    return _wrap(_ensure_contig(_np.linalg.pinv(_as_t(t)._data)), requires_grad=False)
