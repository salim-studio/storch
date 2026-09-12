"""storch — PyTorch-compatible API, faster on CPU.

    import storch
    import storch.nn as nn
    import storch.optim as optim

    model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 2))
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

Why fast (no compiler):
1. float32 + C-contiguous by default (best BLAS/SIMD utilization).
2. Multithreaded element-wise ops for large tensors.
3. Fused ops: Linear(+bias)(+ReLU), in-place SGD/Adam steps.
4. Iterative autograd (no recursion) + no graph nodes under no_grad.
"""
from __future__ import annotations

from ._core import (
    Tensor, tensor, zeros, ones, empty, full, zeros_like, ones_like,
    empty_like, full_like, arange, linspace, eye, randn, rand, randint,
    cat, stack, reshape, squeeze, unsqueeze, transpose, matmul, mm, bmm,
    sum, mean, max, min, abs, exp, log, sqrt, pow, clamp,
    manual_seed, save, load, no_grad, enable_grad, is_grad_enabled,
    is_tensor, allclose, equal, numel,
)
from . import functional, nn, optim
from . import _conv, _ops, linalg
from . import db, io, preprocessing, metrics, datasets, models
from ._ops import (
    where, masked_fill, gather, index_select, argmax, argmin, topk, sort,
    argsort, unique, nonzero, split, chunk, narrow, einsum, norm, std, var,
    cumsum, prod, flip, roll, tile, tril, triu, diag, flatten, unbind,
)
from .utils import data as data
from .utils.data import DataLoader, TensorDataset, Dataset, Subset, ConcatDataset, random_split
from ._parallel import MAX_WORKERS, PARALLEL_THRESHOLD

__version__ = "0.2.0"

# torch.compat aliases
cat = cat
concat = cat
manual_seed = manual_seed
float32 = __import__("numpy").float32
float64 = __import__("numpy").float64
int64 = __import__("numpy").int64
int32 = __import__("numpy").int32
long = __import__("numpy").int64

__all__ = [
    "Tensor", "tensor", "zeros", "ones", "empty", "full", "zeros_like", "ones_like",
    "empty_like", "full_like", "arange", "linspace", "eye", "randn", "rand", "randint",
    "cat", "stack", "reshape", "squeeze", "unsqueeze", "transpose", "matmul", "mm", "bmm",
    "sum", "mean", "max", "min", "abs", "exp", "log", "sqrt", "pow", "clamp",
    "where", "masked_fill", "gather", "index_select", "argmax", "argmin", "topk", "sort",
    "argsort", "unique", "nonzero", "split", "chunk", "narrow", "einsum", "norm", "std", "var",
    "cumsum", "prod", "flip", "roll", "tile", "tril", "triu", "diag", "flatten", "unbind",
    "manual_seed", "save", "load", "no_grad", "enable_grad", "is_grad_enabled",
    "is_tensor", "allclose", "equal", "numel",
    "functional", "nn", "optim", "data", "Dataset", "TensorDataset", "DataLoader",
    "Subset", "ConcatDataset", "random_split",
    "db", "io", "preprocessing", "metrics", "datasets", "models", "linalg", "_ops",
    "MAX_WORKERS", "PARALLEL_THRESHOLD", "__version__",
]


def set_workers(n: int):
    from . import _parallel
    _parallel.MAX_WORKERS = max(1, int(n))
    globals()["MAX_WORKERS"] = _parallel.MAX_WORKERS


def info():
    import os
    import numpy as _np
    return {"version": __version__, "numpy": _np.__version__,
            "workers": MAX_WORKERS, "cpus": os.cpu_count(),
            "parallel_threshold": PARALLEL_THRESHOLD}
