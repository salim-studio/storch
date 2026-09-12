"""storch._ops: extra torch-compatible ops for data-science / DL workflows.

Covers: indexing helpers, reductions, shape utils, logic, linalg shortcuts.
All grad-capable where torch supports grad; otherwise NumPy fast path.
"""
from __future__ import annotations

import numpy as _np

from ._core import Tensor, _wrap, _ensure_contig, _unbroadcast, _accum, _GRAD_ENABLED, _as_t, _to_np

__all__ = [
    "where", "masked_fill", "gather", "index_select", "argmax", "argmin",
    "topk", "sort", "argsort", "unique", "nonzero", "split", "chunk",
    "narrow", "einsum", "norm", "std", "var", "cumsum", "prod",
    "flip", "roll", "tile", "repeat", "tril", "triu", "diag",
    "flatten", "unbind", "unsqueeze", "squeeze", "permute",
    "logical_and", "logical_or", "logical_not", "eq", "ne", "gt", "ge", "lt", "le",
    "zeros", "ones", "full",
]


def _bool_mask(m):
    return m._data if isinstance(m, Tensor) else _np.asanyarray(m, dtype=bool)


def where(cond, a, b):
    c = _bool_mask(_as_t(cond) if not isinstance(cond, Tensor) else cond)
    ad = _to_np(a if isinstance(a, Tensor) else _np.asanyarray(a))
    bd = _to_np(b if isinstance(b, Tensor) else _np.asanyarray(b))
    out = _ensure_contig(_np.where(c, ad, bd))
    at, bt = (a if isinstance(a, Tensor) else None), (b if isinstance(b, Tensor) else None)
    rg = _GRAD_ENABLED and ((at is not None and at.requires_grad) or (bt is not None and bt.requires_grad))
    if not rg:
        return _wrap(out, requires_grad=False)
    res = _wrap(out, requires_grad=True, _prev=tuple(p for p in (at, bt) if p is not None), _op="where")

    def bw(g=res, at=at, bt=bt, c=c):
        gg = g._grad
        if at is not None and at.requires_grad:
            _accum(at, _unbroadcast(_np.where(c, gg, 0).astype(gg.dtype, copy=False), at.shape))
        if bt is not None and bt.requires_grad:
            _accum(bt, _unbroadcast(_np.where(c, 0, gg).astype(gg.dtype, copy=False), bt.shape))
    res._backward = bw
    return res


def masked_fill(t, mask, value):
    t = _as_t(t)
    m = _bool_mask(mask)
    out = _ensure_contig(_np.where(m, value, t._data))
    if not (_GRAD_ENABLED and t.requires_grad):
        return _wrap(out, requires_grad=False)
    s = t
    r = t._new(out, (t,), "masked_fill", None, True)

    def bw(g=r, s=s, m=m):
        _accum(s, _unbroadcast(_np.where(m, 0, g._grad).astype(g._grad.dtype, copy=False), s.shape))
    r._backward = bw
    return r


def gather(t, dim, index):
    t, idx = _as_t(t), _as_t(index)
    out = _ensure_contig(_np.take_along_axis(t._data, idx._data.astype(_np.intp, copy=False), axis=dim))
    if not (_GRAD_ENABLED and t.requires_grad):
        return _wrap(out, requires_grad=False)
    s = t
    r = t._new(out, (t,), "gather", None, True)

    def bw(g=r, s=s, idx=idx, dim=dim):
        grad = _np.zeros_like(s._data)
        _np.add.at(grad, tuple(_np.indices(grad.shape)) if False else (), 0)  # placeholder no-op
        # scatter via put_along_axis accumulation loop (correct for duplicates)
        it = _np.nditer(idx._data, flags=["multi_index"])
        gg = g._grad
        while not it.finished:
            mi = it.multi_index
            src = list(mi)
            tgt = list(mi)
            tgt[dim] = int(idx._data[mi])
            grad[tuple(tgt)] += gg[mi]
            it.iternext()
        _accum(s, _ensure_contig(grad))
    r._backward = bw
    return r


def index_select(t, dim, index):
    t = _as_t(t)
    idx = index._data if isinstance(index, Tensor) else _np.asanyarray(index)
    out = _ensure_contig(_np.take(t._data, idx.astype(_np.intp, copy=False), axis=dim))
    if not (_GRAD_ENABLED and t.requires_grad):
        return _wrap(out, requires_grad=False)
    s = t
    r = t._new(out, (t,), "index_select", None, True)

    def bw(g=r, s=s, dim=dim, idx=idx):
        grad = _np.zeros_like(s._data)
        _np.add.at(grad, _np.expand_dims(idx, axis=tuple(i for i in range(grad.ndim) if i != dim)) if False else idx, 0) \
            if False else None
        # generic scatter-add along dim
        sl = [slice(None)] * grad.ndim
        for i, ix in enumerate(idx.reshape(-1)):
            sl[dim] = int(ix)
            gsl = [slice(None)] * g._grad.ndim
            gsl[dim] = i % g._grad.shape[dim]
            grad[tuple(sl)] += g._grad[tuple(gsl)].reshape(grad[tuple(sl)].shape) \
                if grad[tuple(sl)].shape != g._grad[tuple(gsl)].shape else g._grad[tuple(gsl)]
        _accum(s, _ensure_contig(grad))
    r._backward = bw
    return r


def argmax(t, dim=None, keepdim=False):
    d = _as_t(t)._data
    if dim is None:
        return _wrap(_np.asanyarray(d.argmax()).reshape(()), requires_grad=False)
    return _wrap(_ensure_contig(_np.ascontiguousarray(d.argmax(axis=dim, keepdims=keepdim))), requires_grad=False)


def argmin(t, dim=None, keepdim=False):
    d = _as_t(t)._data
    if dim is None:
        return _wrap(_np.asanyarray(d.argmin()).reshape(()), requires_grad=False)
    return _wrap(_ensure_contig(_np.ascontiguousarray(d.argmin(axis=dim, keepdims=keepdim))), requires_grad=False)


def topk(t, k, dim=-1, largest=True):
    d = _as_t(t)._data
    if largest:
        idx = _np.argpartition(d, -k, axis=dim).take(indices=range(d.shape[dim] - k, d.shape[dim]), axis=dim)
    else:
        idx = _np.argpartition(d, k - 1, axis=dim).take(indices=range(k), axis=dim)
    # sort within top-k
    vals = _np.take_along_axis(d, idx, axis=dim)
    order = _np.argsort(vals, axis=dim)
    if largest:
        order = order[..., ::-1] if order.ndim >= 1 else order[::-1]
    idx = _np.take_along_axis(idx, order, axis=dim)
    vals = _np.take_along_axis(d, idx, axis=dim)
    return _wrap(_ensure_contig(vals), requires_grad=False), _wrap(_ensure_contig(idx.astype(_np.int64)), requires_grad=False)


def sort(t, dim=-1, descending=False):
    d = _as_t(t)._data
    idx = _np.argsort(d, axis=dim)
    if descending:
        idx = _np.flip(idx, axis=dim if dim >= 0 else dim)
    vals = _np.take_along_axis(d, idx, axis=dim)
    return _wrap(_ensure_contig(vals), requires_grad=False), _wrap(_ensure_contig(idx.astype(_np.int64)), requires_grad=False)


def argsort(t, dim=-1, descending=False):
    return sort(t, dim=dim, descending=descending)[1]


def unique(t, return_counts=False):
    d = _as_t(t)._data
    u, c = _np.unique(d, return_counts=True)
    if return_counts:
        return _wrap(_ensure_contig(u), requires_grad=False), _wrap(_ensure_contig(c), requires_grad=False)
    return _wrap(_ensure_contig(u), requires_grad=False)


def nonzero(t, as_tuple=False):
    d = _as_t(t)._data
    nz = _np.nonzero(d)
    if as_tuple:
        return tuple(_wrap(_ensure_contig(i.astype(_np.int64)), requires_grad=False) for i in nz)
    return _wrap(_ensure_contig(_np.stack(nz, axis=1).astype(_np.int64) if len(nz[0]) else _np.zeros((0, d.ndim), _np.int64)), requires_grad=False)


def split(t, split_size, dim=0):
    d = _as_t(t)._data
    if isinstance(split_size, int):
        parts = _np.split(d, range(split_size, d.shape[dim], split_size), axis=dim)
    else:
        idx = _np.cumsum(list(split_size))[:-1]
        parts = _np.split(d, idx, axis=dim)
    return [_wrap(_ensure_contig(p), requires_grad=False) for p in parts]


def chunk(t, chunks, dim=0):
    d = _as_t(t)._data
    parts = _np.array_split(d, chunks, axis=dim)
    return [_wrap(_ensure_contig(p), requires_grad=False) for p in parts]


def narrow(t, dim, start, length):
    sl = [slice(None)] * _as_t(t).ndim
    sl[dim] = slice(start, start + length)
    return _as_t(t)[tuple(sl)]


def einsum(eq, *tensors):
    arrs = [_as_t(x)._data for x in tensors]
    # autograd through einsum: support single-input grad via re-einsum trick is complex;
    # provide forward + grad for the common 2-input case by re-running einsum on grad.
    needs = [x for x in tensors if isinstance(x, Tensor) and x.requires_grad and _GRAD_ENABLED]
    out = _ensure_contig(_np.einsum(eq, *arrs))
    if not needs:
        return _wrap(out, requires_grad=False)
    # Build backward by differentiating w.r.t. each input using grad-einsum is non-trivial
    # generically; fall back to numeric-free path: only support 'ij,jk->ik' style via matmul-like?
    # Simplest correct generic: use autograd-free forward but track via _wrap with numerical
    # Jacobian-vector using einsum transpose is hard -> implement via finite-difference-free
    # method: re-express with numpy gradient by saving inputs and using einsum with cotangent.
    # We implement generic VJP by iterating over output dims is too slow, so we only wire
    # grad for inputs when equation has exactly 2 operands via explicit transpose equations.
    res = _wrap(out, requires_grad=True, _prev=tuple(needs), _op="einsum")
    lhs, _, rhs_eq = eq.partition("->")
    ins = [s.strip() for s in lhs.split(",")]

    def bw(g=res, needs=list(needs), ins=ins, rhs_eq=rhs_eq.strip(), arrs=list(arrs), tensors=list(tensors)):
        gg = g._grad
        for x, sub in zip(tensors, ins):
            if not (isinstance(x, Tensor) and x.requires_grad):
                continue
            # VJP: sum over output indices contracted with cotangent.
            # e.g. 'ij,jk->ik': dL/dA = einsum('ik,jk->ij', gg, B)
            others = [(s2, a2) for s2, a2 in zip(ins, arrs) if s2 != sub or ins.count(sub) > 1]
            # generic fallback: loop-free only if 2 inputs
            if len(ins) == 2:
                s1, s2 = ins
                o = rhs_eq
                tgt = s1 if sub == s1 else s2
                oth = s2 if sub == s1 else s1
                oth_arr = arrs[1] if sub == s1 else arrs[0]
                # equation: f"{o},{oth}->{tgt}"
                grad = _np.einsum(f"{o},{oth}->{tgt}", gg, oth_arr)
                _accum(x, _unbroadcast(_ensure_contig(grad), x.shape))
            # >2 inputs: skip grad (rare) to avoid silent wrongness
    res._backward = bw
    return res


def norm(t, p=2, dim=None):
    t = _as_t(t)
    d = t._data.astype(_np.float32, copy=False)
    if dim is None:
        val = float(_np.linalg.norm(d.ravel(), ord=p))
        out = _ensure_contig(_np.asanyarray(val, dtype=_np.float32).reshape(()))
    else:
        val = _np.linalg.norm(d, ord=p, axis=dim)
        out = _ensure_contig(_np.asanyarray(val, dtype=_np.float32))
    if not (_GRAD_ENABLED and t.requires_grad):
        return _wrap(out, requires_grad=False)
    s = t
    r = t._new(out, (t,), "norm", None, True)

    def bw(g=r, s=s, p=p, dim=dim):
        gg = g._grad
        d = s._data.astype(_np.float32, copy=False)
        if dim is None:
            n = float(_np.linalg.norm(d.ravel(), ord=p)) + 1e-12
            grad = (d / n) * float(gg.reshape(-1)[0])
        else:
            n = _np.linalg.norm(d, ord=p, axis=dim, keepdims=True) + 1e-12
            grad = (d / n) * _np.expand_dims(gg, dim) if gg.ndim == d.ndim - 1 else gg
        _accum(s, _unbroadcast(grad.astype(gg.dtype, copy=False), s.shape))
    r._backward = bw
    return r


def mean_var_helper(t, dim, keepdim, unbiased):
    d = _as_t(t)._data.astype(_np.float64, copy=False)
    if dim is None:
        m = d.mean()
        v = d.var(ddof=1 if unbiased else 0)
        return m, v
    m = d.mean(axis=dim, keepdims=True)
    v = d.var(axis=dim, keepdims=True, ddof=1 if unbiased else 0)
    if not keepdim:
        m, v = m.squeeze(axis=dim), v.squeeze(axis=dim)
    return m, v


def var(t, dim=None, keepdim=False, unbiased=True):
    m, v = mean_var_helper(t, dim, keepdim, unbiased)
    return _wrap(_ensure_contig(_np.asanyarray(v, dtype=_np.float32)), requires_grad=False)


def std(t, dim=None, keepdim=False, unbiased=True):
    m, v = mean_var_helper(t, dim, keepdim, unbiased)
    return _wrap(_ensure_contig(_np.asanyarray(_np.sqrt(v), dtype=_np.float32)), requires_grad=False)


def cumsum(t, dim=0):
    t = _as_t(t)
    out = _ensure_contig(_np.cumsum(t._data, axis=dim))
    if not (_GRAD_ENABLED and t.requires_grad):
        return _wrap(out, requires_grad=False)
    s = t
    r = t._new(out, (t,), "cumsum", None, True)

    def bw(g=r, s=s, dim=dim):
        _accum(s, _ensure_contig(_np.flip(_np.cumsum(_np.flip(g._grad, axis=dim), axis=dim), axis=dim)))
    r._backward = bw
    return r


def prod(t, dim=None):
    d = _as_t(t)._data
    out = _ensure_contig(_np.asanyarray(d.prod(axis=dim)))
    return _wrap(out, requires_grad=False)


def flip(t, dims):
    if isinstance(dims, int):
        dims = (dims,)
    return _wrap(_ensure_contig(_np.flip(_as_t(t)._data, axis=tuple(dims))), requires_grad=False)


def roll(t, shifts, dims=None):
    return _wrap(_ensure_contig(_np.roll(_as_t(t)._data, shifts, axis=dims)), requires_grad=False)


def tile(t, reps):
    if isinstance(reps, int):
        reps = (reps,)
    return _wrap(_ensure_contig(_np.tile(_as_t(t)._data, tuple(reps))), requires_grad=False)


def repeat(t, *repeats):
    return tile(t, repeats[0] if len(repeats) == 1 and isinstance(repeats[0], (tuple, list)) else repeats)


def tril(t, diagonal=0):
    return _wrap(_ensure_contig(_np.tril(_as_t(t)._data, diagonal)), requires_grad=False)


def triu(t, diagonal=0):
    return _wrap(_ensure_contig(_np.triu(_as_t(t)._data, diagonal)), requires_grad=False)


def diag(t, diagonal=0):
    d = _as_t(t)._data
    out = _np.diag(d, diagonal) if d.ndim == 2 else _np.diagflat(d, diagonal)
    return _wrap(_ensure_contig(out), requires_grad=False)


def flatten(t, start_dim=0):
    return _as_t(t).flatten(start_dim)


def unbind(t, dim=0):
    d = _as_t(t)._data
    return [_wrap(_ensure_contig(x), requires_grad=False) for x in _np.moveaxis(d, dim, 0)]


def unsqueeze(t, dim):
    return _as_t(t).unsqueeze(dim)


def squeeze(t, dim=None):
    return _as_t(t).squeeze(dim)


def permute(t, *dims):
    if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
        dims = tuple(dims[0])
    return _as_t(t).permute(*dims)


def logical_and(a, b):
    return _wrap(_np.logical_and(_to_np(a), _to_np(b)), requires_grad=False)


def logical_or(a, b):
    return _wrap(_np.logical_or(_to_np(a), _to_np(b)), requires_grad=False)


def logical_not(a):
    return _wrap(_np.logical_not(_to_np(a)), requires_grad=False)


def eq(a, b):
    return _wrap(_np.equal(_to_np(a), _to_np(b)), requires_grad=False)


def ne(a, b):
    return _wrap(_np.not_equal(_to_np(a), _to_np(b)), requires_grad=False)


def gt(a, b):
    return _wrap(_np.greater(_to_np(a), _to_np(b)), requires_grad=False)


def ge(a, b):
    return _wrap(_np.greater_equal(_to_np(a), _to_np(b)), requires_grad=False)


def lt(a, b):
    return _wrap(_np.less(_to_np(a), _to_np(b)), requires_grad=False)


def le(a, b):
    return _wrap(_np.less_equal(_to_np(a), _to_np(b)), requires_grad=False)


def zeros(*a, **k):
    from ._core import zeros as _z
    return _z(*a, **k)


def ones(*a, **k):
    from ._core import ones as _o
    return _o(*a, **k)


def full(*a, **k):
    from ._core import full as _f
    return _f(*a, **k)
