"""storch core: Tensor + autograd engine + torch-compatible creation/math API.

Design for speed (CPU):
- float32 default + C-contiguous storage -> best BLAS/SIMD utilization.
- multithreaded element-wise for large tensors (see _parallel).
- fused Linear forward (matmul + in-place bias add), fused bias+ReLU.
- iterative topo-sort backward (no recursion), grad accumulation in-place.
- inference fast path: no graph nodes allocated under no_grad().
"""
from __future__ import annotations

import contextlib
import math
import numpy as _np

from ._parallel import parallel_ewise, ewise_add, ewise_mul

__all__ = [
    "Tensor", "tensor", "zeros", "ones", "empty", "full", "zeros_like", "ones_like",
    "empty_like", "full_like", "arange", "linspace", "eye", "randn", "rand", "randint",
    "cat", "stack", "reshape", "squeeze", "unsqueeze", "transpose", "matmul", "mm", "bmm",
    "sum", "mean", "max", "min", "abs", "exp", "log", "sqrt", "pow", "clamp",
    "manual_seed", "save", "load", "no_grad", "enable_grad", "is_grad_enabled",
    "is_tensor", "allclose", "equal", "numel",
]

# Grad enabled flag (global).
_GRAD_ENABLED = True


def is_grad_enabled() -> bool:
    return _GRAD_ENABLED


@contextlib.contextmanager
def no_grad():
    global _GRAD_ENABLED
    prev = _GRAD_ENABLED
    _GRAD_ENABLED = False
    try:
        yield
    finally:
        _GRAD_ENABLED = prev


@contextlib.contextmanager
def enable_grad():
    global _GRAD_ENABLED
    prev = _GRAD_ENABLED
    _GRAD_ENABLED = True
    try:
        yield
    finally:
        _GRAD_ENABLED = prev


inference_mode = no_grad


# ---------- helpers ----------

_DEFAULT_DTYPE = _np.float32


def _to_np(x, dtype=None):
    if isinstance(x, Tensor):
        arr = x._data
        return arr if dtype is None else arr.astype(dtype, copy=False)
    return _np.asanyarray(x, dtype=dtype)


def _ensure_contig(a: _np.ndarray) -> _np.ndarray:
    if isinstance(a, _np.ndarray) and a.flags["C_CONTIGUOUS"]:
        return a
    return _np.ascontiguousarray(a)


def _unbroadcast(g: _np.ndarray, shape: tuple) -> _np.ndarray:
    """Sum grad down to a broadcasted input shape."""
    if g.shape == shape:
        return g
    # leading extra dims
    ndiff = g.ndim - len(shape)
    if ndiff > 0:
        g = g.sum(axis=tuple(range(ndiff)))
    for i, dim in enumerate(shape):
        if dim == 1:
            g = g.sum(axis=i, keepdims=True)
    if g.shape != shape:
        g = g.reshape(shape)
    return _np.ascontiguousarray(g)


def _wrap(data: _np.ndarray, requires_grad=False, _prev=(), _op="", _backward=None) -> "Tensor":
    t = Tensor.__new__(Tensor)
    t._data = _ensure_contig(data)
    t._grad = None
    t.requires_grad = bool(requires_grad)
    t._prev = _prev
    t._op = _op
    t._backward = _backward
    return t


# ---------- Tensor ----------

class Tensor:
    """torch.Tensor-compatible eager tensor with dynamic autograd."""

    __slots__ = ("_data", "_grad", "requires_grad", "_prev", "_op", "_backward")

    def __init__(self, data, dtype=None, requires_grad: bool = False, device=None):
        if isinstance(data, Tensor):
            arr = data._data
            arr = arr.astype(dtype, copy=False) if dtype is not None else _np.array(arr, copy=True, subok=False)
        elif isinstance(data, _np.ndarray):
            arr = data.astype(dtype, copy=False) if dtype is not None else _np.array(data, copy=True, subok=False)
        else:
            arr = _np.asanyarray(data, dtype=dtype if dtype is not None else _DEFAULT_DTYPE)
            arr = _np.array(arr, copy=True, subok=False)
        if arr.dtype == _np.float64 and dtype is None:
            pass  # keep float64 if user explicitly passed doubles? -> keep as-is
        self._data = _ensure_contig(arr)
        self._grad = None
        self.requires_grad = bool(requires_grad)
        self._prev: tuple = ()
        self._op: str = ""
        self._backward = None

    # -- meta --
    @property
    def data(self): return self._data
    @property
    def shape(self): return self._data.shape
    @property
    def ndim(self): return self._data.ndim
    @property
    def dtype(self): return self._data.dtype
    @property
    def device(self): return "cpu"
    @property
    def grad(self): return self._grad
    @grad.setter
    def grad(self, v): self._grad = v
    @property
    def T(self): return self.transpose()
    @property
    def is_leaf(self): return self._backward is None

    def __len__(self): return len(self._data)
    def __repr__(self):
        return f"storch.Tensor({self._data!r}, requires_grad={self.requires_grad})"

    # -- indexing (torch-compatible, grad-capable) --
    def __getitem__(self, idx):
        # normalize Tensor indices -> numpy arrays
        if isinstance(idx, Tensor):
            idx = idx._data
        elif isinstance(idx, tuple):
            idx = tuple(i._data if isinstance(i, Tensor) else i for i in idx)
        out = _ensure_contig(self._data[idx])
        rg = _GRAD_ENABLED and self.requires_grad
        if not rg:
            return _wrap(out, requires_grad=False)
        s = self
        t = self._new(out, (self,), "index", None, True)

        def bw(g=t, s=s, idx=idx, shp=self.shape):
            gg = g._grad
            grad = _np.zeros(shp, dtype=gg.dtype)
            # np.add.at handles duplicate indices correctly
            _np.add.at(grad, idx, gg)
            _accum(s, _ensure_contig(grad))
        t._backward = bw
        return t

    def __setitem__(self, idx, value):
        if isinstance(idx, Tensor):
            idx = idx._data
        v = value._data if isinstance(value, Tensor) else value
        self._data[idx] = v

    def copy_(self, other):
        self._data[:] = other._data if isinstance(other, Tensor) else _np.asanyarray(other)
        return self

    def fill_(self, value):
        self._data.fill(value)
        return self

    def zero_(self):
        self._data.fill(0)
        return self

    def numpy(self): return self._data
    def to_numpy(self): return self._data
    def detach(self): return _wrap(self._data, requires_grad=False)
    def contiguous(self): self._data = _ensure_contig(self._data); return self
    def numel(self): return self._data.size

    def item(self):
        if self._data.size != 1: raise ValueError("only one element tensors can be converted to Python scalars")
        return self._data.reshape(-1)[0].item()

    def tolist(self): return self._data.tolist()

    def to(self, dtype=None, device=None):
        if dtype is not None:
            self._data = _ensure_contig(self._data.astype(dtype, copy=False))
        return self

    def cpu(self): return self
    def float(self): return self.to(_np.float32)
    def double(self): return self.to(_np.float64)
    def long(self): return self.to(_np.int64)
    def int(self): return self.to(_np.int32)

    # -- comparisons (no grad, torch-compatible bool tensors) --
    def __eq__(self, o): return _wrap(_np.equal(self._data, _to_np(o)), requires_grad=False)
    def __ne__(self, o): return _wrap(_np.not_equal(self._data, _to_np(o)), requires_grad=False)
    def __gt__(self, o): return _wrap(_np.greater(self._data, _to_np(o)), requires_grad=False)
    def __ge__(self, o): return _wrap(_np.greater_equal(self._data, _to_np(o)), requires_grad=False)
    def __lt__(self, o): return _wrap(_np.less(self._data, _to_np(o)), requires_grad=False)
    def __le__(self, o): return _wrap(_np.less_equal(self._data, _to_np(o)), requires_grad=False)

    # -- convenience methods used by data-science workflows --
    def flatten(self, start_dim=0):
        sh = self.shape
        if start_dim == 0:
            return self.reshape((-1,))
        return self.reshape((sh[0], -1) if start_dim == 1 else (-1,))

    def argmax(self, dim=None, keepdim=False):
        from . import _ops as _O
        return _O.argmax(self, dim=dim, keepdim=keepdim)

    def argmin(self, dim=None, keepdim=False):
        from . import _ops as _O
        return _O.argmin(self, dim=dim, keepdim=keepdim)

    def masked_fill(self, mask, value):
        from . import _ops as _O
        return _O.masked_fill(self, mask, value)

    def topk(self, k, dim=-1):
        from . import _ops as _O
        return _O.topk(self, k, dim=dim)

    def sort(self, dim=-1, descending=False):
        from . import _ops as _O
        return _O.sort(self, dim=dim, descending=descending)

    def std(self, dim=None, keepdim=False, unbiased=True):
        from . import _ops as _O
        return _O.std(self, dim=dim, keepdim=keepdim, unbiased=unbiased)

    def var(self, dim=None, keepdim=False, unbiased=True):
        from . import _ops as _O
        return _O.var(self, dim=dim, keepdim=keepdim, unbiased=unbiased)

    def norm(self, p=2, dim=None):
        from . import _ops as _O
        return _O.norm(self, p=p, dim=dim)

    def cumsum(self, dim=0):
        from . import _ops as _O
        return _O.cumsum(self, dim=dim)

    def flip(self, dims):
        from . import _ops as _O
        return _O.flip(self, dims)

    def repeat(self, *repeats):
        from . import _ops as _O
        return _O.tile(self, repeats)

    def expand(self, *sizes):
        return _wrap(_ensure_contig(_np.broadcast_to(self._data, tuple(sizes))), requires_grad=False)

    def zero_grad(self):
        self._grad = None
        return self

    def requires_grad_(self, v: bool = True):
        self.requires_grad = bool(v)
        return self

    def retain_grad(self): return self

    # -- internal graph helper --
    def _new(self, data, prev, op, backward, requires_grad):
        return _wrap(data, requires_grad=requires_grad, _prev=prev, _op=op, _backward=backward)

    def _binary(self, other, np_fn, op_name, bwd_fn):
        o_data = other._data if isinstance(other, Tensor) else other
        rg = _GRAD_ENABLED and (self.requires_grad or (isinstance(other, Tensor) and other.requires_grad))
        if self._data.size >= 50_000 and np_fn in ("add", "mul"):
            fn = ewise_add if np_fn == "add" else ewise_mul
            out = fn(self._data, _to_np(o_data, None))
        else:
            if np_fn == "add": out = self._data + _to_np(o_data, None)
            elif np_fn == "sub": out = self._data - _to_np(o_data, None)
            elif np_fn == "mul": out = self._data * _to_np(o_data, None)
            elif np_fn == "div": out = self._data / _to_np(o_data, None)
            elif np_fn == "pow": out = self._data ** _to_np(o_data, None)
            else: raise ValueError(np_fn)
        out = _ensure_contig(out)
        if not rg:
            return _wrap(out, requires_grad=False)
        s, o = self, other
        sd, od = self._data, _to_np(o_data, None)

        def _backward(g):
            gs, go = bwd_fn(g, sd, od, out)
            return gs, go
        t = self._new(out, (self, other if isinstance(other, Tensor) else None), op_name, None, True)

        def bw(g=t):
            gg = g._grad
            gs, go = _backward(gg)
            if gs is not None: _accum(s, gs)
            if go is not None and isinstance(o, Tensor): _accum(o, go)
        t._backward = bw
        return t

    # -- arithmetic --
    def __add__(self, o): return self._binary(o, "add", "add", _bwd_add)
    def __radd__(self, o): return self._binary(o, "add", "add", _bwd_add_swap)
    def __sub__(self, o): return self._binary(o, "sub", "sub", _bwd_sub)
    def __rsub__(self, o):
        r = (-self)._binary(o, "add", "add", _bwd_add_swap)
        return r
    def __mul__(self, o): return self._binary(o, "mul", "mul", _bwd_mul)
    def __rmul__(self, o): return self._binary(o, "mul", "mul", _bwd_mul_swap)
    def __truediv__(self, o): return self._binary(o, "div", "div", _bwd_div)
    def __pow__(self, o): return self._binary(o, "pow", "pow", _bwd_pow)
    def __neg__(self):
        out = _ensure_contig(-self._data)
        rg = _GRAD_ENABLED and self.requires_grad
        if not rg: return _wrap(out, requires_grad=False)
        s = self
        t = self._new(out, (self,), "neg", None, True)
        def bw(g=t, s=s): _accum(s, -g._grad)
        t._backward = bw
        return t

    def __matmul__(self, o):
        od = o._data if isinstance(o, Tensor) else _to_np(o)
        out = _ensure_contig(self._data @ od)
        rg = _GRAD_ENABLED and (self.requires_grad or (isinstance(o, Tensor) and o.requires_grad))
        if not rg: return _wrap(out, requires_grad=False)
        s = self
        t = self._new(out, (self, o if isinstance(o, Tensor) else None), "matmul", None, True)
        def bw(g=t, s=s, o=o, od=od):
            gg = _ensure_contig(g._grad)
            if s.requires_grad:
                gs = gg @ _swap_last2(od)
                _accum(s, _unbroadcast(gs, s.shape))
            if isinstance(o, Tensor) and o.requires_grad:
                go = _swap_last2(s._data) @ gg
                _accum(o, _unbroadcast(go, o.shape))
        t._backward = bw
        return t

    # -- shape ops (grad-capable) --
    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)): shape = tuple(shape[0])
        out = self._data.reshape(shape)
        rg = _GRAD_ENABLED and self.requires_grad
        if not rg: return _wrap(_ensure_contig(out), requires_grad=False)
        s = self; sh = self.shape
        t = self._new(_ensure_contig(out), (self,), "reshape", None, True)
        def bw(g=t, s=s, sh=sh): _accum(s, _ensure_contig(g._grad.reshape(sh)))
        t._backward = bw
        return t

    def view(self, *shape): return self.reshape(*shape)

    def transpose(self, dim0=-2, dim1=-1):
        out = _ensure_contig(_np.swapaxes(self._data, dim0 % self.ndim if self.ndim else 0, dim1 % self.ndim if self.ndim else 0)) if self.ndim >= 2 else self._data
        rg = _GRAD_ENABLED and self.requires_grad
        if not rg: return _wrap(out if isinstance(out, _np.ndarray) else _np.asanyarray(out), requires_grad=False)
        s = self; a, b = dim0, dim1
        t = self._new(out, (self,), "transpose", None, True)
        def bw(g=t, s=s, a=a, b=b): _accum(s, _ensure_contig(_np.swapaxes(g._grad, a % s.ndim, b % s.ndim)))
        t._backward = bw
        return t

    def permute(self, *dims):
        out = _ensure_contig(self._data.transpose(dims))
        rg = _GRAD_ENABLED and self.requires_grad
        if not rg: return _wrap(out, requires_grad=False)
        s = self; inv = _np.argsort(dims)
        t = self._new(out, (self,), "permute", None, True)
        def bw(g=t, s=s, inv=inv): _accum(s, _ensure_contig(g._grad.transpose(inv)))
        t._backward = bw
        return t

    def squeeze(self, dim=None):
        out = _ensure_contig(self._data.squeeze(axis=dim))
        rg = _GRAD_ENABLED and self.requires_grad
        if not rg: return _wrap(out, requires_grad=False)
        s = self; sh = self.shape
        t = self._new(out, (self,), "squeeze", None, True)
        def bw(g=t, s=s, sh=sh): _accum(s, _ensure_contig(g._grad.reshape(sh)))
        t._backward = bw
        return t

    def unsqueeze(self, dim):
        out = _ensure_contig(_np.expand_dims(self._data, axis=dim))
        rg = _GRAD_ENABLED and self.requires_grad
        if not rg: return _wrap(out, requires_grad=False)
        s = self; sh = self.shape
        t = self._new(out, (self,), "unsqueeze", None, True)
        def bw(g=t, s=s, sh=sh): _accum(s, _ensure_contig(g._grad.reshape(sh)))
        t._backward = bw
        return t

    # -- reductions / activations as methods --
    def sum(self, dim=None, keepdim=False): return _reduce(self, "sum", dim, keepdim)
    def mean(self, dim=None, keepdim=False): return _reduce(self, "mean", dim, keepdim)
    def max(self, dim=None, keepdim=False): return _reduce(self, "max", dim, keepdim)
    def min(self, dim=None, keepdim=False): return _reduce(self, "min", dim, keepdim)
    def abs(self): return _unary(self, _np.abs, "abs", _bwd_abs)
    def exp(self): return _unary(self, _np.exp, "exp", _bwd_exp)
    def log(self): return _unary(self, _np.log, "log", _bwd_log)
    def sqrt(self): return _unary(self, _np.sqrt, "sqrt", _bwd_sqrt)
    def tanh(self): return _unary(self, _np.tanh, "tanh", _bwd_tanh)
    def sigmoid(self): return _sigmoid(self)
    def relu(self): return _relu(self)
    def clamp(self, min=None, max=None): return _clamp(self, min, max)
    def pow(self, e): return self ** e
    def softmax(self, dim=-1): from . import functional as F; return F.softmax(self, dim=dim)
    def log_softmax(self, dim=-1): from . import functional as F; return F.log_softmax(self, dim=dim)

    # -- backward --
    def backward(self, gradient=None):
        if not self.requires_grad: return
        if gradient is None:
            if self._data.size != 1: raise RuntimeError("grad can be implicitly created only for scalar outputs")
            g0 = _np.ones_like(self._data)
        else:
            g0 = gradient._data if isinstance(gradient, Tensor) else _np.asanyarray(gradient, dtype=self._data.dtype)
            g0 = _np.ascontiguousarray(g0)
        self._grad = g0 if self._grad is None else self._grad + g0
        # iterative post-order topo sort -> reversed() gives root-first execution
        topo, visited = [], set()
        stack = [(self, False)]
        while stack:
            n, done = stack.pop()
            if done:
                topo.append(n)
                continue
            if id(n) in visited:
                continue
            visited.add(id(n))
            stack.append((n, True))
            for p in n._prev:
                if p is not None and p.requires_grad and id(p) not in visited:
                    stack.append((p, False))
        for n in reversed(topo):
            if n._backward is not None and n._grad is not None:
                n._backward()


def _accum(t: Tensor, g):
    g = _np.ascontiguousarray(g)
    t._grad = g if t._grad is None else _np.ascontiguousarray(t._grad + g)


def _swap_last2(a):
    return _np.swapaxes(a, -1, -2) if a.ndim >= 2 else a


# ---------- element-wise backward fns ----------

def _bwd_add(g, sd, od, out): return _unbroadcast(g, sd.shape), _unbroadcast(g, _np.shape(od))
def _bwd_add_swap(g, sd, od, out): return _unbroadcast(g, _np.shape(od)), _unbroadcast(g, sd.shape)
def _bwd_sub(g, sd, od, out): return g, _unbroadcast(-g, _np.shape(od))
def _bwd_mul(g, sd, od, out): return _unbroadcast(g * od, sd.shape), _unbroadcast(g * sd, _np.shape(od))
def _bwd_mul_swap(g, sd, od, out): return _unbroadcast(g * sd, _np.shape(od)), _unbroadcast(g * od, sd.shape)
def _bwd_div(g, sd, od, out):
    return _unbroadcast(g / od, sd.shape), _unbroadcast(-g * sd / (od * od), _np.shape(od))
def _bwd_pow(g, sd, od, out):
    e = od
    gs = g * e * _np.power(sd, e - 1) if not isinstance(e, _np.ndarray) or e.size == 1 else g * e * _np.power(sd, e - 1)
    return _unbroadcast(gs, sd.shape), None


def _unary(t: Tensor, fn, name, bwd):
    out = _ensure_contig(fn(t._data))
    rg = _GRAD_ENABLED and t.requires_grad
    if not rg: return _wrap(out, requires_grad=False)
    s = t; sd = t._data
    tt = t._new(out, (t,), name, None, True)
    def bw(g=tt, s=s, sd=sd, out=out):
        _accum(s, _unbroadcast(bwd(g._grad, sd, out), s.shape))
    tt._backward = bw
    return tt


def _bwd_abs(g, sd, out): return g * _np.sign(sd)
def _bwd_exp(g, sd, out): return g * out
def _bwd_log(g, sd, out): return g / sd
def _bwd_sqrt(g, sd, out): return g / (2 * out)
def _bwd_tanh(g, sd, out): return g * (1 - out * out)


def _sigmoid(t: Tensor):
    # stable sigmoid, single pass
    d = t._data
    out = _np.empty_like(d)
    pos = d >= 0
    out[pos] = 1 / (1 + _np.exp(-d[pos]))
    e = _np.exp(d[~pos]); out[~pos] = e / (1 + e)
    out = _ensure_contig(out)
    rg = _GRAD_ENABLED and t.requires_grad
    if not rg: return _wrap(out, requires_grad=False)
    s = t
    tt = t._new(out, (t,), "sigmoid", None, True)
    def bw(g=tt, s=s, out=out): _accum(s, _unbroadcast(g._grad * out * (1 - out), s.shape))
    tt._backward = bw
    return tt


def _relu(t: Tensor):
    d = t._data
    if d.size >= 50_000:
        out = parallel_ewise(lambda x, out: _np.maximum(x, 0, out=out), d)
    else:
        out = _np.maximum(d, 0)
    out = _ensure_contig(out)
    rg = _GRAD_ENABLED and t.requires_grad
    if not rg: return _wrap(out, requires_grad=False)
    s = t
    tt = t._new(out, (t,), "relu", None, True)
    def bw(g=tt, s=s, out=out): _accum(s, _unbroadcast(g._grad * (out > 0), s.shape))
    tt._backward = bw
    return tt


def _clamp(t: Tensor, mn, mx):
    out = _ensure_contig(_np.clip(t._data, mn, mx))
    rg = _GRAD_ENABLED and t.requires_grad
    if not rg: return _wrap(out, requires_grad=False)
    s = t; sd = t._data
    tt = t._new(out, (t,), "clamp", None, True)
    def bw(g=tt, s=s, sd=sd, mn=mn, mx=mx):
        m = _np.ones_like(sd)
        if mn is not None: m = m * (sd > mn)
        if mx is not None: m = m * (sd < mx)
        _accum(s, _unbroadcast(g._grad * m, s.shape))
    tt._backward = bw
    return tt


def _reduce(t: Tensor, kind, dim, keepdim):
    d = t._data
    if kind == "sum": out = d.sum(axis=dim, keepdims=keepdim)
    elif kind == "mean": out = d.mean(axis=dim, keepdims=keepdim)
    elif kind == "max": out = d.max(axis=dim, keepdims=keepdim)
    elif kind == "min": out = d.min(axis=dim, keepdims=keepdim)
    out = _ensure_contig(_np.asanyarray(out))
    rg = _GRAD_ENABLED and t.requires_grad
    if not rg: return _wrap(out, requires_grad=False)
    s = t; sh = t.shape
    tt = t._new(out, (t,), kind, None, True)
    def bw(g=tt, s=s, sh=sh):
        gg = g._grad
        if kind == "sum":
            _accum(s, _np.broadcast_to(gg if keepdim else _np.expand_dims(gg, dim) if dim is not None else gg, sh).copy()
                   if dim is not None or keepdim else _np.full(sh, gg.reshape(-1)[0], dtype=gg.dtype))
        elif kind == "mean":
            n = _np.size(s._data) / _np.size(gg)
            _accum(s, _np.broadcast_to((gg / n) if keepdim or dim is None else _np.expand_dims(gg, dim) / n, sh).copy()
                   if dim is not None or keepdim else _np.full(sh, (gg.reshape(-1)[0] / n), dtype=gg.dtype))
        else:  # max/min: route grad to argmax (approx, matches torch for unique max)
            if dim is None:
                m = _np.zeros(sh, dtype=gg.dtype); m.reshape(-1)[_np.argmax(s._data.reshape(-1))] = gg.reshape(-1)[0]
            else:
                idx = s._data.argmax(axis=dim, keepdims=True)
                m = _np.zeros(sh, dtype=gg.dtype)
                _np.put_along_axis(m, idx, _np.broadcast_to(gg if keepdim else _np.expand_dims(gg, dim), idx.shape), axis=dim if dim is not None else -1)
            _accum(s, _ensure_contig(m))
    tt._backward = bw
    return tt


# ---------- creation ops ----------

def tensor(data, dtype=None, requires_grad=False, device=None):
    if isinstance(data, Tensor):
        return data.to(dtype) if dtype is not None else data.detach()
    return Tensor(data, dtype=dtype or _DEFAULT_DTYPE, requires_grad=requires_grad)


def zeros(*shape, dtype=_DEFAULT_DTYPE, requires_grad=False):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)): shape = tuple(shape[0])
    return Tensor(_np.zeros(shape, dtype=dtype), requires_grad=requires_grad)


def ones(*shape, dtype=_DEFAULT_DTYPE, requires_grad=False):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)): shape = tuple(shape[0])
    return Tensor(_np.ones(shape, dtype=dtype), requires_grad=requires_grad)


def empty(*shape, dtype=_DEFAULT_DTYPE, requires_grad=False):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)): shape = tuple(shape[0])
    return Tensor(_np.empty(shape, dtype=dtype), requires_grad=requires_grad)


def full(shape, fill_value, dtype=_DEFAULT_DTYPE, requires_grad=False):
    if isinstance(shape, int): shape = (shape,)
    return Tensor(_np.full(tuple(shape), fill_value, dtype=dtype), requires_grad=requires_grad)


def zeros_like(t, dtype=None): return Tensor(_np.zeros_like(t._data, dtype=dtype or t.dtype))
def ones_like(t, dtype=None): return Tensor(_np.ones_like(t._data, dtype=dtype or t.dtype))
def empty_like(t, dtype=None): return Tensor(_np.empty_like(t._data, dtype=dtype or t.dtype))
def full_like(t, v, dtype=None): return Tensor(_np.full_like(t._data, v, dtype=dtype or t.dtype))


def arange(*args, dtype=None, **kw):
    return Tensor(_np.arange(*args, **kw).astype(dtype or _DEFAULT_DTYPE, copy=False))


def linspace(s, e, steps, dtype=_DEFAULT_DTYPE):
    return Tensor(_np.linspace(s, e, steps).astype(dtype, copy=False))


def eye(n, m=None, dtype=_DEFAULT_DTYPE):
    return Tensor(_np.eye(n, M=m, dtype=dtype))


_rng = _np.random.default_rng()


def manual_seed(seed: int):
    global _rng
    _rng = _np.random.default_rng(seed)
    _np.random.seed(seed)
    return seed


def randn(*shape, requires_grad=False, dtype=_DEFAULT_DTYPE):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)): shape = tuple(shape[0])
    return Tensor(_ensure_contig(_rng.standard_normal(shape, dtype=dtype)), requires_grad=requires_grad)


def rand(*shape, requires_grad=False, dtype=_DEFAULT_DTYPE):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)): shape = tuple(shape[0])
    return Tensor(_ensure_contig(_rng.random(shape, dtype=dtype)), requires_grad=requires_grad)


def randint(low, high=None, size=None, dtype=_np.int64):
    if high is None: low, high = 0, low
    return Tensor(_rng.integers(low, high, size=size).astype(dtype, copy=False))


# ---------- shape / combine / math module-level ----------

def _as_t(x, requires_grad=False):
    return x if isinstance(x, Tensor) else Tensor(x, requires_grad=requires_grad)


def cat(tensors, dim=0):
    ts = [_as_t(t) for t in tensors]
    out = _ensure_contig(_np.concatenate([t._data for t in ts], axis=dim))
    rg = _GRAD_ENABLED and any(t.requires_grad for t in ts)
    if not rg: return _wrap(out, requires_grad=False)
    sizes = [t.shape[dim] for t in ts]
    res = _wrap(out, requires_grad=True, _prev=tuple(ts), _op="cat")
    def bw(g=res, ts=ts, dim=dim, sizes=sizes):
        off = 0; gg = g._grad
        for t, sz in zip(ts, sizes):
            sl = [slice(None)] * gg.ndim; sl[dim] = slice(off, off + sz); off += sz
            if t.requires_grad: _accum(t, _ensure_contig(gg[tuple(sl)]))
    res._backward = bw
    return res


def stack(tensors, dim=0):
    ts = [_as_t(t) for t in tensors]
    out = _ensure_contig(_np.stack([t._data for t in ts], axis=dim))
    rg = _GRAD_ENABLED and any(t.requires_grad for t in ts)
    if not rg: return _wrap(out, requires_grad=False)
    res = _wrap(out, requires_grad=True, _prev=tuple(ts), _op="stack")
    def bw(g=res, ts=ts, dim=dim):
        gg = g._grad
        parts = _np.split(gg, len(ts), axis=dim)
        for t, p in zip(ts, parts):
            if t.requires_grad: _accum(t, _ensure_contig(p.squeeze(axis=dim)))
    res._backward = bw
    return res


def reshape(t, shape): return _as_t(t).reshape(shape)
def squeeze(t, dim=None): return _as_t(t).squeeze(dim)
def unsqueeze(t, dim): return _as_t(t).unsqueeze(dim)
def transpose(t, d0=-2, d1=-1): return _as_t(t).transpose(d0, d1)
def matmul(a, b): return _as_t(a) @ _as_t(b)
def mm(a, b): return _as_t(a) @ _as_t(b)
def bmm(a, b): return _as_t(a) @ _as_t(b)
def sum(t, dim=None, keepdim=False): return _as_t(t).sum(dim, keepdim)
def mean(t, dim=None, keepdim=False): return _as_t(t).mean(dim, keepdim)
def max(t, dim=None, keepdim=False): return _as_t(t).max(dim, keepdim)
def min(t, dim=None, keepdim=False): return _as_t(t).min(dim, keepdim)
def abs(t): return _as_t(t).abs()
def exp(t): return _as_t(t).exp()
def log(t): return _as_t(t).log()
def sqrt(t): return _as_t(t).sqrt()
def pow(t, e): return _as_t(t) ** e
def clamp(t, min=None, max=None): return _as_t(t).clamp(min, max)


def is_tensor(x): return isinstance(x, Tensor)
def numel(t): return _as_t(t).numel()
def equal(a, b): return _np.array_equal(_to_np(a), _to_np(b))
def allclose(a, b, rtol=1e-05, atol=1e-08): return bool(_np.allclose(_to_np(a), _to_np(b), rtol=rtol, atol=atol))


def save(obj, path):
    import pickle
    if isinstance(obj, Tensor):
        _np.savez_compressed(path, data=obj._data)
    elif isinstance(obj, dict):
        npd = {k: (v._data if isinstance(v, Tensor) else _np.asanyarray(v)) for k, v in obj.items()}
        _np.savez_compressed(path, **npd)
    else:
        with open(path if str(path).endswith(".pkl") else str(path) + ".pkl", "wb") as f:
            pickle.dump(obj, f)


def load(path, map_location=None):
    import pickle
    try:
        z = _np.load(path, allow_pickle=True)
        if isinstance(z, _np.lib.npyio.NpzFile):
            d = dict(z)
            if set(d) == {"data"}: return Tensor(d["data"])
            return {k: Tensor(v) for k, v in d.items()}
    except Exception:
        pass
    with open(path, "rb") as f:
        return pickle.load(f)
