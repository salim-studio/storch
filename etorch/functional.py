"""etorch.functional: torch.nn.functional-compatible ops (fused fast paths)."""
from __future__ import annotations

import numpy as _np
from ._core import Tensor, _wrap, _ensure_contig, _unbroadcast, _accum, _GRAD_ENABLED, _as_t
from ._parallel import fused_bias_add_relu

__all__ = [
    "relu", "sigmoid", "tanh", "gelu", "silu", "leaky_relu", "softmax", "log_softmax",
    "linear", "cross_entropy", "mse_loss", "l1_loss", "bce_with_logits",
    "nll_loss", "dropout", "layer_norm", "embedding",
]


def relu(t): return _as_t(t).relu()
def sigmoid(t): return _as_t(t).sigmoid()
def tanh(t): return _as_t(t).tanh()


def leaky_relu(t, negative_slope=0.01):
    t = _as_t(t)
    out = _ensure_contig(_np.where(t._data > 0, t._data, t._data * negative_slope))
    if not (_GRAD_ENABLED and t.requires_grad): return _wrap(out, requires_grad=False)
    s = t
    r = t._new(out, (t,), "leaky_relu", None, True)
    def bw(g=r, s=s, ns=negative_slope): _accum(s, _unbroadcast(g._grad * _np.where(s._data > 0, 1, ns), s.shape))
    r._backward = bw
    return r


def gelu(t):
    t = _as_t(t)
    d = t._data
    out = _ensure_contig(0.5 * d * (1 + _np.tanh(_np.sqrt(2 / _np.pi) * (d + 0.044715 * d ** 3))))
    if not (_GRAD_ENABLED and t.requires_grad): return _wrap(out, requires_grad=False)
    s = t
    r = t._new(out, (t,), "gelu", None, True)
    def bw(g=r, s=s, out=out):
        # tanh-approx gelu backward
        d = s._data
        c = _np.sqrt(2 / _np.pi) * (d + 0.044715 * d ** 3)
        th = _np.tanh(c)
        sech2 = 1 - th * th
        inner = _np.sqrt(2 / _np.pi) * (1 + 3 * 0.044715 * d * d)
        grad = 0.5 * (1 + th) + 0.5 * d * sech2 * inner
        _accum(s, _unbroadcast(g._grad * grad, s.shape))
    r._backward = bw
    return r


def silu(t):
    t = _as_t(t)
    sig = 1 / (1 + _np.exp(-t._data))
    out = _ensure_contig(t._data * sig)
    if not (_GRAD_ENABLED and t.requires_grad): return _wrap(out, requires_grad=False)
    s = t
    r = t._new(out, (t,), "silu", None, True)
    def bw(g=r, s=s, sig=sig, out=out): _accum(s, _unbroadcast(g._grad * (sig + out * (1 - sig)), s.shape))
    r._backward = bw
    return r


def softmax(t, dim=-1):
    t = _as_t(t)
    d = t._data
    m = d.max(axis=dim, keepdims=True)
    e = _np.exp(d - m)
    out = _ensure_contig(e / e.sum(axis=dim, keepdims=True))
    if not (_GRAD_ENABLED and t.requires_grad): return _wrap(out, requires_grad=False)
    s = t
    r = t._new(out, (t,), "softmax", None, True)
    def bw(g=r, s=s, out=out, dim=dim):
        gg = g._grad
        _accum(s, _unbroadcast(out * (gg - (gg * out).sum(axis=dim, keepdims=True)), s.shape))
    r._backward = bw
    return r


def log_softmax(t, dim=-1):
    t = _as_t(t)
    d = t._data
    m = d.max(axis=dim, keepdims=True)
    lse = _np.log(_np.exp(d - m).sum(axis=dim, keepdims=True)) + m
    out = _ensure_contig(d - lse)
    if not (_GRAD_ENABLED and t.requires_grad): return _wrap(out, requires_grad=False)
    s = t
    r = t._new(out, (t,), "log_softmax", None, True)
    def bw(g=r, s=s, out=out, dim=dim):
        gg = g._grad
        sm = _np.exp(out)
        _accum(s, _unbroadcast(gg - sm * gg.sum(axis=dim, keepdims=True), s.shape))
    r._backward = bw
    return r


def linear(x, weight, bias=None, activation=None):
    """Fused linear: y = x @ W.T (+ bias) (+ relu). Single bias pass, in-place relu."""
    x, w = _as_t(x), _as_t(weight)
    b = _as_t(bias) if bias is not None else None
    out = _ensure_contig(x._data @ w._data.T)
    if b is not None:
        out += b._data  # fused in-place bias add (no temp)
    if activation == "relu":
        _np.maximum(out, 0, out=out)
    rg = _GRAD_ENABLED and (x.requires_grad or w.requires_grad or (b is not None and b.requires_grad))
    if not rg: return _wrap(out, requires_grad=False)
    res = _wrap(out, requires_grad=True, _prev=tuple(p for p in (x, w, b) if p is not None), _op="linear")
    def bw(g=res, x=x, w=w, b=b, act=activation):
        gg = g._grad
        if act == "relu":
            gg = gg * (out > 0)
        if x.requires_grad: _accum(x, _unbroadcast(_ensure_contig(gg @ w._data), x.shape))
        if w.requires_grad: _accum(w, _unbroadcast(_ensure_contig(gg.T @ x._data if gg.ndim == 2 else _np.tensordot(gg, x._data, axes=([0], [0]))), w.shape))
        if b is not None and b.requires_grad:
            axes = tuple(range(gg.ndim - 1))
            _accum(b, _unbroadcast(gg.sum(axis=axes) if axes else gg, b.shape))
    res._backward = bw
    return res


def dropout(t, p=0.5, training=True):
    t = _as_t(t)
    if not training or p == 0:
        return t if not (_GRAD_ENABLED and t.requires_grad) else _wrap(_ensure_contig(t._data.copy()), requires_grad=True, _prev=(t,), _op="dropout")
    mask = (_np.random.rand(*t.shape) >= p).astype(t._data.dtype) / (1 - p)
    out = _ensure_contig(t._data * mask)
    if not (_GRAD_ENABLED and t.requires_grad): return _wrap(out, requires_grad=False)
    s = t
    r = t._new(out, (t,), "dropout", None, True)
    def bw(g=r, s=s, mask=mask): _accum(s, _unbroadcast(g._grad * mask, s.shape))
    r._backward = bw
    return r


def layer_norm(t, normalized_shape, weight=None, bias=None, eps=1e-5):
    t = _as_t(t)
    if isinstance(normalized_shape, int): normalized_shape = (normalized_shape,)
    axes = tuple(range(-len(normalized_shape), 0))
    mu = t._data.mean(axis=axes, keepdims=True)
    var = ((t._data - mu) ** 2).mean(axis=axes, keepdims=True)
    xn = (t._data - mu) / _np.sqrt(var + eps)
    out = xn
    if weight is not None: out = out * _as_t(weight)._data
    if bias is not None: out = out + _as_t(bias)._data
    out = _ensure_contig(out)
    if not (_GRAD_ENABLED and t.requires_grad): return _wrap(out, requires_grad=False)
    s = t
    r = t._new(out, (t,), "layernorm", None, True)
    def bw(g=r, s=s, mu=mu, var=var, xn=xn, eps=eps, axes=axes, w=weight):
        gg = g._grad
        if w is not None: gg = gg * _as_t(w)._data
        n = _np.prod([s.shape[a] for a in axes])
        gmu = gg.mean(axis=axes, keepdims=True)
        gxn = (gg - gmu - xn * (gg * xn).mean(axis=axes, keepdims=True)) / _np.sqrt(var + eps)
        _accum(s, _unbroadcast(gxn, s.shape))
    r._backward = bw
    return r


def embedding(indices, weight):
    w = _as_t(weight)
    idx = indices._data if isinstance(indices, Tensor) else _np.asanyarray(indices)
    out = _ensure_contig(w._data[idx])
    if not (_GRAD_ENABLED and w.requires_grad): return _wrap(out, requires_grad=False)
    res = _wrap(out, requires_grad=True, _prev=(w,), _op="embedding")
    def bw(g=res, w=w, idx=idx):
        gw = _np.zeros_like(w._data)
        _np.add.at(gw, idx, g._grad)
        _accum(w, gw)
    res._backward = bw
    return res


def mse_loss(pred, target, reduction="mean"):
    p, tg = _as_t(pred), _as_t(target)
    diff = p._data - tg._data
    if reduction == "mean": val = (diff * diff).mean()
    elif reduction == "sum": val = (diff * diff).sum()
    else: return _wrap(_ensure_contig(diff * diff), requires_grad=False)
    out = _ensure_contig(_np.asanyarray(val).reshape(()))
    if not (_GRAD_ENABLED and p.requires_grad): return _wrap(out, requires_grad=False)
    r = _wrap(out, requires_grad=True, _prev=(p,), _op="mse")
    def bw(g=r, p=p, diff=diff, reduction=reduction):
        n = diff.size if reduction == "mean" else 1.0
        _accum(p, _unbroadcast(g._grad.reshape(-1)[0] * 2 * diff / n, p.shape))
    r._backward = bw
    return r


def l1_loss(pred, target, reduction="mean"):
    p, tg = _as_t(pred), _as_t(target)
    diff = p._data - tg._data
    out = _ensure_contig(_np.asanyarray(_np.abs(diff).mean() if reduction == "mean" else _np.abs(diff).sum()).reshape(()))
    if not (_GRAD_ENABLED and p.requires_grad): return _wrap(out, requires_grad=False)
    r = _wrap(out, requires_grad=True, _prev=(p,), _op="l1")
    def bw(g=r, p=p, diff=diff, reduction=reduction):
        n = diff.size if reduction == "mean" else 1.0
        _accum(p, _unbroadcast(g._grad.reshape(-1)[0] * _np.sign(diff) / n, p.shape))
    r._backward = bw
    return r


def nll_loss(log_probs, target, reduction="mean"):
    lp, tg = _as_t(log_probs), _as_t(target)
    tval = tg._data.astype(_np.int64).reshape(-1)
    lpf = lp._data.reshape(-1, lp.shape[-1])
    losses = -lpf[_np.arange(lpf.shape[0]), tval]
    val = losses.mean() if reduction == "mean" else losses.sum()
    out = _ensure_contig(_np.asanyarray(val).reshape(()))
    if not (_GRAD_ENABLED and lp.requires_grad): return _wrap(out, requires_grad=False)
    r = _wrap(out, requires_grad=True, _prev=(lp,), _op="nll")
    def bw(g=r, lp=lp, tval=tval, lpf=lpf, reduction=reduction):
        n = lpf.shape[0] if reduction == "mean" else 1.0
        gg = _np.zeros_like(lpf); gg[_np.arange(lpf.shape[0]), tval] = -g._grad.reshape(-1)[0] / n
        _accum(lp, _unbroadcast(gg.reshape(lp.shape), lp.shape))
    r._backward = bw
    return r


def cross_entropy(logits, target, reduction="mean"):
    return nll_loss(log_softmax(_as_t(logits)), target, reduction=reduction)


def bce_with_logits(logits, target, reduction="mean"):
    lg, tg = _as_t(logits), _as_t(target)
    d, tval = lg._data, tg._data
    # stable: max(d,0) - d*t + log(1+exp(-|d|))
    loss = _np.maximum(d, 0) - d * tval + _np.log1p(_np.exp(-_np.abs(d)))
    val = loss.mean() if reduction == "mean" else loss.sum()
    out = _ensure_contig(_np.asanyarray(val).reshape(()))
    if not (_GRAD_ENABLED and lg.requires_grad): return _wrap(out, requires_grad=False)
    r = _wrap(out, requires_grad=True, _prev=(lg,), _op="bce")
    def bw(g=r, lg=lg, d=d, tval=tval, reduction=reduction):
        sig = 1 / (1 + _np.exp(-d))
        n = d.size if reduction == "mean" else 1.0
        _accum(lg, _unbroadcast(g._grad.reshape(-1)[0] * (sig - tval) / n, lg.shape))
    r._backward = bw
    return r
