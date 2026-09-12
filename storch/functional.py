"""storch.functional: torch.nn.functional-compatible ops (fused fast paths)."""
from __future__ import annotations

import numpy as _np
from ._core import Tensor, _wrap, _ensure_contig, _unbroadcast, _accum, _GRAD_ENABLED, _as_t
from ._parallel import fused_bias_add_relu

__all__ = [
    "relu", "sigmoid", "tanh", "gelu", "silu", "leaky_relu", "elu", "softplus", "mish",
    "softmax", "log_softmax",
    "linear", "cross_entropy", "mse_loss", "l1_loss", "bce_with_logits",
    "huber_loss", "smooth_l1_loss", "nll_loss", "dropout", "layer_norm", "batch_norm",
    "embedding", "max_pool2d", "avg_pool2d", "adaptive_avg_pool2d",
    "interpolate_nearest", "scaled_dot_product_attention",
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
        if w.requires_grad:
            if gg.ndim == 2:
                wg = _ensure_contig(gg.T @ x._data)
            else:  # (..., in) @ (out,in).T -> flatten leading dims
                gg2 = gg.reshape(-1, gg.shape[-1])
                x2 = x._data.reshape(-1, x.shape[-1])
                wg = _ensure_contig(gg2.T @ x2)
            _accum(w, _unbroadcast(wg, w.shape))
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
    if not (_GRAD_ENABLED and (t.requires_grad or
        (weight is not None and _as_t(weight).requires_grad) or
        (bias is not None and _as_t(bias).requires_grad))): return _wrap(out, requires_grad=False)
    s = t
    ww = _as_t(weight) if weight is not None else None
    bb = _as_t(bias) if bias is not None else None
    r = _wrap(out, requires_grad=True,
              _prev=tuple(p for p in (s, ww, bb) if isinstance(p, Tensor)), _op="layernorm")
    def bw(g=r, s=s, mu=mu, var=var, xn=xn, eps=eps, axes=axes, w=weight, ww=ww, bb=bb):
        gg = g._grad
        wd = _as_t(w)._data if w is not None else 1.0
        nd = len(xn.shape) if isinstance(xn, _np.ndarray) else s.ndim
        # reduce axes = all dims except the last len(normalized_shape) dims
        k = len(axes)
        red = tuple(range(gg.ndim - k - (gg.ndim - s.ndim))) if False else None
        # normalized dims are the last `k` dims of x (since axes = range(-k, 0))
        norm_axes = tuple(range(s.ndim - k, s.ndim))
        red_axes = tuple(a for a in range(gg.ndim) if (a - (gg.ndim - s.ndim)) not in norm_axes)
        if ww is not None and ww.requires_grad:
            _accum(ww, _unbroadcast(_ensure_contig((gg * xn).sum(axis=red_axes, keepdims=False).reshape(ww.shape) if red_axes else (gg * xn)), ww.shape))
        if bb is not None and bb.requires_grad:
            _accum(bb, _unbroadcast(_ensure_contig(gg.sum(axis=red_axes).reshape(bb.shape) if red_axes else gg), bb.shape))
        if s.requires_grad:
            gg2 = gg * wd
            n = _np.prod([s.shape[a] for a in axes])
            gmu = gg2.mean(axis=axes, keepdims=True)
            gxn = (gg2 - gmu - xn * (gg2 * xn).mean(axis=axes, keepdims=True)) / _np.sqrt(var + eps)
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


def huber_loss(pred, target, delta=1.0, reduction="mean"):
    p, tg = _as_t(pred), _as_t(target)
    diff = p._data - tg._data
    ad = _np.abs(diff)
    loss = _np.where(ad < delta, 0.5 * diff * diff, delta * (ad - 0.5 * delta))
    val = loss.mean() if reduction == "mean" else loss.sum()
    out = _ensure_contig(_np.asanyarray(val).reshape(()))
    if not (_GRAD_ENABLED and p.requires_grad): return _wrap(out, requires_grad=False)
    r = _wrap(out, requires_grad=True, _prev=(p,), _op="huber")
    def bw(g=r, p=p, diff=diff, delta=delta, reduction=reduction):
        n = diff.size if reduction == "mean" else 1.0
        grad = _np.where(_np.abs(diff) < delta, diff, delta * _np.sign(diff)) / n
        _accum(p, _unbroadcast(g._grad.reshape(-1)[0] * grad, p.shape))
    r._backward = bw
    return r


def smooth_l1_loss(pred, target, beta=1.0, reduction="mean"):
    return huber_loss(pred, target, delta=beta, reduction=reduction)


def elu(t, alpha=1.0):
    t = _as_t(t)
    out = _ensure_contig(_np.where(t._data > 0, t._data, alpha * (_np.exp(t._data) - 1)))
    if not (_GRAD_ENABLED and t.requires_grad): return _wrap(out, requires_grad=False)
    s = t
    r = t._new(out, (t,), "elu", None, True)
    def bw(g=r, s=s, alpha=alpha, out=out):
        grad = _np.where(s._data > 0, 1.0, out + alpha)
        _accum(s, _unbroadcast(g._grad * grad, s.shape))
    r._backward = bw
    return r


def softplus(t, beta=1.0, threshold=20.0):
    t = _as_t(t)
    x = beta * t._data
    out = _ensure_contig(_np.where(x > threshold, x, _np.log1p(_np.exp(x))) / beta)
    if not (_GRAD_ENABLED and t.requires_grad): return _wrap(out, requires_grad=False)
    s = t
    r = t._new(out, (t,), "softplus", None, True)
    def bw(g=r, s=s, beta=beta, threshold=threshold):
        x = beta * s._data
        sig = _np.where(x > threshold, 1.0, 1 / (1 + _np.exp(-x)))
        _accum(s, _unbroadcast(g._grad * sig, s.shape))
    r._backward = bw
    return r


def mish(t):
    t = _as_t(t)
    sp = _np.log1p(_np.exp(t._data))
    out = _ensure_contig(t._data * _np.tanh(sp))
    if not (_GRAD_ENABLED and t.requires_grad): return _wrap(out, requires_grad=False)
    s = t
    r = t._new(out, (t,), "mish", None, True)
    def bw(g=r, s=s):
        d = s._data
        sig = 1 / (1 + _np.exp(-d))
        th = _np.tanh(_np.log1p(_np.exp(d)))
        sech2 = 1 - th * th
        grad = th + d * sech2 * sig
        _accum(s, _unbroadcast(g._grad * grad, s.shape))
    r._backward = bw
    return r


def batch_norm(x, weight=None, bias=None, eps=1e-5):
    """BatchNorm in training mode (batch statistics). Built from differentiable ops."""
    x = _as_t(x)
    # infer channel dim: (N,C,...) -> stats over N + spatial dims
    axes = (0,) + tuple(range(2, x.ndim))
    mu = x._data.mean(axis=axes, keepdims=True)
    var = ((x._data - mu) ** 2).mean(axis=axes, keepdims=True)
    xn = (x._data - mu) / _np.sqrt(var + eps)
    out = xn
    if weight is not None: out = out * _as_t(weight)._data.reshape((1, -1) + (1,) * (x.ndim - 2))
    if bias is not None: out = out + _as_t(bias)._data.reshape((1, -1) + (1,) * (x.ndim - 2))
    out = _ensure_contig(out)
    if not (_GRAD_ENABLED and (x.requires_grad or
        (weight is not None and _as_t(weight).requires_grad) or
        (bias is not None and _as_t(bias).requires_grad))):
        return _wrap(out, requires_grad=False)
    xx, ww, bb = x, weight, bias
    res = _wrap(out, requires_grad=True,
                _prev=tuple(p for p in (xx, ww, bb) if isinstance(p, Tensor)), _op="batchnorm")
    def bw(g=res, xx=xx, ww=ww, bb=bb, mu=mu, var=var, xn=xn, eps=eps, axes=axes):
        gg = g._grad
        w = _as_t(ww)._data.reshape((1, -1) + (1,) * (xx.ndim - 2)) if ww is not None else 1.0
        gxn = gg * w
        n = _np.prod([xx.shape[a] for a in axes])
        gmu = gxn.mean(axis=axes, keepdims=True)
        gxn_c = (gxn - gmu - xn * (gxn * xn).mean(axis=axes, keepdims=True)) / _np.sqrt(var + eps)
        if xx.requires_grad:
            _accum(xx, _unbroadcast(gxn_c, xx.shape))
        if isinstance(ww, Tensor) and ww.requires_grad:
            gw = (gg * xn).sum(axis=axes)
            _accum(ww, _unbroadcast(gw.reshape(ww.shape), ww.shape))
        if isinstance(bb, Tensor) and bb.requires_grad:
            gb = gg.sum(axis=axes)
            _accum(bb, _unbroadcast(gb.reshape(bb.shape), bb.shape))
    res._backward = bw
    return res


def max_pool2d(t, kernel_size=2, stride=None):
    from . import _conv
    return _conv.max_pool2d(t, kernel_size, stride)


def avg_pool2d(t, kernel_size=2, stride=None):
    from . import _conv
    return _conv.avg_pool2d(t, kernel_size, stride)


def adaptive_avg_pool2d(t, output_size=(1, 1)):
    from . import _conv
    return _conv.adaptive_avg_pool2d(t, output_size)


def interpolate_nearest(t, scale_factor=2):
    t = _as_t(t)
    if isinstance(scale_factor, int): scale_factor = (scale_factor, scale_factor)
    sh, st = scale_factor
    d = t._data
    out = _ensure_contig(_np.repeat(_np.repeat(d, sh, axis=-2), st, axis=-1))
    if not (_GRAD_ENABLED and t.requires_grad): return _wrap(out, requires_grad=False)
    s = t
    r = t._new(out, (t,), "upsample", None, True)
    def bw(g=r, s=s, sh=sh, st=st):
        gg = g._grad
        N, C, H, W = s.shape
        dx = gg.reshape(N, C, H, sh, W, st).sum(axis=(3, 5))
        _accum(s, _ensure_contig(dx))
    r._backward = bw
    return r


def scaled_dot_product_attention(q, k, v, mask=None, dropout_p=0.0, training=False):
    """Attention: softmax(QK^T/sqrt(d))V — fully differentiable."""
    q, k, v = _as_t(q), _as_t(k), _as_t(v)
    d = q.shape[-1]
    scores = (q @ k.transpose(-2, -1)) * (1.0 / _np.sqrt(float(d)))
    if mask is not None:
        m = mask._data if isinstance(mask, Tensor) else _np.asanyarray(mask)
        scores = _as_t(scores).masked_fill(m == 0, -1e9)
    attn = softmax(scores, dim=-1)
    if dropout_p and training:
        attn = dropout(attn, dropout_p, True)
    return attn @ v
