"""storch._conv: vectorized conv2d/pooling used by nn.Conv2d."""
from __future__ import annotations

import numpy as _np
from ._core import _wrap, _ensure_contig, _unbroadcast, _accum, _GRAD_ENABLED, _as_t

__all__ = ["conv2d", "conv1d", "max_pool2d", "avg_pool2d", "adaptive_avg_pool2d"]


def _im2col(x, kh, kw, s0, s1, p0, p1):
    N, C, H, W = x.shape
    if p0 or p1:
        x = _np.pad(x, ((0, 0), (0, 0), (p0, p0), (p1, p1)))
        H, W = H + 2 * p0, W + 2 * p1
    OH, OW = (H - kh) // s0 + 1, (W - kw) // s1 + 1
    stN, stC, stH, stW = x.strides
    shape = (N, C, kh, kw, OH, OW)
    strides = (stN, stC, stH, stW, stH * s0, stW * s1)
    cols = _np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    return _ensure_contig(cols.reshape(N * OH * OW, C * kh * kw)), OH, OW


def conv2d(x, weight, bias=None, stride=(1, 1), padding=(0, 0)):
    x, w = _as_t(x), _as_t(weight)
    b = _as_t(bias) if bias is not None else None
    N, C, H, Wd = x.shape
    OC, _, kh, kw = w.shape
    s0, s1 = stride; p0, p1 = padding
    cols, OH, OW = _im2col(_ensure_contig(x._data), kh, kw, s0, s1, p0, p1)
    wr = w._data.reshape(OC, -1)
    out = _ensure_contig((cols @ wr.T).reshape(N, OH, OW, OC).transpose(0, 3, 1, 2))
    if b is not None:
        out += b._data.reshape(1, -1, 1, 1)
    rg = _GRAD_ENABLED and (x.requires_grad or w.requires_grad or (b is not None and b.requires_grad))
    if not rg: return _wrap(out, requires_grad=False)
    res = _wrap(out, requires_grad=True, _prev=tuple(p for p in (x, w, b) if p is not None), _op="conv2d")
    def bw(g=res, x=x, w=w, b=b, cols=cols, wr=wr, s=(N, C, H, Wd, OH, OW, kh, kw, s0, s1, p0, p1)):
        N, C, H, Wd, OH, OW, kh, kw, s0, s1, p0, p1 = s
        gg = _ensure_contig(g._grad)
        if b is not None and b.requires_grad:
            _accum(b, _unbroadcast(gg.sum(axis=(0, 2, 3)), b.shape))
        gg_r = gg.transpose(0, 2, 3, 1).reshape(-1, w.shape[0])
        if w.requires_grad:
            _accum(w, _unbroadcast(_ensure_contig(gg_r.T @ cols).reshape(w.shape), w.shape))
        if x.requires_grad:
            dcols = gg_r @ wr
            dcols = dcols.reshape(N, OH, OW, C, kh, kw).transpose(0, 3, 4, 5, 1, 2)
            Hp, Wp = H + 2 * p0, Wd + 2 * p1
            dx = _np.zeros((N, C, Hp, Wp), dtype=dcols.dtype)
            for i in range(kh):
                ii = i
                for j in range(kw):
                    dx[:, :, ii:ii + OH * s0:s0, j:j + OW * s1:s1] += dcols[:, :, i, j, :, :]
            if p0 or p1:
                dx = dx[:, :, p0:p0 + H, p1:p1 + Wd]
            _accum(x, _ensure_contig(dx))
    res._backward = bw
    return res


def max_pool2d(x, kernel_size=2, stride=None):
    x = _as_t(x)
    if isinstance(kernel_size, int): kernel_size = (kernel_size, kernel_size)
    if stride is None: stride = kernel_size
    if isinstance(stride, int): stride = (stride, stride)
    kh, kw = kernel_size; s0, s1 = stride
    N, C, H, W = x.shape
    OH, OW = (H - kh) // s0 + 1, (W - kw) // s1 + 1
    stN, stC, stH, stW = x._data.strides
    win = _np.lib.stride_tricks.as_strided(
        x._data, shape=(N, C, OH, OW, kh, kw),
        strides=(stN, stC, stH * s0, stW * s1, stH, stW))
    out = _ensure_contig(win.max(axis=(-2, -1)))
    if not (_GRAD_ENABLED and x.requires_grad): return _wrap(out, requires_grad=False)
    s = x
    res = _wrap(out, requires_grad=True, _prev=(x,), _op="maxpool")
    def bw(g=res, s=s, win=win.copy(), OH=OH, OW=OW, kh=kh, kw=kw, s0=s0, s1=s1):
        gg = g._grad
        mx = win.max(axis=(-2, -1), keepdims=True)
        mask = (win == mx)
        mask = mask / mask.sum(axis=(-2, -1), keepdims=True)
        dwin = mask * gg[..., None, None]
        dx = _np.zeros_like(s._data)
        for i in range(kh):
            for j in range(kw):
                dx[:, :, i:i + OH * s0:s0, j:j + OW * s1:s1] += dwin[:, :, :, :, i, j]
        _accum(s, _ensure_contig(dx))
    res._backward = bw
    return res


def conv1d(x, weight, bias=None, stride=1, padding=0):
    """Conv1d via conv2d reuse: (N,C,L) -> conv2d on (N,C,1,L)."""
    x, w = _as_t(x), _as_t(weight)
    x4 = _ensure_contig(x._data[:, :, None, :])
    w4 = _ensure_contig(w._data[:, :, None, :])
    x4t = _wrap(x4, requires_grad=x.requires_grad, _prev=(x,), _op="unsqueeze")
    if _GRAD_ENABLED and x.requires_grad:
        def _bw(g=x4t, s=x):
            _accum(s, _ensure_contig(g._grad[:, :, 0, :]))
        x4t._backward = _bw
    w4t = _wrap(w4, requires_grad=w.requires_grad, _prev=(w,), _op="unsqueeze")
    if _GRAD_ENABLED and w.requires_grad:
        def _wbw(g=w4t, s=w):
            _accum(s, _ensure_contig(g._grad[:, :, 0, :]))
        w4t._backward = _wbw
    s1 = stride if isinstance(stride, int) else stride[0]
    p1 = padding if isinstance(padding, int) else padding[0]
    y4 = conv2d(x4t, w4t, bias, stride=(1, s1), padding=(0, p1))
    out = _ensure_contig(y4._data[:, :, 0, :])
    if not (_GRAD_ENABLED and y4.requires_grad):
        return _wrap(out, requires_grad=False)
    res = _wrap(out, requires_grad=True, _prev=(y4,), _op="squeeze")

    def bw(g=res, src=y4):
        gg = _ensure_contig(g._grad[:, :, None, :])
        src._grad = gg if src._grad is None else src._grad + gg
        if src._backward is not None:
            src._backward()
    res._backward = bw
    return res


def avg_pool2d(x, kernel_size=2, stride=None):
    x = _as_t(x)
    if isinstance(kernel_size, int): kernel_size = (kernel_size, kernel_size)
    if stride is None: stride = kernel_size
    if isinstance(stride, int): stride = (stride, stride)
    kh, kw = kernel_size; s0, s1 = stride
    N, C, H, W = x.shape
    OH, OW = (H - kh) // s0 + 1, (W - kw) // s1 + 1
    stN, stC, stH, stW = x._data.strides
    win = _np.lib.stride_tricks.as_strided(
        x._data, shape=(N, C, OH, OW, kh, kw),
        strides=(stN, stC, stH * s0, stW * s1, stH, stW))
    out = _ensure_contig(win.mean(axis=(-2, -1)))
    if not (_GRAD_ENABLED and x.requires_grad): return _wrap(out, requires_grad=False)
    s = x
    res = _wrap(out, requires_grad=True, _prev=(x,), _op="avgpool")
    def bw(g=res, s=s, OH=OH, OW=OW, kh=kh, kw=kw, s0=s0, s1=s1):
        gg = g._grad / (kh * kw)
        dx = _np.zeros_like(s._data)
        for i in range(kh):
            for j in range(kw):
                dx[:, :, i:i + OH * s0:s0, j:j + OW * s1:s1] += gg[:, :, :, :]
        _accum(s, _ensure_contig(dx))
    res._backward = bw
    return res


def adaptive_avg_pool2d(x, output_size=(1, 1)):
    x = _as_t(x)
    if isinstance(output_size, int): output_size = (output_size, output_size)
    oh, ow = output_size
    N, C, H, W = x.shape
    if (oh, ow) == (1, 1):
        out = _ensure_contig(x._data.mean(axis=(2, 3), keepdims=True))
        if not (_GRAD_ENABLED and x.requires_grad): return _wrap(out, requires_grad=False)
        s = x
        res = _wrap(out, requires_grad=True, _prev=(x,), _op="adaptiveavg")
        def bw(g=res, s=s, H=H, W=W):
            _accum(s, _ensure_contig(_np.broadcast_to(g._grad / (H * W), s.shape).copy()))
        res._backward = bw
        return res
    kh, kw = H // oh, W // ow
    if H % oh == 0 and W % ow == 0:
        d = _ensure_contig(x._data.reshape(N, C, oh, kh, ow, kw).mean(axis=(3, 5)))
        if not (_GRAD_ENABLED and x.requires_grad): return _wrap(d, requires_grad=False)
        s = x
        res = _wrap(d, requires_grad=True, _prev=(x,), _op="adaptiveavg")
        def bw(g=res, s=s, kh=kh, kw=kw):
            _accum(s, _ensure_contig(_np.repeat(_np.repeat(g._grad / (kh * kw), kh, axis=2), kw, axis=3)))
        res._backward = bw
        return res
    ri = (_np.linspace(0, H, oh + 1)).astype(int)
    ci = (_np.linspace(0, W, ow + 1)).astype(int)
    outs = _np.stack([[x._data[:, :, ri[i]:ri[i + 1], ci[j]:ci[j + 1]].mean(axis=(2, 3))
                        for j in range(ow)] for i in range(oh)], axis=2)
    return _wrap(_ensure_contig(outs.reshape(N, C, oh, ow)), requires_grad=False)
