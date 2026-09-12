"""storch.nn: torch.nn-compatible modules (fast fused Linear etc)."""
from __future__ import annotations

import math
import numpy as _np

from ._core import Tensor, _wrap, _ensure_contig, _GRAD_ENABLED, _as_t, randn
from . import functional as F

__all__ = [
    "Module", "Sequential", "ModuleList", "Linear", "Embedding", "LayerNorm",
    "BatchNorm1d", "BatchNorm2d", "Dropout",
    "ReLU", "Sigmoid", "Tanh", "GELU", "SiLU", "LeakyReLU", "ELU", "Softplus", "Mish",
    "Softmax", "Flatten",
    "Conv1d", "Conv2d", "MaxPool2d", "AvgPool2d", "AdaptiveAvgPool2d", "Upsample",
    "RNNCell", "LSTMCell", "GRUCell", "RNN", "LSTM", "GRU",
    "MultiheadAttention", "TransformerEncoderLayer",
    "MSELoss", "L1Loss", "HuberLoss", "SmoothL1Loss",
    "CrossEntropyLoss", "NLLLoss", "BCEWithLogitsLoss",
]


class Module:
    training: bool = True

    def __init__(self):
        object.__setattr__(self, "_params", {})
        object.__setattr__(self, "_modules", {})

    def __setattr__(self, name, value):
        if isinstance(value, Tensor):
            self._params[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        object.__setattr__(self, name, value)

    def __call__(self, *a, **k): return self.forward(*a, **k)
    def forward(self, *a, **k): raise NotImplementedError

    def parameters(self):
        ps = list(self._params.values())
        for m in self._modules.values():
            ps += m.parameters()
        # Sequential containers store children in _modules too; also check list attrs
        for v in vars(self).values():
            if isinstance(v, list):
                for m in v:
                    if isinstance(m, Module): ps += m.parameters()
        # dedupe by id
        seen, out = set(), []
        for p in ps:
            if id(p) not in seen: seen.add(id(p)); out.append(p)
        return out

    def named_parameters(self, prefix=""):
        for k, p in self._params.items():
            yield (prefix + k, p)
        for nk, m in self._modules.items():
            yield from m.named_parameters(prefix + nk + ".")
        for k, v in vars(self).items():
            if isinstance(v, list):
                for i, m in enumerate(v):
                    if isinstance(m, Module): yield from m.named_parameters(f"{prefix}{k}.{i}.")

    def zero_grad(self):
        for p in self.parameters(): p.zero_grad()

    def train(self, mode=True):
        self.training = mode
        for m in self._modules.values(): m.train(mode)
        for v in vars(self).values():
            if isinstance(v, list):
                for m in v:
                    if isinstance(m, Module): m.train(mode)
        return self

    def eval(self): return self.train(False)

    def state_dict(self): return {k: Tensor(p._data.copy()) for k, p in self.named_parameters()}
    def load_state_dict(self, sd):
        mine = dict(self.named_parameters())
        for k, v in sd.items():
            if k in mine:
                mine[k]._data = _ensure_contig((v._data if isinstance(v, Tensor) else _np.asanyarray(v)).astype(mine[k].dtype, copy=False))

    def to(self, dtype=None, device=None):
        for p in self.parameters(): p.to(dtype)
        return self
    def float(self): return self.to(_np.float32)


class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = list(layers)
        for i, l in enumerate(layers):
            if isinstance(l, Module): self._modules[str(i)] = l

    def forward(self, x):
        for l in self.layers: x = l(x)
        return x

    def append(self, m):
        self.layers.append(m)
        if isinstance(m, Module): self._modules[str(len(self.layers) - 1)] = m
        return self


class ModuleList(Module):
    def __init__(self, modules=None):
        super().__init__()
        self.layers = list(modules or [])
        for i, l in enumerate(self.layers):
            if isinstance(l, Module): self._modules[str(i)] = l

    def __len__(self): return len(self.layers)
    def __getitem__(self, i): return self.layers[i]
    def __iter__(self): return iter(self.layers)
    def append(self, m):
        self.layers.append(m)
        if isinstance(m, Module): self._modules[str(len(self.layers) - 1)] = m
        return self
    def forward(self, x):
        for l in self.layers: x = l(x)
        return x


def _kaiming(n_in, n_out, dtype=_np.float32):
    return (randn(n_out, n_in, dtype=dtype)._data * math.sqrt(2.0 / n_in)).astype(dtype)


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, dtype=_np.float32):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.weight = Tensor(_kaiming(in_features, out_features, dtype), requires_grad=True)
        self.bias = Tensor(_np.zeros(out_features, dtype=dtype), requires_grad=True) if bias else None
        self._params["weight"] = self.weight
        if bias: self._params["bias"] = self.bias

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, dtype=_np.float32):
        super().__init__()
        self.weight = Tensor(randn(num_embeddings, embedding_dim, dtype=dtype)._data, requires_grad=True)
        self._params["weight"] = self.weight

    def forward(self, idx): return F.embedding(idx, self.weight)


class LayerNorm(Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        if isinstance(normalized_shape, int): normalized_shape = (normalized_shape,)
        self.normalized_shape, self.eps = tuple(normalized_shape), eps
        self.weight = Tensor(_np.ones(self.normalized_shape, dtype=_np.float32), requires_grad=True)
        self.bias = Tensor(_np.zeros(self.normalized_shape, dtype=_np.float32), requires_grad=True)
        self._params.update(weight=self.weight, bias=self.bias)

    def forward(self, x): return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)


class Dropout(Module):
    def __init__(self, p=0.5): super().__init__(); self.p = p
    def forward(self, x): return F.dropout(x, self.p, self.training)


class ReLU(Module):
    def forward(self, x): return F.relu(x)
class ELU(Module):
    def __init__(self, alpha=1.0): super().__init__(); self.alpha = alpha
    def forward(self, x): return F.elu(x, self.alpha)
class Softplus(Module):
    def forward(self, x): return F.softplus(x)
class Mish(Module):
    def forward(self, x): return F.mish(x)
class Sigmoid(Module):
    def forward(self, x): return F.sigmoid(x)
class Tanh(Module):
    def forward(self, x): return F.tanh(x)
class GELU(Module):
    def forward(self, x): return F.gelu(x)
class SiLU(Module):
    def forward(self, x): return F.silu(x)
class LeakyReLU(Module):
    def __init__(self, ns=0.01): super().__init__(); self.ns = ns
    def forward(self, x): return F.leaky_relu(x, self.ns)
class Softmax(Module):
    def __init__(self, dim=-1): super().__init__(); self.dim = dim
    def forward(self, x): return F.softmax(x, self.dim)
class Flatten(Module):
    def __init__(self, start_dim=1): super().__init__(); self.start_dim = start_dim
    def forward(self, x):
        x = _as_t(x)
        sh = x.shape
        return x.reshape((sh[0], -1) if self.start_dim == 1 else (-1,))


class _Loss(Module):
    def __init__(self, reduction="mean"): super().__init__(); self.reduction = reduction
class MSELoss(_Loss):
    def forward(self, p, t): return F.mse_loss(p, t, self.reduction)
class HuberLoss(_Loss):
    def __init__(self, reduction="mean", delta=1.0): super().__init__(reduction); self.delta = delta
    def forward(self, p, t): return F.huber_loss(p, t, self.delta, self.reduction)
class SmoothL1Loss(HuberLoss):
    pass
class L1Loss(_Loss):
    def forward(self, p, t): return F.l1_loss(p, t, self.reduction)
class CrossEntropyLoss(_Loss):
    def forward(self, l, t): return F.cross_entropy(l, t, self.reduction)
class NLLLoss(_Loss):
    def forward(self, l, t): return F.nll_loss(l, t, self.reduction)
class BCEWithLogitsLoss(_Loss):
    def forward(self, l, t): return F.bce_with_logits(l, t, self.reduction)


class Conv2d(Module):
    """Vectorized Conv2d (im2col via stride tricks + BLAS matmul)."""

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, bias=True, dtype=_np.float32):
        super().__init__()
        if isinstance(kernel_size, int): kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int): stride = (stride, stride)
        if isinstance(padding, int): padding = (padding, padding)
        self.in_ch, self.out_ch, self.ks, self.stride, self.padding = in_ch, out_ch, kernel_size, stride, padding
        kh, kw = kernel_size
        self.weight = Tensor((randn(out_ch, in_ch, kh, kw, dtype=dtype)._data * math.sqrt(2.0 / (in_ch * kh * kw))).astype(dtype), requires_grad=True)
        self.bias = Tensor(_np.zeros(out_ch, dtype=dtype), requires_grad=True) if bias else None
        self._params["weight"] = self.weight
        if bias: self._params["bias"] = self.bias

    def forward(self, x):
        from . import _conv
        return _conv.conv2d(x, self.weight, self.bias, self.stride, self.padding)


class Conv1d(Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, bias=True, dtype=_np.float32):
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        self.kernel_size, self.stride, self.padding = kernel_size, stride, padding
        self.weight = Tensor((randn(out_ch, in_ch, kernel_size, dtype=dtype)._data
                              * math.sqrt(2.0 / (in_ch * kernel_size))).astype(dtype), requires_grad=True)
        self.bias = Tensor(_np.zeros(out_ch, dtype=dtype), requires_grad=True) if bias else None
        self._params["weight"] = self.weight
        if bias: self._params["bias"] = self.bias

    def forward(self, x):
        from . import _conv
        return _conv.conv1d(x, self.weight, self.bias, self.stride, self.padding)


class BatchNorm1d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features, self.eps, self.momentum = num_features, eps, momentum
        self.weight = Tensor(_np.ones(num_features, dtype=_np.float32), requires_grad=True)
        self.bias = Tensor(_np.zeros(num_features, dtype=_np.float32), requires_grad=True)
        self._params.update(weight=self.weight, bias=self.bias)
        self.running_mean = _np.zeros(num_features, dtype=_np.float32)
        self.running_var = _np.ones(num_features, dtype=_np.float32)

    def forward(self, x):
        from ._core import _GRAD_ENABLED
        xd = _as_t(x)
        if self.training:
            axes = (0,) + tuple(range(2, xd.ndim))
            d = xd._data
            mu = d.mean(axis=axes)
            var = ((d - mu.reshape((1, -1) + (1,) * (xd.ndim - 2))) ** 2).mean(axis=axes)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
            return F.batch_norm(x, self.weight, self.bias, self.eps)
        # eval: normalize with running stats — differentiable in x/weight/bias
        shape = (1, -1) + (1,) * (xd.ndim - 2)
        rm = Tensor(self.running_mean.reshape(shape))
        rs = Tensor((1.0 / _np.sqrt(self.running_var.reshape(shape) + self.eps)).astype(_np.float32))
        xn = (xd + (-rm)) * rs
        w = self.weight.reshape(shape)
        b = self.bias.reshape(shape)
        return xn * w + b


class BatchNorm2d(BatchNorm1d):
    pass


class MaxPool2d(Module):
    def __init__(self, kernel_size=2, stride=None): super().__init__(); self.ks, self.st = kernel_size, stride
    def forward(self, x):
        from . import _conv
        return _conv.max_pool2d(x, self.ks, self.st)


class AvgPool2d(Module):
    def __init__(self, kernel_size=2, stride=None): super().__init__(); self.ks, self.st = kernel_size, stride
    def forward(self, x):
        from . import _conv
        return _conv.avg_pool2d(x, self.ks, self.st)


class AdaptiveAvgPool2d(Module):
    def __init__(self, output_size=(1, 1)): super().__init__(); self.os = output_size
    def forward(self, x):
        from . import _conv
        return _conv.adaptive_avg_pool2d(x, self.os)


class Upsample(Module):
    def __init__(self, scale_factor=2): super().__init__(); self.sf = scale_factor
    def forward(self, x): return F.interpolate_nearest(x, self.sf)


# ---- Recurrent cells (built from differentiable ops -> autograd free) ----

class RNNCell(Module):
    def __init__(self, input_size, hidden_size, nonlinearity="tanh", bias=True):
        super().__init__()
        self.inp, self.hid = input_size, hidden_size
        self.nonlinearity = nonlinearity
        s = 1.0 / math.sqrt(hidden_size)
        self.weight_ih = Tensor((_np.random.rand(hidden_size, input_size).astype(_np.float32) * 2 * s - s), requires_grad=True)
        self.weight_hh = Tensor((_np.random.rand(hidden_size, hidden_size).astype(_np.float32) * 2 * s - s), requires_grad=True)
        self.bias_ih = Tensor(_np.zeros(hidden_size, dtype=_np.float32), requires_grad=True) if bias else None
        self.bias_hh = Tensor(_np.zeros(hidden_size, dtype=_np.float32), requires_grad=True) if bias else None
        self._params.update(weight_ih=self.weight_ih, weight_hh=self.weight_hh)
        if bias: self._params.update(bias_ih=self.bias_ih, bias_hh=self.bias_hh)

    def forward(self, x, h=None):
        from ._core import _as_t as _a
        x, h0 = _a(x), _a(h) if h is not None else None
        if h0 is None:
            h0 = _a(_np.zeros((x.shape[0], self.hid), dtype=_np.float32))
        y = F.linear(x, self.weight_ih, self.bias_ih) + F.linear(h0, self.weight_hh, self.bias_hh)
        return _a(y).tanh() if self.nonlinearity == "tanh" else F.relu(y)


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.inp, self.hid = input_size, hidden_size
        s = 1.0 / math.sqrt(hidden_size)
        self.weight_ih = Tensor((_np.random.rand(4 * hidden_size, input_size).astype(_np.float32) * 2 * s - s), requires_grad=True)
        self.weight_hh = Tensor((_np.random.rand(4 * hidden_size, hidden_size).astype(_np.float32) * 2 * s - s), requires_grad=True)
        self.bias = Tensor(_np.zeros(4 * hidden_size, dtype=_np.float32), requires_grad=True) if bias else None
        self._params.update(weight_ih=self.weight_ih, weight_hh=self.weight_hh)
        if bias: self._params["bias"] = self.bias

    def forward(self, x, state=None):
        from ._core import _as_t as _a
        x = _a(x)
        if state is None:
            h = _a(_np.zeros((x.shape[0], self.hid), dtype=_np.float32))
            c = _a(_np.zeros((x.shape[0], self.hid), dtype=_np.float32))
        else:
            h, c = state
        gates = F.linear(x, self.weight_ih, None) + F.linear(_a(h), self.weight_hh, self.bias)
        # split into 4 gates (each slice is grad-capable via __getitem__)
        gi = gates[:, 0 * self.hid:1 * self.hid]
        gf = gates[:, 1 * self.hid:2 * self.hid]
        gg = gates[:, 2 * self.hid:3 * self.hid]
        go = gates[:, 3 * self.hid:4 * self.hid]
        ii, ff, oo = gi.sigmoid(), gf.sigmoid(), go.sigmoid()
        cc = gg.tanh()
        c_new = ff * _a(c) + ii * cc
        h_new = oo * c_new.tanh()
        return h_new, c_new


class GRUCell(Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.inp, self.hid = input_size, hidden_size
        s = 1.0 / math.sqrt(hidden_size)
        self.weight_ih = Tensor((_np.random.rand(3 * hidden_size, input_size).astype(_np.float32) * 2 * s - s), requires_grad=True)
        self.weight_hh = Tensor((_np.random.rand(3 * hidden_size, hidden_size).astype(_np.float32) * 2 * s - s), requires_grad=True)
        self.bias = Tensor(_np.zeros(3 * hidden_size, dtype=_np.float32), requires_grad=True) if bias else None
        self._params.update(weight_ih=self.weight_ih, weight_hh=self.weight_hh)
        if bias: self._params["bias"] = self.bias

    def forward(self, x, h=None):
        from ._core import _as_t as _a
        x = _a(x)
        h = _a(h) if h is not None else _a(_np.zeros((x.shape[0], self.hid), dtype=_np.float32))
        gates_x = F.linear(x, self.weight_ih, None)
        gates_h = F.linear(h, self.weight_hh, self.bias)
        rx = gates_x[:, 0 * self.hid:1 * self.hid]; rh = gates_h[:, 0 * self.hid:1 * self.hid]
        zx = gates_x[:, 1 * self.hid:2 * self.hid]; zh = gates_h[:, 1 * self.hid:2 * self.hid]
        nx = gates_x[:, 2 * self.hid:3 * self.hid]; nh = gates_h[:, 2 * self.hid:3 * self.hid]
        r, z = (rx + rh).sigmoid(), (zx + zh).sigmoid()
        n = (nx + r * nh).tanh()
        return (1 - z) * n + z * h


class _RNNBase(Module):
    _cell = RNNCell
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True, _nonlinearity="tanh"):
        super().__init__()
        self.hid = hidden_size
        if self._cell is RNNCell:
            self.layers = [RNNCell(input_size if i == 0 else hidden_size, hidden_size, _nonlinearity)
                           for i in range(num_layers)]
        else:
            self.layers = [self._cell(input_size if i == 0 else hidden_size, hidden_size)
                           for i in range(num_layers)]
        self.num_layers = num_layers
        for i, l in enumerate(self.layers): self._modules[str(i)] = l

    def forward(self, x, h=None):
        from ._core import _as_t as _a
        x = _a(x)  # (B,T,F)
        B, T, _ = x.shape
        outs = []
        states = []
        inp = x
        for li, cell in enumerate(self.layers):
            hh = None if h is None else _a(h[li])
            cc = None
            seq = []
            for t in range(T):
                xt = inp[:, t, :]
                if isinstance(cell, LSTMCell):
                    hh, cc = cell(xt, (hh, cc) if hh is not None else None)
                    seq.append(hh)
                else:
                    hh = cell(xt, hh)
                    seq.append(hh)
            from ._core import stack as _st
            inp = _st(seq, dim=1)
            outs.append(inp)
            states.append(hh)
        from ._core import stack as _st
        return inp, _st(states, dim=0)


class RNN(_RNNBase):
    _cell = RNNCell
    def __init__(self, input_size, hidden_size, num_layers=1, nonlinearity="tanh"):
        super().__init__(input_size, hidden_size, num_layers, _nonlinearity=nonlinearity)


class LSTM(_RNNBase):
    _cell = LSTMCell


class GRU(_RNNBase):
    _cell = GRUCell


class MultiheadAttention(Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.E, self.H, self.D = embed_dim, num_heads, embed_dim // num_heads
        self.dropout = dropout
        self.q_proj = Linear(embed_dim, embed_dim)
        self.k_proj = Linear(embed_dim, embed_dim)
        self.v_proj = Linear(embed_dim, embed_dim)
        self.out_proj = Linear(embed_dim, embed_dim)
        for n, m in (("q", self.q_proj), ("k", self.k_proj), ("v", self.v_proj), ("o", self.out_proj)):
            self._modules[n] = m

    def forward(self, query, key=None, value=None, attn_mask=None):
        from ._core import _as_t as _a, stack as _st
        key = query if key is None else key
        value = key if value is None else value
        Q, K, V = self.q_proj(_a(query)), self.k_proj(_a(key)), self.v_proj(_a(value))
        B, Tq, _ = Q.shape
        # differentiable split into heads: (B,T,E) -> (B,T,H,D) -> (B,H,T,D)
        Qh = Q.reshape((B, Tq, self.H, self.D)).permute(0, 2, 1, 3)
        Tk = K.shape[1]
        Kh = K.reshape((B, Tk, self.H, self.D)).permute(0, 2, 1, 3)
        Vh = V.reshape((B, Tk, self.H, self.D)).permute(0, 2, 1, 3)
        outs = []
        for hh in range(self.H):
            qq, kk, vv = Qh[:, hh, :, :], Kh[:, hh, :, :], Vh[:, hh, :, :]
            outs.append(F.scaled_dot_product_attention(qq, kk, vv))
        out = _st(outs, dim=1).permute(0, 2, 1, 3).reshape((B, Tq, self.E))
        return self.out_proj(out)


class TransformerEncoderLayer(Module):
    def __init__(self, d_model, nhead, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.attn = MultiheadAttention(d_model, nhead, dropout)
        self.ln1 = LayerNorm(d_model); self.ln2 = LayerNorm(d_model)
        self.ff1 = Linear(d_model, dim_feedforward); self.ff2 = Linear(dim_feedforward, d_model)
        self.drop = Dropout(dropout)
        for n, m in (("attn", self.attn), ("ln1", self.ln1), ("ln2", self.ln2),
                     ("ff1", self.ff1), ("ff2", self.ff2)):
            self._modules[n] = m

    def forward(self, x):
        a = self.attn(x, x, x)
        x = self.ln1(_as_t(x) + self.drop(a))
        f = self.ff2(F.relu(self.ff1(x)))
        return self.ln2(_as_t(x) + self.drop(f))
