"""etorch.nn: torch.nn-compatible modules (fast fused Linear etc)."""
from __future__ import annotations

import math
import numpy as _np

from ._core import Tensor, _wrap, _ensure_contig, _GRAD_ENABLED, _as_t, randn
from . import functional as F

__all__ = [
    "Module", "Sequential", "Linear", "Embedding", "LayerNorm", "Dropout",
    "ReLU", "Sigmoid", "Tanh", "GELU", "SiLU", "LeakyReLU", "Softmax",
    "Flatten", "MSELoss", "L1Loss", "CrossEntropyLoss", "NLLLoss", "BCEWithLogitsLoss",
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
