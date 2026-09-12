"""storch.optim: torch.optim-compatible SGD / Adam / AdamW with fused in-place steps."""
from __future__ import annotations

import math
import numpy as _np

__all__ = ["Optimizer", "SGD", "Adam", "AdamW"]


class Optimizer:
    def __init__(self, params, lr=1e-3):
        self.param_groups = []
        if isinstance(params, dict):
            self.param_groups.append({"params": list(params["params"]), "lr": params.get("lr", lr)})
        elif isinstance(params, (list, tuple)) and params and isinstance(params[0], dict):
            for g in params:
                self.param_groups.append({"params": list(g["params"]), "lr": g.get("lr", lr), **{k: v for k, v in g.items() if k not in ("params", "lr")}})
        else:
            self.param_groups.append({"params": list(params), "lr": lr})
        self.defaults = {"lr": lr}
        self.state = {}

    def zero_grad(self):
        for g in self.param_groups:
            for p in g["params"]: p.zero_grad()

    def step(self): raise NotImplementedError

    def state_dict(self): return {"param_groups": self.param_groups, "state": self.state}
    def load_state_dict(self, sd): self.param_groups, self.state = sd["param_groups"], sd["state"]


class SGD(Optimizer):
    def __init__(self, params, lr=1e-2, momentum=0.0, weight_decay=0.0, nesterov=False):
        super().__init__(params, lr)
        for g in self.param_groups:
            g.update(momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)

    def step(self):
        for g in self.param_groups:
            lr, mu, wd, nest = g["lr"], g["momentum"], g["weight_decay"], g["nesterov"]
            for p in g["params"]:
                if p.grad is None: continue
                # fused: grad + wd*p in one expression, in-place param update
                gr = p.grad
                if wd: gr = gr + wd * p._data
                st = self.state.setdefault(id(p), {})
                if mu:
                    buf = st.get("buf")
                    buf = gr.copy() if buf is None else (buf * mu + gr * (1 if nest else 1))
                    if id(p) not in st or st.get("buf") is None:
                        st["buf"] = gr.copy() if st.get("buf") is None and mu else buf
                    else:
                        st["buf"] *= mu; st["buf"] += gr
                    d = gr + mu * st["buf"] if nest else st["buf"]
                else:
                    d = gr
                p._data -= lr * d  # in-place, no alloc for params


class Adam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, amsgrad=False):
        super().__init__(params, lr)
        for g in self.param_groups:
            g.update(betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)

    def step(self):
        for g in self.param_groups:
            lr, (b1, b2), eps, wd = g["lr"], g["betas"], g["eps"], g["weight_decay"]
            for p in g["params"]:
                if p.grad is None: continue
                gr = p.grad
                if wd:  # decoupled? No: classic Adam L2 here; AdamW below is decoupled
                    gr = gr + wd * p._data
                st = self.state.setdefault(id(p), {})
                if not st:
                    st.update(m=_np.zeros_like(p._data), v=_np.zeros_like(p._data), t=0)
                    if g["amsgrad"]: st["vmax"] = _np.zeros_like(p._data)
                st["t"] += 1
                # fused in-place moment updates (single pass each)
                st["m"] *= b1; st["m"] += (1 - b1) * gr
                st["v"] *= b2; st["v"] += (1 - b2) * (gr * gr)
                mhat = st["m"] / (1 - b1 ** st["t"])
                vhat = st["v"] / (1 - b2 ** st["t"])
                if g["amsgrad"]:
                    _np.maximum(st["vmax"], vhat, out=st["vmax"]); vhat = st["vmax"]
                p._data -= (lr * mhat / (_np.sqrt(vhat) + eps))


class AdamW(Adam):
    """Decoupled weight decay (matches torch.optim.AdamW)."""

    def step(self):
        for g in self.param_groups:
            lr, (b1, b2), eps, wd = g["lr"], g["betas"], g["eps"], g["weight_decay"]
            for p in g["params"]:
                if p.grad is None: continue
                if wd:
                    p._data -= lr * wd * p._data  # decoupled, in-place
                gr = p.grad
                st = self.state.setdefault(id(p), {})
                if not st:
                    st.update(m=_np.zeros_like(p._data), v=_np.zeros_like(p._data), t=0)
                st["t"] += 1
                st["m"] *= b1; st["m"] += (1 - b1) * gr
                st["v"] *= b2; st["v"] += (1 - b2) * (gr * gr)
                mhat = st["m"] / (1 - b1 ** st["t"])
                vhat = st["v"] / (1 - b2 ** st["t"])
                p._data -= (lr * mhat / (_np.sqrt(vhat) + eps))
