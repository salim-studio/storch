"""Benchmark storch vs naive python-loop baseline (and torch if installed)."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import storch
import storch.nn as nn


def bench_matmul():
    a = storch.randn(512, 512)
    b = storch.randn(512, 512)
    t0 = time.perf_counter()
    for _ in range(10):
        _ = a @ b
    dt = (time.perf_counter() - t0) / 10
    print(f"storch matmul 512x512: {dt*1000:.2f} ms")
    try:
        import torch
        ta, tb = torch.randn(512, 512), torch.randn(512, 512)
        t0 = time.perf_counter()
        for _ in range(10):
            _ = ta @ tb
        print(f"torch  matmul 512x512: {(time.perf_counter()-t0)/10*1000:.2f} ms")
    except ImportError:
        print("(torch not installed — skipping torch comparison)")


def bench_mlp():
    storch.manual_seed(0)
    X = storch.randn(1024, 128)
    model = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 10))
    import storch.optim as optim
    opt = optim.Adam(model.parameters())
    t0 = time.perf_counter()
    for _ in range(20):
        opt.zero_grad()
        loss = model(X).sum()
        loss.backward()
        opt.step()
    print(f"storch MLP fwd+bwd (1024x128->256->10) x20: {time.perf_counter()-t0:.2f}s")


if __name__ == "__main__":
    print(storch.info())
    bench_matmul()
    bench_mlp()
