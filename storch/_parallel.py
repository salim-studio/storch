"""storch parallel engine: multithreaded element-wise + fused ops.

Why faster than naive torch-on-cpu patterns for many workloads:
- chunked ThreadPool for memory-bound element-wise ops (GIL released in C loops),
- fused kernels (bias+relu, mul+add, sgd/adam step) -> fewer passes & allocations,
- float32 + C-contiguous fast path everywhere.
"""
from __future__ import annotations

import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor

_ncpu = os.cpu_count() or 4
MAX_WORKERS = max(1, min(32, _ncpu))
PARALLEL_THRESHOLD = 50_000


def get_workers(n: int | None = None) -> int:
    if n is not None and n < PARALLEL_THRESHOLD:
        return 1
    return MAX_WORKERS


def _ranges(n: int, workers: int):
    chunk = (n + workers - 1) // workers
    return [(i, min(i + chunk, n)) for i in range(0, n, chunk)]


def parallel_ewise(func, *arrays, out=None):
    """Run C-level element-wise func over flat chunks in parallel."""
    arrays = [np.asanyarray(a) for a in arrays]
    try:
        s0 = arrays[0].shape
        if all(a.shape == s0 for a in arrays):
            size = arrays[0].size
            if size < PARALLEL_THRESHOLD or MAX_WORKERS <= 1:
                if out is None:
                    res = np.empty(s0, dtype=np.result_type(*arrays))
                    func(*arrays, out=res)
                    return res
                func(*arrays, out=out)
                return out
            flats = [a.reshape(-1) if a.flags["C_CONTIGUOUS"] else np.ascontiguousarray(a).reshape(-1)
                     for a in arrays]
            res_flat = np.empty(size, dtype=np.result_type(*arrays)) if out is None \
                else np.asanyarray(out).reshape(-1)
            rs = _ranges(size, MAX_WORKERS)

            def _job(r):
                s, e = r
                func(*[f[s:e] for f in flats], out=res_flat[s:e])

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                list(ex.map(_job, rs))
            return res_flat.reshape(s0) if out is None else out
    except Exception:
        pass
    # broadcast fallback
    bcast = [np.ascontiguousarray(a) for a in np.broadcast_arrays(*arrays)]
    shape = bcast[0].shape
    size = bcast[0].size
    if size < PARALLEL_THRESHOLD or MAX_WORKERS <= 1:
        if out is None:
            res = np.empty(shape, dtype=np.result_type(*arrays))
            func(*bcast, out=res.reshape(-1))
            return res
        func(*bcast, out=np.asanyarray(out).reshape(-1))
        return out
    flats = [b.reshape(-1) for b in bcast]
    res_flat = np.empty(size, dtype=np.result_type(*arrays)) if out is None \
        else np.asanyarray(out).reshape(-1)
    rs = _ranges(size, MAX_WORKERS)

    def _job(r):
        s, e = r
        func(*[f[s:e] for f in flats], out=res_flat[s:e])

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        list(ex.map(_job, rs))
    return res_flat.reshape(shape) if out is None else out


# ---- fused kernels (single pass, parallel when large) ----

def _np_add(a, b, out): np.add(a, b, out=out)
def _np_mul(a, b, out): np.multiply(a, b, out=out)
def _np_relu(a, out): np.maximum(a, 0, out=out)
def _np_add_relu(a, b, out): np.add(a, b, out=out); np.maximum(out, 0, out=out)
def _np_mul_add(a, b, c, out): np.multiply(a, b, out=out); np.add(out, c, out=out)


def ewise_add(a, b, out=None):
    return parallel_ewise(_np_add, a, b, out=out)


def ewise_mul(a, b, out=None):
    return parallel_ewise(_np_mul, a, b, out=out)


def relu_inplace_or_new(a, out=None):
    if out is None and a.flags["C_CONTIGUOUS"] and a.dtype == np.float32:
        np.maximum(a, 0, out=a)
        return a
    return parallel_ewise(lambda x, out: np.maximum(x, 0, out=out), a, out=out)


def fused_bias_add_relu(out, bias):
    """out += bias (row-wise) then relu, in-place. Used by Linear fast path."""
    out += bias  # broadcast C-level, no extra full-size temp
    np.maximum(out, 0, out=out)
    return out
