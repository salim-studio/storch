"""storch.io: everyday file loading/saving for data analysts — CSV/JSON/Parquet/NPZ.

Every loader returns Tensors or a Table (from storch.db) with no mandatory pandas
dependency. Parquet requires optional pyarrow or pandas.
"""
from __future__ import annotations

import csv
import json
import os

import numpy as _np

from ._core import Tensor
from .db import Table

__all__ = [
    "load_csv", "save_csv", "load_json", "save_json",
    "load_parquet", "save_tensors", "load_tensors",
    "from_pandas", "to_pandas",
]


def _coerce_col(vals):
    # try int -> float -> str
    non_empty = [v for v in vals if v not in (None, "")]
    if not non_empty:
        return _np.array([float("nan")] * len(vals), dtype=_np.float64)
    try:
        return _np.array([int(v) if v not in (None, "") else 0 for v in vals], dtype=_np.int64)
    except Exception:
        pass
    try:
        out = []
        for v in vals:
            if v in (None, ""):
                out.append(float("nan"))
            else:
                out.append(float(v))
        return _np.array(out, dtype=_np.float64)
    except Exception:
        return _np.array(["" if v is None else str(v) for v in vals], dtype=object)


def load_csv(path, delimiter=",", has_header=True, dtype=_np.float32, target=None, features=None):
    """Load a CSV into a Table. With target/features given, return Tensors (X[, y]) directly."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f, delimiter=delimiter)
        rows = list(reader)
    if not rows:
        raise ValueError("empty csv")
    if has_header:
        header, body = rows[0], rows[1:]
    else:
        header = [f"c{i}" for i in range(len(rows[0]))]
        body = rows
    cols = {h: [] for h in header}
    for r in body:
        for h, v in zip(header, r):
            cols[h].append(v)
    data = {h: _coerce_col(v) for h, v in cols.items()}
    tbl = Table(header, data)
    if target is None and features is None:
        return tbl
    feats = features or [c for c in header if c != target]
    return tbl.to_tensors(feats, target, dtype)


def save_csv(path, table_or_tensors, columns=None, delimiter=","):
    if isinstance(table_or_tensors, Table):
        tbl = table_or_tensors
        cols = tbl.columns
        n = len(tbl)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter=delimiter)
            w.writerow(cols)
            for i in range(n):
                w.writerow([_np.asanyarray(tbl.data[c])[i].tolist()
                            if hasattr(_np.asanyarray(tbl.data[c])[i], "tolist")
                            else _np.asanyarray(tbl.data[c])[i] for c in cols])
        return
    ts = table_or_tensors if isinstance(table_or_tensors, (list, tuple)) else [table_or_tensors]
    mats = [_np.asanyarray(t._data if isinstance(t, Tensor) else t) for t in ts]
    mat = _np.concatenate([m.reshape(len(m), -1) for m in mats], axis=1)
    columns = columns or [f"c{i}" for i in range(mat.shape[1])]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=delimiter)
        w.writerow(columns)
        w.writerows(mat.tolist())


def load_json(path, dtype=_np.float32, target=None, features=None):
    with open(path, encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict):
        cols = list(obj.keys())
        n = len(obj[cols[0]])
        data = {c: _np.asanyarray(obj[c]) for c in cols}
        tbl = Table(cols, data)
    elif isinstance(obj, list) and obj and isinstance(obj[0], dict):
        cols = list(obj[0].keys())
        data = {c: _np.asanyarray([r.get(c) for r in obj]) for c in cols}
        # coerce numerics
        for c in cols:
            a = data[c]
            if a.dtype == object:
                try:
                    data[c] = _np.array([float("nan") if v is None else float(v) for v in a.tolist()])
                except Exception:
                    pass
        tbl = Table(cols, data)
    else:
        raise ValueError("unsupported json structure (expect dict-of-lists or list-of-dicts)")
    if target is None and features is None:
        return tbl
    feats = features or [c for c in tbl.columns if c != target]
    return tbl.to_tensors(feats, target, dtype)


def save_json(path, table_or_dict):
    if isinstance(table_or_dict, Table):
        obj = {c: _np.asanyarray(table_or_dict.data[c]).tolist() for c in table_or_dict.columns}
    elif isinstance(table_or_dict, dict):
        obj = {k: (_np.asanyarray(v).tolist() if isinstance(v, (Tensor, _np.ndarray)) else v)
               for k, v in table_or_dict.items()}
    else:
        raise TypeError("expect Table or dict")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_parquet(path, dtype=_np.float32, target=None, features=None):
    """Load a Parquet file (requires pyarrow or pandas)."""
    tbl = None
    try:
        import pyarrow.parquet as pq
        t = pq.read_table(path)
        cols = t.column_names
        data = {c: _np.asanyarray(t.column(c).to_pylist()) for c in cols}
        # coerce
        for c in cols:
            a = data[c]
            if a.dtype == object:
                try:
                    data[c] = _np.array([float("nan") if v is None else float(v) for v in a.tolist()])
                except Exception:
                    pass
        tbl = Table(cols, data)
    except ImportError:
        try:
            import pandas as pd
            df = pd.read_parquet(path)
            tbl = Table.from_pandas(df)
        except ImportError as e:
            raise ImportError("load_parquet needs pyarrow or pandas. pip install pyarrow") from e
    if target is None and features is None:
        return tbl
    feats = features or [c for c in tbl.columns if c != target]
    return tbl.to_tensors(feats, target, dtype)


def save_tensors(path, tensors: dict):
    arrs = {k: (v._data if isinstance(v, Tensor) else _np.asanyarray(v)) for k, v in tensors.items()}
    _np.savez_compressed(path, **arrs)


def load_tensors(path, dtype=None):
    z = _np.load(path, allow_pickle=True)
    out = {}
    for k in z.files:
        a = _np.ascontiguousarray(z[k])
        if dtype is not None:
            a = a.astype(dtype, copy=False)
        out[k] = Tensor(a)
    return out


def from_pandas(df, target=None, dtype=_np.float32):
    tbl = Table.from_pandas(df)
    if target is None:
        return tbl.to_tensors(tbl.columns, None, dtype)
    feats = [c for c in tbl.columns if c != target]
    return tbl.to_tensors(feats, target, dtype)


def to_pandas(table_or_tensor, columns=None):
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError("pandas is required. pip install pandas") from e
    if isinstance(table_or_tensor, Table):
        return table_or_tensor.to_pandas()
    a = table_or_tensor._data if isinstance(table_or_tensor, Tensor) else _np.asanyarray(table_or_tensor)
    if a.ndim == 1:
        return pd.DataFrame({(columns or ["value"])[0]: a})
    columns = columns or [f"c{i}" for i in range(a.shape[1])]
    return pd.DataFrame(a, columns=columns)
