"""storch.db: database tasks — SQLite/SQL straight into Tensors/Datasets.

Analysts and ML engineers read from a database in two lines::

    import storch.db as edb
    tbl = edb.read_sql("data.db", "SELECT age, income, churn FROM users WHERE age > 18")
    X, y = tbl.to_tensors(features=["age", "income"], target="churn")
    ds = tbl.to_dataset(features=["age", "income"], target="churn")

No pandas required (optional, for conversion only). Built on stdlib sqlite3 + numpy.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as _np

from ._core import Tensor

__all__ = [
    "Table", "connect", "read_sql", "execute", "executemany",
    "create_table", "insert_rows", "insert_tensors", "DatabaseDataset",
]


@dataclass
class Table:
    """A lightweight numpy-backed table (mini-DataFrame) for analysis and training."""
    columns: list
    data: dict = field(default_factory=dict)  # col -> np.ndarray (1-D, shape (N,))

    def __post_init__(self):
        self.columns = list(self.columns)

    def __len__(self):
        if not self.columns:
            return 0
        return len(self.data[self.columns[0]])

    @property
    def shape(self):
        return (len(self), len(self.columns))

    def head(self, n=5):
        n = min(n, len(self))
        return {c: self.data[c][:n].tolist() for c in self.columns}

    def select(self, cols):
        return Table(list(cols), {c: _np.asanyarray(self.data[c]) for c in cols})

    def drop(self, cols):
        keep = [c for c in self.columns if c not in set(cols)]
        return self.select(keep)

    def filter(self, col, op, value):
        d = _np.asanyarray(self.data[col])
        if op in ("==", "="):
            m = d == value
        elif op == "!=":
            m = d != value
        elif op == ">":
            m = d > value
        elif op == ">=":
            m = d >= value
        elif op == "<":
            m = d < value
        elif op == "<=":
            m = d <= value
        else:
            raise ValueError(f"unknown op {op}")
        return Table(self.columns, {c: _np.asanyarray(self.data[c])[m] for c in self.columns})

    def fillna(self, value=0.0):
        out = {}
        for c in self.columns:
            a = _np.asanyarray(self.data[c])
            if a.dtype.kind == "f":
                a = _np.where(_np.isnan(a), value, a)
            elif a.dtype.kind in "iu":
                pass
            else:  # object: replace None
                a = _np.array([value if v is None else v for v in a.tolist()], dtype=object)
            out[c] = a
        return Table(self.columns, out)

    def describe(self):
        info = {}
        for c in self.columns:
            a = _np.asanyarray(self.data[c])
            try:
                af = a.astype(float)
                mask = ~_np.isnan(af) if af.dtype.kind == "f" else _np.ones(len(af), bool)
                v = af[mask] if mask.any() else af[:0]
                info[c] = {
                    "count": int(mask.sum()), "mean": float(v.mean()) if len(v) else None,
                    "std": float(v.std()) if len(v) else None,
                    "min": float(v.min()) if len(v) else None,
                    "max": float(v.max()) if len(v) else None,
                }
            except Exception:
                vals, counts = _np.unique(a, return_counts=True)
                info[c] = {"count": len(a), "unique": len(vals),
                           "top": str(vals[_np.argmax(counts)]) if len(vals) else None}
        return info

    def to_numpy(self, cols=None, dtype=_np.float32):
        cols = cols or self.columns
        arrs = []
        for c in cols:
            a = _np.asanyarray(self.data[c])
            if a.dtype.kind in "fiu":
                arrs.append(a.astype(dtype, copy=False).reshape(-1, 1) if a.ndim == 1 else a.astype(dtype, copy=False))
            else:
                # non-numeric: factorize
                _, inv = _np.unique(a, return_inverse=True)
                arrs.append(inv.astype(dtype, copy=False).reshape(-1, 1))
        return _np.concatenate(arrs, axis=1) if arrs else _np.zeros((len(self), 0), dtype)

    def to_tensors(self, features, target=None, dtype=_np.float32):
        X = Tensor(self.to_numpy(features, dtype=dtype))
        if target is None:
            return X
        t = _np.asanyarray(self.data[target])
        if t.dtype.kind == "f":
            y = Tensor(t.astype(dtype, copy=False))
        elif t.dtype.kind in "iu":
            y = Tensor(t.astype(_np.int64, copy=False))
        else:
            _, inv = _np.unique(t, return_inverse=True)
            y = Tensor(inv.astype(_np.int64, copy=False))
        return X, y

    def to_dataset(self, features, target=None, dtype=_np.float32):
        from .utils.data import TensorDataset
        out = self.to_tensors(features, target, dtype)
        if target is None:
            return TensorDataset(out)
        return TensorDataset(*out)

    def to_pandas(self):
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError("pandas is required for to_pandas(). pip install pandas") from e
        return pd.DataFrame({c: _np.asanyarray(self.data[c]) for c in self.columns})

    @staticmethod
    def from_pandas(df):
        return Table(list(df.columns), {c: _np.asanyarray(df[c].to_numpy()) for c in df.columns})

    @staticmethod
    def from_rows(columns, rows):
        cols = list(columns)
        arr = list(rows)
        data = {}
        for j, c in enumerate(cols):
            col = [r[j] for r in arr]
            try:
                data[c] = _np.asanyarray(col)
            except Exception:
                data[c] = _np.array(col, dtype=object)
        # try numeric coercion for object cols holding numbers/None
        for c in cols:
            a = data[c]
            if a.dtype == object:
                try:
                    data[c] = _np.array([float("nan") if v is None else float(v) for v in a.tolist()], dtype=_np.float64)
                except Exception:
                    pass
        return Table(cols, data)


def connect(path=":memory:"):
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    return con


def _fetch_table(cur) -> Table:
    rows = cur.fetchall()
    cols = [d[0] for d in cur.description] if cur.description else []
    return Table.from_rows(cols, [tuple(r) for r in rows])


def read_sql(path_or_con, sql, params=()):
    """Run a SQL query and return a Table. Accepts a file path or an sqlite3 connection."""
    close = False
    if isinstance(path_or_con, (str, bytes)):
        con = connect(str(path_or_con))
        close = True
    else:
        con = path_or_con
    try:
        cur = con.execute(sql, params)
        return _fetch_table(cur)
    finally:
        if close:
            con.close()


def execute(path_or_con, sql, params=(), commit=True):
    close = False
    if isinstance(path_or_con, (str, bytes)):
        con = connect(str(path_or_con))
        close = True
    else:
        con = path_or_con
    try:
        cur = con.execute(sql, params)
        if commit:
            con.commit()
        return cur
    finally:
        if close:
            con.close()


def executemany(path_or_con, sql, seq, commit=True):
    close = False
    if isinstance(path_or_con, (str, bytes)):
        con = connect(str(path_or_con))
        close = True
    else:
        con = path_or_con
    try:
        con.executemany(sql, seq)
        if commit:
            con.commit()
    finally:
        if close:
            con.close()


def create_table(path_or_con, name, schema: dict, drop_if_exists=False):
    """schema: {col: sqltype} e.g. {"age": "REAL", "label": "INTEGER"}"""
    cols = ", ".join(f'"{k}" {v}' for k, v in schema.items())
    if drop_if_exists:
        execute(path_or_con, f'DROP TABLE IF EXISTS "{name}"')
    execute(path_or_con, f'CREATE TABLE IF NOT EXISTS "{name}" ({cols})')


def insert_rows(path_or_con, name, columns: Sequence[str], rows: Sequence[tuple]):
    ph = ", ".join(["?"] * len(columns))
    cols = ", ".join(f'"{c}"' for c in columns)
    executemany(path_or_con, f'INSERT INTO "{name}" ({cols}) VALUES ({ph})', rows)


def insert_tensors(path_or_con, name, columns: Sequence[str], *tensors: Tensor):
    arrs = [t._data.reshape(len(t._data), -1) if t.ndim > 1 else t._data.reshape(-1, 1)
            if isinstance(t, Tensor) else _np.asanyarray(t).reshape(-1, 1) for t in tensors]
    mat = _np.concatenate(arrs, axis=1) if len(arrs) > 1 else arrs[0]
    # flatten multi-dim beyond 1 col per tensor is not supported for DB insert; require 1-D tensors
    rows = [tuple(float(v) if isinstance(v, _np.floating) else int(v) if isinstance(v, _np.integer) else v
                  for v in row) for row in mat.tolist()]
    insert_rows(path_or_con, name, columns, rows)


class DatabaseDataset:
    """A Dataset that reads directly from a SQL query (eager) — DataLoader compatible."""

    def __init__(self, path_or_con, sql, features, target=None, params=(), dtype=_np.float32):
        self.table = read_sql(path_or_con, sql, params)
        self.features = list(features)
        self.target = target
        self.tensors = self.table.to_tensors(features, target, dtype)

    def __len__(self):
        return len(self.table)

    def __getitem__(self, i):
        if self.target is None:
            X = self.tensors
            return (Tensor(X._data[i]),)
        X, y = self.tensors
        return Tensor(X._data[i]), Tensor(_np.asanyarray(y._data[i]))
