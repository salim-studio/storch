"""etorch.utils.data: torch.utils.data-compatible Dataset/DataLoader."""
from __future__ import annotations

import math
import numpy as _np
from concurrent.futures import ThreadPoolExecutor

from .._core import Tensor, _as_t

__all__ = ["Dataset", "TensorDataset", "DataLoader"]


class Dataset:
    def __len__(self): raise NotImplementedError
    def __getitem__(self, i): raise NotImplementedError


class TensorDataset(Dataset):
    def __init__(self, *tensors):
        self.tensors = [_as_t(t) for t in tensors]
        n = self.tensors[0].shape[0]
        assert all(t.shape[0] == n for t in self.tensors)

    def __len__(self): return self.tensors[0].shape[0]

    def __getitem__(self, i):
        return tuple(Tensor(t._data[i]) for t in self.tensors)


def _collate(batch):
    # batch: list of tuples
    cols = list(zip(*batch))
    out = []
    for c in cols:
        if isinstance(c[0], Tensor):
            import numpy as np
            out.append(Tensor(np.stack([t._data for t in c])))
        else:
            out.append(c)
    return tuple(out) if len(out) > 1 else out[0]


class DataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0,
                 drop_last=False, collate_fn=None):
        self.dataset, self.batch_size = dataset, batch_size
        self.shuffle, self.num_workers = shuffle, num_workers
        self.drop_last, self.collate_fn = drop_last, collate_fn or _collate
        self._pool = ThreadPoolExecutor(max_workers=num_workers) if num_workers else None

    def __len__(self):
        n = len(self.dataset) // self.batch_size
        if not self.drop_last and len(self.dataset) % self.batch_size: n += 1
        return n

    def __iter__(self):
        idx = _np.arange(len(self.dataset))
        if self.shuffle: _np.random.shuffle(idx)
        batches = [idx[i:i + self.batch_size] for i in range(0, len(idx), self.batch_size)]
        if self.drop_last and batches and len(batches[-1]) < self.batch_size:
            batches.pop()
        if self._pool:
            for b in self._pool.map(lambda bi: self.collate_fn([self.dataset[int(i)] for i in bi]), batches):
                yield b
        else:
            for bi in batches:
                yield self.collate_fn([self.dataset[int(i)] for i in bi])
