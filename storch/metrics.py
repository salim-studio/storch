"""storch.metrics: evaluation metrics for classification, regression and clustering — numpy/Tensor."""
from __future__ import annotations

import numpy as _np

from ._core import Tensor

__all__ = [
    "accuracy", "top_k_accuracy", "precision", "recall", "f1_score",
    "confusion_matrix", "classification_report",
    "mse", "rmse", "mae", "r2_score", "mape",
    "roc_auc", "log_loss", "silhouette_score",
]


def _arr(x):
    return x._data if isinstance(x, Tensor) else _np.asanyarray(x)


def accuracy(y_pred, y_true):
    yp = _arr(y_pred)
    yt = _arr(y_true).reshape(-1)
    if yp.ndim > 1:
        yp = yp.argmax(axis=1)
    yp = yp.reshape(-1)
    return float((yp == yt).mean()) if len(yt) else 0.0


def top_k_accuracy(logits, y_true, k=5):
    lg = _arr(logits)
    yt = _arr(y_true).reshape(-1)
    k = min(k, lg.shape[1])
    topk = _np.argpartition(lg, -k, axis=1)[:, -k:]
    return float(sum(yt[i] in topk[i] for i in range(len(yt))) / len(yt)) if len(yt) else 0.0


def _prf(y_pred, y_true, average="binary", pos_label=1, zero_division=0.0):
    yp = _arr(y_pred)
    yt = _arr(y_true).reshape(-1)
    if yp.ndim > 1:
        yp = yp.argmax(axis=1)
    yp = yp.reshape(-1)
    classes = _np.unique(_np.concatenate([yt, yp]))
    def _bin(c):
        tp = int(((yp == c) & (yt == c)).sum())
        fp = int(((yp == c) & (yt != c)).sum())
        fn = int(((yp != c) & (yt == c)).sum())
        p = tp / (tp + fp) if (tp + fp) else zero_division
        r = tp / (tp + fn) if (tp + fn) else zero_division
        f = 2 * p * r / (p + r) if (p + r) else zero_division
        return p, r, f, int((yt == c).sum())
    per = {int(c): _bin(c) for c in classes}
    if average is None:
        return per
    if average == "binary":
        c = pos_label if pos_label in per else (classes[1] if len(classes) > 1 else classes[0])
        return per[int(c)]
    vals = _np.array([[v[0], v[1], v[2]] for v in per.values()])
    if average == "macro":
        return tuple(vals.mean(axis=0).tolist())
    if average == "micro":
        tp = sum(((yp == c) & (yt == c)).sum() for c in classes)
        fp = sum(((yp == c) & (yt != c)).sum() for c in classes)
        fn = sum(((yp != c) & (yt == c)).sum() for c in classes)
        p = tp / (tp + fp) if (tp + fp) else zero_division
        r = tp / (tp + fn) if (tp + fn) else zero_division
        f = 2 * p * r / (p + r) if (p + r) else zero_division
        return (float(p), float(r), float(f))
    if average == "weighted":
        tot = len(yt)
        out = [0.0, 0.0, 0.0]
        for c, (p, r, f, s) in per.items():
            w = s / tot
            out[0] += p * w; out[1] += r * w; out[2] += f * w
        return tuple(out)
    raise ValueError(average)


def precision(y_pred, y_true, average="binary", pos_label=1):
    return _prf(y_pred, y_true, average, pos_label)[0]


def recall(y_pred, y_true, average="binary", pos_label=1):
    return _prf(y_pred, y_true, average, pos_label)[1]


def f1_score(y_pred, y_true, average="binary", pos_label=1):
    return _prf(y_pred, y_true, average, pos_label)[2]


def confusion_matrix(y_pred, y_true, labels=None):
    yp = _arr(y_pred)
    yt = _arr(y_true).reshape(-1)
    if yp.ndim > 1:
        yp = yp.argmax(axis=1)
    yp = yp.reshape(-1)
    labels = _np.unique(_np.concatenate([yt, yp])) if labels is None else _np.asanyarray(labels)
    m = _np.zeros((len(labels), len(labels)), dtype=_np.int64)
    idx = {v: i for i, v in enumerate(labels.tolist())}
    for t, p in zip(yt.tolist(), yp.tolist()):
        m[idx[t], idx[p]] += 1
    return m


def classification_report(y_pred, y_true):
    yp = _arr(y_pred)
    yt = _arr(y_true).reshape(-1)
    if yp.ndim > 1:
        yp = yp.argmax(axis=1)
    per = _prf(yp, yt, average=None)
    lines = ["cls  precision  recall  f1  support"]
    for c, (p, r, f, s) in sorted(per.items()):
        lines.append(f"{c:<4} {p:.3f}      {r:.3f}   {f:.3f}  {s}")
    acc = accuracy(yp, yt)
    macro = _prf(yp, yt, average="macro")
    lines.append(f"accuracy {acc:.3f}")
    lines.append(f"macro-avg {macro[0]:.3f} {macro[1]:.3f} {macro[2]:.3f}")
    return "\n".join(lines)


def mse(y_pred, y_true):
    return float(_np.mean((_arr(y_pred).astype(float) - _arr(y_true).astype(float)) ** 2))


def rmse(y_pred, y_true):
    return float(_np.sqrt(mse(y_pred, y_true)))


def mae(y_pred, y_true):
    return float(_np.mean(_np.abs(_arr(y_pred).astype(float) - _arr(y_true).astype(float))))


def r2_score(y_pred, y_true):
    yt = _arr(y_true).astype(float)
    yp = _arr(y_pred).astype(float)
    ss_res = ((yt - yp) ** 2).sum()
    ss_tot = ((yt - yt.mean()) ** 2).sum()
    return float(1 - ss_res / ss_tot) if ss_tot else 0.0


def mape(y_pred, y_true, eps=1e-8):
    yt = _arr(y_true).astype(float)
    yp = _arr(y_pred).astype(float)
    return float(_np.mean(_np.abs((yt - yp) / _np.maximum(_np.abs(yt), eps))) * 100)


def roc_auc(y_score, y_true, pos_label=1):
    s = _arr(y_score).astype(float).reshape(-1)
    yt = (_arr(y_true).reshape(-1) == pos_label).astype(int)
    order = _np.argsort(s)
    yt = yt[order]
    tp = _np.cumsum(yt[::-1])
    fp = _np.cumsum(1 - yt[::-1])
    P, N = tp[-1], fp[-1]
    if P == 0 or N == 0:
        return 0.5
    tpr = _np.concatenate([[0], tp / P])
    fpr = _np.concatenate([[0], fp / N])
    return float(_np.trapezoid(tpr, fpr))


def log_loss(y_prob, y_true, eps=1e-12):
    p = _np.clip(_arr(y_prob).astype(float), eps, 1 - eps)
    yt = _arr(y_true).reshape(-1)
    if p.ndim > 1:
        return float(-_np.log(p[_np.arange(len(yt)), yt.astype(int)]).mean())
    return float(-(yt * _np.log(p) + (1 - yt) * _np.log(1 - p)).mean())


def silhouette_score(X, labels):
    X = _arr(X).astype(float)
    y = _arr(labels).reshape(-1)
    classes = _np.unique(y)
    if len(classes) < 2 or len(X) < 3:
        return 0.0
    # pairwise distances (ok for moderate N; sample if huge)
    n = len(X)
    if n > 2000:
        idx = _np.random.default_rng(0).choice(n, 2000, replace=False)
        X, y = X[idx], y[idx]
        n = 2000
    from numpy.linalg import norm as _n
    D = _n(X[:, None, :] - X[None, :, :], axis=2)
    sils = []
    for i in range(n):
        same = (y == y[i])
        same[i] = False
        a = D[i][same].mean() if same.any() else 0.0
        b = min(D[i][(y == c)].mean() for c in classes if c != y[i])
        sils.append((b - a) / max(a, b) if max(a, b) > 0 else 0.0)
    return float(_np.mean(sils))
