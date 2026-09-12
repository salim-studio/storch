<p align="center">
  <img src="assets/banner.svg" alt="storch — from database to deep learning" width="100%"/>
</p>

<p align="center">
  <a href="https://github.com/salim-studio/storch/actions/workflows/ci.yml"><img src="https://github.com/salim-studio/storch/actions/workflows/ci.yml/badge.svg" alt="CI"/></a>
  <img src="https://img.shields.io/badge/version-0.2.0-orange" alt="version"/>
  <img src="https://img.shields.io/badge/python-%3E%3D3.9-blue" alt="python"/>
  <img src="https://img.shields.io/badge/license-MIT-green" alt="license"/>
  <a href="https://github.com/salim-studio/storch/stargazers"><img src="https://img.shields.io/github/stars/salim-studio/storch?style=social" alt="stars"/></a>
</p>

# storch — from database to deep learning

> **PyTorch-compatible API + data-science toolkit. Fast on CPU. NumPy-only core.**
> One library covering the whole journey: **SQL/CSV → analysis → classical ML → deep learning**.

*(Previously named `etorch` — `import etorch` still works via a deprecation shim, but please use `import storch`.)*

```bash
pip install storch            # core (numpy only)
pip install "storch[io]"      # + pandas/pyarrow for Parquet
pip install "storch[all]"     # everything
```

## Why storch?

| Audience | What you get |
|---|---|
| Developers | PyTorch-like API (`Tensor`, `nn`, `optim`, `DataLoader`) + `save`/`load`, `no_grad`, `einsum`, `linalg`, `topk`, full autograd-capable indexing |
| Data analysts | `storch.db` (SQL/SQLite → table/tensor in 2 lines) + `storch.io` (CSV/JSON/Parquet) + `Table.describe/head/filter/fillna` |
| Data scientists | `preprocessing` (scalers/encoders/imputers/stratified split) + `metrics` (accuracy/F1/confusion/R²/silhouette…) + `datasets` + sklearn-style `models` |
| ML/DL engineers | `Linear`, `Conv1d/Conv2d`, pooling, `BatchNorm`, `RNN/LSTM/GRU`, `MultiheadAttention`, `Transformer`, `AdamW/SGD` with fused CPU kernels |

## 1) Database → trained model (data engineers)

```python
import storch.db as db
import storch.preprocessing as pp
import storch.models as models, storch.metrics as metrics

con = db.connect("shop.db")
tbl = db.read_sql(con, "SELECT age, income, churn FROM customers WHERE age > 18")
print(tbl.describe()["income"], tbl.head(3))

X, y = tbl.to_tensors(["age", "income"], "churn")   # Tensors, directly
ds = tbl.to_dataset(["age", "income"], "churn")     # or a Dataset for DataLoader
Xtr, Xte, ytr, yte = pp.train_test_split(X, y, test_size=0.2, seed=0, stratify=y)
print(metrics.accuracy(models.KNNClassifier(5).fit(Xtr, ytr).predict(Xte), yte))
```

## 2) File analysis (analysts)

```python
import storch.io as sio
tbl = sio.load_csv("sales.csv")          # or load_json / load_parquet
tbl = tbl.fillna(0.0).filter("year", ">=", 2020)
X = tbl.to_numpy(["price", "qty"])
```

## 3) Classical ML (data scientists)

```python
from storch import datasets, preprocessing as pp, models, metrics
X, y = datasets.load_iris()
Xtr, Xte, ytr, yte = pp.train_test_split(X, y, test_size=0.3, seed=0, stratify=y)
Xtr = pp.StandardScaler().fit_transform(Xtr)
clf = models.LogisticRegression(epochs=300).fit(Xtr, ytr)
print(metrics.classification_report(clf.predict(Xte), yte))
# LinearRegression / Ridge / KNN / KMeans / GaussianNB / PCA — same style
```

## 4) Deep learning, PyTorch style

```python
import storch, storch.nn as nn, storch.optim as optim
storch.manual_seed(0)
model = nn.Sequential(nn.Linear(8, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Linear(64, 2))
opt = optim.AdamW(model.parameters(), lr=1e-2)
loss_fn = nn.CrossEntropyLoss()
X, y = storch.randn(256, 8), storch.randint(0, 2, size=256)
dl = storch.DataLoader(storch.TensorDataset(X, y), batch_size=32, shuffle=True)
for xb, yb in dl:
    opt.zero_grad(); loss_fn(model(xb), yb).backward(); opt.step()

cnn = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Flatten(), nn.Linear(8*14*14, 10))
lstm = nn.LSTM(16, 32)                        # RNN / GRU too
trans = nn.TransformerEncoderLayer(16, 4)     # attention + transformer
```

## Module map

```
storch/
  _core.py         Tensor + autograd + creation/math ops
  _ops.py          where/masked_fill/gather/topk/sort/einsum/norm/...
  linalg.py        inv/det/solve/svd/qr/eigh/pinv
  nn.py            Linear/Conv1d-2d/Pool/BatchNorm/RNN/LSTM/GRU/Attention/Transformer + losses
  functional.py    activations/losses/norms/pooling/attention (all with autograd)
  optim.py         SGD/Adam/AdamW (in-place, fused)
  db.py            SQLite/SQL -> Table/Tensor/Dataset + DatabaseDataset
  io.py            CSV/JSON/Parquet/NPZ + optional pandas interop
  preprocessing.py scalers/encoders/imputer/polynomial/train_test_split
  metrics.py       accuracy/F1/confusion/RMSE/R²/MAPE/ROC-AUC/silhouette
  models.py        LinearRegression/Ridge/Logistic/KNN/KMeans/NB/PCA (sklearn style)
  datasets.py      make_* generators / load_iris / load_wine / CSVDataset
  utils/data.py    Dataset/TensorDataset/DataLoader/Subset/ConcatDataset/random_split
```

## PyTorch compatibility

| torch | storch |
|---|---|
| `tensor/zeros/ones/randn/randint/cat/stack/...` | ✔ same names |
| `x[i]`, `x[:, 1:]`, `x[mask]`, `x[idx] = v` | ✔ autograd-capable indexing |
| `where/masked_fill/gather/topk/sort/einsum/norm` | ✔ |
| `torch.linalg.*` | ✔ `storch.linalg` |
| `nn.{Linear,Conv1d,Conv2d,BatchNorm1d/2d,MaxPool2d,AvgPool2d,AdaptiveAvgPool2d,RNN,LSTM,GRU,MultiheadAttention,TransformerEncoderLayer,...}` | ✔ |
| `F.{relu,gelu,silu,elu,softplus,mish,softmax,batch_norm,scaled_dot_product_attention,...}` | ✔ |
| `optim.{SGD,Adam,AdamW}` | ✔ |
| `utils.data.{Dataset,TensorDataset,DataLoader,Subset,random_split}` | ✔ |
| `save/load/state_dict` | ✔ (npz) |

## Why fast on CPU

1. `float32 + C-contiguous` by default (best BLAS/SIMD use).
2. Multithreaded element-wise ops above ~50k elements.
3. Fused `Linear` (matmul + in-place bias), in-place SGD/Adam steps.
4. Iterative autograd + zero graph nodes under `no_grad`.
5. `Conv2d` via im2col + BLAS.

## Run it

```bash
pip install -e ".[dev]"
pytest tests/ -v
python examples/01_db_to_ml.py
python examples/04_deep_learning.py
python benchmarks/bench_storch.py
```

More walkthroughs in [`examples/`](examples/): `01_db_to_ml.py` · `02_data_analysis.py` · `03_classical_ml.py` · `04_deep_learning.py`

## Roadmap

- [ ] `storch.viz` (lightweight training curves), real `datasets` (MNIST/CSV URLs)
- [ ] `nn.ConvTranspose2d` + `GroupNorm` + `Dropout2d`
- [ ] LR schedulers (`StepLR`, `CosineAnnealing`)
- [ ] Optional zero-copy `torch.Tensor` interop when torch is installed

## Contributing & license

PRs welcome — please add a test in `tests/` for every feature. See [CONTRIBUTING.md](CONTRIBUTING.md). MIT licensed.
