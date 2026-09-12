# CHANGELOG — storch

## 0.2.0 (2026-09-12)
### Renamed: `etorch` → `storch`
- The package, imports, repo and docs are now **storch** (`pip install storch`).
- `import etorch` keeps working through a `DeprecationWarning` shim (removed in a future release).
- New home: https://github.com/salim-studio/storch — README rewritten in English, new logo/banner in `assets/`.

### Added — the road to fame: DB → Data → ML → DL
- **Core**: full indexing `x[i]/x[mask]` with autograd + `__setitem__/copy_/fill_/zero_` + comparisons + `flatten/expand/topk/sort/std/var/norm/cumsum/flip/tile`.
- **`storch._ops`**: `where/masked_fill/gather/index_select/argmax/topk/sort/unique/nonzero/split/chunk/einsum/norm/...`.
- **`storch.linalg`**: `norm/inv/det/solve/svd/qr/eigh/matrix_rank/pinv`.
- **`storch.db`** (new): `Table` (select/filter/fillna/describe/to_tensors/to_dataset) + `read_sql/execute/create_table/insert_*` + `DatabaseDataset`.
- **`storch.io`** (new): `load/save_csv/json`, `load_parquet` (optional), `save/load_tensors`, `from/to_pandas`.
- **`storch.preprocessing`** (new): `Standard/MinMax/MaxAbs/Robust` scalers + `OneHot/Label` encoders + `SimpleImputer` + stratified `train_test_split` + `shuffle`.
- **`storch.metrics`** (new): `accuracy/top_k/f1/confusion/report/mse/rmse/mae/r2/mape/roc_auc/log_loss/silhouette`.
- **`storch.models`** (new): `LinearRegression/Ridge/LogisticRegression/KNN/KMeans/GaussianNB/PCA`, sklearn style.
- **`storch.datasets`** (new): `make_classification/regression/blobs/moons/circles`, `load_iris/load_wine`, `ToyDataset/CSVDataset`.
- **nn/functional**: `BatchNorm1d/2d` (differentiable train+eval), `Conv1d`, `Max/Avg/AdaptiveAvgPool2d`, `Upsample`, `ELU/Softplus/Mish`, `HuberLoss`, `RNN/LSTM/GRU` (+Cells), `MultiheadAttention`, `TransformerEncoderLayer`, `scaled_dot_product_attention`.
- **Data**: `Subset/ConcatDataset/random_split`.
- **Packaging**: extras `[io/viz/dev/all]` + `examples/` + English docs + CI workflow.

### Fixed
- `F.linear` backward for 3D inputs (Batch×Seq×Feat) — used to crash with Transformer/Attention.
- `F.layer_norm` now differentiates w.r.t. weight/bias (numerically verified) — previously input-only.

## 0.1.0 (as etorch)
- Tensor + autograd + nn (Linear/Embedding/LayerNorm/Conv2d/…) + SGD/Adam/AdamW + DataLoader.
