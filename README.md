# etorch — مطابقة PyTorch، أسرع على CPU

```python
import etorch
import etorch.nn as nn
import etorch.optim as optim

etorch.manual_seed(0)
model = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 2))
opt = optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

X = etorch.randn(64, 4)
y = etorch.randint(0, 2, size=64)
for _ in range(100):
    opt.zero_grad()
    loss = loss_fn(model(X), y)
    loss.backward()
    opt.step()

with etorch.no_grad():
    print(model(X).numpy().argmax(1)[:10])
```

## مطابقة PyTorch

| torch | etorch |
|---|---|
| `torch.tensor`, `zeros/ones/empty/full/randn/rand` | نفس الأسماء |
| `x.backward()`, `.grad`, `.detach()`, `no_grad()` | مدعوم |
| `torch.nn.{Linear,Embedding,LayerNorm,Conv2d,Sequential,MSELoss,CrossEntropyLoss,...}` | مدعوم |
| `torch.nn.functional.{relu,softmax,cross_entropy,...}` | مدعوم |
| `torch.optim.{SGD,Adam,AdamW}` | مدعوم |
| `torch.utils.data.{Dataset,TensorDataset,DataLoader}` | مدعوم |
| `torch.save/load`, `state_dict()` | مدعوم (npz) |

## لماذا أسرع؟

1. **float32 + C-contiguous** افتراضياً.
2. **element-wise متوازي** (ThreadPool) فوق ~50k عنصر.
3. **fused**: `Linear` (matmul + in-place bias)، `bias+ReLU`، خطوات SGD/Adam in-place بدون نسخ.
4. **autograd تكراري** + لا عقد رسم تحت `no_grad`.
5. **Conv2d** عبر im2col + BLAS بدل حلقات بايثون.

## تشغيل

```
pip install -e .
pytest tests/ -v
python benchmarks/bench_etorch.py
```
