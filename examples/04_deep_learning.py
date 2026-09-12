"""04 — تعلم عميق (MLP + CNN + LSTM + Transformer) بأسلوب PyTorch."""
import storch
import storch.nn as nn
import storch.optim as optim

storch.manual_seed(0)

# --- MLP ثنائي التصنيف ---
X = storch.randn(256, 8)
y = storch.Tensor((X.numpy()[:, 0] + X.numpy()[:, 1] > 0).astype("int64"))
ds = storch.TensorDataset(X, y)
tr, te = storch.random_split(ds, [200, 56], seed=0)
dl = storch.DataLoader(tr, batch_size=32, shuffle=True)

model = nn.Sequential(nn.Linear(8, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Linear(64, 2))
opt = optim.AdamW(model.parameters(), lr=1e-2)
loss_fn = nn.CrossEntropyLoss()
for epoch in range(5):
    for xb, yb in dl:
        opt.zero_grad()
        loss_fn(model(xb), yb).backward()
        opt.step()

# --- CNN على صور وهمية ---
cnn = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Flatten(), nn.Linear(8 * 14 * 14, 10))
with storch.no_grad():
    print("cnn out:", cnn(storch.randn(4, 1, 28, 28)).shape)

# --- LSTM + Transformer على تسلسلات ---
seq = storch.randn(4, 10, 16)
lstm = nn.LSTM(16, 32)
print("lstm out:", lstm(seq)[0].shape)
trans = nn.TransformerEncoderLayer(16, 4)
print("transformer out:", trans(seq).shape)
print("OK - deep learning blocks run")
