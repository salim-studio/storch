"""02 — تحليل بيانات CSV (محللو البيانات): describe/fillna/scale/PCA/KMeans."""
import storch.io as eio
import storch.preprocessing as pp
import storch.models as models
import storch.metrics as metrics
import storch.datasets as datasets

import tempfile, os
tmp = os.path.join(tempfile.gettempdir(), "storch_blobs.csv")
X, y = datasets.make_blobs(n_samples=150, centers=3, seed=0)
eio.save_csv(tmp, [X, y], columns=["x0", "x1", "label"])

tbl = eio.load_csv(tmp)
print("describe:", tbl.describe()["x0"])
print("head:", tbl.head(3))

Xn = pp.StandardScaler().fit_transform(tbl.to_numpy(["x0", "x1"]))
Z = models.PCA(2).fit_transform(Xn)
km = models.KMeans(n_clusters=3, seed=0).fit(Xn)
print("silhouette:", round(metrics.silhouette_score(Xn, km.labels_), 3))
print("pca shape:", Z.shape)
