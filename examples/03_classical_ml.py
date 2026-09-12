"""03 — مقارنة نماذج كلاسيكية (علماء البيانات)."""
import storch.datasets as datasets
import storch.preprocessing as pp
import storch.models as models
import storch.metrics as metrics

X, y = datasets.load_iris()
Xtr, Xte, ytr, yte = pp.train_test_split(X, y, test_size=0.3, seed=0, stratify=y)
Xtr, Xte = pp.StandardScaler().fit_transform(Xtr), pp.StandardScaler().fit(Xtr).transform(Xte)

for name, clf in [
    ("LogisticRegression", models.LogisticRegression(epochs=300)),
    ("KNN(5)", models.KNNClassifier(5)),
    ("GaussianNB", models.GaussianNB()),
]:
    clf.fit(Xtr, ytr)
    p = clf.predict(Xte)
    print(f"{name}: acc={metrics.accuracy(p, yte):.3f} f1-macro={metrics.f1_score(p, yte, average='macro'):.3f}")

reg = models.Ridge(alpha=1.0).fit(*datasets.make_regression(seed=0)[:1], datasets.make_regression(seed=0)[1])
print("Ridge fitted, coef:", reg.coef_.shape)
