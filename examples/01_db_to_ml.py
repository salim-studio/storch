"""01 — من قاعدة البيانات إلى نموذج ML في دقائق (SQLite -> storch)."""
import storch.db as db
import storch.preprocessing as pp
import storch.metrics as metrics
import storch.models as models

con = db.connect(":memory:")
db.create_table(con, "customers", {"age": "REAL", "income": "REAL", "churn": "INTEGER"})
db.insert_rows(con, "customers", ["age", "income", "churn"], [
    (25, 50.0, 0), (35, 80.0, 1), (45, 120.0, 1), (22, 30.0, 0),
    (52, 95.0, 1), (28, 55.0, 0), (40, 110.0, 1), (31, 70.0, 0),
])

tbl = db.read_sql(con, "SELECT * FROM customers WHERE age > 20")
print("shape:", tbl.shape, "| describe:", tbl.describe()["income"])

X, y = tbl.to_tensors(features=["age", "income"], target="churn")
Xtr, Xte, ytr, yte = pp.train_test_split(X, y, test_size=0.25, seed=0, stratify=y)

sc = pp.StandardScaler().fit(Xtr.numpy())
Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)

clf = models.LogisticRegression(epochs=200).fit(Xtr_s, ytr)
pred = clf.predict(Xte_s)
print("accuracy:", metrics.accuracy(pred, yte))
print(metrics.classification_report(pred, yte))
