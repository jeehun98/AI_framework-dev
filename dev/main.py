import numpy as np
from ml.preprocessing import (
    SimpleImputer, OneHotEncoder, StandardScaler,
    ColumnTransformer, Pipeline, make_column_selector,
)

# 혼합 데이터
X = np.array([
    [25.0, 3000.0, "red",   "S"],
    [None, 5200.0, "blue",  None],
    [43.0, None,   "red",   "M"],
    [36.0, 4100.0, "green", "L"],
], dtype=object)
y = np.array([0, 1, 0, 1])

num_sel = make_column_selector(dtype_include=np.number)
cat_sel = make_column_selector(dtype_include=np.object_)

num_pipe = Pipeline([
    ("imp", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler()),
])
cat_pipe = Pipeline([
    ("imp", SimpleImputer(strategy="most_frequent")),
    ("oh", OneHotEncoder(handle_unknown="ignore")),
])

pre = ColumnTransformer([
    ("num", num_pipe, num_sel),
    ("cat", cat_pipe, cat_sel),
], remainder="drop")

# 데모용 분류기(너의 로지스틱/트리로 교체 권장)
class DummyClf:
    def fit(self, X, y): self.mean_=X.mean(0); return self
    def predict(self, X): return (X@np.sign(self.mean_)>0).astype(int)

clf = Pipeline([
    ("pre", pre),
    ("est", DummyClf()),
])

clf.fit(X, y)
print("Pipeline OK. Transformed shape:", pre.fit_transform(X).shape)
print("Predict:", clf.predict(X))
