import numpy as np
from ml.preprocessing import (
    SimpleImputer, OneHotEncoder, StandardScaler,
    ColumnTransformer, Pipeline, make_column_selector,
)
from ml.model_selection import GridSearchCV, RandomizedSearchCV
from ml.metrics import accuracy_score

# 데모용 작은 로지스틱 (네 진짜 모델로 교체 OK)
class TinyLogReg:
    def __init__(self, lr=0.1, epochs=300):
        self.lr=lr; self.epochs=epochs; self.w=None
    def get_params(self, deep=True): return {"lr": self.lr, "epochs": self.epochs}
    def set_params(self, **p):
        for k,v in p.items(): setattr(self, k, v); return self
    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y, float)
        Xb = np.c_[np.ones(len(X)), X]
        self.w = np.zeros(Xb.shape[1])
        for _ in range(self.epochs):
            p = 1/(1+np.exp(-(Xb@self.w)))
            grad = Xb.T @ (p - y) / len(y)
            self.w -= self.lr*grad
        return self
    def predict_proba(self, X):
        X = np.asarray(X, float); Xb = np.c_[np.ones(len(X)), X]
        p = 1/(1+np.exp(-(Xb@self.w)))
        return np.c_[1-p, p]
    def predict(self, X): return (self.predict_proba(X)[:,1] >= 0.5).astype(int)

# 혼합 데이터
X = np.array([
    [25.0, 3000.0, "red",   "S"],
    [None, 5200.0, "blue",  None],
    [43.0, None,   "red",   "M"],
    [36.0, 4100.0, "green", "L"],
    [29.0, 4800.0, "red",   "M"],
    [33.0, 3900.0, "blue",  "S"],
    [41.0, 4500.0, "green", "L"],
    [27.0, 5100.0, "red",   "S"],
], dtype=object)
y = np.array([0,1,0,1,1,0,1,0])

num_sel = make_column_selector(dtype_include=np.number)
cat_sel = make_column_selector(dtype_include=np.object_)

num_pipe = Pipeline([("imp", SimpleImputer("mean")), ("sc", StandardScaler())])
cat_pipe = Pipeline([("imp", SimpleImputer("most_frequent")), ("oh", OneHotEncoder("ignore"))])

pre = ColumnTransformer([("num", num_pipe, num_sel), ("cat", cat_pipe, cat_sel)], remainder="drop")
clf = Pipeline([("pre", pre), ("est", TinyLogReg())])

# ─ GridSearchCV
param_grid = {
    "est__lr": [0.05, 0.1, 0.2],
    "est__epochs": [200, 400, 800],
    # 전처리 하이퍼파라미터도 함께 탐색 가능
    "pre__num__sc__with_std": [True, False],
}

gs = GridSearchCV(clf, param_grid=param_grid, scoring="accuracy", cv=3, random_state=42, refit=True, return_train_score=True, verbose=1)
gs.fit(X, y)

print("Grid best score:", gs.best_score_)
print("Grid best params:", gs.best_params_)
best_model = gs.best_estimator_

# ─ RandomizedSearchCV
param_dist = {
    "est__lr": [0.01, 0.02, 0.05, 0.1, 0.2],
    "est__epochs": [200, 300, 400, 600, 800, 1200],
    "pre__num__sc__with_std": [True, False],
}
rs = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=6, scoring="roc_auc", cv=3, random_state=0, refit=True, verbose=1)
rs.fit(X, y)
print("Random best score (AUC):", rs.best_score_)
print("Random best params:", rs.best_params_)
