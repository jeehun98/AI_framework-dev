import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np

from backend import CPUBackend
from ml.preprocessing import (
    SimpleImputer, OneHotEncoder, StandardScaler,
    ColumnTransformer, Pipeline, make_column_selector,
)
from ml.linear_model import LogisticRegression, Ridge
from ml.model_selection import cross_val_score, GridSearchCV, train_test_split
from ml.metrics import accuracy_score, r2_score


def test_classification_pipeline_cpu():
    be = CPUBackend()

    # 혼합 데이터셋 (숫자+문자열)
    X = np.array([
        [25.0, 3000.0, "red",   "S"],
        [None, 5200.0, "blue",  None],
        [43.0, None,   "red",   "M"],
        [36.0, 4100.0, "green", "L"],
        [29.0, 4800.0, "red",   "M"],
        [33.0, 3900.0, "blue",  "S"],
        [41.0, 4500.0, "green", "L"],
        [27.0, 5100.0, "red",   "S"],
        [31.0, 4700.0, "blue",  "M"],
        [38.0, 4300.0, "green", "L"],
    ], dtype=object)
    y = np.array([0,1,0,1,1,0,1,0,1,0])

    num_sel = make_column_selector(dtype_include=np.number)
    cat_sel = make_column_selector(dtype_include=np.object_)

    num_pipe = Pipeline([
        ("imp", SimpleImputer("mean")),
        ("sc", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imp", SimpleImputer("most_frequent")),
        ("oh", OneHotEncoder("ignore")),
    ])

    pre = ColumnTransformer([
        ("num", num_pipe, num_sel),
        ("cat", cat_pipe, cat_sel),
    ])

    clf = Pipeline([
        ("pre", pre),
        ("est", LogisticRegression(lr=0.2, max_iter=800, backend=be)),
    ])

    # 1) 교차검증
    scores = cross_val_score(clf, X, y, cv=3, scoring="accuracy", random_state=42)
    print("CV accuracy:", scores, "mean=", scores.mean())
    assert scores.size == 3

    # 2) 그리드서치 (하이퍼파라미터 몇 개만)
    param_grid = {
        "est__C": [0.5, 1.0, 2.0],
        "est__lr": [0.05, 0.1, 0.2],
        "pre__num__sc__with_std": [True, False],
    }
    gs = GridSearchCV(clf, param_grid=param_grid, scoring="accuracy", cv=3, random_state=0, refit=True, return_train_score=True)
    gs.fit(X, y)
    print("Grid best score:", gs.best_score_)
    print("Grid best params:", gs.best_params_)
    assert isinstance(gs.best_score_, float)

    # 3) 홀드아웃 평가
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    best_model = gs.best_estimator_
    y_pred = best_model.predict(X_te)
    print("Holdout acc:", accuracy_score(y_te, y_pred))
    assert y_pred.shape == y_te.shape


def test_regression_pipeline_cpu():
    be = CPUBackend()

    # 단순 회귀 예시
    X = np.array([[1.,2.],[2.,0.],[3.,1.],[4.,3.],[5.,4.],[6.,5.]])
    y = np.array([3.2, 2.1, 3.0, 4.9, 6.2, 7.0])

    model = Ridge(alpha=1.0, backend=be)
    model.fit(X, y)
    yhat = model.predict(X)
    print("Ridge R2:", r2_score(y, yhat))
    assert yhat.shape == y.shape


# (선택) CuPy가 있다면 GPU 파이프라인도 스모크 테스트
def test_classification_pipeline_gpu_optional():
    try:
        import cupy as cp
        from backend import CUDABackend
    except Exception:
        # CuPy/GPU가 없으면 스킵
        print("CuPy not available; skipping GPU test.")
        return

    be = CUDABackend()
    X = np.array([
        [25.0, 3000.0, "red",   "S"],
        [31.0, 4700.0, "blue",  "M"],
        [38.0, 4300.0, "green", "L"],
        [27.0, 5100.0, "red",   "S"],
    ], dtype=object)
    y = np.array([0,1,1,0])

    # 간단히 숫자만 써서 GPU 경로 확인
    Xn = np.array([[25.,3000.],[31.,4700.],[38.,4300.],[27.,5100.]])
    clf = LogisticRegression(lr=0.2, max_iter=300, backend=be)
    clf.fit(be.to_device(Xn), be.to_device(y))
    proba = clf.predict_proba(be.to_device(Xn))
    print("GPU proba shape:", proba.shape)
    assert proba.shape == (4, 2)


if __name__ == "__main__":
    test_classification_pipeline_cpu()
    test_regression_pipeline_cpu()
    test_classification_pipeline_gpu_optional()
    print("End-to-end pipeline OK")
