import numpy as np
from backend import CUDABackend
from ml.preprocessing import SimpleImputer, StandardScaler, OneHotEncoder, ColumnTransformer, Pipeline, make_column_selector

be = CUDABackend()

import numpy as np

def test_gpu_pipeline_smoke():
    try:
        import cupy as cp
        from backend import CUDABackend
    except Exception:
        print("CuPy/CUDA not available; skipping.")
        return

    from ml.preprocessing import SimpleImputer, StandardScaler, OneHotEncoder, ColumnTransformer, Pipeline, make_column_selector
    from ml.linear_model import LogisticRegression

    be = CUDABackend()

    X = np.array([
        [25.0, 3000.0, "red"],
        [31.0, 4700.0, "blue"],
        [38.0, 4300.0, "green"],
        [27.0, 5100.0, "red"],
    ], dtype=object)
    y = np.array([0,1,1,0])

    num_sel = make_column_selector(dtype_include=np.number)
    cat_sel = make_column_selector(dtype_include=np.object_)

    num = Pipeline([
        ("imp", SimpleImputer(strategy="mean", backend=be)),
        ("sc",  StandardScaler(backend=be)),
    ], backend=be, device_policy="device")

    cat = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent", backend=be)),
        ("oh",  OneHotEncoder(device_output="device", backend=be)),
    ], backend=be, device_policy="device")

    pre = ColumnTransformer([
        ("num", num, num_sel),
        ("cat", cat, cat_sel),
    ], backend=be, device_policy="device")

    # 전처리만 먼저 확인 (GPU 상주?)
    Z = pre.fit_transform(X, y)
    assert isinstance(Z, cp.ndarray)
    assert Z.shape[0] == 4

    # 파이프라인 + 로지스틱
    clf = Pipeline([
        ("pre", pre),
        ("est", LogisticRegression(lr=0.2, max_iter=400, backend=be)),
    ], backend=be, device_policy="auto")

    clf.fit(X, y)
    P = clf.predict_proba(X)
    assert isinstance(P, np.ndarray)  # predict_proba는 host 반환 설계
    assert P.shape == (4, 2)

if __name__ == "__main__":
    test_gpu_pipeline_smoke()
    print("CPU linear models OK")
