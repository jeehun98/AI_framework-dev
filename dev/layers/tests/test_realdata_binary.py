# filename: test_realdata_binary_and_xor.py
import os
import sys
import csv
import math
import urllib.request
import numpy as np

# ===== CUDA DLL 경로 (Windows, 필요 시 조정) =====
try:
    os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin")
except Exception:
    pass

# ===== 프로젝트/바인딩 모듈 경로 =====
# graph_executor .pyd 경로 (프로젝트 구조에 맞춰 조정)
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend", "graph_executor", "build", "lib.win-amd64-cpython-312"))
# 프로젝트 루트
sys.path.insert(0, os.path.abspath("C:/Users/owner/Desktop/AI_framework-dev"))

# ===== AI Framework =====
from dev.models.sequential import Sequential
from dev.layers.dense import Dense
from dev.layers.activation_layer import Activation
from dev.layers.flatten import Flatten

# (선택) 그래프 바인딩 점검 도우미가 있으면 호출
def _try_debug_graph(model):
    try:
        import graph_executor as ge  # noqa
        uses_param = {ge.OpType.MATMUL, ge.OpType.ADD, ge.OpType.CONV2D}
        graph_param_ids = sorted({op.param_id for op in model.E if (op.op_type in uses_param) and getattr(op, "param_id", "")})
        weight_keys = sorted(getattr(model, "weights", {}).keys())
        bias_keys   = sorted(getattr(model, "biases", {}).keys())
        shape_keys  = sorted(getattr(model, "shapes", {}).keys())
        tensor_keys = sorted(set(weight_keys) | set(bias_keys))

        print("\n==== Graph/Param wiring check ====")
        print("Graph param_ids:", graph_param_ids)
        print("weights keys:", weight_keys)
        print("biases  keys:", bias_keys)
        print("shapes  keys (first 20):", shape_keys[:20])
        print("tensor keys:", tensor_keys)

        miss_tensors = [k for k in graph_param_ids if k not in tensor_keys]
        miss_shapes  = [k for k in graph_param_ids if k not in shape_keys]
        if miss_tensors: print("⚠️  MISSING in tensors:", miss_tensors)
        if miss_shapes:  print("⚠️  MISSING in shapes:", miss_shapes)
        if not graph_param_ids:
            print("❌ Graph has NO trainable params (no MATMUL/ADD/CONV2D with param_id).")
        elif not tensor_keys:
            print("❌ Model has NO param buffers in weights/biases.")
        elif not miss_tensors and not miss_shapes:
            print("✅ Wiring looks good.")
    except Exception as e:
        print(f"(debug_graph_bindings skip) {e}")

# ---------------------------------------------
# 실데이터: UCI Breast Cancer Wisconsin (Diagnostic)
# ---------------------------------------------
BC_WDBC_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
BC_LOCAL_FILENAME = "wdbc.data"  # 같은 폴더에 두면 우선 사용

def _ensure_wdbc_csv(path_hint=None):
    """
    wdbc.data 를 준비한다.
    1) path_hint 또는 로컬 디렉터리에서 찾기
    2) 없으면 UCI에서 다운로드
    """
    # 1) 명시 경로
    if path_hint and os.path.isfile(path_hint):
        return path_hint

    # 2) 현재 파일 기준 같은 폴더
    here = os.path.dirname(os.path.abspath(__file__))
    local_path = os.path.join(here, BC_LOCAL_FILENAME)
    if os.path.isfile(local_path):
        return local_path

    # 3) 다운로드 시도
    try:
        print(f"Downloading WDBC from UCI → {local_path}")
        urllib.request.urlretrieve(BC_WDBC_URL, local_path)
        return local_path
    except Exception as e:
        raise RuntimeError(
            "WDBC CSV를 찾을 수 없고 다운로드도 실패했습니다. "
            "로컬에 wdbc.data를 두거나, 인터넷 연결 후 다시 실행하세요."
        ) from e

def _load_wdbc_numpy(csv_path):
    """
    WDBC:
      - 569 행, 32열
      - 열: ID, diagnosis(M/B), 30개의 float 특성
    반환: X (numpy float32) shape (N, F), y (numpy float32) shape (N, 1)  [M=1, B=0]
    """
    X_list, y_list = [], []
    with open(csv_path, "r", newline="") as f:
        r = csv.reader(f)
        for row in r:
            # row[0]=ID, row[1]=M/B, row[2:]=30 features
            if len(row) < 32:
                continue
            y_list.append(1.0 if row[1].strip().upper() == "M" else 0.0)
            feats = [float(v) for v in row[2:]]
            X_list.append(feats)
    X = np.asarray(X_list, dtype=np.float32)  # (N, 30)
    y = np.asarray(y_list, dtype=np.float32).reshape(-1, 1)  # (N, 1)
    return X, y

def _standardize_train_test(X_train, X_test, eps=1e-8):
    mean = X_train.mean(axis=0, keepdims=True)
    std  = X_train.std(axis=0, keepdims=True)
    std  = np.where(std < eps, 1.0, std)
    X_train_z = (X_train - mean) / std
    X_test_z  = (X_test  - mean) / std
    return X_train_z.astype(np.float32), X_test_z.astype(np.float32)

def _train_val_split(X, y, val_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    idx = np.arange(N)
    rng.shuffle(idx)
    n_val = int(math.floor(N * val_ratio))
    val_idx = idx[:n_val]
    tr_idx  = idx[n_val:]
    return X[tr_idx], y[tr_idx], X[val_idx], y[val_idx]

def _binarize(y_pred, thr=0.5):
    return (y_pred >= thr).astype(np.float32)

def _accuracy(y_true, y_pred_bin):
    y_true = y_true.reshape(-1)
    y_pred_bin = y_pred_bin.reshape(-1)
    return float((y_true == y_pred_bin).mean())

def _confusion(y_true, y_pred_bin):
    y_true = y_true.reshape(-1).astype(np.int32)
    y_pred_bin = y_pred_bin.reshape(-1).astype(np.int32)
    tp = int(((y_true == 1) & (y_pred_bin == 1)).sum())
    tn = int(((y_true == 0) & (y_pred_bin == 0)).sum())
    fp = int(((y_true == 0) & (y_pred_bin == 1)).sum())
    fn = int(((y_true == 1) & (y_pred_bin == 0)).sum())
    return tp, fp, tn, fn

def build_binary_classifier(input_features, lr=0.01, optimizer="adam"):
    model = Sequential(input_shape=(1, 1, input_features))
    model.add(Flatten(input_shape=(1, 1, input_features)))
    model.add(Dense(units=32, activation=None, initializer="xavier"))
    model.add(Activation("relu"))
    model.add(Dense(units=16, activation=None, initializer="xavier"))
    model.add(Activation("relu"))
    model.add(Dense(units=1, activation=None, initializer="xavier"))
    model.add(Activation("sigmoid"))
    model.compile(optimizer=optimizer, loss="bce", learning_rate=lr)
    return model

def test_wdbc_binary(path_hint=None, epochs=300, batch=32, lr=0.01, optimizer="adam", seed=7):
    """
    실제 UCI WDBC(유방암 진단) 데이터셋으로 학습 검증.
    기대: 검증 정확도 90% 이상 (모델/초깃값/러닝레이트에 따라 조금 오차 가능)
    """
    print(f"\n=== [TEST] WDBC Binary Classification — {optimizer.upper()} (lr={lr}, epochs={epochs}, batch={batch}) ===")
    np.random.seed(seed)

    csv_path = _ensure_wdbc_csv(path_hint)
    X, y = _load_wdbc_numpy(csv_path)
    print(f"Loaded WDBC: X={X.shape}, y={y.shape}, malignancy rate={float(y.mean()):.3f}")

    # split + standardize
    X_tr, y_tr, X_va, y_va = _train_val_split(X, y, val_ratio=0.2, seed=seed)
    X_tr, X_va = _standardize_train_test(X_tr, X_va)

    # NCHW (B, 1, 1, F)
    Xtr4 = X_tr.reshape(X_tr.shape[0], 1, 1, X_tr.shape[1]).astype(np.float32)
    Xva4 = X_va.reshape(X_va.shape[0], 1, 1, X_va.shape[1]).astype(np.float32)

    model = build_binary_classifier(input_features=X.shape[1], lr=lr, optimizer=optimizer)
    _try_debug_graph(model)

    # baseline
    loss_before = float(model.evaluate(Xtr4, y_tr))
    print(f"[BEFORE] BCE(train)={loss_before:.6f}")

    # train
    for e in range(epochs):
        model.fit(Xtr4, y_tr, epochs=1, batch_size=batch)
        if (e+1) % 25 == 0 or e == 0:
            tr_loss = float(model.evaluate(Xtr4, y_tr))
            va_loss = float(model.evaluate(Xva4, y_va))
            y_hat_va = model.predict(Xva4)
            acc_va = _accuracy(y_va, _binarize(y_hat_va, 0.5))
            print(f"  [epoch {e+1:3d}] loss(tr)={tr_loss:.5f}  loss(va)={va_loss:.5f}  acc(va)={acc_va*100:.2f}%")

    # final eval
    y_hat_tr = model.predict(Xtr4)
    y_hat_va = model.predict(Xva4)
    tr_acc = _accuracy(y_tr, _binarize(y_hat_tr, 0.5))
    va_acc = _accuracy(y_va, _binarize(y_hat_va, 0.5))
    tp, fp, tn, fn = _confusion(y_va, _binarize(y_hat_va, 0.5))

    print("\n=== [RESULT] WDBC ===")
    print(f"Train Acc: {tr_acc*100:.2f}%")
    print(f"Valid Acc: {va_acc*100:.2f}%")
    print(f"Confusion (Valid) — TP:{tp}  FP:{fp}  TN:{tn}  FN:{fn}")

    # 간단한 기준선 체크 (너무 빡세게 잡지 않음)
    assert va_acc > 0.85, f"Validation accuracy too low: {va_acc:.3f} (expected > 0.85)"

    return va_acc

if __name__ == "__main__":
    # 1) 실제 데이터셋 테스트 (Adam + 표준화)
    try:
        acc = test_wdbc_binary(path_hint=None, epochs=300, batch=32, lr=0.01, optimizer="adam", seed=7)
        print(f"\nWDBC test done ✅  (Valid Acc ≈ {acc*100:.2f}%)")
    except AssertionError as e:
        print("ASSERT:", e)
    except Exception as e:
        print("ERROR:", e)

    print("\nAll tests finished ✅")
