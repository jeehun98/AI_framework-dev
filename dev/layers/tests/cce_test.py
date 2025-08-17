# dev/layers/tests/cce_only_check.py
import sys, os
import numpy as np

# (선택) 첫 커널 에러 즉시 표면화
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# CUDA DLL 경로
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin")

# Pybind11 .pyd 경로 (graph_executor가 빌드된 폴더)
sys.path.append(os.path.join(os.path.dirname(__file__), "build", "lib.win-amd64-cpython-312"))

# AI framework 루트 및 테스트 경로
sys.path.insert(0, os.path.abspath("C:/Users/owner/Desktop/AI_framework-dev"))
sys.path.append("C:/Users/owner/Desktop/AI_framework-dev/dev/backend/graph_executor/test")

from dev.models.sequential import Sequential
from dev.layers.dense import Dense
from dev.layers.activation_layer import Activation
from dev.layers.flatten import Flatten

np.set_printoptions(suppress=True, linewidth=120)

def assert_loss_decreases(name, before, after, factor=0.7):
    print(f"[{name}] loss before={before:.6f} → after={after:.6f}")
    if not (after < before * factor):
        raise AssertionError(f"{name}: loss did not decrease enough. (×{after/before:.3f})")

def make_simple_3class(seed=123, n_per_class=64, noise=0.15):
    rng = np.random.default_rng(seed)
    centers = np.array([[1,0,0], [0,1,0], [0,0,1]], dtype=np.float32)

    X_list, Y_list = [], []
    for c in range(3):
        base = centers[c]
        pts = base + noise * rng.normal(size=(n_per_class, 3)).astype(np.float32)
        X_list.append(pts)
        Y = np.zeros((n_per_class, 3), dtype=np.float32)
        Y[:, c] = 1.0
        Y_list.append(Y)

    X = np.vstack(X_list).astype(np.float32).reshape(-1, 1, 1, 3)  # B,1,1,3
    Y = np.vstack(Y_list).astype(np.float32)                       # B,3 (one-hot)
    return X, Y

def test_softmax_cce(epochs=1500, lr=3e-3, seed=123, noise=0.15, n_per_class=64):
    print("\n=== [TEST] softmax × CCE (3-class) ===")
    X, Y = make_simple_3class(seed=seed, n_per_class=n_per_class, noise=noise)

    model = Sequential(input_shape=(1,1,3))
    model.add(Flatten(input_shape=(1,1,3)))
    model.add(Dense(units=32, activation=None, initializer="xavier"))
    model.add(Activation("gelu"))
    model.add(Dense(units=3, activation=None, initializer="xavier"))
    model.add(Activation("softmax"))

    # ✅ 핵심: CCE 사용 (fused 경로 활성화 기대)
    model.compile(optimizer="adam", loss="cce", learning_rate=lr)

    before = model.evaluate(X, Y)
    model.fit(X, Y, epochs=epochs, batch_size=64)
    after = model.evaluate(X, Y)

    assert_loss_decreases("softmax/cce", before, after, factor=0.7)

    # 러프 정확도
    y_pred = model.predict(X)
    acc = (np.argmax(y_pred, axis=1) == np.argmax(Y, axis=1)).mean()
    print(f"[softmax×cce] rough acc: {acc*100:.1f}%")

def main():
    # 기본 파라미터로 CCE만 테스트
    test_softmax_cce(epochs=1500, lr=3e-3)

if __name__ == "__main__":
    main()
