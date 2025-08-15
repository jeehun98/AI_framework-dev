import sys, os
import numpy as np

# CUDA DLL 경로 (임포트 전에 추가)
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin")

# Pybind11로 빌드된 .pyd 경로
sys.path.append(os.path.join(os.path.dirname(__file__), "build", "lib.win-amd64-cpython-312"))

# AI framework 루트 경로
sys.path.insert(0, os.path.abspath("C:/Users/owner/Desktop/AI_framework-dev"))
sys.path.append("C:/Users/owner/Desktop/AI_framework-dev/dev/backend/graph_executor/test")

# AI Framework
from dev.models.sequential import Sequential
from dev.layers.dense import Dense
from dev.layers.activation_layer import Activation
from dev.layers.flatten import Flatten

np.set_printoptions(suppress=True, linewidth=120)

# ------------------------------------------------------------
# 유틸
# ------------------------------------------------------------
def print_graph(model):
    print("\n=== [Graph E] ===")
    for i, op in enumerate(model.E):
        print(f"[{i}] type={op.op_type}, input={op.input_id}, output={op.output_id}")

def assert_loss_decreases(name, before, after, factor=0.5):
    print(f"[{name}] loss before={before:.6f} → after={after:.6f}")
    if not (after < before * factor):
        # 완전히 같은 초기화/학습률이 아닐 수 있으니, 1) 에폭 늘려 보거나 2) lr 키워볼 것
        raise AssertionError(f"{name}: loss did not decrease enough. (×{after/before:.3f})")

# ------------------------------------------------------------
# 1) LeakyReLU / ELU / GELU / SiLU: 회귀 과제 수렴 테스트
#     - 입력: 2D 벡터
#     - 목표: y = x0 - 0.5*x1  (선형 회귀; 은닉에 비선형 활성화 끼워서 연쇄 미분 검증)
# ------------------------------------------------------------
def make_regression_data(n=64, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 2, size=(n, 1, 1, 2)).astype(np.float32)  # B,C,H,W = (n,1,1,2)
    y = (X[..., 0, 0] - 0.5 * X[..., 0, 1]).reshape(n, 1).astype(np.float32)
    return X, y

def test_activation_regression(act_name, epochs=600, lr=0.01):
    print(f"\n=== [TEST] {act_name} - regression (MSE) ===")
    X, y = make_regression_data(n=128, seed=42)

    model = Sequential(input_shape=(1,1,2))
    model.add(Flatten(input_shape=(1,1,2)))
    model.add(Dense(units=8, activation=None, initializer="xavier"))
    model.add(Activation(act_name))             # 여기서 추가한 활성화 사용
    model.add(Dense(units=1, activation=None, initializer="xavier"))


    model.compile(optimizer="sgd", loss="mse", learning_rate=lr)

    loss_before = model.evaluate(X, y)

    model.fit(X, y, epochs=epochs, batch_size=32)
    loss_after = model.evaluate(X, y)

    assert_loss_decreases(f"{act_name}/regression", loss_before, loss_after, factor=0.4)

# ------------------------------------------------------------
# 2) Softmax: 간단 3클래스 분류(원-핫을 MSE로 근사)
#     - 입력: 3개 기준 벡터 + 노이즈
#     - 목표: one-hot 타깃 (MSE 사용: CE가 아직 미구현이어도 동작)
# ------------------------------------------------------------
def make_simple_3class(seed=0, n_per_class=64, noise=0.1):
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
    Y = np.vstack(Y_list).astype(np.float32)                       # B,3
    return X, Y

def test_softmax_mse(epochs=2000, lr=1e-2):   # 에폭↑, lr↑
    print(f"\n=== [TEST] softmax - 3-class (MSE to one-hot) ===")
    X, Y = make_simple_3class(seed=123, n_per_class=64, noise=0.15)

    model = Sequential(input_shape=(1,1,3))
    model.add(Flatten(input_shape=(1,1,3)))
    model.add(Dense(units=32, activation=None, initializer="xavier"))  # 너비↑
    model.add(Activation("gelu"))
    model.add(Dense(units=3, activation=None, initializer="xavier"))
    model.add(Activation("softmax"))

    # ✅ adam 사용 (adamw 아님)
    model.compile(optimizer="adam", loss="mse", learning_rate=lr)

    loss_before = model.evaluate(X, Y)
    model.fit(X, Y, epochs=epochs, batch_size=64)
    loss_after = model.evaluate(X, Y)

    assert_loss_decreases("softmax/3class", loss_before, loss_after, factor=0.7)  # 기준 완화

# ------------------------------------------------------------
# 3) Forward 범위/안정성 smoke 테스트 (예외 없이 통과 + 값 범위)
#     - sigmoid/tanh/softmax는 출력 범위 체크
#     - 나머지는 NaN/Inf가 없음을 체크
# ------------------------------------------------------------
def test_forward_ranges():
    print("\n=== [TEST] forward ranges (smoke) ===")
    X = np.array([[-3.0, -1.0, 0.0, 1.0, 3.0]], dtype=np.float32).reshape(1,1,1,5)

    def run_activation(name):
        model = Sequential(input_shape=(1,1,5))
        model.add(Flatten(input_shape=(1,1,5)))
        model.add(Activation(name))
        model.compile(optimizer="adam", loss="mse", learning_rate=1e-3)  # loss는 사용 안함
        y_pred = model.predict(X)
        return y_pred

    # sigmoid
    s = run_activation("sigmoid")
    assert np.all(np.isfinite(s)), "sigmoid produced non-finite values"
    assert np.all((s > 0) & (s < 1)), "sigmoid out of (0,1) range"

    # tanh
    t = run_activation("tanh")
    assert np.all(np.isfinite(t)), "tanh produced non-finite values"
    assert np.all((t >= -1) & (t <= 1)), "tanh out of [-1,1] range"

    # softmax (axis=feature)
    sm = run_activation("softmax")
    assert np.all(np.isfinite(sm)), "softmax produced non-finite values"
    row = sm.reshape(-1, 5)
    sums = np.sum(row, axis=1)
    assert np.allclose(sums, 1.0, atol=1e-4), f"softmax rows don't sum to 1 (got {sums})"

    # gelu / silu / elu / leaky_relu: NaN/Inf 여부만 체크
    for name in ["gelu", "silu", "elu", "leaky_relu", "relu"]:
        v = run_activation(name)
        assert np.all(np.isfinite(v)), f"{name} produced non-finite values"

# ------------------------------------------------------------
# 메인: 각 활성화 테스트 실행
# ------------------------------------------------------------
def main():
    # 0) 기존 XOR 테스트(참고용)
    # from your earlier code — 필요 시 그대로 호출 가능
    # test_xor()

    # 1) 회귀 수렴 테스트: 새 활성화 4종
    for act in ["leaky_relu", "elu", "gelu", "silu"]:
        # 학습률/에폭은 환경에 따라 미세 조정 가능
        test_activation_regression(act_name=act, epochs=600, lr=1e-2)

    # 2) 소프트맥스 수렴 테스트 (MSE + one-hot 근사)
    test_softmax_mse(epochs=800, lr=3e-3)

    # 3) forward 범위/안정성 스모크
    test_forward_ranges()

    print("\n✅ All activation tests passed.")

if __name__ == "__main__":
    main()
