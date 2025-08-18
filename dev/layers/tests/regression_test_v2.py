# regression_backtracking_relaxed_test.py
import os, sys, numpy as np, cupy as cp
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin")

# 프로젝트 경로 맞게
sys.path.insert(0, os.path.abspath("C:/Users/owner/Desktop/AI_framework-dev"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend", "graph_executor", "build", "lib.win-amd64-cpython-312"))

from dev.models.sequential import Sequential
from dev.layers.flatten import Flatten
from dev.layers.dense import Dense
import graph_executor as ge

# ---------- 유틸 ----------
def standardize_1d(x):
    x = x.astype(np.float32, copy=False).reshape(-1, 1, 1, 1)
    z = x.reshape(-1)
    mu, sigma = float(z.mean()), float(z.std() + 1e-8)
    z = (z - mu) / sigma
    return z.reshape(x.shape).astype(np.float32), mu, sigma

def build_reg_model(lr, optimizer):
    # 1층 선형 회귀 (bias 포함)
    model = Sequential(input_shape=(1, 1, 1))
    model.add(Flatten(input_shape=(1, 1, 1)))
    model.add(Dense(units=1, activation=None, initializer="xavier"))  # bias 기본 포함
    model.compile(optimizer=optimizer, loss="mse", learning_rate=lr)
    return model

def get_params(model):
    p = {}
    p.update(getattr(model, "weights", {}))
    p.update(getattr(model, "biases", {}))
    return p

def flatten_params(pd):
    flats = []
    for k in sorted(pd.keys()):
        arr = pd[k]
        if isinstance(arr, cp.ndarray):
            arr = cp.asnumpy(arr)
        flats.append(arr.ravel().astype(np.float32))
    return np.concatenate(flats) if flats else np.zeros(0, np.float32)

# ---------- 백트래킹(완화판) ----------
def train_with_relaxed_backtracking(model, x, y, epochs=600, init_lr=1e-2,
                                    tol_rel=1e-4, tol_abs=1e-7,
                                    max_halves=6, min_lr=1e-5,
                                    rewarm_every=10, rewarm_factor=1.25, verbose=True):
    # 모델의 lr를 강제로 우리가 관리
    model.learning_rate = init_lr
    lr = init_lr
    best = float("inf")
    good_streak = 0

    loss_prev = float(model.evaluate(x, y))
    if verbose:
        print(f"  MSE(before): {loss_prev:.6f}")

    for e in range(1, epochs+1):
        # 현재 파라미터 스냅샷
        params_before = {k: cp.array(v, copy=True) for k, v in get_params(model).items()}

        # 1 스텝 시도
        model.fit(x, y, epochs=1, batch_size=min(32, x.shape[0]), verbose=0)
        loss_now = float(model.evaluate(x, y))

        # 수용 조건(상대/절대)
        accept = (loss_now <= loss_prev * (1.0 + tol_rel) + tol_abs)

        retries = 0
        while (not accept) and retries < max_halves:
            # 롤백 + lr 반감 + 재시도
            for k, v in params_before.items():
                get_params(model)[k][...] = v  # in-place 복원
            lr = max(lr * 0.5, min_lr)
            model.learning_rate = lr

            model.fit(x, y, epochs=1, batch_size=min(32, x.shape[0]), verbose=0)
            loss_now = float(model.evaluate(x, y))
            accept = (loss_now <= loss_prev * (1.0 + tol_rel) + tol_abs)
            retries += 1

        # 수용/거부 처리
        if accept:
            # 좋은 스텝(엄밀 감소면 streak 증가)
            if loss_now < loss_prev - 1e-12:
                good_streak += 1
                best = min(best, loss_now)
                # rewarm: 일정 횟수 연속으로 좋았으면 살짝 올려줌(초기 lr 한도)
                if rewarm_every and (good_streak % rewarm_every == 0):
                    lr = min(init_lr, lr * rewarm_factor)
                    model.learning_rate = lr
            else:
                good_streak = 0
            loss_prev = loss_now
        else:
            # 끝까지 반감해도 못받으면 롤백하고 아주 작은 업뎃만 허용
            for k, v in params_before.items():
                get_params(model)[k][...] = v
            good_streak = 0

        if verbose and (e % 50 == 0 or e == epochs):
            print(f"  [epoch+{e}] MSE={loss_prev:.6f} (lr={lr:.3e}, retries={retries})")

    return loss_prev

# ---------- 실행 ----------
if __name__ == "__main__":
    np.random.seed(0)

    # 데이터 생성 ([-2,2], 약간의 노이즈)
    x = np.linspace(-2, 2, 64).astype(np.float32)
    y = (3.0 * x - 1.0).astype(np.float32)
    y += np.random.randn(*y.shape).astype(np.float32) * 0.05

    # 표준화 입력(권장). y는 그대로 두고 bias가 intercept를 학습하도록.
    x_std, mu_x, std_x = standardize_1d(x)
    y = y.reshape(-1, 1).astype(np.float32)

    # 모델
    for opt, init_lr in [("sgd", 1e-2), ("adam", 1e-3)]:
        print(f"\n=== Linear Regression y=3x-1 / {opt} (init_lr={init_lr}) ===")
        model = build_reg_model(lr=init_lr, optimizer=opt)

        # 완화된 백트래킹 학습
        final_loss = train_with_relaxed_backtracking(
            model, x_std, y,
            epochs=600, init_lr=init_lr,
            tol_rel=1e-4, tol_abs=1e-7,
            max_halves=6, min_lr=1e-5,
            rewarm_every=20, rewarm_factor=1.25,
            verbose=True
        )

        # 최종 파라미터와 예측 확인
        preds = model.predict(x_std).reshape(-1)
        print(f"  MSE(after):  {final_loss:.6f}")
        # (표준화 입력이라 실제 기울기는 3/std_x, 절편은 -1 - 3*mu_x/std_x 로 환산 가능)
        # 여기서는 그냥 학습된 (W,b) 직접 확인
        params = get_params(model)
        # Dense 1x1 가정
        W = float(cp.asnumpy(params[next(k for k in params if k.endswith("_W"))]).reshape(-1)[0])
        b = float(cp.asnumpy(params[next(k for k in params if k.endswith("_b"))]).reshape(-1)[0])
        print(f"  learned (W, b) on x_std ≈ ({W:.3f}, {b:.3f})")
