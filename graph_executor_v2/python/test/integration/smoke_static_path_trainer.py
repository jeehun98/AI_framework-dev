# tests/smoke_static_path_trainer.py
from __future__ import annotations
import os, sys, math
import cupy as cp

THIS = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(THIS, "..",".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from graph_executor_v2.layers.sequential import Sequential
from graph_executor_v2.layers.rnn import RNN
from graph_executor_v2.layers.flatten import Flatten
from graph_executor_v2.layers.dense_gemm import Dense
from graph_executor_v2.layers.activations import ActivationLayer
from graph_executor_v2.layers.conv2d import Conv2D  # CNN 스모크용

from graph_executor_v2.losses.softmax_ce import SoftmaxCrossEntropy
from graph_executor_v2.optim.adamw import AdamWOpt


# ------------------------
# Model builders (STATIC)
# ------------------------
def make_model_static_mlp(
    *, N=8, T=12, I=16, H=32, hidden=64, classes=7
) -> Sequential:
    """
    정적 MLP-like 시퀀스:
      RNN -> Flatten -> Dense(hidden) -> ReLU -> Dense(classes)
    """
    model = Sequential(
        RNN(hidden_size=H, activation="tanh", with_bias=True, save_z_in_fwd=True),
        Flatten(),
        Dense(hidden, activation="none", initializer="he", use_native_bwd=True),
        ActivationLayer(act="relu", save_y=True, name="MLPAct"),
        Dense(classes, activation="none", initializer="xavier", use_native_bwd=True),
    )
    model.build((N, T, I))
    model.train(True)
    return model


def make_model_static_cnn(
    *, N=8, C=3, H=16, W=16, K=16, classes=7, ks=3, stride=1, padding=1
) -> Sequential:
    """
    정적 CNN-like 시퀀스:
      Conv2D(C->K) -> ReLU -> Flatten -> Dense(classes)
    """
    model = Sequential(
        Conv2D(out_channels=K, kernel_size=(ks, ks),
            stride=(stride, stride), padding=(padding, padding), dilation=(1, 1), groups=1),
        ActivationLayer(act="relu", save_y=False, name="ConvAct"),
        Flatten(),
        Dense(classes, activation="none", initializer="xavier", use_native_bwd=True),
    )
    model.build((N, C, H, W))
    model.train(True)
    return model


# ------------------------
# Helpers
# ------------------------
def _adamw(lr=1e-3, wd=1e-4):
    opt = AdamWOpt([], lr=lr, wd=wd)
    if hasattr(opt, "ensure_initialized"):
        opt.ensure_initialized()
    return opt


# ------------------------
# Smoke runners (STATIC)
# ------------------------
def run_smoke_static_mlp(tag: str):
    """
    - compile()로 정적 캡처
    - 동일 배치로 2회 replay: 손실 유효 + logits 포인터 안정성 확인
    """
    C = 7
    cp.random.seed(2025)

    N, T, I = 8, 12, 16
    model = make_model_static_mlp(N=N, T=T, I=I, H=32, hidden=64, classes=C)

    loss = SoftmaxCrossEntropy()
    opt = _adamw()

    # 정적 캡처 준비
    tg = model.compile((N, T, I), loss=loss, optimizer=opt)

    # 배치 준비
    X = cp.random.randn(N, T, I).astype(cp.float32)
    y = cp.random.randint(0, C, size=(N,), dtype=cp.int32)

    # 1회차
    L1 = model.one_step(X, y)
    assert isinstance(L1, float) and math.isfinite(L1)
    logits_ptr = int(model.tg.logits.data.ptr)

    # 2회차 (같은 그래프 재생 → 포인터 동일)
    L2 = model.one_step(X, y)
    assert isinstance(L2, float) and math.isfinite(L2)
    assert int(model.tg.logits.data.ptr) == logits_ptr, \
        "logits buffer ptr must be stable for static graph replay"

    print(f"[OK][static_mlp:{tag}] L1={L1:.6f} L2={L2:.6f}")


def run_static_shape_guard(tag: str):
    """
    - 정적 그래프의 shape 가드 동작 확인:
      compile 시점 shape과 다른 입력을 넣으면 AssertionError 발생
    """
    C = 7
    cp.random.seed(2025)

    N, T, I = 8, 12, 16
    model = make_model_static_mlp(N=N, T=T, I=I, H=32, hidden=64, classes=C)

    loss = SoftmaxCrossEntropy()
    opt = _adamw()

    model.compile((N, T, I), loss=loss, optimizer=opt)

    # 다른 N으로 시도 → 실패해야 함
    N_bad = N * 2
    X_bad = cp.random.randn(N_bad, T, I).astype(cp.float32)
    y_bad = cp.random.randint(0, C, size=(N_bad,), dtype=cp.int32)

    try:
        model.one_step(X_bad, y_bad)
        raise AssertionError("static shape guard must raise on mismatched (N,T,I)")
    except AssertionError:
        # 기대한 실패
        pass

    print(f"[OK][static_shape_guard:{tag}] mismatch correctly rejected")


def run_smoke_static_cnn(tag: str):
    """
    - Conv2D를 포함한 정적 경로도 capture/replay가 정상 동작하는지 확인
      (workspace 확보, forward/backward, opt step까지)
    """
    C = 7
    cp.random.seed(2025)

    N, Cin, H, W = 8, 3, 16, 16
    model = make_model_static_cnn(N=N, C=Cin, H=H, W=W, K=16, classes=C)

    loss = SoftmaxCrossEntropy()
    opt = _adamw()

    model.compile((N, Cin, H, W), loss=loss, optimizer=opt)

    X = cp.random.randn(N, Cin, H, W).astype(cp.float32)
    y = cp.random.randint(0, C, size=(N,), dtype=cp.int32)

    L1 = model.one_step(X, y)
    assert isinstance(L1, float) and math.isfinite(L1)
    logits_ptr = int(model.tg.logits.data.ptr)

    # 한 번 더
    L2 = model.one_step(X, y)
    assert isinstance(L2, float) and math.isfinite(L2)
    assert int(model.tg.logits.data.ptr) == logits_ptr, \
        "logits buffer ptr must be stable for static CNN graph replay"

    print(f"[OK][static_cnn:{tag}] L1={L1:.6f} L2={L2:.6f}")


# ------------------------
# Main
# ------------------------
def main():
    print("== Static path trainer smoke tests ==")
    run_smoke_static_mlp("mlp-1")
    run_static_shape_guard("guard-1")
    run_smoke_static_cnn("cnn-1")
    print("[ALL OK] static path smoke completed.")
    print("Tip) Nsight 타임라인에서 [CAPTURE][static] / [REPLAY][static] 태그를 확인하세요.")

if __name__ == "__main__":
    main()
