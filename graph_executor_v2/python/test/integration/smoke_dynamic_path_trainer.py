# tests/smoke_dynamic_path_trainer.py
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
from graph_executor_v2.layers.dropout import Dropout
from graph_executor_v2.layers.conditional import If, Repeat, EarlyExit

from graph_executor_v2.losses.softmax_ce import SoftmaxCrossEntropy
from graph_executor_v2.optim.adamw import AdamWOpt


# ------------------------
# Model builders (with dynamic path)
# ------------------------
def make_model_dynamic_if(
    *, N=8, T=12, I=16, H=32, hidden=64, classes=7,
    p_drop_then=0.3, p_drop_else=0.0, seed=0x1111
) -> Sequential:
    """
    분기 If 레이어:
      - then: Dropout -> Dense
      - else: Dense (no dropout)
    then/else 출력 shape는 동일해야 하므로 동일한 Dense를 배치한다.
    """
    then_block = Sequential(
        Dropout(p=p_drop_then, scale_in_train=True, seed=seed ^ 0xAAAA),
        Dense(hidden, activation="none", initializer="he", use_native_bwd=True),
        ActivationLayer(act="relu", save_y=True, name="ThenAct"),
    )
    else_block = Sequential(
        Dense(hidden, activation="none", initializer="he", use_native_bwd=True),
        ActivationLayer(act="relu", save_y=True, name="ElseAct"),
    )

    model = Sequential(
        RNN(hidden_size=H, activation="tanh", with_bias=True, save_z_in_fwd=True),
        Flatten(),
        If(lambda X, ctx: X.shape[0] >= 32, then_block=then_block, else_block=else_block),
        Dense(classes, activation="none", initializer="xavier", use_native_bwd=True),
    )
    model.build((N, T, I))
    model.train(True)
    return model


def make_model_dynamic_repeat(
    *, N=8, T=12, I=16, H=32, hidden=64, classes=7
) -> Sequential:
    """
    Repeat 레이어:
      - body: Dense(hidden)->relu
      - steps_fn: 배치 합계에 따라 1~3회 반복(단, 캡처는 1-step 그래프를 생성하고
                  실행 시에만 T번 launch)
    """
    body = Sequential(
        Dense(hidden, activation="none", initializer="he", use_native_bwd=True),
        ActivationLayer(act="relu", save_y=False, name="RAct"),
    )

    def steps_fn(X, ctx):
        # X의 평균으로 간단히 반복 횟수 결정 (테스트용)
        m = float(cp.mean(X).get()) if hasattr(X, "device") else float(X.mean())
        if m > 0.2:
            return 3
        elif m > -0.2:
            return 2
        else:
            return 1

    model = Sequential(
        RNN(hidden_size=H, activation="tanh", with_bias=True, save_z_in_fwd=True),
        Flatten(),
        Repeat(body, steps_fn=steps_fn),
        Dense(classes, activation="none", initializer="xavier", use_native_bwd=True),
    )
    model.build((N, T, I))
    model.train(True)
    return model


def make_model_dynamic_earlyexit(
    *, N=8, T=12, I=16, H=32, hidden=64, classes=7
) -> Sequential:
    """
    EarlyExit 레이어:
      - stage0: 가벼운 Dense
      - stage1: 추가 Dense
    최소 구현에서는 전체 스테이지를 선형화하므로, exit_fn은 ctx 기록용으로만 둠.
    """
    stage0 = Sequential(
        Dense(hidden, activation="none", initializer="he", use_native_bwd=True),
        ActivationLayer(act="relu", save_y=False, name="EE0"),
    )
    stage1 = Sequential(
        Dense(hidden, activation="none", initializer="he", use_native_bwd=True),
        ActivationLayer(act="relu", save_y=False, name="EE1"),
    )
    def exit_fn(ctx):
        # 향후 부분 캡처 확장을 위한 자리 — 현재는 ctx 플래그만 남김
        ctx["earlyexit_checked"] = True
        return False

    model = Sequential(
        RNN(hidden_size=H, activation="tanh", with_bias=True, save_z_in_fwd=True),
        Flatten(),
        EarlyExit([stage0, stage1], exit_fn=exit_fn),
        Dense(classes, activation="none", initializer="xavier", use_native_bwd=True),
    )
    model.build((N, T, I))
    model.train(True)
    return model


# ------------------------
# Helpers
# ------------------------
def _clone_params_flat(model: Sequential):
    vecs = []
    for lyr in model.layers:
        for name in ("W", "B", "b"):  # 대소문자 혼용 대비
            if hasattr(lyr, name):
                w = getattr(lyr, name)
                if w is not None:
                    vecs.append(w.ravel())
    return None if not vecs else cp.concatenate(vecs)


# ------------------------
# Smoke runners
# ------------------------
def run_smoke_dynamic_if(tag: str):
    """
    - 배치 크기에 따라 분기 선택(else → then) 테스트
    - 같은 분기에서는 그래프 캐시 재사용(고정 logits 포인터 동일)
    - 분기를 바꾸면 다른 그래프가 사용됨(포인터 달라짐)
    """
    C = 7
    cp.random.seed(2025)

    # else-branch (N=8 < 32)
    N, T, I = 8, 12, 16
    model = make_model_dynamic_if(N=N, T=T, I=I, H=32, hidden=64, classes=C)

    loss = SoftmaxCrossEntropy()
    opt = AdamWOpt([], lr=1e-3, wd=1e-4)
    if hasattr(opt, "ensure_initialized"):
        opt.ensure_initialized()

    # 첫 입력: else-branch
    X_small = cp.random.randn(N, T, I).astype(cp.float32)
    y_small = cp.random.randint(0, C, size=(N,), dtype=cp.int32)

    ctx = {"variant": {"unroll": 1, "amp": "fp32"}}
    L1 = model.one_step_dynamic(X_small, y_small, loss=loss, optimizer=opt, ctx=ctx)
    assert isinstance(L1, float) and math.isfinite(L1)

    logits_ptr_else = int(model.tg.logits.data.ptr)

    # 동일 분기 재사용 → 포인터 동일
    L2 = model.one_step_dynamic(X_small, y_small, loss=loss, optimizer=opt, ctx=ctx)
    assert isinstance(L2, float) and math.isfinite(L2)
    assert int(model.tg.logits.data.ptr) == logits_ptr_else, \
        "logits buffer ptr must be stable for cached else-branch graph"

    # then-branch (N=64 ≥ 32, 분기 변경)
    N_big = 64
    X_big = cp.random.randn(N_big, T, I).astype(cp.float32)
    y_big = cp.random.randint(0, C, size=(N_big,), dtype=cp.int32)

    L3 = model.one_step_dynamic(X_big, y_big, loss=loss, optimizer=opt,
                                ctx={"variant": {"unroll": 1, "amp": "fp32"}})
    assert isinstance(L3, float) and math.isfinite(L3)
    logits_ptr_then = int(model.tg.logits.data.ptr)
    assert logits_ptr_then != logits_ptr_else, \
        "different branch should create/use different cached graph (different logits buffer)"

    print(f"[OK][dynamic_if:{tag}] L1={L1:.6f} L2={L2:.6f} L3={L3:.6f}")


def run_smoke_dynamic_repeat(tag: str):
    """
    - Repeat 본문은 1-step으로 캡처되고, 실행 시 T회 launch
    - steps_fn 결과에 따라 반복 횟수가 바뀌어도 캡처 재사용(같은 서명/variant면 동일 그래프)
    """
    C = 7
    cp.random.seed(2025)

    N, T, I = 8, 12, 16
    model = make_model_dynamic_repeat(N=N, T=T, I=I, H=32, hidden=64, classes=C)

    loss = SoftmaxCrossEntropy()
    opt = AdamWOpt([], lr=1e-3, wd=1e-4)
    if hasattr(opt, "ensure_initialized"):
        opt.ensure_initialized()

    X = cp.random.randn(N, T, I).astype(cp.float32)
    y = cp.random.randint(0, C, size=(N,), dtype=cp.int32)

    ctx = {"variant": {"unroll": 1}}
    L1 = model.one_step_dynamic(X, y, loss=loss, optimizer=opt, ctx=ctx)
    assert isinstance(L1, float) and math.isfinite(L1)

    logits_ptr_1 = int(model.tg.logits.data.ptr)

    # X를 살짝 이동시켜 steps_fn 결과(반복 횟수)가 달라지게 하되, shape/dtype 동일
    X2 = X + 0.5
    L2 = model.one_step_dynamic(X2, y, loss=loss, optimizer=opt, ctx=ctx)
    assert isinstance(L2, float) and math.isfinite(L2)

    # 동일 서명/variant → 같은 캡처 재사용 → 포인터 동일해야 함
    assert int(model.tg.logits.data.ptr) == logits_ptr_1, \
        "repeat with same signature/variant should reuse the same cached graph"

    print(f"[OK][dynamic_repeat:{tag}] L1={L1:.6f} L2={L2:.6f}")


def run_smoke_dynamic_earlyexit(tag: str):
    """
    - EarlyExit는 최소 구현에서 모든 스테이지를 선형화하여 캡처
    - 재생/손실 유효성만 확인
    """
    C = 7
    cp.random.seed(2025)

    N, T, I = 8, 12, 16
    model = make_model_dynamic_earlyexit(N=N, T=T, I=I, H=32, hidden=64, classes=C)

    loss = SoftmaxCrossEntropy()
    opt = AdamWOpt([], lr=1e-3, wd=1e-4)
    if hasattr(opt, "ensure_initialized"):
        opt.ensure_initialized()

    X = cp.random.randn(N, T, I).astype(cp.float32)
    y = cp.random.randint(0, C, size=(N,), dtype=cp.int32)

    ctx = {"variant": {"unroll": 1}}
    L = model.one_step_dynamic(X, y, loss=loss, optimizer=opt, ctx=ctx)
    assert isinstance(L, float) and math.isfinite(L)

    # 재실행도 정상
    L2 = model.one_step_dynamic(X, y, loss=loss, optimizer=opt, ctx=ctx)
    assert isinstance(L2, float) and math.isfinite(L2)

    print(f"[OK][dynamic_earlyexit:{tag}] L={L:.6f} L2={L2:.6f}")


# ------------------------
# Main
# ------------------------
def main():
    print("== Dynamic path trainer smoke tests ==")
    run_smoke_dynamic_if("case-if-1")
    run_smoke_dynamic_repeat("case-repeat-1")
    run_smoke_dynamic_earlyexit("case-ee-1")
    print("[ALL OK] dynamic path smoke completed.")

if __name__ == "__main__":
    main()
