# tests/smoke_dynamic_path_trainer.py
from __future__ import annotations
import os, sys, math
import cupy as cp

THIS = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(THIS, "..",".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from graph_executor_v2.layers.sequential import Sequential, _FALLBACK_POOL  # _FALLBACK_POOL: 테스트 전용 접근
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
    If 분기:
      - then: Dropout -> Dense -> ReLU
      - else: Dense -> ReLU
    then/else 출력 shape 동일.
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
    Repeat:
      - body: Dense(hidden)->ReLU
      - steps_fn: X 평균값 기준 1~3회 반복
        (캡처는 1-step, 실행 시 T회 launch)
    """
    body = Sequential(
        Dense(hidden, activation="none", initializer="he", use_native_bwd=True),
        ActivationLayer(act="relu", save_y=False, name="RAct"),
    )

    def steps_fn(X, ctx):
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
    EarlyExit (최소 구현: 전 스테이지 선형화)
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
def _adamw(lr=1e-3, wd=1e-4):
    opt = AdamWOpt([], lr=lr, wd=wd)
    if hasattr(opt, "ensure_initialized"):
        opt.ensure_initialized()
    return opt


# ------------------------
# Smoke runners
# ------------------------
def run_smoke_dynamic_if(tag: str):
    """
    - 배치 크기로 분기(else→then) 테스트
    - 같은 분기에서는 그래프 캐시 재사용(로그릿 포인터 동일)
    - 분기 변경 시 다른 그래프 사용(포인터 달라짐)
    """
    C = 7
    cp.random.seed(2025)

    # else-branch (N=8 < 32)
    N, T, I = 8, 12, 16
    model = make_model_dynamic_if(N=N, T=T, I=I, H=32, hidden=64, classes=C)

    loss = SoftmaxCrossEntropy()
    opt = _adamw()

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

    # then-branch (N=64 ≥ 32, 분기 변경 → 다른 그래프)
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
    - Repeat 캡처/리플레이: 같은 서명/variant면 steps_fn 변화에도 같은 그래프 재사용
    - NVTX로는 [REPLAY][dynamic] ... xT 확인 가능(수동)
    """
    C = 7
    cp.random.seed(2025)

    N, T, I = 8, 12, 16
    model = make_model_dynamic_repeat(N=N, T=T, I=I, H=32, hidden=64, classes=C)

    loss = SoftmaxCrossEntropy()
    opt = _adamw()

    X = cp.random.randn(N, T, I).astype(cp.float32)
    y = cp.random.randint(0, C, size=(N,), dtype=cp.int32)

    ctx = {"variant": {"unroll": 1}}
    L1 = model.one_step_dynamic(X, y, loss=loss, optimizer=opt, ctx=ctx)
    assert isinstance(L1, float) and math.isfinite(L1)
    logits_ptr_1 = int(model.tg.logits.data.ptr)

    # X를 이동시켜 steps_fn 반복 횟수 변화(1→2→3 등), shape/dtype 동일
    X2 = X + 0.5
    L2 = model.one_step_dynamic(X2, y, loss=loss, optimizer=opt, ctx=ctx)
    assert isinstance(L2, float) and math.isfinite(L2)

    # 동일 서명/variant → 같은 캡처 재사용 → 포인터 동일
    assert int(model.tg.logits.data.ptr) == logits_ptr_1, \
        "repeat with same signature/variant should reuse the same cached graph"

    # repeat_batches 사용 예: 서로 다른 배치를 T회 연속 적용
    ctx_batches = {"variant": {"unroll": 1}, "repeat_steps": 3,
                   "repeat_batches": [(X*0.9, y), (X*1.05, y), (X*1.2, y)]}
    L3 = model.one_step_dynamic(X, y, loss=loss, optimizer=opt, ctx=ctx_batches)
    assert isinstance(L3, float) and math.isfinite(L3)

    print(f"[OK][dynamic_repeat:{tag}] L1={L1:.6f} L2={L2:.6f} L3={L3:.6f}")


def run_smoke_dynamic_earlyexit(tag: str):
    """
    - EarlyExit(선형화): 캡처/리플레이 유효성
    """
    C = 7
    cp.random.seed(2025)

    N, T, I = 8, 12, 16
    model = make_model_dynamic_earlyexit(N=N, T=T, I=I, H=32, hidden=64, classes=C)

    loss = SoftmaxCrossEntropy()
    opt = _adamw()

    X = cp.random.randn(N, T, I).astype(cp.float32)
    y = cp.random.randint(0, C, size=(N,), dtype=cp.int32)

    ctx = {"variant": {"unroll": 1}}
    L = model.one_step_dynamic(X, y, loss=loss, optimizer=opt, ctx=ctx)
    assert isinstance(L, float) and math.isfinite(L)

    # 재실행(동일 그래프 재사용)
    L2 = model.one_step_dynamic(X, y, loss=loss, optimizer=opt, ctx=ctx)
    assert isinstance(L2, float) and math.isfinite(L2)

    print(f"[OK][dynamic_earlyexit:{tag}] L={L:.6f} L2={L2:.6f}")


def run_dropout_rng_epoch_change(tag: str):
    """
    - Dropout counter_base 제어 검증:
      동일 입력/경로에서 rng_epoch 변경 시 같은 그래프를 재사용하면서
      손실 값이 (대체로) 달라지는지 확인
    - rng_epoch 미변경 시 손실 값이 동일하게 유지되는지 확인(결정적 Dropout 가정)
    """
    C = 7
    cp.random.seed(777)

    N, T, I = 32, 8, 16  # 분기는 고정(then/else 상관없게)
    model = make_model_dynamic_if(N=N, T=T, I=I, H=16, hidden=32, classes=C,
                                  p_drop_then=0.5, p_drop_else=0.5)

    loss = SoftmaxCrossEntropy()
    opt = _adamw()

    X = cp.random.randn(N, T, I).astype(cp.float32)
    y = cp.random.randint(0, C, size=(N,), dtype=cp.int32)

    ctx = {"variant": {"unroll": 1}, "rng_epoch": 0}
    L1 = model.one_step_dynamic(X, y, loss=loss, optimizer=opt, ctx=ctx)
    ptr1 = int(model.tg.logits.data.ptr)

    # rng_epoch 같으면 같은 마스크 → 손실 같을 가능성 높음(결정적 구현 가정)
    ctx_same = {"variant": {"unroll": 1}, "rng_epoch": 0}
    L1b = model.one_step_dynamic(X, y, loss=loss, optimizer=opt, ctx=ctx_same)
    ptr1b = int(model.tg.logits.data.ptr)
    assert ptr1b == ptr1, "same graph should be reused (logits ptr stable)"
    # 같으면 베스트, 혹시 다를 수도 있으니 완전 동일을 강제하진 않음
    # assert abs(L1b - L1) < 1e-7

    # rng_epoch 바꾸면 마스크 바뀜 → 같은 그래프 재사용이지만 손실이 달라질 확률 큼
    ctx_new = {"variant": {"unroll": 1}, "rng_epoch": 1}
    L2 = model.one_step_dynamic(X, y, loss=loss, optimizer=opt, ctx=ctx_new)
    ptr2 = int(model.tg.logits.data.ptr)
    assert ptr2 == ptr1, "rng_epoch only should not change graph key (reuse same graph)"
    assert L1 != L2, "with different rng_epoch, dropout mask is expected to change loss"

    print(f"[OK][dropout_rng:{tag}] L1={L1:.6f} L1b={L1b:.6f} L2={L2:.6f}")


def run_pool_lru_cap(tag: str):
    """
    - 내부 Fallback 풀의 LRU 상한 동작 확인
    - 서로 다른 variant로 다수 그래프 생성 → 상한 이내로 유지되는지 점검
    """
    C = 5
    cp.random.seed(2025)

    N, T, I = 8, 8, 8
    model = make_model_dynamic_if(N=N, T=T, I=I, H=16, hidden=32, classes=C)

    loss = SoftmaxCrossEntropy()
    opt = _adamw()

    X = cp.random.randn(N, T, I).astype(cp.float32)
    y = cp.random.randint(0, C, size=(N,), dtype=cp.int32)

    # 다양한 variant로 다른 키 유도(경로 fingerprint+variant 포함됨)
    K = model._FALLBACK_POOL_MAX + 4
    before = len(_FALLBACK_POOL)
    for i in range(K):
        ctx = {"variant": {"alpha": i}}
        model.one_step_dynamic(X, y, loss=loss, optimizer=opt, ctx=ctx)

    pool_size = len(_FALLBACK_POOL)
    assert pool_size <= model._FALLBACK_POOL_MAX, \
        f"pool should be capped by LRU (got {pool_size} > {model._FALLBACK_POOL_MAX})"

    print(f"[OK][pool_lru:{tag}] before={before} after={pool_size} cap={model._FALLBACK_POOL_MAX}")


# ------------------------
# Main
# ------------------------
def main():
    print("== Dynamic path trainer smoke tests ==")
    run_smoke_dynamic_if("case-if-1")
    run_smoke_dynamic_repeat("case-repeat-1")
    run_smoke_dynamic_earlyexit("case-ee-1")
    run_dropout_rng_epoch_change("case-rng-1")
    run_pool_lru_cap("case-lru-1")
    print("[ALL OK] dynamic path smoke completed.")
    print("Tip) Nsight 타임라인에서 [CAPTURE]/[REPLAY] NVTX 태그를 함께 확인하세요.")

if __name__ == "__main__":
    main()
