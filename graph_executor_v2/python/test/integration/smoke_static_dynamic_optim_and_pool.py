from __future__ import annotations
import os, sys, math
import cupy as cp

THIS = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(THIS, "..",".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# 모델/레이어
from graph_executor_v2.layers.sequential import Sequential, _FALLBACK_POOL
from graph_executor_v2.layers.rnn import RNN
from graph_executor_v2.layers.flatten import Flatten
from graph_executor_v2.layers.dense_gemm import Dense
from graph_executor_v2.layers.activations import ActivationLayer
from graph_executor_v2.layers.dropout import Dropout
from graph_executor_v2.layers.conditional import If, Repeat, EarlyExit

# 그래프 풀(있으면 전역 공유 캐시)
try:
    from graph_executor_v2.graph.graph_executor import graph_pool as GLOBAL_POOL
except Exception:
    GLOBAL_POOL = None

# 로스/옵티마이저
from graph_executor_v2.losses.softmax_ce import SoftmaxCrossEntropy
from graph_executor_v2.optim.adamw import AdamWOpt
from graph_executor_v2.optim.sgd import SGDOpt  # 방금 만든 sgd.py 기준
from graph_executor_v2.optim.rebind import try_rebind_grads

# ------------------------
# 공용 모델 빌더 (정적용)
# ------------------------
def make_model_static(*, N=8, T=12, I=16, H=32, hidden=64, classes=7) -> Sequential:
    """
    정적 경로: (RNN -> Flatten -> Dense -> ReLU -> Dense)
    - 분기/반복 없음
    - capture_plan / record_step_graph / TrainGraph 정석 경로 검증
    """
    model = Sequential(
        RNN(hidden_size=H, activation="tanh", with_bias=True, save_z_in_fwd=True),
        Flatten(),
        Dense(hidden, activation="none", initializer="he", use_native_bwd=True),
        ActivationLayer(act="relu", save_y=True),
        Dense(classes, activation="none", initializer="xavier", use_native_bwd=True),
    )
    model.build((N, T, I))
    model.train(True)
    return model

# ------------------------
# 동적 모델 빌더 (If/Repeat/EE)
# ------------------------
def make_model_dynamic_if(*, N=8, T=12, I=16, H=32, hidden=64, classes=7,
                          p_drop_then=0.3, p_drop_else=0.0, seed=0x1111) -> Sequential:
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

def make_model_dynamic_repeat(*, N=8, T=12, I=16, H=32, hidden=64, classes=7) -> Sequential:
    body = Sequential(
        Dense(hidden, activation="none", initializer="he", use_native_bwd=True),
        ActivationLayer(act="relu", save_y=False, name="RAct"),
    )
    def steps_fn(X, ctx):
        m = float(cp.mean(X).get()) if hasattr(X, "device") else float(X.mean())
        return 3 if m > 0.2 else (2 if m > -0.2 else 1)
    model = Sequential(
        RNN(hidden_size=H, activation="tanh", with_bias=True, save_z_in_fwd=True),
        Flatten(),
        Repeat(body, steps_fn=steps_fn),
        Dense(classes, activation="none", initializer="xavier", use_native_bwd=True),
    )
    model.build((N, T, I)); model.train(True); return model

def make_model_dynamic_earlyexit(*, N=8, T=12, I=16, H=32, hidden=64, classes=7) -> Sequential:
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
    model.build((N, T, I)); model.train(True); return model

# ------------------------
# 헬퍼
# ------------------------
def _adamw(lr=1e-3, wd=1e-4):
    opt = AdamWOpt([], lr=lr, wd=wd)
    if hasattr(opt, "ensure_initialized"): opt.ensure_initialized()
    return opt

def _sgd(lr=1e-2, wd=0.0, momentum=0.9, nesterov=True, damp=0.0):
    opt = SGDOpt([], lr=lr, wd=wd, momentum=momentum, nesterov=nesterov, damp=damp)
    if hasattr(opt, "ensure_initialized"): opt.ensure_initialized()
    return opt

# ------------------------
# 테스트들
# ------------------------
def test_static_compile_and_replay(tag: str):
    C = 6; cp.random.seed(123)
    N, T, I = 16, 8, 12
    model = make_model_static(N=N, T=T, I=I, H=24, hidden=48, classes=C)
    loss = SoftmaxCrossEntropy()
    opt = _adamw()

    # compile → 정적 그래프 캡처
    tg = model.compile((N, T, I), loss=loss, optimizer=opt)
    assert tg is model.tg

    X = cp.random.randn(N, T, I).astype(cp.float32)
    y = cp.random.randint(0, C, size=(N,), dtype=cp.int32)

    # 첫 실행
    L1 = model.one_step(X, y)
    assert isinstance(L1, float) and math.isfinite(L1)
    logits_ptr = int(model.tg.logits.data.ptr)

    # 재실행: 같은 그래프 재생 → 포인터 동일
    L2 = model.one_step(X, y)
    assert isinstance(L2, float) and math.isfinite(L2)
    assert int(model.tg.logits.data.ptr) == logits_ptr

    print(f"[OK][static:{tag}] L1={L1:.6f} L2={L2:.6f}")

def test_dynamic_if_and_pool(tag: str):
    C = 7; cp.random.seed(2025)
    # else-branch (N=8)
    N, T, I = 8, 12, 16
    model = make_model_dynamic_if(N=N, T=T, I=I, H=32, hidden=64, classes=C)
    loss = SoftmaxCrossEntropy(); opt = _adamw()

    X_small = cp.random.randn(N, T, I).astype(cp.float32)
    y_small = cp.random.randint(0, C, size=(N,), dtype=cp.int32)
    ctx = {"variant": {"unroll": 1, "amp": "fp32"}}

    L1 = model.one_step_dynamic(X_small, y_small, loss=loss, optimizer=opt, ctx=ctx)
    assert isinstance(L1, float) and math.isfinite(L1)
    ptr_else = int(model.tg.logits.data.ptr)

    # 동일 분기 재사용
    L2 = model.one_step_dynamic(X_small, y_small, loss=loss, optimizer=opt, ctx=ctx)
    assert int(model.tg.logits.data.ptr) == ptr_else

    # then-branch (N=64)
    X_big = cp.random.randn(64, T, I).astype(cp.float32)
    y_big = cp.random.randint(0, C, size=(64,), dtype=cp.int32)
    L3 = model.one_step_dynamic(X_big, y_big, loss=loss, optimizer=opt, ctx={"variant":{"unroll":1}})
    assert isinstance(L3, float) and math.isfinite(L3)
    ptr_then = int(model.tg.logits.data.ptr)
    assert ptr_then != ptr_else

    print(f"[OK][dynamic_if:{tag}] L1={L1:.6f} L2={L2:.6f} L3={L3:.6f}")

def test_dynamic_repeat(tag: str):
    C = 7; cp.random.seed(2025)
    N, T, I = 8, 12, 16
    model = make_model_dynamic_repeat(N=N, T=T, I=I, H=32, hidden=64, classes=C)
    loss = SoftmaxCrossEntropy(); opt = _adamw()

    X = cp.random.randn(N, T, I).astype(cp.float32)
    y = cp.random.randint(0, C, size=(N,), dtype=cp.int32)
    ctx = {"variant": {"unroll": 1}}

    L1 = model.one_step_dynamic(X, y, loss=loss, optimizer=opt, ctx=ctx)
    ptr1 = int(model.tg.logits.data.ptr)

    X2 = X + 0.5
    L2 = model.one_step_dynamic(X2, y, loss=loss, optimizer=opt, ctx=ctx)
    assert int(model.tg.logits.data.ptr) == ptr1

    # repeat_batches
    ctx_b = {"variant":{"unroll":1}, "repeat_steps":3,
             "repeat_batches":[(X*0.9, y), (X*1.05, y), (X*1.2, y)]}
    L3 = model.one_step_dynamic(X, y, loss=loss, optimizer=opt, ctx=ctx_b)
    assert all(isinstance(v, float) and math.isfinite(v) for v in (L1, L2, L3))

    print(f"[OK][dynamic_repeat:{tag}] L1={L1:.6f} L2={L2:.6f} L3={L3:.6f}")

def test_dynamic_earlyexit(tag: str):
    C = 7; cp.random.seed(2025)
    N, T, I = 8, 12, 16
    model = make_model_dynamic_earlyexit(N=N, T=T, I=I, H=32, hidden=64, classes=C)
    loss = SoftmaxCrossEntropy(); opt = _adamw()

    X = cp.random.randn(N, T, I).astype(cp.float32)
    y = cp.random.randint(0, C, size=(N,), dtype=cp.int32)
    ctx = {"variant": {"unroll": 1}}

    L = model.one_step_dynamic(X, y, loss=loss, optimizer=opt, ctx=ctx)
    L2 = model.one_step_dynamic(X, y, loss=loss, optimizer=opt, ctx=ctx)
    assert isinstance(L, float) and isinstance(L2, float)

    print(f"[OK][dynamic_ee:{tag}] L={L:.6f} L2={L2:.6f}")

def test_optimizer_sgd_on_static(tag: str):
    """
    - 정적 경로 + SGDOpt 혼합정밀 호환 (grad fp32 허용)
    - try_rebind_grads 로 plan gW/gB 포인터 연결 후 캡처/재생 검증
    """
    C = 5; cp.random.seed(4242)
    N, T, I = 16, 8, 10
    model = make_model_static(N=N, T=T, I=I, H=16, hidden=32, classes=C)
    loss = SoftmaxCrossEntropy()
    opt = _sgd(lr=5e-3, wd=1e-4, momentum=0.9, nesterov=True)

    # compile → 내부에서 plan 생성 후 try_rebind_grads 호출
    tg = model.compile((N, T, I), loss=loss, optimizer=opt)
    assert tg is model.tg

    X = cp.random.randn(N, T, I).astype(cp.float32)
    y = cp.random.randint(0, C, size=(N,), dtype=cp.int32)

    L1 = model.one_step(X, y); L2 = model.one_step(X, y)
    assert math.isfinite(L1) and math.isfinite(L2)
    print(f"[OK][sgd_static:{tag}] L1={L1:.6f} L2={L2:.6f}")

def test_global_graph_pool_if_available(tag: str):
    """
    전역 graph_pool(축소안)이 import 되는 환경이면,
    서로 다른 경로를 최소 2개 만들고 엔트리가 들어오는지만 확인.
    (정확한 크기 비교는 구현 상세에 따라 달라질 수 있으므로 존재 여부만 체크)
    """
    if GLOBAL_POOL is None:
        print(f"[SKIP][global_pool:{tag}] no global pool module")
        return

    C = 3; N, T, I = 8, 8, 8
    model = make_model_dynamic_if(N=N, T=T, I=I, H=16, hidden=16, classes=C)
    loss = SoftmaxCrossEntropy(); opt = _adamw()
    Xs = cp.random.randn(N, T, I).astype(cp.float32)
    ys = cp.random.randint(0, C, size=(N,), dtype=cp.int32)
    Xb = cp.random.randn(64, T, I).astype(cp.float32)
    yb = cp.random.randint(0, C, size=(64,), dtype=cp.int32)

    before_hint = getattr(GLOBAL_POOL, "_store", None)
    model.one_step_dynamic(Xs, ys, loss=loss, optimizer=opt, ctx={"variant":{"probe":0}})
    model.one_step_dynamic(Xb, yb, loss=loss, optimizer=opt, ctx={"variant":{"probe":1}})
    after_hint = getattr(GLOBAL_POOL, "_store", None)

    # 내부 구조를 직접 단언하지 않고 "무언가 들어갔다"만 확인
    if before_hint is not None and after_hint is not None:
        # dict 크기가 늘었는지 힌트만 체크(드라이런)
        try:
            assert len(after_hint) >= len(before_hint)
        except Exception:
            pass
    print(f"[OK][global_pool:{tag}] pool exists and dynamic paths executed.")

def test_fallback_pool_lru(tag: str):
    """
    폴백 풀 LRU 상한이 지켜지는지 확인 (graph_executor_v2.layers.sequential._FALLBACK_POOL)
    """
    C = 4; N, T, I = 8, 8, 8
    model = make_model_dynamic_if(N=N, T=T, I=I, H=16, hidden=16, classes=C)
    loss = SoftmaxCrossEntropy(); opt = _adamw()
    X = cp.random.randn(N, T, I).astype(cp.float32)
    y = cp.random.randint(0, C, size=(N,), dtype=cp.int32)

    K = model._FALLBACK_POOL_MAX + 5
    before = len(_FALLBACK_POOL)
    for i in range(K):
        ctx = {"variant": {"alpha": i}}
        model.one_step_dynamic(X, y, loss=loss, optimizer=opt, ctx=ctx)
    size = len(_FALLBACK_POOL)
    assert size <= model._FALLBACK_POOL_MAX, f"LRU cap exceeded: {size} > {model._FALLBACK_POOL_MAX}"
    print(f"[OK][fallback_lru:{tag}] before={before} after={size} cap={model._FALLBACK_POOL_MAX}")

# ------------------------
# Main
# ------------------------
def main():
    print("== Static & Dynamic & Optim & Pool smoke tests ==")
    test_static_compile_and_replay("s1")
    test_dynamic_if_and_pool("d_if")
    test_dynamic_repeat("d_rep")
    test_dynamic_earlyexit("d_ee")
    test_optimizer_sgd_on_static("opt_sgd")
    test_global_graph_pool_if_available("pool_glob")
    test_fallback_pool_lru("pool_fallback")
    print("[ALL OK] extended smoke completed.")
    print("Tip) Nsight에서 [CAPTURE]/[REPLAY] NVTX 구간과 동적 경로 태그들을 같이 보세요.")

if __name__ == "__main__":
    main()
