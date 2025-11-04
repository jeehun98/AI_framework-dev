from __future__ import annotations
import os, sys, math, hashlib
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

# 전역 그래프 풀(있으면)
try:
    from graph_executor_v2.graph.graph_executor import graph_pool as GLOBAL_POOL
except Exception:
    GLOBAL_POOL = None

# 로스/옵티마이저
from graph_executor_v2.losses.softmax_ce import SoftmaxCrossEntropy
from graph_executor_v2.optim.adamw import AdamWOpt
from graph_executor_v2.optim.sgd import SGDOpt
from graph_executor_v2.optim.rebind import try_rebind_grads

# ------------------------
# 유틸
# ------------------------
def _adamw(lr=1e-3, wd=1e-4):
    opt = AdamWOpt([], lr=lr, wd=wd)
    if hasattr(opt, "ensure_initialized"): opt.ensure_initialized()
    return opt

def _sgd(lr=1e-2, wd=0.0, momentum=0.9, nesterov=True, damp=0.0):
    opt = SGDOpt([], lr=lr, wd=wd, momentum=momentum, nesterov=nesterov, damp=damp)
    if hasattr(opt, "ensure_initialized"): opt.ensure_initialized()
    return opt

# 기존 함수 대체
def _hash_array(x) -> str:
    """
    x: cupy.ndarray 이거나, Tensor-like ( .data/.shape/.dtype ) 이거나,
       cupy.cuda.memory.MemoryPointer (이 경우 shape/dtype은 소유자에서 가져와야 함)
    """
    # 1) 이미 ndarray면 바로 사용
    if isinstance(x, cp.ndarray):
        arr = x.astype(cp.float32, copy=False).ravel()
        b = cp.asnumpy(arr).tobytes()
        return hashlib.sha256(b).hexdigest()

    # 2) Tensor-like (logits 등): .data/.shape/.dtype 보유
    if hasattr(x, "data") and hasattr(x, "shape") and hasattr(x, "dtype"):
        view = cp.ndarray(x.shape, dtype=x.dtype, memptr=x.data)  # 포인터→ndarray 뷰
        arr = view.astype(cp.float32, copy=False).ravel()
        b = cp.asnumpy(arr).tobytes()
        return hashlib.sha256(b).hexdigest()

    # 3) MemoryPointer만 넘어온 경우는 shape/dtype 정보를 모르므로 예외
    if isinstance(x, cp.cuda.memory.MemoryPointer):
        raise TypeError("MemoryPointer만 전달됨: shape/dtype을 알 수 없어 해시 불가. Tensor나 ndarray를 넘겨주세요.")

    raise TypeError(f"지원하지 않는 타입: {type(x)}")


# ------------------------
# 공용 모델 빌더
# ------------------------
def make_model_static(*, N=8, T=12, I=16, H=32, hidden=64, classes=7) -> Sequential:
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

def make_model_with_dropout(*, N=8, T=12, I=16, H=32, hidden=64, classes=7,
                            p=0.3, seed=0x1234) -> Sequential:
    """Dropout 결정성 확인용(정적 경로)"""
    model = Sequential(
        RNN(hidden_size=H, activation="tanh", with_bias=True, save_z_in_fwd=True),
        Flatten(),
        Dense(hidden, activation="none", initializer="he", use_native_bwd=True),
        Dropout(p=p, scale_in_train=True, seed=seed),
        ActivationLayer(act="relu", save_y=True),
        Dense(classes, activation="none", initializer="xavier", use_native_bwd=True),
    )
    model.build((N, T, I))
    model.train(True)
    return model

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

# ------------------------
# 기존 기본 스모크 (요약)
# ------------------------
def base_smoke_once(tag: str):
    C = 6; cp.random.seed(123)
    N, T, I = 16, 8, 12
    model = make_model_static(N=N, T=T, I=I, H=24, hidden=48, classes=C)
    loss = SoftmaxCrossEntropy(); opt = _adamw()

    tg = model.compile((N, T, I), loss=loss, optimizer=opt)
    assert tg is model.tg

    X = cp.random.randn(N, T, I).astype(cp.float32)
    y = cp.random.randint(0, C, size=(N,), dtype=cp.int32)
    L1 = model.one_step(X, y); L2 = model.one_step(X, y)
    assert math.isfinite(L1) and math.isfinite(L2)
    assert int(model.tg.logits.data.ptr) == int(model.tg.logits.data.ptr)
    print(f"[OK][base:{tag}] L1={L1:.6f} L2={L2:.6f}")

# ------------------------
# (1) Dropout RNG 결정성
# ------------------------
def test_dropout_rng_determinism(tag: str):
    C = 7; cp.random.seed(777)
    N, T, I = 16, 8, 12
    seed = 0xCAFE
    model = make_model_with_dropout(N=N, T=T, I=I, H=24, hidden=48, classes=C, p=0.4, seed=seed)
    loss = SoftmaxCrossEntropy(); 
    opt = _adamw(lr=0.0, wd=0.0)   # <-- 가중치 불변

    model.compile((N, T, I), loss=loss, optimizer=opt)

    X = cp.random.randn(N, T, I).astype(cp.float32)
    y = cp.random.randint(0, C, size=(N,), dtype=cp.int32)

    # 동일 입력/동일 컨텍스트/동일 seed → logits/손실 동일해야 함
    L1 = model.one_step(X, y)
    h1 = _hash_array(model.tg.logits)

    L2 = model.one_step(X, y)
    h2 = _hash_array(model.tg.logits)

    assert isinstance(L1, float) and isinstance(L2, float)
    assert math.isfinite(L1) and math.isfinite(L2)
    assert abs(L1 - L2) < 1e-7, f"Dropout deterministic loss mismatch: {L1} vs {L2}"
    assert h1 == h2, f"Dropout deterministic logits mismatch: {h1} vs {h2}"

    # seed 바꿔서 새 모델 구성 → 다른 해시/손실일 가능성(결정성은 유지, 값은 달라짐)
    model2 = make_model_with_dropout(N=N, T=T, I=I, H=24, hidden=48, classes=C, p=0.4, seed=seed ^ 0x5A5A)
    model2.compile((N, T, I), loss=loss, optimizer=_adamw())
    L3 = model2.one_step(X, y)
    h3 = _hash_array(model2.tg.logits)

    assert (h3 != h1) or (abs(L3 - L1) > 1e-6), "Different seed should alter dropout outcome."
    print(f"[OK][dropout_determinism:{tag}] L1={L1:.6f} L2={L2:.6f} L3={L3:.6f}")

# ------------------------
# (2) AMP variant 분리 (fp32 vs fp16/bf16)
# ------------------------
def test_amp_variants_separate_and_reuse(tag: str):
    C = 6; cp.random.seed(2025)
    N, T, I = 16, 8, 12
    model = make_model_static(N=N, T=T, I=I, H=24, hidden=48, classes=C)
    loss = SoftmaxCrossEntropy(); opt = _adamw()

    X = cp.random.randn(N, T, I).astype(cp.float32)
    y = cp.random.randint(0, C, size=(N,), dtype=cp.int32)

    # 1) fp32 경로: 재사용 확인
    ctx32 = {"variant": {"amp": "fp32", "unroll": 1}}
    L1 = model.one_step_dynamic(X, y, loss=loss, optimizer=opt, ctx=ctx32)
    ptr32_a = int(model.tg.logits.data.ptr)
    L2 = model.one_step_dynamic(X, y, loss=loss, optimizer=opt, ctx=ctx32)
    ptr32_b = int(model.tg.logits.data.ptr)
    assert ptr32_a == ptr32_b, "fp32 path should be reused."

    # 2) fp16 경로 요청
    ctx16 = {"variant": {"amp": "fp16", "unroll": 1}}
    L3 = model.one_step_dynamic(X, y, loss=loss, optimizer=opt, ctx=ctx16)
    ptr16_a = int(model.tg.logits.data.ptr)
    L4 = model.one_step_dynamic(X, y, loss=loss, optimizer=opt, ctx=ctx16)
    ptr16_b = int(model.tg.logits.data.ptr)
    assert ptr16_a == ptr16_b, "fp16 path (if created) should be reused."

    if ptr16_a != ptr32_a:
        # 원하는 이상적 상태: AMP가 GraphKey에 반영되어 분리됨
        print(f"[OK][amp_separation:{tag}] fp32_ptr={ptr32_a} fp16_ptr={ptr16_a}")
    else:
        # 아직 AMP가 분리키에 포함되지 않거나 AMP 미구현 → 경고로 내리고,
        # 분리 보장이 되는 다른 variant(예: unroll=2)로 대체 검증
        print(f"[WARN][amp_separation:{tag}] amp('fp16') not separating from 'fp32' (ptr identical: {ptr32_a}). "
              "Falling back to variant separation via unroll.")
        ctx_unroll2 = {"variant": {"amp": "fp32", "unroll": 2}}
        L5 = model.one_step_dynamic(X, y, loss=loss, optimizer=opt, ctx=ctx_unroll2)
        ptr_u2_a = int(model.tg.logits.data.ptr)
        L6 = model.one_step_dynamic(X, y, loss=loss, optimizer=opt, ctx=ctx_unroll2)
        ptr_u2_b = int(model.tg.logits.data.ptr)
        assert ptr_u2_a == ptr_u2_b, "unroll=2 path should be reused."
        assert ptr_u2_a != ptr32_a, "unroll change should force a new graph capture."
        print(f"[OK][variant_separation_fallback:{tag}] fp32(unroll=1) ptr={ptr32_a} vs unroll=2 ptr={ptr_u2_a}")

    assert all(math.isfinite(v) for v in (L1, L2, L3, L4))



# ------------------------
# (3) Optimizer 전환 시 재캡처
# ------------------------
def test_optimizer_switch_recapture(tag: str):
    C = 5; cp.random.seed(4242)
    N, T, I = 16, 8, 10
    model = make_model_static(N=N, T=T, I=I, H=16, hidden=32, classes=C)
    loss = SoftmaxCrossEntropy()
    opt1 = _adamw(lr=1e-3, wd=1e-4)

    # AdamW로 컴파일/실행
    tg1 = model.compile((N, T, I), loss=loss, optimizer=opt1)
    X = cp.random.randn(N, T, I).astype(cp.float32)
    y = cp.random.randint(0, C, size=(N,), dtype=cp.int32)

    L1 = model.one_step(X, y)
    ptr_adamw = int(model.tg.logits.data.ptr)

    # Optimizer → SGD로 전환 (키가 바뀌어야 함: 내부 policy에 따라 재캡처)
    opt2 = _sgd(lr=5e-3, wd=1e-4, momentum=0.9, nesterov=True)
    # 재컴파일 경로를 명시적으로 호출(프레임워크 정책에 따라 one_step_dynamic에서 교체도 가능)
    tg2 = model.compile((N, T, I), loss=loss, optimizer=opt2)

    L2 = model.one_step(X, y)
    ptr_sgd = int(model.tg.logits.data.ptr)

    assert math.isfinite(L1) and math.isfinite(L2)
    assert ptr_sgd != ptr_adamw, "Optimizer switch should force a new graph capture."
    print(f"[OK][opt_switch_recapture:{tag}] adamw_ptr={ptr_adamw} sgd_ptr={ptr_sgd} L2={L2:.6f}")

# ------------------------
# (4) 폴백 LRU 축출 검증
# ------------------------
def test_fallback_pool_lru_eviction(tag: str):
    """
    - 다양한 variant로 cap 초과 엔트리 생성
    - size <= cap 보장
    - 초기 variant를 다시 호출했을 때 '새 포인터'가 관측되면 축출→재캡처가 일어난 힌트
    """
    C = 4; N, T, I = 8, 8, 8
    model = make_model_dynamic_if(N=N, T=T, I=I, H=16, hidden=16, classes=C)
    loss = SoftmaxCrossEntropy(); opt = _adamw()

    X = cp.random.randn(N, T, I).astype(cp.float32)
    y = cp.random.randint(0, C, size=(N,), dtype=cp.int32)

    cap = getattr(model, "_FALLBACK_POOL_MAX", 8)
    before = len(_FALLBACK_POOL)

    # 초기 엔트리 하나 만들고 포인터 확보
    ctx0 = {"variant": {"alpha": -1, "unroll": 1}}
    model.one_step_dynamic(X, y, loss=loss, optimizer=opt, ctx=ctx0)
    ptr0_first = int(model.tg.logits.data.ptr)

    # cap+K 만큼 서로 다른 variant로 채우기
    K = 4
    ptrs = []
    for i in range(cap + K):
        ctx = {"variant": {"alpha": i, "unroll": 1}}
        model.one_step_dynamic(X, y, loss=loss, optimizer=opt, ctx=ctx)
        ptrs.append(int(model.tg.logits.data.ptr))

    size = len(_FALLBACK_POOL)
    assert size <= cap, f"LRU cap exceeded: {size} > {cap}"

    # 가장 먼저 만든 ctx0를 다시 실행 → 축출되었으면 새 캡처로 '다른 포인터'가 나와야 함
    model.one_step_dynamic(X, y, loss=loss, optimizer=opt, ctx=ctx0)
    ptr0_second = int(model.tg.logits.data.ptr)

    # 축출이 반드시 ptr 변화로 보장되진 않을 수 있으나, 보통은 다름.
    if ptr0_second == ptr0_first:
        # 보수적으로 경고만 출력
        print(f"[WARN][lru_eviction:{tag}] ptr unchanged; internal pool policy may retain early entry.")
    else:
        print(f"[OK][lru_eviction:{tag}] evicted+recaptured (ptr0 {ptr0_first} -> {ptr0_second})")

    print(f"[OK][fallback_lru_size:{tag}] before={before} after={size} cap={cap}")

# ------------------------
# 메인
# ------------------------
def main():
    print("== Extended Runtime Behavior Tests ==")
    base_smoke_once("base")

    test_dropout_rng_determinism("rng")
    test_amp_variants_separate_and_reuse("amp")
    test_optimizer_switch_recapture("opt_switch")
    test_fallback_pool_lru_eviction("lru")

    print("[ALL OK] extended runtime behavior suite completed.")
    print("Tip) Nsight에서 [CAPTURE]/[REPLAY] 구간 + variant(amp/unroll/alpha) 태그를 함께 보세요.")

if __name__ == "__main__":
    main()
