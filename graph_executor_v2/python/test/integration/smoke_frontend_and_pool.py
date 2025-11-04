# -*- coding: utf-8 -*-
# File: graph_executor_v2/python/test/integration/smoke_frontend_and_pool.py

from __future__ import annotations
import os, sys, math
import cupy as cp

THIS = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(THIS, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# 모델/레이어/옵티마이저/로스
from graph_executor_v2.layers.sequential import Sequential, _FALLBACK_POOL
from graph_executor_v2.layers.rnn import RNN
from graph_executor_v2.layers.flatten import Flatten
from graph_executor_v2.layers.dense_gemm import Dense
from graph_executor_v2.layers.activations import ActivationLayer
from graph_executor_v2.layers.dropout import Dropout
from graph_executor_v2.layers.conditional import If, Repeat, EarlyExit

from graph_executor_v2.losses.softmax_ce import SoftmaxCrossEntropy
from graph_executor_v2.optim.adamw import AdamWOpt
from graph_executor_v2.optim.sgd import SGDOpt

from graph_executor_v2.graph.graph_executor import graph_pool as GP

# ===== 공용 헬퍼 =====
def _adamw(lr=1e-3, wd=1e-4):
    opt = AdamWOpt([], lr=lr, wd=wd)
    if hasattr(opt, "ensure_initialized"): opt.ensure_initialized()
    return opt

def _sgd(lr=5e-3, wd=1e-4, momentum=0.9, nesterov=True):
    opt = SGDOpt([], lr=lr, wd=wd, momentum=momentum, nesterov=nesterov)
    if hasattr(opt, "ensure_initialized"): opt.ensure_initialized()
    return opt

def _logits_ptr(tg) -> int:
    """TrainGraph의 logits 버퍼 포인터를 안전하게 획득."""
    io = getattr(tg, "io", None)
    if io is None:
        io = getattr(tg, "_io")  # 일부 구현에서 private일 수 있음
    logits = io["logits"]
    raw = getattr(logits, "data", logits)
    return int(raw.ptr)

class ToyLoader:
    """간단한 배치 제너레이터 (고정 shape)"""
    def __init__(self, *, batches=20, N=16, T=8, I=12, C=6, seed=0):
        self.batches = batches
        self.N, self.T, self.I, self.C = N, T, I, C
        self.rng = cp.random.RandomState(seed)

    def __iter__(self):
        for _ in range(self.batches):
            X = self.rng.randn(self.N, self.T, self.I).astype(cp.float32)
            y = self.rng.randint(0, self.C, size=(self.N,), dtype=cp.int32)
            yield X, y

    def peek_shape(self):
        return (self.N, self.T, self.I)

# ===== 테스트용 모델 빌더 =====
def make_model_dynamic_if(*, N=8, T=12, I=16, H=32, hidden=64, classes=7,
                          p_drop_then=0.3, p_drop_else=0.0, seed=0x2222) -> Sequential:
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

def make_model_static(*, N=16, T=8, I=12, H=24, hidden=48, classes=6) -> Sequential:
    return Sequential(
        RNN(hidden_size=H, activation="tanh", with_bias=True, save_z_in_fwd=True),
        Flatten(),
        Dense(hidden, activation="none", initializer="he", use_native_bwd=True),
        ActivationLayer(act="relu", save_y=True),
        Dense(classes, activation="none", initializer="xavier", use_native_bwd=True),
    ).train(True)

# ===== 개별 테스트 =====
def test_fit_dynamic_and_pool_stats(tag="fit_dyn"):
    cp.random.seed(123)
    C = 6
    loader = ToyLoader(batches=30, N=16, T=8, I=12, C=C, seed=123)
    model = make_model_dynamic_if(N=16, T=8, I=12, H=24, hidden=48, classes=C)
    loss = SoftmaxCrossEntropy()
    opt = _adamw()

    ctx = {"variant": {"unroll": 1, "amp": "fp32"}}
    model.fit(loader, loss=loss, optimizer=opt, ctx=ctx, epochs=1, use_dynamic=True, report_every=0)

    assert model.tg is not None
    stats = model.pool_stats()
    assert isinstance(stats, dict) and ("fallback_size" in stats)
    print(f"[OK][{tag}] pool_stats={stats}")

def test_warmup_and_replay_hotloop(tag="hotloop"):
    cp.random.seed(2025)
    C = 6
    model = make_model_dynamic_if(N=16, T=8, I=12, H=24, hidden=48, classes=C)
    loss = SoftmaxCrossEntropy()
    opt = _adamw()

    X0 = cp.random.randn(16, 8, 12).astype(cp.float32)
    y0 = cp.random.randint(0, C, size=(16,), dtype=cp.int32)
    X1 = X0 * 1.05
    y1 = y0

    ctx32_u1 = {"variant": {"unroll": 1, "amp": "fp32"}}
    warms = model.warmup([(X0, y0, ctx32_u1)], loss=loss, optimizer=opt)

    expected_var = tuple(sorted(
        (str(k), Sequential._freeze_value(v)) for k, v in ctx32_u1["variant"].items()
    ))

    assert len(warms) >= 1, "warmup should return at least one captured TrainGraph"
    assert expected_var in warms, f"expected variant key {expected_var} not found in warmup result"

    before_ptr = _logits_ptr(model.tg)
    batches = [(X0, y0), (X1, y1), (X0, y0), (X1, y1)]
    model.replay_loop(batches, steps=len(batches))
    after_ptr = _logits_ptr(model.tg)

    assert before_ptr == after_ptr, "hotloop should reuse the same captured graph/logits buffer"
    print(f"[OK][{tag}] replay kept same ptr={before_ptr}")

def test_amp_variant_separation_now_on_key(tag="amp_key"):
    """_make_pool_key에 amp 문자열이 반영되었는지 확인 (fp32 vs fp16 분리)."""
    cp.random.seed(7)
    C = 6
    model = make_model_static(N=16, T=8, I=12, H=24, hidden=48, classes=C)
    loss = SoftmaxCrossEntropy()
    opt = _adamw()

    X = cp.random.randn(16, 8, 12).astype(cp.float32)
    y = cp.random.randint(0, C, size=(16,), dtype=cp.int32)

    ctx32 = {"variant": {"unroll": 1, "amp": "fp32"}}
    L32a = model.one_step_dynamic(X, y, loss=loss, optimizer=opt, ctx=ctx32)
    ptr32 = _logits_ptr(model.tg)
    L32b = model.one_step_dynamic(X, y, loss=loss, optimizer=opt, ctx=ctx32)
    ptr32_b = _logits_ptr(model.tg)
    assert ptr32 == ptr32_b and math.isfinite(L32a) and math.isfinite(L32b)

    ctx16 = {"variant": {"unroll": 1, "amp": "fp16"}}
    L16a = model.one_step_dynamic(X, y, loss=loss, optimizer=opt, ctx=ctx16)
    ptr16 = _logits_ptr(model.tg)
    assert ptr16 != ptr32, "amp('fp16') should now produce a different GraphKey/graph"

    L16b = model.one_step_dynamic(X, y, loss=loss, optimizer=opt, ctx=ctx16)
    ptr16_b = _logits_ptr(model.tg)
    assert ptr16 == ptr16_b and math.isfinite(L16a) and math.isfinite(L16b)
    print(f"[OK][{tag}] fp32 ptr={ptr32}, fp16 ptr={ptr16}")

def test_static_fit_mode(tag="fit_static"):
    """use_dynamic=False 경로에서 compile → one_step 재생 확인."""
    cp.random.seed(99)
    C = 5
    model = make_model_static(N=16, T=8, I=12, H=20, hidden=40, classes=C)
    loss = SoftmaxCrossEntropy()
    opt = _sgd()

    loader = ToyLoader(batches=5, N=16, T=8, I=12, C=C, seed=99)
    model.fit(loader, loss=loss, optimizer=opt, use_dynamic=False,
              static_input_shape=loader.peek_shape(), epochs=1, report_every=0)

    X, y = next(iter(ToyLoader(batches=1, N=16, T=8, I=12, C=C, seed=101)))
    L1 = model.one_step(X, y)
    ptr_a = _logits_ptr(model.tg)
    L2 = model.one_step(X, y)
    ptr_b = _logits_ptr(model.tg)
    assert ptr_a == ptr_b and math.isfinite(L1) and math.isfinite(L2)
    print(f"[OK][{tag}] static replay ptr={ptr_a}")

def test_fallback_lru_with_monkeypatch(tag="lru_fallback"):
    """
    글로벌 풀 대신 폴백 LRU를 강제해 축출을 검증.
    - graph_pool을 임시 무력화
    - cap을 2로 낮춰 여러 변형을 넣고 축출 유도
    """
    cp.random.seed(4242)
    C = 4
    model = make_model_static(N=8, T=6, I=10, H=16, hidden=24, classes=C)
    loss = SoftmaxCrossEntropy()
    opt = _adamw()

    import graph_executor_v2.layers.sequential as seqmod
    old_gp = getattr(seqmod, "graph_pool", None)
    old_cap = model._FALLBACK_POOL_MAX
    try:
        seqmod.graph_pool = None  # force fallback
        model._FALLBACK_POOL_MAX = 2

        X = cp.random.randn(8, 6, 10).astype(cp.float32)
        y = cp.random.randint(0, C, size=(8,), dtype=cp.int32)

        ptrs = []
        for amp in ("fp32", "fp16", "bf16", "fp32"):  # 3개 이상으로 넘겨 cap=2 초과
            ctx = {"variant": {"unroll": 1, "amp": amp}}
            _ = model.one_step_dynamic(X, y, loss=loss, optimizer=opt, ctx=ctx)
            p = _logits_ptr(model.tg)
            ptrs.append((amp, p))

        size_after = len(_FALLBACK_POOL)
        assert size_after <= model._FALLBACK_POOL_MAX
        print(f"[OK][{tag}] fallback_size={size_after} cap={model._FALLBACK_POOL_MAX} ptrs={ptrs}")

        removed = model.evict_pool(predicate=lambda k,e: any("bf16" in str(k) for _ in [0]))
        print(f"[OK][{tag}] evict_pool removed={removed}")
    finally:
        model._FALLBACK_POOL_MAX = old_cap
        seqmod.graph_pool = old_gp

def test_pool_evicted_summaries(tag="evict_summary"):
    # (cap을 낮추거나 다양한 키를 넣어 evict 발생 유도하는 시나리오 실행 후)
    st = GP.stats() if hasattr(GP, "stats") else {}
    summaries = st.get("last_evicted_summaries", [])
    assert isinstance(summaries, list)
    if summaries:
        s0 = summaries[-1]
        assert "key_hex" in s0
    print(f"[OK][{tag}] last_evicted_summaries_len={len(summaries)}")

def test_get_graph_key_preview_and_real_capture(tag="key_preview"):
    """preview로 본 키와 동일 ctx에서 실제 캡처가 안정적으로 재사용되는지만 확인."""
    cp.random.seed(1717)
    C = 6
    model = make_model_dynamic_if(N=16, T=8, I=12, H=24, hidden=48, classes=C)
    loss = SoftmaxCrossEntropy(); opt = _adamw()
    X = cp.random.randn(16, 8, 12).astype(cp.float32)
    y = cp.random.randint(0, C, size=(16,), dtype=cp.int32)
    ctx = {"variant": {"unroll": 1, "amp": "fp32"}}

    key1 = model.get_graph_key_preview(X, ctx=ctx, loss=loss)
    L1 = model.one_step_dynamic(X, y, loss=loss, optimizer=opt, ctx=ctx)
    p1 = _logits_ptr(model.tg)
    L2 = model.one_step_dynamic(X, y, loss=loss, optimizer=opt, ctx=ctx)
    p2 = _logits_ptr(model.tg)
    assert p1 == p2 and math.isfinite(L1) and math.isfinite(L2)
    print(f"[OK][{tag}] preview_key_type={type(key1).__name__} ptr={p1}")

def test_telemetry_counters(tag="telemetry"):
    cp.random.seed(555)
    C = 6
    from graph_executor_v2.graph.graph_executor import graph_pool as GP2

    model = make_model_static(N=16, T=8, I=12, H=24, hidden=48, classes=C)
    loss = SoftmaxCrossEntropy(); opt = _adamw()
    X = cp.random.randn(16, 8, 12).astype(cp.float32)
    y = cp.random.randint(0, C, size=(16,), dtype=cp.int32)

    if hasattr(GP2, "reset_stats"):
        GP2.reset_stats()
    model.reset_telemetry()

    nonce = int(cp.random.randint(0, 1 << 30))
    ctx = {"variant": {"unroll": 1, "amp": "fp32", "nonce": nonce}}

    _ = model.one_step_dynamic(X, y, loss=loss, optimizer=opt, ctx=ctx)  # miss→capture
    _ = model.one_step_dynamic(X, y, loss=loss, optimizer=opt, ctx=ctx)  # hit

    tm = model.telemetry()
    ps = model.pool_stats()

    assert tm["capture_count"] >= 1, f"expected at least one capture, got {tm}"
    assert tm["replay_count"] >= 2, f"expected at least two replays, got {tm}"
    assert tm["pool_hit"] >= 1 and tm["pool_miss"] >= 1, f"expected both hit and miss, got {tm}"

    if ps.get("global", False) and all(k in ps for k in ("global_hits","global_misses")):
        # 텔레메트리 패치가 적용된 환경에서만 검증
        if ps["global_hits"] is not None and ps["global_misses"] is not None:
            assert ps["global_hits"] >= 1 and ps["global_misses"] >= 1

    print(f"[OK][{tag}] local_tm={tm} global={ps.get('global',False)} "
          f"global_stats={{hits:{ps.get('global_hits')}, misses:{ps.get('global_misses')}, "
          f"puts:{ps.get('global_puts')}, evicts:{ps.get('global_evicts')}}}")

def test_global_pool_eviction_summary(tag="global_evict_summary"):
    """전역 풀의 last_evicted_summaries가 실제로 채워지는지 강제 확인.

    핵심: graph_executor와 layers.sequential 양쪽의 graph_pool을 '같은 인스턴스'로 패치해야
    Sequential이 실제로 그 풀에 put/get을 수행합니다.
    """
    import graph_executor_v2.graph.graph_executor as ge
    from graph_executor_v2.graph.graph_executor import MultiGraphPool
    import graph_executor_v2.layers.sequential as seqmod

    old_gp_ge = ge.graph_pool
    old_gp_seq = getattr(seqmod, "graph_pool", None)

    new_pool = MultiGraphPool(max_size=2)  # 매우 작은 풀로 축출 유도
    ge.graph_pool = new_pool
    seqmod.graph_pool = new_pool  # ★ layers.sequential도 동일 인스턴스로 바꿔줌

    try:
        C = 5
        model = make_model_static(N=16, T=8, I=12, H=16, hidden=32, classes=C)
        loss = SoftmaxCrossEntropy(); opt = _adamw()
        X = cp.random.randn(16,8,12).astype(cp.float32)
        y = cp.random.randint(0, C, size=(16,), dtype=cp.int32)

        # 텔레메트리 초기화(선택)
        if hasattr(new_pool, "reset_stats"):
            new_pool.reset_stats()

        # 서로 다른 variant로 3회 miss → cap=2라 최소 1회 evict 발생해야 함
        for amp in ("fp32", "fp16", "bf16"):
            ctx = {"variant": {"unroll": 1, "amp": amp}}
            _ = model.one_step_dynamic(X, y, loss=loss, optimizer=opt, ctx=ctx)

        st = new_pool.stats() if hasattr(new_pool, "stats") else {}
        ev = st.get("evicts", 0)
        summaries = st.get("last_evicted_summaries", [])

        assert ev >= 1, f"expected at least one eviction, stats={st}"
        assert isinstance(summaries, list) and len(summaries) >= 1, "expected non-empty evicted summaries"
        assert "key_hex" in summaries[-1], f"unexpected summary shape: {summaries[-1]}"

        print(f"[OK][{tag}] evicts={ev} last={summaries[-1]}")
    finally:
        # 원복 철저히
        ge.graph_pool = old_gp_ge
        seqmod.graph_pool = old_gp_seq


# ===== 메인 =====
def main():
    print("== Frontend & Pool/Key integration tests ==")
    test_fit_dynamic_and_pool_stats("fit_dyn")
    test_warmup_and_replay_hotloop("hotloop")
    test_amp_variant_separation_now_on_key("amp_key")
    test_static_fit_mode("fit_static")
    test_fallback_lru_with_monkeypatch("lru_fallback")
    test_get_graph_key_preview_and_real_capture("key_preview")
    test_telemetry_counters("telemetry")
    print("[ALL OK] frontend + pool/key suite completed.")
    print("Tip) Nsight에서 [CAPTURE]/[REPLAY] 범위와 variant(amp/unroll) 태그를 함께 확인하세요.")

    test_pool_evicted_summaries("evict_summary")
    test_global_pool_eviction_summary("global_evict_summary")

if __name__ == "__main__":
    main()
