# -*- coding: utf-8 -*-
# File: graph_executor_v2/python/test/integration/smoke_capture_plan_and_static_graph.py

from __future__ import annotations
import os, sys, math, json
import cupy as cp

# 🔴 캡처 전에 디버그 표면 ON (plan/key/tags 노출)
os.environ.setdefault("GEV2_EXPOSE_DEBUG", "1")

THIS = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(THIS, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ===== 프레임워크 import =====
from graph_executor_v2.layers.sequential import Sequential
from graph_executor_v2.layers.rnn import RNN
from graph_executor_v2.layers.flatten import Flatten
from graph_executor_v2.layers.dense_gemm import Dense
from graph_executor_v2.layers.activations import ActivationLayer
from graph_executor_v2.layers.dropout import Dropout
from graph_executor_v2.layers.conditional import If

from graph_executor_v2.losses.softmax_ce import SoftmaxCrossEntropy
from graph_executor_v2.optim.adamw import AdamWOpt
from graph_executor_v2.optim.sgd import SGDOpt


# ===== 유틸 =====
def _adamw(lr=1e-3, wd=1e-4):
    opt = AdamWOpt([], lr=lr, wd=wd)
    if hasattr(opt, "ensure_initialized"):
        opt.ensure_initialized()
    return opt

def _sgd(lr=5e-3, wd=1e-4, momentum=0.9, nesterov=True):
    opt = SGDOpt([], lr=lr, wd=wd, momentum=momentum, nesterov=nesterov)
    if hasattr(opt, "ensure_initialized"):
        opt.ensure_initialized()
    return opt

def _ptr_of_logits(tg) -> int:
    io = getattr(tg, "io", None) or getattr(tg, "_io", None)
    assert io is not None and "logits" in io, "TrainGraph.io에 logits가 없습니다."
    logits = io["logits"]
    raw = getattr(logits, "data", logits)
    return int(raw.ptr)

def _print_kv(title, d):
    print(f"--- {title} ---")
    for k, v in d.items():
        print(f"{k}: {v}")

def _freeze(v):
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return tuple(_freeze(x) for x in v)
    if isinstance(v, dict):
        return tuple(sorted((str(k), _freeze(val)) for k, val in v.items()))
    return str(v)

def _normalize_key(k):
    """
    다양한 구현(튜플/GraphKey 객체/기타)을 공통 튜플로 정규화.
    형태: ('dyn'|'static', shape, dtype, layout, branch_id, variant_tuple_of_pairs)
    """
    if k is None:
        return None
    # 이미 튜플 기반
    if isinstance(k, tuple) and len(k) >= 6 and isinstance(k[5], (tuple, list)):
        tag = str(k[0])
        shape = tuple(k[1]) if isinstance(k[1], (list, tuple)) else tuple()
        dtype = str(k[2])
        layout = str(k[3])
        branch_id = str(k[4] or "default")
        variant = _freeze(k[5])
        return (tag, shape, dtype, layout, branch_id, variant)

    # 객체 기반(GraphKey 등) 추정
    sig = getattr(k, "signature", None)
    shape = tuple(getattr(sig, "shape", ())) if sig is not None else tuple()
    dtype = str(getattr(sig, "dtype", "")) if sig is not None else ""
    layout = str(getattr(sig, "layout", "")) if sig is not None else ""
    branch_id = str(getattr(k, "branch_id", getattr(k, "branch", "default")) or "default")
    variant = getattr(k, "variant", getattr(k, "variant_dict", ()))
    variant = _freeze(variant)
    tag = getattr(k, "tag", "dyn")
    return (str(tag), shape, dtype, layout, branch_id, variant)

def _variant_to_dict(variant_tuple):
    """
    variant_tuple: (('amp','fp32'), ('dtype','float32'), ...)
    → dict 로 변환
    """
    vd = {}
    if isinstance(variant_tuple, (list, tuple)):
        for item in variant_tuple:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                vd[str(item[0])] = item[1]
    return vd


# ===== 테스트용 모델 =====
def make_model_static(*, N=16, T=8, I=12, H=24, hidden=48, classes=6) -> Sequential:
    return Sequential(
        RNN(hidden_size=H, activation="tanh", with_bias=True, save_z_in_fwd=True),
        Flatten(),
        Dense(hidden, activation="none", initializer="he", use_native_bwd=True),
        ActivationLayer(act="relu", save_y=True),
        Dense(classes, activation="none", initializer="xavier", use_native_bwd=True),
    ).train(True)

def make_model_if(*, N=16, T=8, I=12, H=24, hidden=48, classes=6) -> Sequential:
    then_block = Sequential(
        Dropout(p=0.2, scale_in_train=True, seed=0x1111),
        Dense(hidden, activation="none", initializer="he", use_native_bwd=True),
        ActivationLayer(act="relu", save_y=True, name="ThenAct"),
    )
    else_block = Sequential(
        Dense(hidden, activation="none", initializer="he", use_native_bwd=True),
        ActivationLayer(act="relu", save_y=True, name="ElseAct"),
    )
    m = Sequential(
        RNN(hidden_size=H, activation="tanh", with_bias=True, save_z_in_fwd=True),
        Flatten(),
        If(lambda X, ctx: X.shape[0] >= 32, then_block=then_block, else_block=else_block),
        Dense(classes, activation="none", initializer="xavier", use_native_bwd=True),
    )
    m.train(True)
    return m


# ===== 1) Capture Plan Dump & Invariants =====
def test_capture_plan_dump_and_invariants(tag="plan_dump"):
    cp.random.seed(2025)
    C = 6
    model = make_model_static(N=16, T=8, I=12, H=24, hidden=48, classes=C)
    loss = SoftmaxCrossEntropy()
    opt = _adamw()

    X = cp.random.randn(16,8,12).astype(cp.float32)
    y = cp.random.randint(0, C, size=(16,), dtype=cp.int32)

    ctx = {"variant": {"unroll": 1, "amp": "fp32"}}
    L = model.one_step_dynamic(X, y, loss=loss, optimizer=opt, ctx=ctx)
    assert math.isfinite(L)

    # Plan 핸들 획득
    plan = getattr(model, "_last_plan", None)
    if plan is None and getattr(model, "tg", None) is not None:
        plan = getattr(model.tg, "plan", None)
    if plan is None and hasattr(model, "debug_capture_plan"):
        try:
            plan = model.debug_capture_plan()
        except Exception:
            plan = None

    info = {}
    if plan is not None:
        per_layer = list(getattr(plan, "per_layer", []))
        tensor_count = len(per_layer)
        workspace_count = sum(1 for per in per_layer if getattr(per, "work", None) is not None)

        exec_plan = getattr(plan, "exec_plan", None)
        stream_count = int(getattr(exec_plan, "num_streams", 0) or 0)

        rng = {"seed": getattr(plan, "seed", None), "step": getattr(plan, "rng_step", None)}
        info.update({
            "tensor_count": tensor_count,
            "workspace_count": workspace_count,
            "stream_count": stream_count,
            "rng": rng,
        })
    else:
        info.update({"note": "plan handle not exposed on this build"})

    _print_kv(f"[OK][{tag}] CapturePlan summary", info)

    # 가벼운 불변식 점검: y 포인터 유효성
    if plan is not None and info.get("tensor_count", 0) > 0:
        check_cnt = 0
        for per in getattr(plan, "per_layer", []):
            y = getattr(per, "y", None)
            if y is None:
                continue
            raw = getattr(y, "data", y)
            if hasattr(raw, "ptr"):
                assert int(raw.ptr) != 0, "some per-layer y buffer has null ptr"
                check_cnt += 1
                if check_cnt >= 5:
                    break
        print(f"[OK][{tag}] basic tensor ptr checks={check_cnt}")


# ===== 2) Record Step Graph & NVTX/ptr 안정성 =====
def test_record_step_graph_and_ptr_stability(tag="record_graph"):
    cp.random.seed(7)
    C = 6
    model = make_model_static(N=16, T=8, I=12, H=24, hidden=48, classes=C)
    loss = SoftmaxCrossEntropy()
    opt = _adamw()

    X = cp.random.randn(16,8,12).astype(cp.float32)
    y = cp.random.randint(0, C, size=(16,), dtype=cp.int32)
    ctx = {"variant": {"unroll": 1, "amp": "fp32"}}

    _ = model.one_step_dynamic(X, y, loss=loss, optimizer=opt, ctx=ctx)   # miss→capture
    ptr_a = _ptr_of_logits(model.tg)
    _ = model.one_step_dynamic(X, y, loss=loss, optimizer=opt, ctx=ctx)   # hit→replay
    ptr_b = _ptr_of_logits(model.tg)

    assert ptr_a == ptr_b, "logits buffer pointer changed across replay"
    out = {"ptr": ptr_a}

    tags = getattr(model.tg, "tags", None)
    if isinstance(tags, dict) and tags:
        out["nvtx"] = tags

    _print_kv(f"[OK][{tag}] graph capture & replay stability", out)


# ===== 2.5) Repeat 경로: RNG step & ptr 안정성 =====
def test_repeat_rng_and_ptr_stability(tag="repeat_rng"):
    cp.random.seed(123)
    C = 6
    model = make_model_static(N=16, T=8, I=12, H=24, hidden=48, classes=C)
    loss = SoftmaxCrossEntropy()
    opt  = _adamw()

    X = cp.random.randn(16,8,12).astype(cp.float32)
    y = cp.random.randint(0, C, size=(16,), dtype=cp.int32)

    ctx = {"variant": {"unroll": 1, "amp": "fp32"}, "repeat_steps": 3}
    _ = model.one_step_dynamic(X, y, loss=loss, optimizer=opt, ctx=ctx)  # miss → capture & replay x3
    ptr_a = _ptr_of_logits(model.tg)

    rng1 = dict(ctx.get("rng", {}))
    step1 = int(rng1.get("step", -1))
    assert step1 == 3, f"rng.step should be 3 after first run (T=3), got {step1}"

    _ = model.one_step_dynamic(X, y, loss=loss, optimizer=opt, ctx=ctx)  # hit → replay x3
    ptr_b = _ptr_of_logits(model.tg)
    assert ptr_a == ptr_b, "logits buffer pointer changed across repeat replay"

    rng2 = dict(ctx.get("rng", {}))
    step2 = int(rng2.get("step", -1))
    assert step2 == 6, f"rng.step should be 6 after two runs (3+3), got {step2}"

    plan = getattr(model.tg, "plan", None)
    meta = {}
    if plan is not None:
        meta = {"plan_seed": getattr(plan, "seed", None), "plan_step": getattr(plan, "rng_step", None)}

    _print_kv(f"[OK][{tag}] repeat path replay & rng", {
        "ptr": ptr_a,
        "rng_after_first": rng1,
        "rng_after_second": rng2,
        "plan_meta": meta
    })


# ===== 3) Graph Key Preview == Real Key =====
def test_graph_key_preview_matches_real(tag="key_preview"):
    """
    핵심 필드(shape/dtype/layout/branch_id)와 variant 서브셋(amp/dtype/loss_kind/training/unroll/path_fp)을 비교.
    추가 내부 필드가 실키에 존재해도 통과하도록 부분집합 비교를 사용.
    """
    cp.random.seed(1717)
    C = 6
    model = make_model_static(N=16, T=8, I=12, H=24, hidden=48, classes=C)
    loss = SoftmaxCrossEntropy(); opt = _adamw()
    X = cp.random.randn(16, 8, 12).astype(cp.float32)
    y = cp.random.randint(0, C, size=(16,), dtype=cp.int32)
    ctx = {"variant": {"unroll": 1, "amp": "fp32"}}

    preview = model.get_graph_key_preview(X, ctx=ctx, loss=loss) if hasattr(model, "get_graph_key_preview") else None

    L1 = model.one_step_dynamic(X, y, loss=loss, optimizer=opt, ctx=ctx)
    L2 = model.one_step_dynamic(X, y, loss=loss, optimizer=opt, ctx=ctx)
    assert math.isfinite(L1) and math.isfinite(L2)

    real_key = None
    if getattr(model, "tg", None) is not None:
        real_key = getattr(model.tg, "key", None) or getattr(model.tg, "graph_key", None)

    n_preview = _normalize_key(preview) if preview is not None else None
    n_real = _normalize_key(real_key) if real_key is not None else None

    out = {
        "preview_type": type(preview).__name__ if preview is not None else None,
        "real_key_type": type(real_key).__name__ if real_key is not None else None,
        "preview_norm": n_preview,
        "real_key_norm": n_real,
    }
    _print_kv(f"[OK][{tag}] key preview vs real (normalized)", out)

    # 비교: 핵심 필드 동일성
    if n_preview is not None and n_real is not None:
        p_tag, p_shape, p_dtype, p_layout, p_branch, p_var = n_preview
        r_tag, r_shape, r_dtype, r_layout, r_branch, r_var = n_real

        assert p_shape == r_shape, "shape mismatch"
        assert p_dtype == r_dtype, "dtype mismatch"
        assert p_layout == r_layout, "layout mismatch"
        assert str(p_branch) == str(r_branch), "branch_id mismatch"

        pvd = _variant_to_dict(p_var)
        rvd = _variant_to_dict(r_var)
        must_keys = {"amp", "dtype", "loss_kind", "training", "unroll", "path_fp"}

        for k in must_keys:
            if k not in pvd:
                continue
            pv = pvd[k]
            rv = rvd.get(k, None)
            if k == "path_fp":
                if pv in ((), None):
                    continue
                assert rv == pv, f"variant field 'path_fp' mismatch: {rv} != {pv}"
            else:
                assert rv == pv, f"variant field '{k}' mismatch: {rv} != {pv}"


# ===== 3.5) If/Else 분기: GraphKey 분리 검증 =====
def test_if_branch_key_isolation(tag="if_branch"):
    cp.random.seed(456)
    C = 6
    model = make_model_if(N=16, T=8, I=12, H=24, hidden=48, classes=C)
    loss = SoftmaxCrossEntropy()
    opt  = _sgd()

    # else-branch (N=16 < 32)
    X_e = cp.random.randn(16,8,12).astype(cp.float32)
    y_e = cp.random.randint(0, C, size=(16,), dtype=cp.int32)
    ctx_e = {"variant": {"unroll": 1, "amp": "fp32"}}
    _ = model.one_step_dynamic(X_e, y_e, loss=loss, optimizer=opt, ctx=ctx_e)
    key_else = getattr(model.tg, "key", None)

    # then-branch (N=32 >= 32)
    X_t = cp.random.randn(32,8,12).astype(cp.float32)
    y_t = cp.random.randint(0, C, size=(32,), dtype=cp.int32)
    ctx_t = {"variant": {"unroll": 1, "amp": "fp32"}}
    _ = model.one_step_dynamic(X_t, y_t, loss=loss, optimizer=opt, ctx=ctx_t)
    key_then = getattr(model.tg, "key", None)

    n_else = _normalize_key(key_else)
    n_then = _normalize_key(key_then)
    assert n_else is not None and n_then is not None, "graph keys must be available"

    vd_e = _variant_to_dict(n_else[5])
    vd_t = _variant_to_dict(n_then[5])
    pfp_e = tuple(vd_e.get("path_fp", ()))
    pfp_t = tuple(vd_t.get("path_fp", ()))

    assert pfp_e != pfp_t, f"path_fingerprint must differ between branches, got {pfp_e} vs {pfp_t}"

    _print_kv(f"[OK][{tag}] if/else key isolation", {
        "else.path_fp": pfp_e,
        "then.path_fp": pfp_t,
        "else.branch": n_else[4],
        "then.branch": n_then[4],
    })


# ===== 4) IR 구조/노드 덤프(가능 시) =====
def test_ir_dump_minimal(tag="ir_dump"):
    cp.random.seed(11)
    C = 5
    model = make_model_static(N=16, T=8, I=12, H=20, hidden=40, classes=C)
    loss = SoftmaxCrossEntropy(); opt = _sgd()

    X = cp.random.randn(16,8,12).astype(cp.float32)
    y = cp.random.randint(0, C, size=(16,), dtype=cp.int32)
    ctx = {"variant": {"unroll": 1, "amp": "fp32"}}

    _ = model.one_step_dynamic(X, y, loss=loss, optimizer=opt, ctx=ctx)

    ir = getattr(model, "last_graph_ir", None) or getattr(model, "_last_graph_ir", None)
    if ir is None and hasattr(model, "debug_dump_ir"):
        try:
            ir = model.debug_dump_ir()
        except Exception:
            ir = None

    if ir is None:
        print(f"[OK][{tag}] IR not exposed on this build (skipped).")
        return

    def _ir_len(ir_obj):
        try:
            return len(ir_obj)
        except Exception:
            return None

    summary = {"ir_type": type(ir).__name__, "node_count": _ir_len(ir)}
    preview = []
    try:
        for i, n in enumerate(ir):
            if i >= 5: break
            if isinstance(n, dict):
                preview.append({k: (str(v) if not isinstance(v, (int, float, str)) else v)
                                for k, v in n.items() if k in ("name","op","inputs","outputs")})
            else:
                name = getattr(n, "name", None) or getattr(n, "op", None) or f"node_{i}"
                ins = getattr(n, "inputs", None); outs = getattr(n, "outputs", None)
                preview.append({"name": str(name), "inputs": str(ins), "outputs": str(outs)})
    except Exception:
        pass

    _print_kv(f"[OK][{tag}] IR summary", summary)
    print(f"[OK][{tag}] IR preview (<=5): {json.dumps(preview, ensure_ascii=False)}")


# ===== 메인 =====
def main():
    print("== Static Graph Builder & Capture Plan — integration smoke ==")
    test_capture_plan_dump_and_invariants("plan_dump")
    test_record_step_graph_and_ptr_stability("record_graph")
    test_repeat_rng_and_ptr_stability("repeat_rng")
    test_if_branch_key_isolation("if_branch")
    test_graph_key_preview_matches_real("key_preview")
    test_ir_dump_minimal("ir_dump")
    print("[ALL OK] static-graph capture plan suite completed.")
    print("Tip) Nsight Systems에서 [CAPTURE]/[REPLAY] 범위를 확인하고, ptr 안정성과 variant 태그를 함께 보세요.")

if __name__ == "__main__":
    main()
