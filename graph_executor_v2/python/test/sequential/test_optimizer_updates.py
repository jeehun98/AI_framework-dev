# File: test_optimizer_updates.py
from __future__ import annotations
import os, sys, math, traceback

# --- Path setup (adjust if your repo layout differs) ---
THIS = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(THIS, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import cupy as cp

from graph_executor_v2.layers.sequential import Sequential
from graph_executor_v2.layers.dense_gemm import Dense
from graph_executor_v2.losses.softmax_ce import SoftmaxCrossEntropy

# Try SGDOpt, fallback to AdamWOpt
try:
    from graph_executor_v2.optim.sgd import SGDOpt  # type: ignore
    OPT_CLS = SGDOpt
    OPT_KW = dict(lr=0.1)
    OPT_NAME = "SGDOpt"
except Exception:
    from graph_executor_v2.optim.adamw import AdamWOpt  # type: ignore
    OPT_CLS = AdamWOpt
    OPT_KW = dict(lr=1e-1)
    OPT_NAME = "AdamWOpt"

def _param_obj(item):
    """(p,g,tag)/(p,g)/p → p"""
    if isinstance(item, tuple):
        return item[0]
    return item

def _grad_obj(item):
    """(p,g,tag)/(p,g)/p → g or p.grad/None"""
    if isinstance(item, tuple):
        if len(item) >= 2:
            return item[1]
        item = item[0]
    return getattr(item, "grad", None)

def _tag(item, default="(no-tag)"):
    if isinstance(item, tuple):
        if len(item) == 3:
            return str(item[2])
        return default
    return default

def _arr(x):
    p = _param_obj(x)
    # ✅ CuPy 배열이면 그대로
    if isinstance(p, cp.ndarray):
        return p
    # ✅ 커스텀 Parameter 처럼 .data가 CuPy 배열인 경우만 .data 사용
    d = getattr(p, "data", None)
    if isinstance(d, cp.ndarray):
        return d
    # 마지막 수단: 원본 리턴
    return p


def _norm(x) -> float:
    try:
        return float(cp.linalg.norm(cp.asarray(x)))
    except Exception:
        return float("nan")

def _ensure_grad_buffers(model: Sequential):
    """
    워크어라운드:
      - Dense 등에서 build시 dW/db를 만들지 않았다면 zeros_like로 생성
      - 그 외 레이어도 W/b, weight/bias 쌍이 있으면 grad 버퍼 만들어줌
    """
    for lyr in getattr(model, "layers", []):
        # Dense 규약 우선
        if isinstance(lyr, Dense):
            if getattr(lyr, "W", None) is not None and getattr(lyr, "dW", None) is None:
                lyr.dW = cp.zeros_like(lyr.W)
            if getattr(lyr, "b", None) is not None and getattr(lyr, "db", None) is None:
                lyr.db = cp.zeros_like(lyr.b)
        # 덕 타이핑: (W,dW) / (weight,dweight) / (b,db) / (bias,dbias)
        for p_name, g_name in (("W","dW"), ("weight","dweight"), ("b","db"), ("bias","dbias")):
            if hasattr(lyr, p_name) and getattr(lyr, p_name) is not None:
                if not hasattr(lyr, g_name) or getattr(lyr, g_name) is None:
                    try:
                        setattr(lyr, g_name, cp.zeros_like(getattr(lyr, p_name)))
                    except Exception:
                        pass

def main():
    print("== Optimizer update smoke (verbose) ==")
    cp.random.seed(5)

    # ---------- Config ----------
    N, Din, C = 16, 8, 3
    FORCE_NATIVE_BWD = None  # set to True/False to force, or None to keep layer default
    ACT = "none"             # "none"으로 단순 경로부터 검증

    # ---------- Data ----------
    X = cp.random.standard_normal((N, Din), dtype=cp.float32)
    y = cp.random.randint(0, C, size=(N,), dtype=cp.int32)
    print(f"[data] X: shape={X.shape} dtype={X.dtype} contiguous={X.flags.c_contiguous}")
    print(f"[data] y: shape={y.shape} dtype={y.dtype}")

    # ---------- Model ----------
    dense = Dense(C, activation=ACT, use_native_bwd=(FORCE_NATIVE_BWD if FORCE_NATIVE_BWD is not None else True))
    m = Sequential(dense)
    m.build(input_shape=(N, Din))
    m.train(True)

    # grad 버퍼가 없으면 만들어줌 (워크어라운드)
    _ensure_grad_buffers(m)

    # grad 핸들 attach + zero
    if hasattr(m, "attach_grads"):
        m.attach_grads()
    if hasattr(m, "zero_grad"):
        m.zero_grad()

    # Dense internals quick view
    try:
        print(f"[dense] W: shape={dense.W.shape if dense.W is not None else None}, "
              f"dtype={dense.W.dtype if dense.W is not None else None}, "
              f"contig={dense.W.flags.c_contiguous if dense.W is not None else None}")
        print(f"[dense] b: shape={dense.b.shape if dense.b is not None else None}, "
              f"dtype={dense.b.dtype if dense.b is not None else None}, "
              f"contig={dense.b.flags.c_contiguous if dense.b is not None else None}")
        print(f"[dense] dW: {'None' if dense.dW is None else dense.dW.shape}, "
              f"db: {'None' if dense.db is None else dense.db.shape}")
    except Exception:
        print("[dense] failed to print W/b/dW/db details")
        traceback.print_exc()

    # ---------- Parameters discovery ----------
    params = list(m.parameters())
    print(f"[params] discovered count={len(params)}")
    for i, it in enumerate(params):
        p = _param_obj(it)
        g = _grad_obj(it)
        tag = _tag(it, default=f"param[{i}]")
        shape = getattr(p, "shape", None)
        dtype = getattr(p, "dtype", None)
        contig = getattr(p, "flags", None).c_contiguous if hasattr(p, "flags") else None
        gshape = getattr(g, "shape", None)
        gdtype = getattr(g, "dtype", None)
        print(f"  - [{i}] tag={tag:>16}  p.shape={shape} dtype={dtype} contig={contig} "
              f"| g.shape={gshape} g.dtype={gdtype}")

    if len(params) == 0:
        raise AssertionError("model has no parameters (Sequential.parameters() returned empty)")

    # ---------- Optimizer ----------
    opt = OPT_CLS(params, **OPT_KW)
    if hasattr(opt, "ensure_initialized"):
        try:
            opt.ensure_initialized()
        except Exception:
            pass
    print(f"[opt] {OPT_NAME} with hyper={OPT_KW}")

    # ---------- Snapshots ----------
    snap = [cp.asarray(_arr(p)).copy() for p in params]
    for i, s in enumerate(snap):
        print(f"[snap0] i={i} ||p||={_norm(s):.6e}")

    # ---------- Forward ----------
    print("[forward] begin")
    try:
        logits = m(X)
        print(f"[forward] logits: shape={logits.shape} dtype={logits.dtype} contig={logits.flags.c_contiguous}")
    except Exception as e:
        print("[forward] FAILED in model(X)")
        traceback.print_exc()
        raise

    # ---------- Loss ----------
    loss = SoftmaxCrossEntropy()
    try:
        L, dlogits = loss(logits, y)
        print(f"[loss] L={float(L):.6f}, dlogits: shape={getattr(dlogits,'shape',None)} "
              f"dtype={getattr(dlogits,'dtype',None)} contig={getattr(getattr(dlogits,'flags',None),'c_contiguous',None)}")
    except Exception:
        print("[loss] FAILED in SoftmaxCrossEntropy(logits, y)")
        traceback.print_exc()
        raise

    if not math.isfinite(float(L)):
        raise AssertionError("loss is not finite")

    # ---------- Backward ----------
    dlogits = cp.asarray(dlogits, dtype=cp.float32, order="C")
    try:
        g_in = m.backward(dlogits)
        print(f"[backward] ok; returned grad for input: shape={getattr(g_in,'shape',None)}")
    except Exception:
        print("[backward] FAILED in m.backward(dlogits)")
        traceback.print_exc()
        raise

    # 🔧 NEW: backward 이후 p.grad 연결 (필수)
    if hasattr(m, "attach_grads"):
        m.attach_grads()

    # Grad norms
    for i, it in enumerate(params):
        g = _grad_obj(it)
        print(f"[grad] param[{i}] tag={_tag(it)} ||grad||={_norm(g):.6e}")

    # ---------- Step ----------
    try:
        # support both step() and step(params)
        try:
            opt.step()
        except TypeError:
            opt.step(list(m.parameters()))  # 새 튜플로 전달 (혹시 참조 업데이트가 필요한 옵티마이저용)
        if hasattr(m, "zero_grad"):
            m.zero_grad()
    except Exception:
        print("[step] FAILED in optimizer step")
        traceback.print_exc()
        raise

    # ---------- Compare ----------
    changed = False
    total_delta = 0.0
    for i, (it, before) in enumerate(zip(params, snap)):
        after = cp.asarray(_arr(it))
        delta = _norm(after - before)
        total_delta += delta
        print(f"[compare] param[{i}] tag={_tag(it)} Δ||p||={delta:.6e} "
              f"(||before||={_norm(before):.6e} → ||after||={_norm(after):.6e})")
        if not bool(cp.allclose(after, before)):
            changed = True

    print(f"[result] any_changed={changed}, total Δ||p||={total_delta:.6e}")
    if not changed:
        raise AssertionError("optimizer step should change at least one parameter")

    print("[ALL OK]")

if __name__ == "__main__":

    try:
        main()
        sys.exit(0)
    except Exception as e:
        print(f"[FAIL] {type(e).__name__}: {e}")
        sys.exit(1)
