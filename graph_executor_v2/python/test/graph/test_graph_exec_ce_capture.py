# python/test/graph/test_graph_exec_ce_capture.py
import os, sys, traceback
THIS = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(THIS, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import cupy as cp
from graph_executor_v2.ops import cross_entropy as ce_ops


# ======================== env ========================
def env_info():
    try:
        dev = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(dev.id)
        name = props.get("name")
        name = name.decode() if isinstance(name, (bytes, bytearray)) else name
        drv = cp.cuda.runtime.driverGetVersion()
        rt  = cp.cuda.runtime.runtimeGetVersion()
        return f"Device: {name}, CC {props.get('major')}.{props.get('minor')} | Driver {drv}, Runtime {rt} | CuPy {cp.__version__}"
    except Exception as e:
        return f"(env info failed: {e})"


# ===================== capture utils ====================
def try_capture(fn, desc):
    """Capture and replay fn() on a non-blocking stream using CuPy 13.x Graph.launch()."""
    s = cp.cuda.Stream(non_blocking=True)

    # 1) warmup
    try:
        fn()
        cp.cuda.get_current_stream().synchronize()
    except Exception:
        print(f"[{desc}] warmup: FAILED")
        traceback.print_exc()
        return None

    # 2) capture + replay
    try:
        with s:
            s.begin_capture()
            fn()
            g = s.end_capture()
        print(f"[{desc}] capture: OK  (graph={type(g).__name__})")

        with s:
            # upload() may not exist depending on CuPy minor; guard it.
            if hasattr(g, "upload"):
                try:
                    g.upload()
                except Exception:
                    pass
            g.launch()
        s.synchronize()
        print(f"[{desc}] replay: OK")
        return g
    except Exception:
        print(f"[{desc}] capture/replay: FAILED")
        traceback.print_exc()
        return None


# ======================== tests =========================
def _make_data(M, N, from_logits=True, ignore_index=-1, seed=123):
    rs = cp.random.RandomState(seed)
    if from_logits:
        X = (rs.standard_normal((M, N)).astype(cp.float32) * 0.5)
    else:
        # Make proper probabilities (no zeros)
        X = rs.standard_normal((M, N)).astype(cp.float32)
        X = cp.exp(X - X.max(axis=1, keepdims=True))
        X = X / X.sum(axis=1, keepdims=True)
        X = cp.clip(X, 1e-5, 1.0)
        X = X / X.sum(axis=1, keepdims=True)

    T = rs.randint(0, N, size=(M,), dtype=cp.int32)
    if ignore_index >= 0:
        # randomly mark ~25% as ignore_index
        mask = rs.rand(M) < 0.25
        T = T.copy()
        T[mask] = cp.int32(ignore_index)
    return X, T


def _check_finite(x: cp.ndarray, name: str):
    if not cp.isfinite(x).all():
        raise AssertionError(f"{name} contains inf/nan")


def test_ce_from_logits(M=8, N=13, reduction="mean", ls_eps=0.1, ignore_index=-1):
    print(f"\n=== CrossEntropy (from_logits=True, reduction={reduction}, ls_eps={ls_eps}, ignore_index={ignore_index}) ===")
    X, T = _make_data(M, N, from_logits=True, ignore_index=ignore_index, seed=2025)

    # forward buffer
    out_shape = (M,) if reduction == "none" else (1,)
    loss_out = cp.empty(out_shape, dtype=cp.float32)

    # backward buffer
    dX_out = cp.empty_like(X)

    def fwd_body():
        ce_ops.forward(
            X, T, from_logits=True, reduction=reduction,
            ignore_index=ignore_index, ls_eps=ls_eps, out=loss_out
        )

    def bwd_body():
        ce_ops.backward(
            X, T, from_logits=True, reduction=reduction,
            ignore_index=ignore_index, ls_eps=ls_eps, out=dX_out
        )

    # Warm correctness (not strict numeric compare, just sanity)
    fwd_body(); _check_finite(loss_out, "loss_out")
    bwd_body(); _check_finite(dX_out, "dX_out")

    try_capture(fwd_body, f"CE-FWD(logits,{reduction})")
    try_capture(bwd_body, f"CE-BWD(logits,{reduction})")


def test_ce_from_probs(M=7, N=11, reduction="sum", ls_eps=0.05, ignore_index=-1):
    print(f"\n=== CrossEntropy (from_logits=False, reduction={reduction}, ls_eps={ls_eps}, ignore_index={ignore_index}) ===")
    X, T = _make_data(M, N, from_logits=False, ignore_index=ignore_index, seed=2026)

    loss_out = cp.empty((M,), dtype=cp.float32) if reduction == "none" else cp.empty((1,), dtype=cp.float32)
    dX_out = cp.empty_like(X)

    def fwd_body():
        ce_ops.forward(
            X, T, from_logits=False, reduction=reduction,
            ignore_index=ignore_index, ls_eps=ls_eps, eps=1e-7, out=loss_out
        )

    def bwd_body():
        ce_ops.backward(
            X, T, from_logits=False, reduction=reduction,
            ignore_index=ignore_index, ls_eps=ls_eps, eps=1e-7, out=dX_out
        )

    fwd_body(); _check_finite(loss_out, "loss_out")
    bwd_body(); _check_finite(dX_out, "dX_out")

    try_capture(fwd_body, f"CE-FWD(probs,{reduction})")
    try_capture(bwd_body, f"CE-BWD(probs,{reduction})")


def test_all_variants():
    # from logits
    test_ce_from_logits(M=16, N=19, reduction="none", ls_eps=0.0, ignore_index=-1)
    test_ce_from_logits(M=16, N=19, reduction="mean", ls_eps=0.1, ignore_index= -1)
    test_ce_from_logits(M=16, N=19, reduction="sum",  ls_eps=0.2, ignore_index= -1)
    test_ce_from_logits(M=12, N=9,  reduction="mean", ls_eps=0.1, ignore_index=0)   # with ignore_index=0

    # from probs
    test_ce_from_probs(M=15, N=10, reduction="none", ls_eps=0.0, ignore_index=-1)
    test_ce_from_probs(M=15, N=10, reduction="mean", ls_eps=0.05, ignore_index=-1)
    test_ce_from_probs(M=15, N=10, reduction="sum",  ls_eps=0.10, ignore_index= -1)
    test_ce_from_probs(M=10, N=7,  reduction="mean", ls_eps=0.05, ignore_index=3)   # with ignore_index=3


# ======================== main ============================
if __name__ == "__main__":
    cp.random.seed(20251008)
    print("[ENV]", env_info())

    test_all_variants()

    print("\n[OK] cross_entropy capture tests finished.")
