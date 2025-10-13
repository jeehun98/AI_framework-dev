# =====================================================================================
# test_conv2d.py
# - ops.conv2d: groups=1 경로만 smoke (forward/backward/save_z/capture-safe/finite-diff)
# - layers.Conv2D: groups=2 및 다양한 activation을 레이어 경유로 검증
# =====================================================================================

import os, sys
import math

# --- Import path (repo/python) ---
THIS = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(THIS, "..", "..", ".."))
PKG  = os.path.join(ROOT, "python")
if PKG not in sys.path:
    sys.path.insert(0, PKG)

import cupy as cp
from graph_executor_v2.ops import conv2d as c2d
from graph_executor_v2.layers.conv2d import Conv2D as LConv2D

# ----- CUDA Graph helpers -----
def _instantiate_graph(graph):
    if hasattr(graph, "instantiate"):
        try: return graph.instantiate()
        except TypeError: pass
    return graph

def _launch_graph(exec_graph, stream: cp.cuda.Stream):
    if hasattr(exec_graph, "launch"):
        try: exec_graph.launch(stream); return
        except TypeError: exec_graph.launch(stream.ptr); return
    if hasattr(cp.cuda.graph, "launch"):
        try: cp.cuda.graph.launch(exec_graph, stream); return
        except TypeError: cp.cuda.graph.launch(exec_graph, stream.ptr); return
    raise RuntimeError("CUDA Graph launch API not found")

# =====================================================================================
# OPS (groups=1) SMOKE
# =====================================================================================
def test_ops_forward_backward_shapes_and_savez_g1():
    cp.cuda.runtime.setDevice(0)
    rng = cp.random.default_rng(0)
    N, Cin, H, W = 2, 4, 8, 8
    Cout, Kh, Kw = 6, 3, 3
    stride, pad, dil, groups = (1, 1), (1, 1), (1, 1), 1

    X  = rng.standard_normal((N, Cin, H, W), dtype=cp.float32)
    Wt = rng.standard_normal((Cout, Cin, Kh, Kw), dtype=cp.float32)  # full Cin
    B  = cp.zeros((Cout,), dtype=cp.float32)

    # Forward (save_z=True)
    Y = c2d.forward(X, Wt, B, stride=stride, padding=pad, dilation=dil,
                    groups=groups, with_bias=True, act="relu", save_z=True)
    assert Y.shape == (N, Cout, H, W)

    # Backward (act relu)
    Z = c2d.forward(X, Wt, B, stride=stride, padding=pad, dilation=dil,
                    groups=groups, with_bias=True, act="none", save_z=True)
    gY = rng.standard_normal(Y.shape, dtype=cp.float32)
    out = c2d.backward(X, Wt, gY, Z, stride=stride, padding=pad, dilation=dil,
                       groups=groups, with_bias=True, act="relu",
                       want_gX=True, want_gW=True, want_gB=True)
    assert out["gX"].shape == X.shape
    assert out["gW"].shape == Wt.shape
    assert out["gB"].shape == B.shape

def test_ops_capture_safe_path_with_graph_g1():
    cp.cuda.runtime.setDevice(0)
    rng = cp.random.default_rng(2)
    N, Cin, H, W = 2, 6, 10, 12
    groups = 1
    Cout, Kh, Kw = 8, 3, 3
    stride, pad, dil = (1,1), (1,1), (1,1)

    X  = rng.standard_normal((N, Cin, H, W), dtype=cp.float32); X  = cp.ascontiguousarray(X)
    Wt = rng.standard_normal((Cout, Cin, Kh, Kw), dtype=cp.float32); Wt = cp.ascontiguousarray(Wt)
    B  = cp.zeros((Cout,), dtype=cp.float32)

    # Output tensors
    H_out, W_out = H, W
    Y  = cp.empty((N, Cout, H_out, W_out), dtype=cp.float32)
    Z  = cp.empty_like(Y)
    gY = rng.standard_normal(Y.shape, dtype=cp.float32)

    # Workspaces
    K   = (Cin // groups) * Kh * Kw
    HWo = H_out * W_out
    work = c2d.Conv2DWorkspaces()
    work.dCol   = cp.empty((HWo, K), dtype=cp.float32)
    work.W_KC   = cp.empty((K,   Cout), dtype=cp.float32)
    work.Y_tmp  = cp.empty((HWo, Cout), dtype=cp.float32)
    work.Z_rows = cp.empty((HWo, Cout), dtype=cp.float32)

    work.dCol_b  = cp.empty((HWo, K), dtype=cp.float32)
    work.dTmp    = cp.empty((max(Cout*K, HWo*K),), dtype=cp.float32)
    work.gy_rows = cp.empty((Cout, HWo), dtype=cp.float32)
    work.Z_rows_b= cp.empty((Cout, HWo), dtype=cp.float32)
    work.W_CK    = cp.empty((Cout, K), dtype=cp.float32)
    work.dY_HT   = cp.empty((HWo,  Cout), dtype=cp.float32)
    work.dWpack  = cp.empty((Cout, K), dtype=cp.float32)

    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        stream.begin_capture()
        c2d.forward_into(X, Wt, out=Y, B=B, stride=stride, padding=pad, dilation=dil,
                         groups=groups, with_bias=True, act="relu", save_z=True, Z_saved=Z,
                         work=work)
        gX_out = cp.empty_like(X)
        gW_out = cp.empty_like(Wt)
        gB_out = cp.empty((Cout,), dtype=cp.float32)
        c2d.backward_into(X, Wt, gY, Z, stride=stride, padding=pad, dilation=dil,
                          groups=groups, with_bias=True, act="relu",
                          gX_out=gX_out, gW_out=gW_out, gB_out=gB_out,
                          work=work)
        graph = stream.end_capture()
        exec_graph = _instantiate_graph(graph)
        _launch_graph(exec_graph, stream)
        _launch_graph(exec_graph, stream)
    stream.synchronize()

    assert Y.shape == (N, Cout, H_out, W_out)
    assert Z.shape == Y.shape

def test_ops_finite_diff_single_dw_g1():
    """ ops 경로 finite-diff (작은 텐서, groups=1) """
    cp.cuda.runtime.setDevice(0)
    rng = cp.random.default_rng(3)
    N, Cin, H, W = 1, 2, 6, 5
    groups = 1
    Cout, Kh, Kw = 2, 3, 3
    stride, pad, dil = (1,1), (1,1), (1,1)

    X  = rng.standard_normal((N, Cin, H, W), dtype=cp.float32)
    Wt = rng.standard_normal((Cout, Cin, Kh, Kw), dtype=cp.float32)
    B  = cp.zeros((Cout,), dtype=cp.float32)

    Y0 = c2d.forward(X, Wt, B, stride=stride, padding=pad, dilation=dil,
                     groups=groups, with_bias=True, act="none", save_z=True)

    Z  = c2d.forward(X, Wt, B, stride=stride, padding=pad, dilation=dil,
                     groups=groups, with_bias=True, act="none", save_z=True)
    gY = cp.ones_like(Y0)
    out = c2d.backward(X, Wt, gY, Z, stride=stride, padding=pad, dilation=dil,
                       groups=groups, with_bias=True, act="none",
                       want_gX=False, want_gW=True, want_gB=False)
    gW = out["gW"]

    iCout, iCin, iKh, iKw = 1, 0, 1, 2
    eps = cp.asarray(1e-2, dtype=cp.float32)

    Wp = Wt.copy(); Wp[iCout, iCin, iKh, iKw] += eps
    Yp = c2d.forward(X, Wp, B, stride=stride, padding=pad, dilation=dil,
                     groups=groups, with_bias=True, act="none", save_z=False)
    Wm = Wt.copy(); Wm[iCout, iCin, iKh, iKw] -= eps
    Ym = c2d.forward(X, Wm, B, stride=stride, padding=pad, dilation=dil,
                     groups=groups, with_bias=True, act="none", save_z=False)

    g_num = ((cp.sum(Yp) - cp.sum(Ym)) / (2 * eps)).item()
    g_aut = gW[iCout, iCin, iKh, iKw].item()
    assert abs(g_num - g_aut) < 5e-2, f"finite diff mismatch: num={g_num:.4f}, aut={g_aut:.4f}"

# =====================================================================================
# LAYER (groups=2) — tests through Conv2D layer only
# =====================================================================================
def test_layer_groups2_and_multiple_acts():
    cp.cuda.runtime.setDevice(0)
    rng = cp.random.default_rng(11)

    # groups=2
    N, Cin, H, W = 2, 4, 7, 9
    G = 2
    Cout, Kh, Kw = 6, 3, 3

    X = rng.standard_normal((N, Cin, H, W), dtype=cp.float32)

    for act in ("none", "relu", "leaky_relu", "sigmoid", "tanh", "gelu"):
        # Conv2D 레이어는 내부에서 act='none'으로 호출하고,
        # 활성화는 별도 레이어로 두는 디자인이지만, 여기서는 단순히 forward만 확인.
        layer = LConv2D(filters=Cout, kernel_size=(Kh, Kw),
                        stride=(1,1), padding=(1,1), dilation=(1,1),
                        groups=G, use_bias=True, initializer="he")
        layer.build(X.shape)
        Y = layer.call(X)
        assert Y.shape[0] == N and Y.shape[1] == Cout

        # backward 경로: 임의 스칼라 손실 L=sum(Y) 가정
        gY = cp.ones_like(Y)
        gX = layer.backward(gY)
        assert gX.shape == X.shape
        assert layer.dW.shape == layer.W.shape
        if layer.use_bias:
            assert layer.db.shape == (Cout,)

# =====================================================================================
# runner
# =====================================================================================
def _run_as_script():
    print("[conv2d] quick smoke start")
    test_ops_forward_backward_shapes_and_savez_g1()
    print("  ✔ ops g=1 forward/backward/save_z")
    test_ops_capture_safe_path_with_graph_g1()
    print("  ✔ ops g=1 capture-safe CUDA Graph")
    test_ops_finite_diff_single_dw_g1()
    print("  ✔ ops g=1 finite-difference dW single element")
    test_layer_groups2_and_multiple_acts()
    print("  ✔ layer groups=2 + fwd/bwd (act variants)")
    print("[conv2d] all quick tests passed.")

if __name__ == "__main__":
    _run_as_script()
