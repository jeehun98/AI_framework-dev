# File: python/graph_executor_v2/graph/exec_utils.py
from __future__ import annotations
from typing import Optional, Tuple, Any
import cupy as cp

# Conv2D / WS 유틸
from graph_executor_v2.layers.conv2d import Conv2D
from graph_executor_v2.ops import conv2d as convops

# ===== NVTX (optional) =====
try:
    from graph_executor_v2.backends.cuda.ops.gemm.detail.nvtx_shim import nvtx_range  # type: ignore
except Exception:
    class _DummyNvtx:
        def __call__(self, *_a, **_k):
            class _Ctx:
                def __enter__(self): return None
                def __exit__(self, *args): return False
            return _Ctx()
    nvtx_range = _DummyNvtx()  # type: ignore

# (선택) BN2d 타입 감지용 (런타임에서 isinstance 체크 용도)
try:
    from graph_executor_v2.layers.batchnorm import BatchNorm2d as _BN2d  # noqa: F401
except Exception:
    _BN2d = None  # type: ignore


def _out_hw(
    H: int, W: int, KH: int, KW: int,
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    dilation: Tuple[int, int],
) -> Tuple[int, int]:
    """Conv2D 출력 H/W 계산 (PyTorch 동일 공식을 정수 연산으로)."""
    sH, sW = stride
    pH, pW = padding
    dH, dW = dilation
    H_out = (H + 2 * pH - dH * (KH - 1) - 1) // sH + 1
    W_out = (W + 2 * pW - dW * (KW - 1) - 1) // sW + 1
    return H_out, W_out


def _alloc_conv2d_ws(HWo: int, K: int, Cout: int) -> convops.Conv2DWorkspaces:
    """Conv2D용 워크스페이스 일괄 할당 (capture-safe)."""
    ws = convops.Conv2DWorkspaces()
    # Forward
    ws.dCol   = cp.empty((HWo, K),    dtype=cp.float32)
    ws.W_KC   = cp.empty((K,   Cout), dtype=cp.float32)
    ws.Y_tmp  = cp.empty((HWo, Cout), dtype=cp.float32)
    ws.Z_rows = cp.empty((HWo, Cout), dtype=cp.float32)
    # Backward
    ws.dCol_b  = cp.empty((HWo, K), dtype=cp.float32)
    ws.dTmp    = cp.empty((max(Cout*K, HWo*K),), dtype=cp.float32)
    ws.gy_rows = cp.empty((Cout, HWo), dtype=cp.float32)
    ws.Z_rows_b= cp.empty((Cout, HWo), dtype=cp.float32)
    ws.W_CK    = cp.empty((Cout, K), dtype=cp.float32)
    ws.dY_HT   = cp.empty((HWo,  Cout), dtype=cp.float32)
    ws.dWpack  = cp.empty((Cout, K), dtype=cp.float32)
    return ws


def _ensure_conv2d_ws_for_forward(per, lyr: Conv2D, cur_shape: tuple[int, int, int, int]) -> convops.Conv2DWorkspaces:
    """Forward 직전에 Conv2D WS 준비/재사용."""
    ws = getattr(per, "work", None)
    if ws is not None:
        return ws
    _, Cin, H, W = map(int, cur_shape)
    KH, KW = lyr.kernel_size
    Cout   = int(lyr.out_channels)
    groups = int(lyr.groups)
    H_out, W_out = _out_hw(H, W, KH, KW, lyr.stride, lyr.padding, lyr.dilation)
    HWo = H_out * W_out
    K   = (Cin // groups) * KH * KW
    ws = _alloc_conv2d_ws(HWo, K, Cout)
    try:
        setattr(per, "work", ws)
    except Exception:
        pass
    return ws


def _ensure_conv2d_ws_for_backward(per, lyr: Conv2D) -> convops.Conv2DWorkspaces:
    """Backward 직전에 Conv2D WS 준비/재사용."""
    ws = getattr(per, "work", None)
    if ws is not None:
        return ws
    # per.y: (N, Cout, H_out, W_out)
    _, Cout, H_out, W_out = map(int, per.y.shape)
    HWo = H_out * W_out
    # lyr.W: (Cout, Cin, KH, KW)
    _, Cin, KH, KW = map(int, lyr.W.shape)
    groups = int(lyr.groups)
    K = (Cin // groups) * KH * KW
    ws = _alloc_conv2d_ws(HWo, K, Cout)
    try:
        setattr(per, "work", ws)
    except Exception:
        pass
    return ws


def _zero_bwd_buffers(plan) -> None:
    """Backward 누적 방지: gA/gW/gB 0 세팅."""
    for p in plan.per_layer:
        if p.gA is not None: p.gA.fill(0)
        if p.gW is not None: p.gW.fill(0)
        if p.gB is not None: p.gB.fill(0)


def _loss_forward(loss_fn, logits, labels):
    """loss.forward 호환 셔틀: (loss, dY) 튜플을 반환."""
    try:
        return loss_fn.forward(logits, labels, return_scalar=False)
    except TypeError:
        out = loss_fn.forward(logits, labels)
        if isinstance(out, tuple) and len(out) == 2:
            return out
        raise TypeError(
            "[exec_utils] loss.forward(logits, labels) must return (loss, dY) "
            "or accept return_scalar=False"
        )
