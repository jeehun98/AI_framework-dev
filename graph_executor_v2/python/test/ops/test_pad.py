# test_pad.py
import os, sys, argparse
import numpy as np

# === Import path & CUDA DLL 경로 (Windows) ===
THIS = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(THIS, "..", "..", ".."))
PKG  = os.path.join(ROOT, "python")
if PKG not in sys.path:
    sys.path.insert(0, PKG)

cuda_bins = [
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin",
]
if hasattr(os, "add_dll_directory"):
    for d in cuda_bins:
        if os.path.isdir(d):
            os.add_dll_directory(d)

from graph_executor_v2.ops import require

# CuPy로 디바이스 포인터 전달/검증
try:
    import cupy as cp
    HAS_CUPY = True
except Exception:
    HAS_CUPY = False
    cp = None

ops_pad = require("pad")  # -> _ops_pad
PadSpec = ops_pad.PadSpec

# ---------------- helpers ----------------
def numpy_pad_const(x: np.ndarray, before, after, value: float):
    """np.pad 래핑 (const) — before/after는 각 축에 대한 정수 리스트"""
    assert len(before) == x.ndim and len(after) == x.ndim
    pad_width = [(int(b), int(a)) for b, a in zip(before, after)]
    return np.pad(x, pad_width=pad_width, mode="constant", constant_values=value)

def make_pads(rank, rng, max_pad=2):
    """각 차원별 before/after를 0..max_pad 범위에서 랜덤 생성 (출력은 리스트[int])"""
    before = rng.integers(low=0, high=max_pad+1, size=(rank,), dtype=np.int64).tolist()
    after  = rng.integers(low=0, high=max_pad+1, size=(rank,), dtype=np.int64).tolist()
    return [int(x) for x in before], [int(x) for x in after]

def apply_spec_to_shape(x_shape, before, after):
    y_shape = [int(d + b + a) for d, b, a in zip(x_shape, before, after)]
    return y_shape

def run_case_forward_backward(x_h: np.ndarray, before, after, value: float, stream=0, check_ref=True):
    """한 케이스 실행: forward 정밀 비교 + backward 슬라이스 확인"""
    assert HAS_CUPY, "CuPy not available"

    # --- host → device
    x_d = cp.asarray(x_h)
    y_shape = apply_spec_to_shape(list(x_h.shape), before, after)
    y_d = cp.empty(y_shape, dtype=cp.float32)

    # --- spec
    spec = PadSpec()
    spec.before = [int(v) for v in before]
    spec.after  = [int(v) for v in after]
    spec.value  = float(value)

    # --- forward
    ops_pad.forward(
        int(x_d.data.ptr), list(x_d.shape),
        int(y_d.data.ptr), y_shape,
        spec,
        stream
    )
    y_h = cp.asnumpy(y_d)

    if check_ref:
        y_ref = numpy_pad_const(x_h, before, after, value)
        max_abs = float(np.max(np.abs(y_h - y_ref)))
        print(f"  forward max_abs: {max_abs:.3e}")
        assert y_ref.shape == tuple(y_shape)
        assert max_abs < 5e-6, f"forward mismatch: max_abs={max_abs}"

    # --- backward: dX = slice(dY, spec)
    # 임의의 dY로 역전파 체크 (패드 제외 영역만 dX로 복사되어야 함)
    rng = np.random.default_rng(123)
    dy_h = rng.standard_normal(size=y_shape, dtype=np.float32)
    dy_d = cp.asarray(dy_h)
    dx_d = cp.zeros_like(x_d)

    ops_pad.backward(
        int(dy_d.data.ptr), y_shape,
        int(dx_d.data.ptr), list(x_h.shape),
        spec,
        stream
    )
    dx_h = cp.asnumpy(dx_d)

    # 참조 dX: y_ref의 원본 X가 들어가는 위치만 잘라서 dY 복사
    # forward에서 y_ref = pad(x_h, spec)이므로, backward는 그 역:
    # 각 축별로 [before[i]:before[i]+x_shape[i]) 슬라이스
    slc = tuple(slice(b, b + s) for b, s in zip(before, x_h.shape))
    dx_ref = dy_h[slc]
    max_abs_dx = float(np.max(np.abs(dx_h - dx_ref)))
    print(f"  backward(dX) max_abs: {max_abs_dx:.3e}")
    assert max_abs_dx < 5e-6, f"backward dX mismatch: max_abs={max_abs_dx}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--sweep", action="store_true", help="여러 rank/shape 조합을 랜덤 스윕")
    args = ap.parse_args()

    print("LOADED:", ops_pad.__file__)
    if not HAS_CUPY:
        print("SKIP: CuPy not available. _ops_pad expects device pointers.")
        sys.exit(0)

    rng = np.random.default_rng(args.seed)

    # === 기본 소형 케이스 (NCHW) ===
    print("Case: NCHW small")
    N, C, H, W = 2, 3, 5, 4
    x_h = rng.standard_normal(size=(N, C, H, W), dtype=np.float32)
    before = [0, 0, 1, 2]
    after  = [0, 0, 0, 1]
    value  = 0.25
    run_case_forward_backward(x_h, before, after, value)

    # === 2D 케이스 (행렬) ===
    print("Case: 2D matrix")
    H2, W2 = 7, 8
    x2_h = rng.standard_normal(size=(H2, W2), dtype=np.float32)
    before2 = [1, 0]
    after2  = [2, 3]
    value2  = -1.0
    run_case_forward_backward(x2_h, before2, after2, value2)

    # === 1D 케이스 ===
    print("Case: 1D vector")
    L = 11
    x1_h = rng.standard_normal(size=(L,), dtype=np.float32)
    before1 = [2]
    after1  = [3]
    value1  = 1.5
    run_case_forward_backward(x1_h, before1, after1, value1)

    # === 선택: 다중 랜덤 스윕 ===
    if args.sweep:
        print("Random sweep...")
        for rank in [1, 2, 3, 4]:
            for _ in range(4):
                # 각 축 길이를 2..6에서 랜덤 선택
                shape = tuple(int(v) for v in rng.integers(low=2, high=7, size=(rank,), dtype=np.int64))
                before, after = make_pads(rank, rng, max_pad=2)
                value = float(rng.uniform(-2.0, 2.0))
                print(f"  rank={rank}, shape={shape}, before={before}, after={after}, value={value:.2f}")
                x_h = rng.standard_normal(size=shape, dtype=np.float32)
                run_case_forward_backward(x_h, before, after, value)

    print("OK: pad forward/backward basic checks passed.")

if __name__ == "__main__":
    main()
