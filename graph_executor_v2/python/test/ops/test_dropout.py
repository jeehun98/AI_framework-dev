# test_dropout.py
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

# === 동적 로더(require) 사용 (gemm 테스트와 동일 스타일) ===
from graph_executor_v2.ops import require
ops_dropout = require("dropout")  # -> _ops_dropout

# === 유틸: pyd 탐색/표시 ===
def list_all_pyd():
    roots = [
        os.path.join(ROOT, "python", "graph_executor_v2", "ops"),
        os.path.dirname(os.__file__),  # ...\Lib
    ]
    found = []
    for base in roots:
        for r, _, files in os.walk(base):
            for f in files:
                if f.startswith("_ops_dropout") and f.endswith(".pyd"):
                    found.append(os.path.join(r, f))
    for sp in sys.path:
        try:
            for r, _, files in os.walk(sp):
                for f in files:
                    if f.startswith("_ops_dropout") and f.endswith(".pyd"):
                        p = os.path.join(r, f)
                        if p not in found:
                            found.append(p)
        except Exception:
            pass
    return sorted(set(found))

def check_binary_has_no_dbg(pyd_path: str, needles=(b"[DROP dbg]",)):
    try:
        with open(pyd_path, "rb") as f:
            blob = f.read()
        return all(n not in blob for n in needles)
    except Exception:
        return False

# === 수학적 기대값 ===
def train_scale(p: float, scale_in_train: bool) -> float:
    return (1.0 / (1.0 - p)) if scale_in_train else 1.0

def forward_expect(X: np.ndarray, M: np.ndarray, p: float, scale_in_train: bool) -> np.ndarray:
    s = train_scale(p, scale_in_train)
    return X.astype(np.float32) * (M.astype(np.int32) != 0) * s

def backward_expect(dY: np.ndarray, M: np.ndarray, p: float, scale_in_train: bool) -> np.ndarray:
    s = train_scale(p, scale_in_train)
    return dY.astype(np.float32) * (M.astype(np.int32) != 0) * s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=8)
    ap.add_argument("--N", type=int, default=7)
    ap.add_argument("--p", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scale-in-train", action="store_true", default=True)
    ap.add_argument("--check-determinism", action="store_true", help="같은 seed로 마스크 동일성 검사")
    ap.add_argument("--tol", type=float, default=1e-6, help="np.allclose 허용 오차")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # === 모듈 로드 경로 확인 ===
    print("LOADED:", ops_dropout.__file__)

    # === 중복 pyd 탐지 ===
    all_pyd = list_all_pyd()
    if all_pyd:
        print("FOUND_PYDS:")
        for p in all_pyd:
            mark = " <-- LOADED" if os.path.abspath(p) == os.path.abspath(ops_dropout.__file__) else ""
            print("  ", p, mark)

    # === 디버그 문자열 없음 확인(있으면 실패)
    ok_no_dbg = check_binary_has_no_dbg(ops_dropout.__file__)
    print("BINARY_HAS_NO_[DROP dbg]:", ok_no_dbg)
    assert ok_no_dbg, "Loaded pyd contains debug marker(s)!"

    # === 입력 준비 ===
    M, N = int(args.M), int(args.N)
    X  = rng.standard_normal(size=(M, N)).astype(np.float32)
    dY = rng.standard_normal(size=(M, N)).astype(np.float32)

    # === Forward: Y, Mask ===
    Y, Mask = ops_dropout.dropout(X, p=args.p, return_mask=True, seed=args.seed, scale_in_train=args.scale_in_train)
    print("Y.shape:", Y.shape, "Mask.shape:", Mask.shape)
    assert Y.shape == (M, N) and Mask.shape == (M, N)
    assert Mask.dtype == np.int32, f"mask dtype must be int32, got {Mask.dtype}"

    # 값 검증: Y ≈ X * (Mask!=0) * scale
    Y_expect = forward_expect(X, Mask, args.p, args.scale_in_train)
    ok_forward = np.allclose(Y, Y_expect, atol=args.tol, rtol=0)
    print("FORWARD_OK:", ok_forward)
    if not ok_forward:
        # 최대 오차 리포트
        max_err = float(np.max(np.abs(Y - Y_expect)))
        print("FORWARD_MAX_ERR:", max_err)
    assert ok_forward, "Forward mismatch against expected masked/scaled output."

    # === Backward: dX ===
    dX = ops_dropout.dropout_backward(dY, Mask, p=args.p, seed=args.seed, scale_in_train=args.scale_in_train)
    print("dX.shape:", dX.shape)
    assert dX.shape == (M, N)

    dX_expect = backward_expect(dY, Mask, args.p, args.scale_in_train)
    ok_backward = np.allclose(dX, dX_expect, atol=args.tol, rtol=0)
    print("BACKWARD_OK:", ok_backward)
    if not ok_backward:
        max_err = float(np.max(np.abs(dX - dX_expect)))
        print("BACKWARD_MAX_ERR:", max_err)
    assert ok_backward, "Backward mismatch against expected masked/scaled gradient."

    # === 결정성 검증(선택): 같은 seed -> 같은 Mask?
    if args.check_determinism:
        _, Mask2 = ops_dropout.dropout(X, p=args.p, return_mask=True, seed=args.seed, scale_in_train=args.scale_in_train)
        same = np.array_equal(Mask, Mask2)
        print("DETERMINISTIC_MASK_WITH_SAME_SEED:", same)
        assert same, "Same seed did not reproduce identical mask."

    print("OK: dropout forward/backward tests passed.")

if __name__ == "__main__":
    main()
