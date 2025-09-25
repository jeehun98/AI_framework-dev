import os, sys, numpy as np
import numpy as np

# === Import path & DLL 경로 설정 ===
THIS = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(THIS, "..", ".."))
PKG  = os.path.join(ROOT, "python")
if PKG not in sys.path:
    sys.path.insert(0, PKG)

# CUDA DLL (Windows) 힌트 경로
cuda_bins = [
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin",
]
if hasattr(os, "add_dll_directory"):
    for d in cuda_bins:
        if os.path.isdir(d):
            os.add_dll_directory(d)


from graph_executor_v2 import _core as ge

np.random.seed(0)

def part_a_smoke():
    """
    PART A: ge.contiguous_materialize 가 있을 때의 스모크 테스트.
    이 함수는 내부에서 H2D -> contiguous_copy -> D2H 왕복만 확인함.
    """
    print("\n[PART A] smoke with ge.contiguous_materialize (row-major roundtrip)")

    # 2D
    X = np.random.randn(5, 7).astype(np.float32)
    Y = ge.contiguous_materialize(X)   # 존재한다고 가정 (제안한 바인딩)
    ok = np.allclose(Y, X, atol=1e-6)
    print("  2D roundtrip close:", ok)

    # 4D (NCHW)
    X4 = np.random.randn(2, 3, 4, 5).astype(np.float32)
    Y4 = ge.contiguous_materialize(X4)
    ok4 = np.allclose(Y4, X4, atol=1e-6)
    print("  4D roundtrip close:", ok4)

def part_b_true_view():
    """
    PART B: 진짜 비연속 view를 테스트.
    전제: 바인딩에 다음 헬퍼가 존재한다고 가정.
      ge.contiguous_from_perm(X: np.ndarray, axes: tuple[int,...]) -> np.ndarray
    동작: X는 C-contiguous로 H2D 업로드하되, src 텐서의 stride를
         np.transpose(X, axes)와 동일하게 세팅한 뒤 contiguous_copy 실행.
    반환은 (permute 결과를 실제로 materialize 한) 연속 ndarray.
    """
    if not hasattr(ge, "contiguous_from_perm"):
        print("\n[PART B] contiguous_from_perm helper not found; skipping (add the helper to bindings).")
        return

    print("\n[PART B] true non-contiguous view test via ge.contiguous_from_perm")

    # 4D permute: NCHW -> NHWC
    N, C, H, W = 2, 3, 4, 5
    X = np.random.randn(N, C, H, W).astype(np.float32)

    axes = (0, 2, 3, 1)  # NCHW -> NHWC
    ref = np.ascontiguousarray(np.transpose(X, axes))  # numpy 기준 정답

    Y = ge.contiguous_from_perm(X, axes)  # 커널로 materialize
    ok = np.allclose(Y, ref, atol=1e-6)
    print(f"  permute {axes} materialize close:", ok)
    print("  shapes: ref", ref.shape, " Y", Y.shape)

    # 3D permute 예시: (B, M, N) -> (N, B, M)
    B, M, N_ = 3, 5, 7
    X3 = np.random.randn(B, M, N_).astype(np.float32)
    axes3 = (2, 0, 1)
    ref3 = np.ascontiguousarray(np.transpose(X3, axes3))
    Y3 = ge.contiguous_from_perm(X3, axes3)
    ok3 = np.allclose(Y3, ref3, atol=1e-6)
    print(f"  permute {axes3} materialize close:", ok3, " shape:", Y3.shape)

if __name__ == "__main__":
    part_a_smoke()
    part_b_true_view()
