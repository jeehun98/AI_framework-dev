try:
    import cupy as cp
except Exception as e:
    cp = None

from .base import Backend


class CUDABackend(Backend):
    """
    기본 구현은 CuPy를 사용해 cuBLAS/cuSOLVER를 자동 활용한다.
    나중에 네 커스텀 CUDA 커널을 사용하려면, 아래 메서드 본문만
    네 래퍼 호출로 교체하면 된다.
    """
    def __init__(self):
        if cp is None:
            raise ImportError("CuPy가 필요합니다. `pip install cupy-cuda12x` 등으로 설치하세요.")

    # ---- BLAS-like ----
    def gemm(self, A, B, transA: bool = False, transB: bool = False):
        A = A.T if transA else A
        B = B.T if transB else B
        return A @ B  # cuBLAS sgemm/dgemm 경유

    def gemv(self, A, x, transA: bool = False):
        A = A.T if transA else A
        return A @ x  # cuBLAS gemv

    def axpy(self, a: float, x, y):
        return a * x + y  # elementwise (커널 1회)

    def dot(self, x, y) -> float:
        return float(cp.dot(x, y).get())  # host float 반환

    # ---- elementwise / reduction ----
    def sigmoid(self, z):
        z = cp.clip(z, -35.0, 35.0)
        return 1.0 / (1.0 + cp.exp(-z))

    def softmax(self, Z, axis: int = 1):
        Z = Z - cp.max(Z, axis=axis, keepdims=True)
        E = cp.exp(cp.clip(Z, -60.0, 60.0))
        return E / cp.sum(E, axis=axis, keepdims=True)

    def sum(self, X, axis=None):
        return cp.sum(X, axis=axis)

    # ---- optional: 해석해법 / prox ----
    def cholesky_solve(self, AtA, Aty, alpha: float = 0.0):
        A = AtA.copy()
        if alpha != 0.0:
            # 대각 성분에 정규화 추가
            diag_idx = cp.arange(A.shape[0])
            A[diag_idx, diag_idx] += alpha
        try:
            L = cp.linalg.cholesky(A)
            y = cp.linalg.solve(L, Aty)
            w = cp.linalg.solve(L.T, y)
        except cp.linalg.LinAlgError:
            w = cp.linalg.solve(A, Aty)
        return w

    def prox_soft_threshold(self, w, tau: float):
        return cp.sign(w) * cp.maximum(cp.abs(w) - tau, 0.0)

    # ---- device I/O ----
    def to_device(self, x):
        return x if isinstance(x, cp.ndarray) else cp.asarray(x)

    def to_host(self, x):
        return x if not isinstance(x, cp.ndarray) else cp.asnumpy(x)
