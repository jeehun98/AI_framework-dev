import numpy as np
from .base import Backend


class CPUBackend(Backend):
    def gemm(self, A, B, transA: bool = False, transB: bool = False):
        A = A.T if transA else A
        B = B.T if transB else B
        return A @ B

    def gemv(self, A, x, transA: bool = False):
        A = A.T if transA else A
        return A @ x

    def axpy(self, a: float, x, y):
        return a * x + y

    def dot(self, x, y) -> float:
        return float(np.dot(x, y))

    def sigmoid(self, z):
        z = np.clip(z, -35.0, 35.0)
        return 1.0 / (1.0 + np.exp(-z))

    def softmax(self, Z, axis: int = 1):
        Z = Z - np.max(Z, axis=axis, keepdims=True)
        E = np.exp(np.clip(Z, -60.0, 60.0))
        return E / np.sum(E, axis=axis, keepdims=True)

    def sum(self, X, axis=None):
        return np.sum(X, axis=axis)

    # 선택: 해석해법 (Ridge/OLS)
    def cholesky_solve(self, AtA, Aty, alpha: float = 0.0):
        A = AtA.copy()
        if alpha != 0.0:
            A[np.diag_indices_from(A)] += alpha
        try:
            L = np.linalg.cholesky(A)
            # L L^T w = Aty
            y = np.linalg.solve(L, Aty)
            w = np.linalg.solve(L.T, y)
        except np.linalg.LinAlgError:
            # fallback
            w = np.linalg.solve(A, Aty)
        return w

    def prox_soft_threshold(self, w, tau: float):
        return np.sign(w) * np.maximum(np.abs(w) - tau, 0.0)
