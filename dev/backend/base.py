from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable, Optional, Any


class Backend(ABC):
    """
    ML/DL 공통 수학 연산의 추상 인터페이스.
    상위 알고리즘(Linear/Ridge/Logistic/Lasso 등)은 이 API만 사용하면
    CPU/GPU 전환이 쉬워진다.
    """

    # ---------- BLAS-like ----------
    @abstractmethod
    def gemm(self, A, B, transA: bool = False, transB: bool = False):
        """C = op(A) @ op(B). 2D x 2D"""
        ...

    @abstractmethod
    def gemv(self, A, x, transA: bool = False):
        """y = op(A) @ x. 2D x 1D"""
        ...

    @abstractmethod
    def axpy(self, a: float, x, y):
        """y <- a * x + y  (새 배열 반환)"""
        ...

    @abstractmethod
    def dot(self, x, y) -> float:
        """벡터 내적"""
        ...

    # ---------- elementwise / reduction ----------
    @abstractmethod
    def sigmoid(self, z):
        ...

    @abstractmethod
    def softmax(self, Z, axis: int = 1):
        ...

    @abstractmethod
    def sum(self, X, axis: Optional[int] = None):
        ...

    # ---------- optional: 고급 ----------
    def cholesky_solve(self, AtA, Aty, alpha: float = 0.0):
        """
        (AtA + alpha*I) w = Aty 를 푸는 해법.
        구현되지 않았으면 NotImplementedError. (GPU에선 cupy.linalg 사용 권장)
        """
        raise NotImplementedError

    def prox_soft_threshold(self, w, tau: float):
        """
        L1 정규화에서 쓰는 soft-threshold(shrinkage).
        구현되지 않았으면 NotImplementedError.
        """
        raise NotImplementedError

    # ---------- device I/O ----------
    def to_device(self, x):  # GPU 백엔드에서 H2D
        return x

    def to_host(self, x):    # GPU 백엔드에서 D2H
        return x

    # ---------- 공통: Conjugate Gradient (CPU/GPU 겸용 기본 구현) ----------
    def conjugate_gradient(
        self,
        A_mv: Callable[[Any], Any],  # v -> A v
        b,
        x0=None,
        tol: float = 1e-6,
        max_iter: int = 1000,
    ):
        """
        Ax=b 반복해법. A_mv/axpy/dot 만 있으면 CPU/GPU 모두 동작.
        """
        x = b * 0 if x0 is None else x0
        r = self.axpy(1.0, b, self.axpy(-1.0, A_mv(x), b))  # r = b - A x
        p = r
        rs_old = self.dot(r, r)

        for _ in range(max_iter):
            Ap = A_mv(p)
            alpha = rs_old / max(self.dot(p, Ap), 1e-30)
            x = self.axpy(alpha, p, x)          # x += alpha*p
            r = self.axpy(-alpha, Ap, r)        # r -= alpha*Ap
            rs_new = self.dot(r, r)
            if rs_new**0.5 < tol:
                break
            beta = rs_new / max(rs_old, 1e-30)
            p = self.axpy(1.0, r, self.axpy(beta, p, r*0))  # p = r + beta*p
            rs_old = rs_new
        return x
