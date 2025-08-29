import numpy as np
from typing import Optional, Literal, Tuple
from backend import CPUBackend  # 기본 백엔드 (GPU 쓰면 CUDABackend 넘기면 됨)


# ======================
# 공통 유틸
# ======================
def _check_backend(backend):
    return backend or CPUBackend()

def _to_device(be, *arrays):
    return tuple(be.to_device(a) for a in arrays)

def _sigmoid(be, z):
    return be.sigmoid(z)

def _softmax(be, Z):
    return be.softmax(Z, axis=1)


# ======================
# LinearRegression (OLS)
# ======================
class LinearRegression:
    def __init__(self, fit_intercept: bool = True, copy_X: bool = True, backend=None):
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.backend = _check_backend(backend)
        self.coef_ = None
        self.intercept_ = 0.0

    def get_params(self, deep=True):
        return {"fit_intercept": self.fit_intercept, "copy_X": self.copy_X, "backend": self.backend}

    def set_params(self, **params):
        for k, v in params.items(): setattr(self, k, v)
        return self

    def fit(self, X, y, sample_weight: Optional[np.ndarray] = None):
        be = self.backend
        Xd, yd = _to_device(be, X, y)

        if self.fit_intercept:
            # 중심화 후 해 구하고, 인터셉트 복원
            X_mean = Xd.mean(axis=0)
            y_mean = yd.mean()
            Xc = Xd - X_mean
            yc = yd - y_mean
            XtX = be.gemm(Xc, Xc, transA=True)
            Xty = be.gemv(Xc, yc, transA=True)
            w = be.cholesky_solve(XtX, Xty, alpha=0.0)
            b = y_mean - (X_mean @ w)
            self.coef_ = be.to_host(w)
            self.intercept_ = float(be.to_host(b))
        else:
            XtX = be.gemm(Xd, Xd, transA=True)
            Xty = be.gemv(Xd, yd, transA=True)
            w = be.cholesky_solve(XtX, Xty, alpha=0.0)
            self.coef_ = be.to_host(w)
            self.intercept_ = 0.0
        return self

    def predict(self, X):
        be = self.backend
        Xd = be.to_device(X)
        yhat = Xd @ be.to_device(self.coef_) + self.intercept_
        return be.to_host(yhat)


# ======================
# Ridge Regression (L2)
# ======================
class Ridge:
    def __init__(self, alpha: float = 1.0, fit_intercept: bool = True, backend=None):
        self.alpha = float(alpha)
        self.fit_intercept = fit_intercept
        self.backend = _check_backend(backend)
        self.coef_ = None
        self.intercept_ = 0.0

    def get_params(self, deep=True):
        return {"alpha": self.alpha, "fit_intercept": self.fit_intercept, "backend": self.backend}

    def set_params(self, **params):
        for k, v in params.items(): setattr(self, k, v)
        return self

    def fit(self, X, y):
        be = self.backend
        Xd, yd = _to_device(be, X, y)

        if self.fit_intercept:
            X_mean = Xd.mean(axis=0)
            y_mean = yd.mean()
            Xc = Xd - X_mean
            yc = yd - y_mean
            XtX = be.gemm(Xc, Xc, transA=True)
            Xty = be.gemv(Xc, yc, transA=True)
            w = be.cholesky_solve(XtX, Xty, alpha=self.alpha)
            b = y_mean - (X_mean @ w)
            self.coef_ = be.to_host(w)
            self.intercept_ = float(be.to_host(b))
        else:
            XtX = be.gemm(Xd, Xd, transA=True)
            Xty = be.gemv(Xd, yd, transA=True)
            w = be.cholesky_solve(XtX, Xty, alpha=self.alpha)
            self.coef_ = be.to_host(w)
            self.intercept_ = 0.0
        return self

    def predict(self, X):
        be = self.backend
        Xd = be.to_device(X)
        return be.to_host(Xd @ be.to_device(self.coef_) + self.intercept_)


# ======================
# Lasso (L1) — FISTA(prox-grad)
# ======================
class Lasso:
    def __init__(self, alpha: float = 1.0, fit_intercept: bool = True,
                 max_iter: int = 1000, tol: float = 1e-4, backend=None):
        self.alpha = float(alpha)
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.backend = _check_backend(backend)
        self.coef_ = None
        self.intercept_ = 0.0

    def get_params(self, deep=True):
        return {"alpha": self.alpha, "fit_intercept": self.fit_intercept,
                "max_iter": self.max_iter, "tol": self.tol, "backend": self.backend}

    def set_params(self, **params):
        for k, v in params.items(): setattr(self, k, v)
        return self

    def _power_lipschitz(self, be, X, iters=20):
        # L ≈ largest eigenvalue of XᵀX → power iteration via A_mv(v)=Xᵀ(Xv)
        d = X.shape[1]
        v = be.to_device(np.random.randn(d).astype(np.float64))
        v = v / (np.sqrt(be.dot(v, v)) + 1e-12)
        for _ in range(iters):
            Xv = be.gemv(X, v)                # (n,)
            Av = be.gemv(X, Xv, transA=True)  # (d,)
            norm = np.sqrt(be.dot(Av, Av)) + 1e-12
            v = Av / norm
        # Rayleigh quotient
        Xv = be.gemv(X, v)
        Av = be.gemv(X, Xv, transA=True)
        return float(be.dot(v, Av))

    def fit(self, X, y):
        be = self.backend
        Xd, yd = _to_device(be, X, y)

        # (선택) 인터셉트는 중심화로 처리
        if self.fit_intercept:
            X_mean = Xd.mean(axis=0)
            y_mean = yd.mean()
            Xc = Xd - X_mean
            yc = yd - y_mean
        else:
            Xc, yc = Xd, yd

        d = Xc.shape[1]
        w = be.to_device(np.zeros(d, dtype=np.float64))
        z = w
        t = 1.0

        L = self._power_lipschitz(be, Xc) + 1e-12
        eta = 1.0 / L

        for _ in range(self.max_iter):
            # grad = Xᵀ (X z - y)
            Xz = be.gemv(Xc, z)
            grad = be.gemv(Xc, be.axpy(1.0, Xz, -yc), transA=True)
            w_next = be.prox_soft_threshold(be.axpy(-eta, grad, z), eta * self.alpha)
            t_next = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
            z = be.axpy(1.0, w_next, be.axpy((t - 1) / t_next, be.axpy(1.0, w_next, -w), w_next*0))
            # 수렴
            diff = be.to_host(np.max(np.abs(be.axpy(1.0, w_next, -w))))
            w, t = w_next, t_next
            if diff < self.tol:
                break

        if self.fit_intercept:
            b = y_mean - (X_mean @ be.to_device(self.coef_) if self.coef_ is not None else X_mean @ w)
        else:
            b = 0.0
        self.coef_ = be.to_host(w)
        self.intercept_ = float(be.to_host(b))
        return self

    def predict(self, X):
        be = self.backend
        Xd = be.to_device(X)
        return be.to_host(Xd @ be.to_device(self.coef_) + self.intercept_)


# ======================
# LogisticRegression (ovr & softmax)
# ======================
class LogisticRegression:
    def __init__(self,
                 penalty: Literal["none", "l2"] = "l2",
                 C: float = 1.0,
                 lr: float = 0.1,
                 max_iter: int = 1000,
                 tol: float = 1e-6,
                 fit_intercept: bool = True,
                 multi_class: Literal["ovr", "softmax"] = "ovr",
                 backend=None):
        self.penalty = penalty
        self.C = float(C)
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.multi_class = multi_class
        self.backend = _check_backend(backend)

        self.coef_ = None
        self.intercept_ = None
        self.classes_ = None

    def get_params(self, deep=True):
        return {"penalty": self.penalty, "C": self.C, "lr": self.lr, "max_iter": self.max_iter,
                "tol": self.tol, "fit_intercept": self.fit_intercept,
                "multi_class": self.multi_class, "backend": self.backend}

    def set_params(self, **params):
        for k, v in params.items(): setattr(self, k, v)
        return self

    def _fit_binary(self, X, y_bin) -> Tuple:
        be = self.backend
        Xd, yd = _to_device(be, X, y_bin)
        n, d = Xd.shape
        w = be.to_device(np.zeros(d, dtype=np.float64))
        b = 0.0

        alpha = 0.0 if self.penalty == "none" else (1.0 / self.C)

        for _ in range(self.max_iter):
            z = be.axpy(1.0, be.gemv(Xd, w), b)  # z = Xw + b
            p = _sigmoid(be, z)
            r = be.axpy(1.0, p, -yd)            # r = p - y
            g_w = be.gemv(Xd, r, transA=True)   # Xᵀ r
            if alpha != 0.0:
                g_w = be.axpy(alpha, w, g_w)    # + α w (bias 제외)
            g_b = float(be.sum(r))              # ∑(p - y)

            # 업데이트
            w = be.axpy(-self.lr, g_w, w)
            b = b - self.lr * g_b

            # 수렴 체크(grad norm)
            if np.linalg.norm(be.to_host(g_w)) < self.tol:
                break

        return be.to_host(w), float(b)

    def _fit_softmax(self, X, y):
        be = self.backend
        Xd, yd = _to_device(be, X, y)
        n, d = Xd.shape
        classes = np.unique(be.to_host(yd))
        K = int(classes.max() + 1) if np.array_equal(classes, np.arange(len(classes))) else len(classes)

        W = be.to_device(np.zeros((d, K), dtype=np.float64))
        b = be.to_device(np.zeros((K,), dtype=np.float64))
        alpha = 0.0 if self.penalty == "none" else (1.0 / self.C)

        # one-hot (host에서 만들고 device로 올려도 OK)
        Y = np.zeros((n, K), dtype=np.float64)
        y_int = be.to_host(yd).astype(int)
        Y[np.arange(n), y_int] = 1.0
        Yd = be.to_device(Y)

        for _ in range(self.max_iter):
            Z = be.gemm(Xd, W) + b  # (n,K) + (K,) broadcasting
            P = _softmax(be, Z)     # (n,K)

            R = P - Yd              # (n,K)
            # dW = Xᵀ R / n + α W
            dW = be.gemm(Xd, R, transA=True) * (1.0 / n)
            if alpha != 0.0:
                dW = dW + alpha * W
            db = be.sum(R, axis=0) * (1.0 / n)

            W = W - self.lr * dW
            b = b - self.lr * db

            if np.linalg.norm(be.to_host(dW)) < self.tol:
                break

        self.coef_ = be.to_host(W).T  # (K,d)
        self.intercept_ = be.to_host(b)
        return self

    def fit(self, X, y):
        be = self.backend
        # y가 CPU든 GPU든 상관없이 "호스트에서" 클래스만 계산
        y_host = be.to_host(y)
        self.classes_ = np.unique(y_host)

        if len(self.classes_) == 2 and self.multi_class == "ovr":
            w, b = self._fit_binary(X, (y_host == self.classes_[1]).astype(int))
            self.coef_ = w
            self.intercept_ = b
            return self

        if self.multi_class == "ovr":
            Ws, bs = [], []
            for cls in self.classes_:
                w, b = self._fit_binary(X, (y_host == cls).astype(int))
                Ws.append(w); bs.append(b)
            self.coef_ = np.vstack(Ws)
            self.intercept_ = np.array(bs)
            return self

        # softmax
        return self._fit_softmax(X, y)  # _fit_softmax 내부에서 to_device 처리함


    def predict_proba(self, X):
        be = self.backend
        Xd = be.to_device(X)
        if self.multi_class == "ovr" and self.coef_.ndim == 1:
            z = be.axpy(1.0, be.gemv(Xd, be.to_device(self.coef_)), self.intercept_)
            p1 = _sigmoid(be, z)
            P = np.c_[be.to_host(1.0 - p1), be.to_host(p1)]
            return P
        if self.multi_class == "ovr":
            # 독립 시그모이드 후 정규화
            W = be.to_device(self.coef_)        # (K,d)
            Z = be.gemm(Xd, W.T) + self.intercept_
            P = _sigmoid(be, Z)
            P = be.to_host(P)
            s = P.sum(axis=1, keepdims=True); s[s == 0] = 1.0
            return P / s
        # softmax
        W = be.to_device(self.coef_.T)          # (d,K)
        Z = be.gemm(Xd, W) + self.intercept_
        return be.to_host(_softmax(be, Z))

    def predict(self, X):
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return self.classes_[idx]
