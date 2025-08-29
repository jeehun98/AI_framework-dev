# ml/preprocessing/preprocessing.py
from __future__ import annotations
import numpy as np
from typing import Optional, Literal, List, Any, Sequence
from backend import CPUBackend

try:
    import cupy as cp
except Exception:
    cp = None


# -------------------------
# 공통 헬퍼
# -------------------------
def _check_backend(be):
    return be or CPUBackend()

def _is_cupy_array(x) -> bool:
    if cp is None:
        return False
    return isinstance(x, cp.ndarray)

def _to_host(x):
    if _is_cupy_array(x):
        return cp.asnumpy(x)
    return x

def _to_device(be, x):
    return be.to_device(x)

def _xp_of(x):
    """x가 cupy면 cp, 아니면 np 반환"""
    return cp if _is_cupy_array(x) else np

def _ensure_2d(X):
    if X.ndim == 1:
        return X.reshape(-1, 1)
    return X

def _is_object_like(X) -> bool:
    """dtype이 object 또는 문자열/파이썬 객체성분이면 True"""
    # cupy는 object dtype을 지원하지 않으니 host에서 판정
    Xh = _to_host(X)
    return Xh.dtype == object

def _nan_mask(x):
    xp = _xp_of(x)
    # object/문자 문자열은 여기 안 옴(숫자 전용)
    return xp.isnan(x)

def _safe_unique(x):
    """cupy/numpy 공통 unique"""
    xp = _xp_of(x)
    return xp.unique(x)

def _hstack(blocks: List[Any]):
    """cupy/numpy 공통 hstack"""
    if any(_is_cupy_array(b) for b in blocks):
        if cp is None:
            # 이론상 오지 않지만, 안전망
            return np.hstack([_to_host(b) for b in blocks])
        return cp.hstack([_to_device(CPUBackend(), b) if not _is_cupy_array(b) else b for b in blocks])
    return np.hstack(blocks)

def _concat_colwise(blocks: List[Any]):
    if any(_is_cupy_array(b) for b in blocks):
        return cp.concatenate(blocks, axis=1)
    return np.concatenate(blocks, axis=1)


# ============================================================
# SimpleImputer (GPU: 숫자형 mean/constant, CPU fallback: most_frequent/객체형)
# ============================================================
class SimpleImputer:
    """
    strategy: "mean" | "most_frequent" | "constant"
      - 숫자형 + mean/constant 는 GPU 지원 (cupy)
      - most_frequent 또는 객체/문자 데이터는 CPU 경로에서 처리
    fill_value: strategy="constant" 일 때 채울 값
    """
    def __init__(self,
                 strategy: Literal["mean", "most_frequent", "constant"] = "mean",
                 fill_value: Optional[Any] = None,
                 backend=None):
        self.strategy = strategy
        self.fill_value = fill_value
        self.backend = _check_backend(backend)
        # learned
        self.statistics_ = None  # (n_features,)
        self.is_numeric_ = True

    def fit(self, X, y=None):
        be = self.backend
        X = _ensure_2d(X)
        # 객체/문자형 또는 most_frequent → CPU 경로
        if self.strategy == "most_frequent" or _is_object_like(X):
            Xh = _to_host(X)
            stats = []
            for j in range(Xh.shape[1]):
                col = Xh[:, j]
                # 결측 취급: None 또는 np.nan
                col = np.array([c for c in col if c is not None and not (isinstance(c, float) and np.isnan(c))],
                               dtype=object)
                if col.size == 0:
                    stats.append(self.fill_value if self.fill_value is not None else 0)
                else:
                    # 최빈값
                    vals, counts = np.unique(col, return_counts=True)
                    stats.append(vals[np.argmax(counts)])
            self.statistics_ = np.asarray(stats, dtype=object)
            self.is_numeric_ = False
            return self

        # 숫자형 & mean/constant (GPU 가능)
        xp = _xp_of(X)
        if self.strategy == "mean":
            # NaN 무시 평균
            stats = xp.nanmean(X, axis=0)
        elif self.strategy == "constant":
            fill = self.fill_value if self.fill_value is not None else 0.0
            stats = xp.array([fill] * X.shape[1], dtype=X.dtype)
        else:
            raise ValueError(f"Unsupported strategy for numeric: {self.strategy}")

        self.statistics_ = stats
        self.is_numeric_ = True
        return self

    def transform(self, X):
        assert self.statistics_ is not None, "Call fit() first."
        X = _ensure_2d(X)
        # CPU 경로 (객체/문자형 또는 most_frequent)
        if not self.is_numeric_:
            Xh = _to_host(X).copy()
            stats = self.statistics_
            for j in range(Xh.shape[1]):
                stat = stats[j]
                col = Xh[:, j]
                for i, v in enumerate(col):
                    if v is None or (isinstance(v, float) and np.isnan(v)):
                        Xh[i, j] = stat
            return Xh

        # 숫자형(GPU 가능)
        xp = _xp_of(X)
        Xc = X.copy()
        m = xp.isnan(Xc)
        if m.any():
            Xc[m] = xp.take(self.statistics_, xp.nonzero(m)[1])
        return Xc

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


# ============================================================
# StandardScaler (GPU 지원, 숫자형 전용 / 객체형은 CPU로 내려 처리)
# ============================================================
class StandardScaler:
    """
    with_mean: 평균 제거
    with_std : 표준편차 나눔(0이면 1로 클리핑)
    """
    def __init__(self, with_mean: bool = True, with_std: bool = True, backend=None):
        self.with_mean = with_mean
        self.with_std = with_std
        self.backend = _check_backend(backend)

        self.mean_ = None
        self.scale_ = None

    def fit(self, X, y=None):
        X = _ensure_2d(X)
        if _is_object_like(X):
            # CPU 경로: 객체형은 스케일링 대상 아님 → 0/1로 세팅, 패스스루
            Xh = _to_host(X).astype(object)
            self.mean_ = np.zeros(Xh.shape[1], dtype=float)
            self.scale_ = np.ones(Xh.shape[1], dtype=float)
            return self

        xp = _xp_of(X)
        mean = xp.nanmean(X, axis=0) if self.with_mean else xp.zeros(X.shape[1], dtype=X.dtype)
        var = xp.nanvar(X, axis=0) if self.with_std else xp.zeros(X.shape[1], dtype=X.dtype)
        scale = xp.sqrt(var)
        # 0 division 보호
        if _is_cupy_array(scale):
            scale = cp.where(scale == 0, 1.0, scale)
        else:
            scale = np.where(scale == 0, 1.0, scale)
        self.mean_ = mean
        self.scale_ = scale
        return self

    def transform(self, X):
        assert self.mean_ is not None and self.scale_ is not None, "Call fit() first."
        X = _ensure_2d(X)
        if _is_object_like(X):
            return _to_host(X)  # 객체형은 그대로 반환
        xp = _xp_of(X)
        Z = X
        if self.with_mean:
            Z = Z - self.mean_
        if self.with_std:
            Z = Z / self.scale_
        return Z

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


# ============================================================
# OneHotEncoder (혼합 지원: 숫자형은 GPU 가능, 객체/문자형은 CPU에서 처리)
# ============================================================
class OneHotEncoder:
    """
    handle_unknown: "error" | "ignore"
    categories: "auto" (열별로 unique 학습)
    dtype: 출력 dtype (np.float32 권장)
    device_output:
        - None: 입력 배열의 장치에 맞춤 (GPU 입력이면 GPU 출력 시도)
        - "host": 항상 numpy 반환
        - "device": 가능하면 cupy 반환 (객체형/CPU 경로는 host)
    """
    def __init__(self,
                 handle_unknown: Literal["error", "ignore"] = "ignore",
                 categories: Literal["auto"] = "auto",
                 dtype: Any = np.float32,
                 device_output: Optional[Literal["host", "device"]] = None,
                 backend=None):
        self.handle_unknown = handle_unknown
        self.categories = categories
        self.dtype = dtype
        self.device_output = device_output
        self.backend = _check_backend(backend)

        # learned
        self.categories_: List[Any] = None   # per-column categories (np or cp arrays)
        self.col_is_numeric_: List[bool] = None

    def fit(self, X, y=None):
        X = _ensure_2d(X)
        n, d = X.shape
        cats = []
        col_is_num = []

        for j in range(d):
            col = X[:, j]
            if _is_object_like(col):
                # CPU: 문자열/객체
                colh = _to_host(col).astype(object)
                cj = np.unique(colh)
                cats.append(cj)
                col_is_num.append(False)
            else:
                # 숫자형: 입력이 GPU면 GPU에서 unique
                xp = _xp_of(col)
                cj = xp.unique(col)
                cats.append(cj)
                col_is_num.append(True)

        self.categories_ = cats
        self.col_is_numeric_ = col_is_num
        return self

    def transform(self, X):
        assert self.categories_ is not None, "Call fit() first."
        X = _ensure_2d(X)
        n, d = X.shape

        blocks = []
        for j in range(d):
            col = X[:, j]
            Cj = self.categories_[j]
            is_num = self.col_is_numeric_[j]

            # --- CPU (객체/문자형) ---
            if not is_num:
                colh = _to_host(col).astype(object)
                Cjh = _to_host(Cj).astype(object)

                # 각 값 -> one-hot
                Kj = len(Cjh)
                Oj = np.zeros((n, Kj), dtype=self.dtype)
                # 빠르게: 해시맵(카테고리 -> index)
                idx = {val: k for k, val in enumerate(Cjh)}
                for i, v in enumerate(colh):
                    k = idx.get(v, -1)
                    if k == -1:
                        if self.handle_unknown == "error":
                            raise ValueError(f"Unknown category {v} in column {j}")
                        else:
                            continue  # ignore -> all zeros
                    Oj[i, k] = 1.0
                blocks.append(Oj)
                continue

            # --- GPU/CPU 숫자형 ---
            xp = _xp_of(col)
            Cj_arr = Cj  # xp array(np or cp)
            Kj = Cj_arr.shape[0]

            # (n,1) vs (1,K) 브로드캐스팅 비교 → (n,K) bool
            eq = (col.reshape(-1, 1) == Cj_arr.reshape(1, -1))
            Oj = eq.astype(self.dtype)

            # unknown 처리: eq에서 모두 False → all zeros (ignore), error이면 체크
            if self.handle_unknown == "error":
                row_has_one = eq.any(axis=1)
                if bool(_to_host(row_has_one == False).any()):
                    # 어떤 행이라도 매칭 없음
                    raise ValueError(f"Unknown numeric category detected in column {j}")

            blocks.append(Oj)

        O = _concat_colwise(blocks)

        # device_output 정책
        if self.device_output == "host":
            return _to_host(O)
        if self.device_output == "device":
            return _to_device(self.backend, O)
        # None: 입력 장치에 맞춘다
        if any(_is_cupy_array(X[:, j]) for j in range(X.shape[1])) or _is_cupy_array(X):
            return _to_device(self.backend, O)
        return _to_host(O)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

# --- Add to ml/preprocessing/preprocessing.py (StandardScaler 아래에 붙이기) ---

class MinMaxScaler:
    """
    Scales each feature to a given range (default [0, 1]).
    - 숫자형은 GPU(CuPy) 지원.
    - 객체/문자형 컬럼은 변환하지 않고 그대로 통과합니다.
    """
    def __init__(self, feature_range=(0.0, 1.0), clip=False, backend=None):
        assert len(feature_range) == 2
        self.feature_range = (float(feature_range[0]), float(feature_range[1]))
        self.clip = bool(clip)
        self.backend = _check_backend(backend)

        self.data_min_ = None
        self.data_max_ = None
        self.data_range_ = None  # max - min

    def fit(self, X, y=None):
        X = _ensure_2d(X)
        if _is_object_like(X):
            # 객체형은 스케일링 대상 아님 → 0..1로 가정, 그대로 통과
            Xh = _to_host(X)
            self.data_min_ = np.zeros(Xh.shape[1], dtype=float)
            self.data_max_ = np.ones(Xh.shape[1], dtype=float)
            self.data_range_ = self.data_max_ - self.data_min_
            return self

        xp = _xp_of(X)
        self.data_min_ = xp.nanmin(X, axis=0)
        self.data_max_ = xp.nanmax(X, axis=0)
        self.data_range_ = self.data_max_ - self.data_min_
        # 0 division 방지
        if _is_cupy_array(self.data_range_):
            self.data_range_ = cp.where(self.data_range_ == 0, 1.0, self.data_range_)
        else:
            self.data_range_ = np.where(self.data_range_ == 0, 1.0, self.data_range_)
        return self

    def transform(self, X):
        assert self.data_min_ is not None, "Call fit() first."
        X = _ensure_2d(X)

        if _is_object_like(X):
            return _to_host(X)  # 객체형은 그대로 반환

        xp = _xp_of(X)
        Xs = (X - self.data_min_) / self.data_range_

        lo, hi = self.feature_range
        if hi != 1.0 or lo != 0.0:
            Xs = Xs * (hi - lo) + lo

        if self.clip:
            if _is_cupy_array(Xs):
                Xs = cp.clip(Xs, lo, hi)
            else:
                Xs = np.clip(Xs, lo, hi)
        return Xs

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
