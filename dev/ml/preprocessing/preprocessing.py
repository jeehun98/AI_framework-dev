import numpy as np
from itertools import combinations, combinations_with_replacement


# ------------------------
# Scaling / Normalization
# ------------------------

class StandardScaler:
    """평균=0, 표준편차=1 로 표준화"""
    def __init__(self, with_mean=True, with_std=True):
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0) if self.with_mean else np.zeros(X.shape[1])
        if self.with_std:
            scale = X.std(axis=0, ddof=0)
            scale[scale == 0] = 1.0
            self.scale_ = scale
        else:
            self.scale_ = np.ones(X.shape[1])
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class MinMaxScaler:
    """데이터를 [min, max] 구간으로 선형 스케일"""
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.min_ = None
        self.scale_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        dmin = X.min(axis=0)
        dmax = X.max(axis=0)
        drange = dmax - dmin
        drange[drange == 0] = 1.0
        self.min_ = dmin
        self.scale_ = drange
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        X_std = (X - self.min_) / self.scale_
        a, b = self.feature_range
        return X_std * (b - a) + a

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class RobustScaler:
    """중앙값 + IQR 기반 스케일링 (이상치에 강건)"""
    def __init__(self, with_centering=True, with_scaling=True):
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.center_ = None
        self.scale_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.center_ = np.median(X, axis=0) if self.with_centering else np.zeros(X.shape[1])
        if self.with_scaling:
            q75, q25 = np.percentile(X, [75, 25], axis=0)
            iqr = q75 - q25
            iqr[iqr == 0] = 1.0
            self.scale_ = iqr
        else:
            self.scale_ = np.ones(X.shape[1])
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        return (X - self.center_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class Normalizer:
    """샘플 단위 정규화 (L1, L2, max)"""
    def __init__(self, norm="l2"):
        assert norm in ("l1", "l2", "max")
        self.norm = norm

    def fit(self, X):
        return self  # 학습 파라미터 없음

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        if self.norm == "l1":
            norms = np.sum(np.abs(X), axis=1, keepdims=True)
        elif self.norm == "l2":
            norms = np.sqrt(np.sum(X**2, axis=1, keepdims=True))
        else:  # max
            norms = np.max(np.abs(X), axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return X / norms

    def fit_transform(self, X):
        return self.fit(X).transform(X)


# ------------------------
# Imputation
# ------------------------

class SimpleImputer:
    """
    결측치 대치: strategy in {'mean','median','most_frequent','constant'}
    - numeric에 mean/median 권장
    - constant는 fill_value 사용
    """
    def __init__(self, strategy="mean", fill_value=None):
        assert strategy in ("mean", "median", "most_frequent", "constant")
        self.strategy = strategy
        self.fill_value = fill_value
        self.statistics_ = None  # 각 컬럼별 대치값

    def fit(self, X):
        X = np.asarray(X, dtype=object)  # 숫자/문자 혼재 가능성
        n_features = X.shape[1]
        stats = []

        for j in range(n_features):
            col = X[:, j]
            mask = self._isnan(col)
            valid = col[~mask]

            if self.strategy == "mean":
                valid_num = valid.astype(np.float64)
                stats.append(np.nan if valid_num.size == 0 else np.mean(valid_num))
            elif self.strategy == "median":
                valid_num = valid.astype(np.float64)
                stats.append(np.nan if valid_num.size == 0 else np.median(valid_num))
            elif self.strategy == "most_frequent":
                if valid.size == 0:
                    stats.append(np.nan)
                else:
                    # 최빈값 (동률 시 첫 번째)
                    values, counts = np.unique(valid, return_counts=True)
                    stats.append(values[np.argmax(counts)])
            else:  # constant
                stats.append(self.fill_value)

        # NaN 남아있으면 0/"" 등으로 보수적으로 처리
        for i, v in enumerate(stats):
            if isinstance(v, float) and np.isnan(v):
                stats[i] = 0.0
            if v is None:
                stats[i] = 0.0

        self.statistics_ = np.array(stats, dtype=object)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=object)
        X_out = X.copy()
        for j in range(X.shape[1]):
            col = X_out[:, j]
            mask = self._isnan(col)
            if np.any(mask):
                col[mask] = self.statistics_[j]
                X_out[:, j] = col
        return X_out

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    @staticmethod
    def _isnan(col):
        # 숫자/문자 혼재 호환: None 또는 np.nan
        return np.array([ (c is None) or (isinstance(c, float) and np.isnan(c)) for c in col ])


# ------------------------
# Categorical Encoding
# ------------------------

class OneHotEncoder:
    """
    범주형 → 원-핫.
    - handle_unknown: {'error', 'ignore'} (미학습 범주는 무시)
    - categories: 'auto' 또는 각 열별 카테고리 리스트의 리스트
    """
    def __init__(self, handle_unknown="ignore", categories="auto", drop=None, dtype=np.float64):
        assert handle_unknown in ("ignore", "error")
        self.handle_unknown = handle_unknown
        self.categories = categories
        self.drop = drop  # ('first' 지원 또는 None). 특정 카테고리 드롭은 확장 가능.
        self.dtype = dtype

        self.categories_ = None   # 학습된 각 열별 카테고리 배열
        self.feature_indices_ = None  # 출력에서 각 원-핫 블록 시작 인덱스

    def fit(self, X):
        X = self._to_2d_object(X)
        n_features = X.shape[1]
        cats = []

        if self.categories == "auto":
            for j in range(n_features):
                col = X[:, j]
                # None/NaN 제거 후 카테고리 수집
                mask = SimpleImputer._isnan(col)
                vals = col[~mask]
                cats_j = np.unique(vals)
                cats.append(cats_j)
        else:
            # 사용자가 직접 제공
            assert isinstance(self.categories, (list, tuple)) and len(self.categories) == n_features
            cats = [np.array(c) for c in self.categories]

        # drop='first' 지원: 각 열에서 첫 카테고리 제거
        if self.drop == "first":
            cats = [c[1:] if len(c) > 0 else c for c in cats]

        self.categories_ = cats

        # 출력 feature 인덱스 누적합 기록
        sizes = [len(c) for c in self.categories_]
        self.feature_indices_ = np.cumsum([0] + sizes)
        return self

    def transform(self, X):
        X = self._to_2d_object(X)
        n_samples, n_features = X.shape
        out_dim = self.feature_indices_[-1]
        out = np.zeros((n_samples, out_dim), dtype=self.dtype)

        for j in range(n_features):
            cats_j = self.categories_[j]
            if len(cats_j) == 0:
                continue
            start = self.feature_indices_[j]
            end = self.feature_indices_[j+1]
            span = end - start

            col = X[:, j]
            for i, val in enumerate(col):
                if (val is None) or (isinstance(val, float) and np.isnan(val)):
                    # 결측은 전부 0 (따로 Imputer 쓰는 걸 권장)
                    continue
                # 카테고리 매칭
                idx = np.where(cats_j == val)[0]
                if idx.size == 0:
                    if self.handle_unknown == "error":
                        raise ValueError(f"Unknown category {val} in column {j}")
                    else:
                        continue  # ignore → 전부 0
                out[i, start + int(idx[0])] = 1
        return out

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    @staticmethod
    def _to_2d_object(X):
        X = np.asarray(X, dtype=object)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X


# ------------------------
# Feature Expansion
# ------------------------

class PolynomialFeatures:
    """
    다항 특성 확장.
    - degree: 최대 차수
    - include_bias: 상수항(1) 포함 여부
    - interaction_only: 상호작용항만(제곱/세제곱 등 자기항 제외)
    """
    def __init__(self, degree=2, include_bias=True, interaction_only=False):
        assert degree >= 1
        self.degree = degree
        self.include_bias = include_bias
        self.interaction_only = interaction_only
        self.n_input_features_ = None
        self.n_output_features_ = None
        self.powers_ = None  # 각 출력 특성의 지수 벡터 목록 (shape: [n_out, n_in])

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n_features = X.shape[1]
        self.n_input_features_ = n_features

        # 지수 조합 생성
        powers = []
        if self.include_bias:
            powers.append(np.zeros(n_features, dtype=int))  # bias term

        # degree 1..d
        if self.interaction_only:
            # 각 차수 k에서 중복 없는 조합
            for k in range(1, self.degree + 1):
                for comb in combinations(range(n_features), k):
                    p = np.zeros(n_features, dtype=int)
                    for idx in comb:
                        p[idx] += 1
                    powers.append(p)
        else:
            # 중복 허용 조합 (자기항 제곱 등 포함)
            for k in range(1, self.degree + 1):
                for comb in combinations_with_replacement(range(n_features), k):
                    p = np.zeros(n_features, dtype=int)
                    for idx in comb:
                        p[idx] += 1
                    powers.append(p)

        self.powers_ = np.vstack(powers) if len(powers) > 0 else np.zeros((0, n_features), dtype=int)
        self.n_output_features_ = self.powers_.shape[0]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n_samples = X.shape[0]
        Z = np.ones((n_samples, self.n_output_features_), dtype=np.float64)

        for j, p in enumerate(self.powers_):
            if np.all(p == 0):
                Z[:, j] = 1.0  # bias
            else:
                # 각 입력 특성에 대해 거듭제곱 후 곱
                cols = []
                for f_idx, power in enumerate(p):
                    if power == 0:
                        continue
                    cols.append(np.power(X[:, f_idx], power))
                prod = cols[0]
                for c in cols[1:]:
                    prod = prod * c
                Z[:, j] = prod
        return Z

    def fit_transform(self, X):
        return self.fit(X).transform(X)
