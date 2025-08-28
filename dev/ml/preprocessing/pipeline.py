import numpy as np
from typing import List, Tuple, Any, Callable, Sequence, Optional

# -----------------
# helpers
# -----------------
def _has_method(obj, name: str) -> bool:
    return callable(getattr(obj, name, None))

def _maybe_fit(est, X, y=None):
    try:
        return est.fit(X, y)
    except TypeError:
        return est.fit(X)

def _maybe_transform(trans, X, y=None):
    if _has_method(trans, "transform"):
        try:
            return trans.transform(X, y)
        except TypeError:
            return trans.transform(X)
    raise AttributeError(f"{trans.__class__.__name__} has no transform()")

def _slice_columns(X: np.ndarray, cols) -> np.ndarray:
    if callable(cols):
        cols = cols(X)
    return X[:, cols]

def make_column_selector(dtype_include=None, dtype_exclude=None) -> Callable[[np.ndarray], Sequence[int]]:
    """
    간단한 컬럼 선택기: numpy dtype 기준으로 인덱스 반환.
    사용 예) num_sel = make_column_selector(dtype_include=np.number)
            cat_sel = make_column_selector(dtype_include=np.object_)
    """
    def _selector(X: np.ndarray):
        idxs = []
        for j in range(X.shape[1]):
            dt = X[:, j].dtype
            inc_ok = True if dtype_include is None else any(np.issubdtype(dt, inc) for inc in np.atleast_1d(dtype_include))
            exc_ok = False if dtype_exclude is None else any(np.issubdtype(dt, exc) for exc in np.atleast_1d(dtype_exclude))
            if inc_ok and not exc_ok:
                idxs.append(j)
        return idxs
    return _selector

# -----------------
# Pipeline
# -----------------
class Pipeline:
    """
    순차 파이프라인.
    steps: List[(name, transformer_or_estimator)]
      - 마지막 스텝은 보통 추정기(estimator)
      - 앞쪽은 변환기(transformer: fit/transform 필요)
    """
    def __init__(self, steps: List[Tuple[str, Any]]):
        if not steps:
            raise ValueError("Pipeline needs at least one step.")
        names = [n for n, _ in steps]
        if len(names) != len(set(names)):
            raise ValueError("Step names must be unique.")
        self.steps = steps

    # utils
    def _iter_transformers(self):
        for name, step in self.steps[:-1]:
            yield name, step

    def _final_step(self):
        return self.steps[-1][1]

    # params (경량)
    def get_params(self, deep=True):
        params = {}
        for name, step in self.steps:
            params[name] = step
            if deep and hasattr(step, "get_params"):
                sub = step.get_params()
                for k, v in sub.items():
                    params[f"{name}__{k}"] = v
        return params

    def set_params(self, **params):
        for k, v in params.items():
            if "__" in k:
                name, subk = k.split("__", 1)
                step = dict(self.steps)[name]
                if hasattr(step, "set_params"):
                    step.set_params(**{subk: v})
                else:
                    setattr(step, subk, v)
            else:
                names = [n for n, _ in self.steps]
                if k in names:
                    i = names.index(k)
                    self.steps[i] = (k, v)
                else:
                    setattr(self, k, v)
        return self

    # core
    def fit(self, X, y=None):
        Xt = X
        for _, trans in self._iter_transformers():
            _maybe_fit(trans, Xt, y)
            Xt = _maybe_transform(trans, Xt, y)
        final = self._final_step()
        _maybe_fit(final, Xt, y)
        return self

    def predict(self, X):
        Xt = X
        for _, trans in self._iter_transformers():
            Xt = _maybe_transform(trans, Xt)
        final = self._final_step()
        if not _has_method(final, "predict"):
            raise AttributeError("Final step has no predict()")
        return final.predict(Xt)

    def predict_proba(self, X):
        Xt = X
        for _, trans in self._iter_transformers():
            Xt = _maybe_transform(trans, Xt)
        final = self._final_step()
        if not _has_method(final, "predict_proba"):
            raise AttributeError("Final step has no predict_proba()")
        return final.predict_proba(Xt)

    def transform(self, X):
        Xt = X
        for _, trans in self.steps:
            Xt = _maybe_transform(trans, Xt)
        return Xt

    def fit_transform(self, X, y=None):
        Xt = X
        for _, trans in self._iter_transformers():
            _maybe_fit(trans, Xt, y)
            Xt = _maybe_transform(trans, Xt, y)
        final = self._final_step()
        if _has_method(final, "fit_transform"):
            return final.fit_transform(Xt, y)
        _maybe_fit(final, Xt, y)
        if _has_method(final, "transform"):
            return final.transform(Xt)
        return Xt

# -----------------
# ColumnTransformer
# -----------------
class ColumnTransformer:
    """
    여러 변환기를 서로 다른 컬럼 서브셋에 적용 후 수평 결합.
    transformers: List[(name, transformer, columns)]
      - columns: list[int] | slice | np.ndarray[bool] | callable(X)->indices
    remainder: 'drop' | 'passthrough'
    """
    def __init__(self, transformers: List[Tuple[str, Any, Any]], remainder: str = "drop", dtype=np.float64):
        names = [n for n, _, _ in transformers]
        if len(names) != len(set(names)):
            raise ValueError("Transformer names must be unique.")
        assert remainder in ("drop", "passthrough")
        self.transformers = transformers
        self.remainder = remainder
        self.dtype = dtype
        self._fitted = False
        self._pass_cols: Optional[np.ndarray] = None

    def _resolve_columns(self, X: np.ndarray, cols_spec) -> np.ndarray:
        if callable(cols_spec):
            cols_spec = cols_spec(X)
        if isinstance(cols_spec, slice):
            idxs = np.arange(X.shape[1])[cols_spec]
        elif isinstance(cols_spec, (list, tuple, np.ndarray)):
            arr = np.array(cols_spec)
            if arr.dtype == bool:
                idxs = np.where(arr)[0]
            else:
                idxs = arr.astype(int)
        else:
            raise ValueError("Unsupported columns spec")
        return idxs

    def _compute_passthrough(self, X, used_cols: Sequence[int]) -> np.ndarray:
        all_cols = np.arange(X.shape[1])
        mask = np.ones_like(all_cols, dtype=bool)
        mask[used_cols] = False
        return all_cols[mask]

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=object)  # 혼합 dtype 허용
        used_all = []
        for _, trans, cols in self.transformers:
            idxs = self._resolve_columns(X, cols)
            used_all.extend(list(idxs))
            X_sub = X[:, idxs]
            _maybe_fit(trans, X_sub, y)
        if self.remainder == "passthrough":
            self._pass_cols = self._compute_passthrough(X, used_all)
        self._fitted = True
        return self

    def transform(self, X):
        if not self._fitted:
            raise RuntimeError("ColumnTransformer must be fit() before transform().")
        X = np.asarray(X, dtype=object)
        blocks = []
        for _, trans, cols in self.transformers:
            idxs = self._resolve_columns(X, cols)
            X_sub = X[:, idxs]
            X_tr = _maybe_transform(trans, X_sub)
            blocks.append(np.asarray(X_tr))
        if self.remainder == "passthrough" and self._pass_cols is not None and self._pass_cols.size > 0:
            passthrough = np.asarray(X[:, self._pass_cols], dtype=self.dtype)
            blocks.append(passthrough)
        if not blocks:
            return np.empty((X.shape[0], 0), dtype=self.dtype)
        return np.hstack(blocks)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    # 경량 param API
    def get_params(self, deep=True):
        params = {"remainder": self.remainder, "dtype": self.dtype}
        for name, trans, cols in self.transformers:
            params[name] = trans
            params[f"{name}__columns"] = cols
            if deep and hasattr(trans, "get_params"):
                sub = trans.get_params()
                for k, v in sub.items():
                    params[f"{name}__{k}"] = v
        return params

    def set_params(self, **params):
        tr_map = {n: (n, t, c) for n, t, c in self.transformers}
        for k, v in params.items():
            if k in ("remainder", "dtype"):
                setattr(self, k, v)
                continue
            if "__" in k:
                name, subk = k.split("__", 1)
                n, t, c = tr_map[name]
                if subk == "columns":
                    tr_map[name] = (n, t, v)
                elif hasattr(t, "set_params"):
                    t.set_params(**{subk: v})
                else:
                    setattr(t, subk, v)
            else:
                if k in tr_map:
                    n, _, c = tr_map[k]
                    tr_map[k] = (n, v, c)
        self.transformers = list(tr_map.values())
        return self
