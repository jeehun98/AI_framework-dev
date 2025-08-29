# ml/preprocessing/pipeline.py

import numpy as np
from typing import List, Any, Callable, Sequence, Optional
from backend import CPUBackend
import inspect

# 기존 _maybe_fit 함수 교체
def _maybe_fit(est, X, y=None):
    """
    y 전달 여부를 시그니처로 '정적' 판별.
    내부 런타임 예외(TypeError 포함)는 절대 삼키지 않고 그대로 올린다.
    """
    try:
        sig = inspect.signature(est.fit)
    except Exception:
        # 시그니처를 못 읽으면 보수적으로 y 없이 호출
        return est.fit(X) if y is None else est.fit(X, y)

    params = sig.parameters
    if "y" in params and y is not None:
        return est.fit(X, y)
    else:
        return est.fit(X)
# -----------------
# Backend helpers
# -----------------
def _check_backend(be):
    return be or CPUBackend()

def _is_cuda_backend(be) -> bool:
    return be.__class__.__name__.lower().startswith("cuda")

def _maybe_to_device(be, x, device_policy: str):
    if device_policy == "device":
        return be.to_device(x)
    return x

def _maybe_to_host(be, x):
    return be.to_host(x)

def _propagate_backend_to_step(step, backend):
    """스텝(변환기/추정기/하위 파이프라인)에 backend 속성이 있으면 주입"""
    if hasattr(step, "backend"):
        setattr(step, "backend", backend)
    return step

# -----------------
# small utils
# -----------------
def _has_method(obj, name: str) -> bool:
    return callable(getattr(obj, name, None))

def _to_host_if_cupy(x):
    try:
        import cupy as cp
        if isinstance(x, cp.ndarray):
            return cp.asnumpy(x)
    except Exception:
        pass
    return x

def _prepare_xy_for_step(step, X, y=None):
    """
    step이 backend 속성을 가지면 (우리 GPU/CPU 추정기) 그대로 전달.
    그 외(NumPy 전용 전처리기 등)는 cupy -> numpy로 안전 변환.
    """
    if hasattr(step, "backend"):
        return X, y
    Xh = _to_host_if_cupy(X)
    yh = _to_host_if_cupy(y)
    return Xh, yh

def _maybe_fit(est, X, y=None):
    Xp, yp = _prepare_xy_for_step(est, X, y)
    try:
        return est.fit(Xp, yp)
    except TypeError:
        return est.fit(Xp)

def _maybe_transform(trans, X, y=None):
    Xp, yp = _prepare_xy_for_step(trans, X, y)
    if _has_method(trans, "transform"):
        try:
            return trans.transform(Xp, yp)
        except TypeError:
            return trans.transform(Xp)
    raise AttributeError(f"{trans.__class__.__name__} has no transform()")

def _maybe_fit_transform(trans, X, y=None):
    """fit_transform이 y를 안 받는 경우를 포함해 안전 호출"""
    Xp, yp = _prepare_xy_for_step(trans, X, y)
    if _has_method(trans, "fit_transform"):
        try:
            return trans.fit_transform(Xp, yp)
        except TypeError:
            return trans.fit_transform(Xp)
    if _has_method(trans, "fit") and _has_method(trans, "transform"):
        _maybe_fit(trans, Xp, yp)
        return _maybe_transform(trans, Xp, yp)
    raise AttributeError(f"{trans.__class__.__name__} has neither fit_transform nor (fit+transform)")

def _slice_columns(X: np.ndarray, cols) -> np.ndarray:
    if callable(cols):
        cols = cols(X)
    return X[:, cols]

def _safe_concatenate(arrs: List[Any], axis: int, be) -> Any:
    """GPU 배열 있으면 cupy.concatenate, 아니면 numpy.concatenate"""
    try:
        import cupy as cp
        if any(isinstance(a, cp.ndarray) for a in arrs):
            arrs_dev = [be.to_device(a) for a in arrs]
            return cp.concatenate(arrs_dev, axis=axis)
    except Exception:
        pass
    return np.concatenate([_to_host_if_cupy(a) for a in arrs], axis=axis)

# -----------------
# Column selector
# -----------------
def make_column_selector(dtype_include=None, dtype_exclude=None) -> Callable[[np.ndarray], Sequence[int]]:
    """
    간단한 컬럼 선택기: numpy dtype 기준으로 인덱스 반환.
    사용 예) num_sel = make_column_selector(dtype_include=np.number)
            cat_sel = make_column_selector(dtype_include=np.object_)
    """
    def _selector(X: np.ndarray):
        Xh = _to_host_if_cupy(X)  # dtype 판정은 host에서
        idxs = []
        for j in range(Xh.shape[1]):
            dt = Xh[:, j].dtype
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
    def __init__(self, steps, backend=None, device_policy: str = "host"):
        """
        steps: [("name", transformer_or_estimator), ...]
        backend: CPUBackend() | CUDABackend()
        device_policy:
            - "host": transform 결과를 항상 host(np)로 유지 (기본)
            - "device": transform 결과를 device(cupy)로 유지
            - "auto": 마지막 단계가 GPU면 device 유지, 아니면 host
        """
        self.steps = list(steps)
        self.backend = _check_backend(backend)
        assert device_policy in ("host", "device", "auto")
        self.device_policy = device_policy

        # 백엔드 전파
        for i, (name, step) in enumerate(self.steps):
            self.steps[i] = (name, _propagate_backend_to_step(step, self.backend))

    def _last(self):
        return self.steps[-1]

    def _final_device_policy(self, last_estimator_backend_is_gpu: bool) -> str:
        if self.device_policy in ("host", "device"):
            return self.device_policy
        return "device" if last_estimator_backend_is_gpu else "host"

    def fit(self, X, y=None):
        Xt = X
        # 앞 단계: 변환기들
        for name, step in self.steps[:-1]:
            if hasattr(step, "fit_transform"):
                Xt = _maybe_fit_transform(step, Xt, y)
            else:
                if hasattr(step, "fit"):
                    _maybe_fit(step, Xt, y)
                Xt = _maybe_transform(step, Xt, y)
        # 마지막 단계: 추정기 (있다면)
        last_name, last_step = self._last()
        last_is_estimator = (_has_method(last_step, "fit")
                             and (_has_method(last_step, "predict") or _has_method(last_step, "transform")))
        if last_is_estimator:
            _maybe_fit(last_step, Xt, y)
        return self

    def transform(self, X):
        be = self.backend
        Xt = X
        for name, step in self.steps:
            if hasattr(step, "transform"):
                Xt = _maybe_transform(step, Xt)
            elif hasattr(step, "predict"):
                break
        # device_policy 적용
        last_backend = getattr(self._last()[1], "backend", self.backend)
        want = self._final_device_policy(_is_cuda_backend(last_backend))
        if want == "device":
            Xt = be.to_device(Xt)
        else:
            Xt = _maybe_to_host(be, Xt)
        return Xt

    def fit_transform(self, X, y=None):
        be = self.backend
        Xt = X
        for name, step in self.steps[:-1]:
            if hasattr(step, "fit_transform"):
                Xt = _maybe_fit_transform(step, Xt, y)
            else:
                if hasattr(step, "fit"):
                    _maybe_fit(step, Xt, y)
                Xt = _maybe_transform(step, Xt, y)
        # 마지막 스텝이 변환기면 거기까지 변환
        last_name, last_step = self._last()
        if hasattr(last_step, "transform") and not hasattr(last_step, "predict"):
            if hasattr(last_step, "fit"):
                _maybe_fit(last_step, Xt, y)
            Xt = _maybe_transform(last_step, Xt, y)

        # device_policy 적용
        last_backend = getattr(last_step, "backend", self.backend)
        want = self._final_device_policy(_is_cuda_backend(last_backend))
        if want == "device":
            Xt = be.to_device(Xt)
        else:
            Xt = _maybe_to_host(be, Xt)
        return Xt

    def predict(self, X):
        Xt = X
        for name, step in self.steps[:-1]:
            if hasattr(step, "transform"):
                Xt = _maybe_transform(step, Xt)
        last_name, last_step = self._last()
        if hasattr(last_step, "predict"):
            return last_step.predict(Xt)
        elif hasattr(last_step, "transform"):
            return last_step.transform(Xt)
        else:
            raise AttributeError("Last step has neither predict nor transform")

    def predict_proba(self, X):
        Xt = X
        for name, step in self.steps[:-1]:
            if hasattr(step, "transform"):
                Xt = _maybe_transform(step, Xt)
        last_name, last_step = self._last()
        if hasattr(last_step, "predict_proba"):
            return last_step.predict_proba(Xt)
        raise AttributeError("Last step has no predict_proba()")

# -----------------
# ColumnTransformer
# -----------------
class ColumnTransformer:
    def __init__(self, transformers, remainder: str = "drop", backend=None, device_policy: str = "host"):
        """
        transformers: [("name", transformer_or_pipeline, column_selector), ...]
                      column_selector: callable(X)->indices | list/array of indices
        remainder: "drop" | "passthrough"
        backend: CPUBackend | CUDABackend
        device_policy: "host" | "device" | "auto"  (transform 출력 유지 정책)
        """
        self.transformers = list(transformers)
        assert remainder in ("drop", "passthrough")
        self.remainder = remainder
        self.backend = _check_backend(backend)
        assert device_policy in ("host", "device", "auto")
        self.device_policy = device_policy

        # 내부 변환기/파이프라인에도 backend 전파
        for i, (name, trans, cols) in enumerate(self.transformers):
            self.transformers[i] = (name, _propagate_backend_to_step(trans, self.backend), cols)

    def _selected_indices(self, X):
        """모든 transformer가 선택한 컬럼 인덱스 합집합 반환"""
        Xh = _to_host_if_cupy(X)
        chosen = set()
        for _, _, cols in self.transformers:
            idxs = cols(Xh) if callable(cols) else cols
            for j in np.atleast_1d(idxs):
                chosen.add(int(j))
        return sorted(chosen)

    def fit(self, X, y=None):
        Xh = _to_host_if_cupy(X)  # 전처리는 기본적으로 host에서
        for name, trans, cols in self.transformers:
            X_sub = _slice_columns(Xh, cols if not callable(cols) else cols)
            if callable(cols):
                X_sub = _slice_columns(Xh, cols)
            else:
                X_sub = Xh[:, cols]
            if hasattr(trans, "fit"):
                _maybe_fit(trans, X_sub, y)
        return self

    def transform(self, X):
        be = self.backend
        Xh = _to_host_if_cupy(X)  # 전처리는 host에서 수행
        outs = []
        for name, trans, cols in self.transformers:
            X_sub = _slice_columns(Xh, cols if not callable(cols) else cols)
            if callable(cols):
                X_sub = _slice_columns(Xh, cols)
            else:
                X_sub = Xh[:, cols]
            if hasattr(trans, "transform"):
                outs.append(_maybe_transform(trans, X_sub))
            elif hasattr(trans, "fit_transform"):
                outs.append(_maybe_fit_transform(trans, X_sub))
            else:
                outs.append(X_sub)

        # remainder 처리
        if self.remainder == "passthrough":
            all_idx = set(range(Xh.shape[1]))
            used_idx = set(self._selected_indices(Xh))
            rem_idx = sorted(all_idx - used_idx)
            if len(rem_idx) > 0:
                outs.append(Xh[:, rem_idx])

        # concat
        out = _safe_concatenate(outs, axis=1, be=be)

        # device_policy 적용
        if self.device_policy == "device":
            return be.to_device(out)
        if self.device_policy == "host":
            return _maybe_to_host(be, out)
        # "auto": 호출자 선택에 맡김(그대로 반환)
        return out

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
