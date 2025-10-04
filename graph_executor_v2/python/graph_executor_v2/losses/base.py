# graph_executor_v2/losses/base.py
from __future__ import annotations
from typing import Tuple, Union

try:
    import cupy as cp  # type: ignore
    _HAS_CUPY = True
except Exception:
    cp = None  # type: ignore
    _HAS_CUPY = False

import numpy as np

Array = Union[np.ndarray, "cp.ndarray"]


class Loss:
    """
    공통 Loss 베이스.
    - forward(y_pred, y_true) -> (loss_scalar: float, grad: Array) 를 구현하세요.
    - __call__ 에서 return_grad 로 스칼라만/튜플 반환을 선택할 수 있습니다.
    """

    def __call__(self, y_pred: Array, y_true: Array, *, return_grad: bool = True):
        """
        기본 호출자.
          - return_grad=True  -> (loss_scalar, grad) 튜플 반환
          - return_grad=False -> loss_scalar (float)만 반환
        """
        loss, grad = self.forward(y_pred, y_true)
        return (loss, grad) if return_grad else loss

    def forward(self, y_pred: Array, y_true: Array) -> Tuple[float, Array]:
        """
        모든 하위 클래스는 (loss_scalar, grad) 를 반환해야 합니다.
          - loss_scalar: Python float (장비 무관)
          - grad: y_pred 와 동일 백엔드(CuPy/NumPy), 동일 shape(dtype=float32 권장)
        """
        raise NotImplementedError

    # --------- 유틸 (선택 사용) ---------
    @staticmethod
    def _as_backend(x: np.ndarray, ref: Array) -> Array:
        """NumPy 배열 x를 ref와 동일 백엔드(CuPy/NumPy)로 변환."""
        if _HAS_CUPY and isinstance(ref, cp.ndarray):  # type: ignore
            return cp.asarray(x)  # type: ignore
        return x

    @staticmethod
    def _ensure_float32(a: Array) -> Array:
        """float32 강제(복사 없이)."""
        if a.dtype == np.float32:
            return a
        if _HAS_CUPY and isinstance(a, cp.ndarray):  # type: ignore
            return a.astype(cp.float32, copy=False)  # type: ignore
        return a.astype(np.float32, copy=False)
