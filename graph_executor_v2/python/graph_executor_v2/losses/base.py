# python/graph_executor_v2/losses/base.py
from __future__ import annotations

from typing import Tuple, Union, TYPE_CHECKING
import numpy as np

# 타입 검사기에는 CuPy 타입을 인식시키되, 런타임에서는 의존성 강제가 없도록 처리
if TYPE_CHECKING:
    import cupy as cp
else:
    cp = None  # type: ignore[assignment]

Array = Union[np.ndarray, "cp.ndarray"]  # Pylance OK (cp는 TYPE_CHECKING에서만 의미 있음)


class Loss:
    """
    공통 Loss 베이스 클래스.

    규약:
      - forward(y_pred, y_true) -> (loss_scalar: float, grad: Array)
      - __call__(..., return_grad=True) -> (float, grad) 또는 float

    반환 규약:
      - loss_scalar: Python float (장비/백엔드 무관)
      - grad: y_pred와 동일 백엔드(CuPy/NumPy), dtype=float32 권장
    """

    def __call__(self, y_pred: Array, y_true: Array, *, return_grad: bool = True):
        loss, grad = self.forward(y_pred, y_true)
        return (loss, grad) if return_grad else loss

    def forward(self, y_pred: Array, y_true: Array) -> Tuple[float, Array]:
        """
        모든 하위 클래스는 (loss_scalar, grad)를 반환해야 합니다.
        """
        raise NotImplementedError

    # ---------- 유틸리티 ----------

    @staticmethod
    def _as_backend(x: np.ndarray, ref: Array) -> Array:
        """
        NumPy 배열 x를 ref와 동일 백엔드(CuPy/NumPy)로 변환.
        """
        # 런타임에서는 cp가 None일 수 있으므로 안전하게 체크
        try:
            import cupy as cp  # type: ignore
            if isinstance(ref, cp.ndarray):  # type: ignore[attr-defined]
                return cp.asarray(x)  # type: ignore[attr-defined]
        except Exception:
            pass
        return x

    @staticmethod
    def _ensure_float32(a: Array) -> Array:
        """
        dtype을 float32로 강제(가능하면 복사 없이).
        """
        if a.dtype == np.float32:
            return a
        try:
            import cupy as cp  # type: ignore
            if isinstance(a, cp.ndarray):  # type: ignore[attr-defined]
                return a.astype(cp.float32, copy=False)  # type: ignore[attr-defined]
        except Exception:
            pass
        return a.astype(np.float32, copy=False)


__all__ = ["Array", "Loss"]
