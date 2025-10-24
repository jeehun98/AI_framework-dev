# python/graph_executor_v2/losses/mse.py
from __future__ import annotations

from typing import Tuple, Optional
import numpy as np

from .base import Loss, Array

try:
    import cupy as cp  # type: ignore
    _HAS_GPU = True
except Exception:
    cp = None  # type: ignore[assignment]
    _HAS_GPU = False


def _mse_numpy(
    x: np.ndarray,
    y: np.ndarray,
    *,
    reduction: str,
    weight: Optional[np.ndarray] = None,
) -> Tuple[float, np.ndarray]:
    x = x.astype(np.float32, copy=False)
    y = y.astype(np.float32, copy=False)
    diff = x - y
    diff2 = diff * diff

    if weight is not None:
        w = np.asarray(weight, dtype=np.float32)
        diff2 = diff2 * w
        grad = 2.0 * diff * w
    else:
        grad = 2.0 * diff

    if reduction == "sum":
        loss_scalar = float(diff2.sum())
    elif reduction == "none":
        loss_scalar = float(diff2.mean())  # 표시용 mean
    else:  # "mean"
        loss_scalar = float(diff2.mean())
        # 평균 기준의 grad 스케일: 요소 수로 나누기
        grad = grad / float(diff.size)

    return loss_scalar, grad.astype(np.float32, copy=False)


class MeanSquaredError(Loss):
    """
    Mean Squared Error (MSE) Loss

    Args:
      reduction: 'mean' | 'sum' | 'none'
      weight: 선택적 가중치 (x/y와 브로드캐스트 가능)
    """
    def __init__(
        self,
        reduction: str = "mean",
        weight: Optional[Array] = None,
    ):
        assert reduction in ("mean", "sum", "none")
        self.reduction = reduction
        self.weight = weight

    def forward(self, y_pred: Array, y_true: Array) -> Tuple[float, Array]:
        # GPU 경로
        if _HAS_GPU:
            try:
                if isinstance(y_pred, cp.ndarray):  # type: ignore[attr-defined]
                    x = y_pred.astype(cp.float32, copy=False)  # type: ignore[attr-defined]
                    y = cp.asarray(y_true, dtype=cp.float32)  # type: ignore[attr-defined]

                    if self.weight is not None:
                        w = cp.asarray(self.weight, dtype=cp.float32)  # type: ignore[attr-defined]
                        diff = x - y
                        diff2 = (diff * diff) * w
                        grad = 2.0 * diff * w
                    else:
                        diff = x - y
                        diff2 = diff * diff
                        grad = 2.0 * diff

                    if self.reduction == "sum":
                        loss_scalar = float(diff2.sum().item())  # type: ignore[call-arg]
                    elif self.reduction == "none":
                        loss_scalar = float(diff2.mean().item())
                    else:
                        loss_scalar = float(diff2.mean().item())
                        grad = grad / float(diff.size)

                    return loss_scalar, grad  # type: ignore[return-value]
            except Exception:
                pass  # 실패 시 CPU 폴백

        # CPU 폴백
        x_np = np.asarray(y_pred, dtype=np.float32)
        y_np = np.asarray(y_true, dtype=np.float32)
        w_np = np.asarray(self.weight, dtype=np.float32) if self.weight is not None else None
        return _mse_numpy(x_np, y_np, reduction=self.reduction, weight=w_np)


__all__ = ["MeanSquaredError"]
