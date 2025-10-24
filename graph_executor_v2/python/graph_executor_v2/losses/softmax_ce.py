# python/graph_executor_v2/losses/softmax_ce.py
from __future__ import annotations

from typing import Tuple
import numpy as np

# 공통 타입/베이스
from .base import Loss, Array

# CUDA 경로: CuPy + ops.cross_entropy 사용 가능 여부 확인
try:
    import cupy as cp  # type: ignore
    from graph_executor_v2.ops import cross_entropy as ce_ops  # type: ignore
    _HAS_GPU = True
except Exception:
    cp = None  # type: ignore[assignment]
    ce_ops = None  # type: ignore[assignment]
    _HAS_GPU = False


# --------- NumPy 폴백 구현 ---------
def _numpy_logsumexp(x: np.ndarray, axis: int = -1, keepdims: bool = False) -> np.ndarray:
    m = x.max(axis=axis, keepdims=True)
    z = x - m
    y = np.log(np.exp(z).sum(axis=axis, keepdims=True)) + m
    return y if keepdims else y.squeeze(axis)


def _numpy_forward(
    logits: np.ndarray,
    target_idx: np.ndarray,
    *,
    from_logits: bool,
    reduction: str,
    ignore_index: int,
    eps: float,
    ls_eps: float,
) -> Tuple[float, np.ndarray]:
    x = logits.astype(np.float32, copy=False)
    t = target_idx.astype(np.int64, copy=False)
    N, C = x.shape

    valid = (t != ignore_index)
    n_valid = int(valid.sum())
    n_valid = max(n_valid, 1)  # div0 방지

    # one-hot + label smoothing
    oh = np.zeros((N, C), np.float32)
    t_clip = np.clip(t, 0, C - 1)
    oh[np.arange(N), t_clip] = 1.0
    if ls_eps > 0.0:
        oh = (1.0 - ls_eps) * oh + (ls_eps / C)

    if from_logits:
        # log-softmax
        lse = _numpy_logsumexp(x, axis=1, keepdims=True)
        logp = x - lse
        loss_all = -(oh * logp).sum(axis=1)
        # grad = softmax - target
        p = np.exp(logp)
        dx = (p - oh)
    else:
        # 입력이 확률
        p = np.clip(x, eps, 1.0)
        loss_all = -(oh * np.log(p)).sum(axis=1)
        dx = -(oh / p)

    # ignore_index 마스킹
    loss_all = np.where(valid, loss_all, 0.0).astype(np.float32)
    dx *= valid[:, None].astype(np.float32)

    # scalar reduction (항상 Python float 반환)
    if reduction == "sum":
        loss_scalar = float(loss_all.sum())
    elif reduction == "none":
        # 베이스 규약상 스칼라는 항상 float -> 표시용 mean 반환
        loss_scalar = float(loss_all.mean())
    else:  # "mean"
        loss_scalar = float(loss_all.sum() / n_valid)

    # backward 스케일: reduction='mean'이면 n_valid로 나눔
    if reduction == "mean":
        dx /= n_valid
    # 'sum' / 'none' -> 스케일 없음

    return loss_scalar, dx.astype(np.float32, copy=False)


# --------- 공개 Loss 클래스 ---------
class SoftmaxCrossEntropy(Loss):
    """
    Softmax + Cross-Entropy (GPU가용 시 CUDA 커널 사용, 불가 시 NumPy 폴백)

    Args:
      label_smoothing: float in [0, 1]
      reduction: 'mean' | 'sum' | 'none'
      ignore_index: 무시할 라벨 인덱스(해당 샘플은 loss/grad에서 제외)
      from_logits: True면 내부에서 log-softmax 처리 (일반적)
      eps: 확률 입력일 때(clipped 확률) 안정성 epsilon
    """
    def __init__(
        self,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
        ignore_index: int = -1,
        from_logits: bool = True,
        eps: float = 1e-7,
    ):
        assert reduction in ("mean", "sum", "none")
        self.ls_eps = float(label_smoothing)
        self.reduction = reduction
        self.ignore_index = int(ignore_index)
        self.from_logits = bool(from_logits)
        self.eps = float(eps)

    def forward(self, logits: Array, target_idx: Array) -> Tuple[float, Array]:
        # ---------- GPU 경로 ----------
        if _HAS_GPU:
            try:
                if isinstance(logits, cp.ndarray):  # type: ignore[attr-defined]
                    x = logits
                    t = target_idx
                    if x.dtype != cp.float32:  # type: ignore[attr-defined]
                        x = x.astype(cp.float32, copy=False)  # type: ignore[attr-defined]
                    # 정수 인덱스는 int32 (CUDA 커널과 합의)
                    if t.dtype != cp.int32:  # type: ignore[attr-defined]
                        t = t.astype(cp.int32, copy=False)  # type: ignore[attr-defined]

                    loss_dev = ce_ops.forward(  # type: ignore[union-attr, call-arg]
                        x, t,
                        from_logits=self.from_logits,
                        reduction=self.reduction,
                        ignore_index=self.ignore_index,
                        eps=self.eps,
                        ls_eps=self.ls_eps,
                    )
                    dx = ce_ops.backward(  # type: ignore[union-attr, call-arg]
                        x, t,
                        from_logits=self.from_logits,
                        reduction=self.reduction,
                        ignore_index=self.ignore_index,
                        eps=self.eps,
                        ls_eps=self.ls_eps,
                    )

                    # 항상 Python float 반환
                    if self.reduction == "none":
                        loss_scalar = float(cp.mean(loss_dev))  # type: ignore[attr-defined]
                    else:
                        loss_scalar = float(loss_dev.ravel()[0])  # type: ignore[call-arg]
                    return loss_scalar, dx
            except Exception:
                # GPU 경로 실패 시 폴백 (안전장치)
                pass

        # ---------- CPU 폴백 ----------
        logits_np = np.asarray(logits, dtype=np.float32)
        target_np = np.asarray(target_idx)
        return _numpy_forward(
            logits_np,
            target_np,
            from_logits=self.from_logits,
            reduction=self.reduction,
            ignore_index=self.ignore_index,
            eps=self.eps,
            ls_eps=self.ls_eps,
        )


__all__ = ["SoftmaxCrossEntropy"]
