# python/graph_executor_v2/losses/softmax_ce.py
from __future__ import annotations
from typing import Tuple
import numpy as np

# (선택) Loss 베이스가 있으면 임포트, 없으면 더미 베이스로 대체
try:
    from .base import Loss  # type: ignore
except Exception:
    class Loss:
        pass

# CUDA 경로: CuPy + ops.cross_entropy 사용 가능 여부 확인
try:
    import cupy as cp  # type: ignore
    from graph_executor_v2.ops import cross_entropy as ce_ops  # type: ignore
    _HAS_GPU = True
except Exception:
    cp = None
    ce_ops = None
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
    n_valid = max(n_valid, 1)

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

    # reduction
    if reduction == "none":
        loss_scalar = float(loss_all.mean())  # 관찰 편의를 위해 평균값을 반환(스칼라)
    elif reduction == "sum":
        loss_scalar = float(loss_all.sum())
    else:  # "mean"
        loss_scalar = float(loss_all.sum() / n_valid)

    # backward 스케일: reduction='mean'이면 n_valid로 나눔, 'sum'이면 그대로, 'none'이면 추가 스케일 없음
    if reduction == "mean":
        dx /= n_valid

    return loss_scalar, dx.astype(np.float32, copy=False)


# --------- 공개 Loss 클래스 ---------
class SoftmaxCrossEntropy(Loss):
    """
    Softmax + Cross-Entropy (GPU가용시 CUDA 커널 사용, 불가 시 NumPy 폴백)

    Args:
      label_smoothing: float in [0,1]
      reduction: 'mean' | 'sum' | 'none'
      ignore_index: 무시할 라벨 인덱스
      from_logits: True면 내부에서 log-softmax 처리
      eps: 확률 입력일 때의 안정성 epsilon
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

    def forward(self, logits, target_idx, *, return_scalar: bool = True):
        if _HAS_GPU and isinstance(logits, cp.ndarray):
            x = logits
            t = target_idx
            if x.dtype != cp.float32:
                x = x.astype(cp.float32, copy=False)
            if t.dtype != cp.int32:
                t = t.astype(cp.int32, copy=False)

            loss_dev = ce_ops.forward(
                x, t,
                from_logits=self.from_logits,
                reduction=self.reduction,
                ignore_index=self.ignore_index,
                eps=self.eps,
                ls_eps=self.ls_eps,
            )
            dx = ce_ops.backward(
                x, t,
                from_logits=self.from_logits,
                reduction=self.reduction,
                ignore_index=self.ignore_index,
                eps=self.eps,
                ls_eps=self.ls_eps,
            )

            if not return_scalar:
                return loss_dev, dx

            if self.reduction == "none":
                return float(cp.mean(loss_dev)), dx
            return float(loss_dev.ravel()[0]), dx