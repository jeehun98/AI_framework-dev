# python/graph_executor_v2/losses/softmax_ce.py
from __future__ import annotations

from typing import Tuple
import numpy as np

# 공통 타입/베이스
from .base import Loss, Array

# --- GPU 사용 가능성 감지 (cupy 유무와 C++ 커널 유무를 분리) ---
try:
    import cupy as cp  # type: ignore
    CUPY_AVAILABLE = True
except Exception:
    cp = None  # type: ignore[assignment]
    CUPY_AVAILABLE = False

try:
    from graph_executor_v2.ops import cross_entropy as ce_ops  # type: ignore
    CE_OPS_AVAILABLE = True
except Exception:
    ce_ops = None  # type: ignore[assignment]
    CE_OPS_AVAILABLE = False


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
    return_scalar: bool,
):
    x = logits.astype(np.float32, copy=False)
    t = target_idx.astype(np.int64, copy=False)
    N, C = x.shape

    valid = (t != ignore_index)
    n_valid = int(valid.sum())
    n_valid = max(n_valid, 1)

    oh = np.zeros((N, C), np.float32)
    t_clip = np.clip(np.where(valid, t, 0), 0, C - 1)
    oh[np.arange(N), t_clip] = 1.0
    if ls_eps > 0.0:
        oh = (1.0 - ls_eps) * oh + (ls_eps / C)

    if from_logits:
        lse = _numpy_logsumexp(x, axis=1, keepdims=True)
        logp = x - lse
        loss_all = -(oh * logp).sum(axis=1).astype(np.float32)
        p = np.exp(logp)
        dx = (p - oh).astype(np.float32)
    else:
        p = np.clip(x, eps, 1.0)
        loss_all = -(oh * np.log(p)).sum(axis=1).astype(np.float32)
        dx = (-(oh / p)).astype(np.float32)

    loss_all = np.where(valid, loss_all, 0.0).astype(np.float32)
    dx *= valid[:, None].astype(np.float32)

    if reduction == "sum":
        loss_val = loss_all.sum(dtype=np.float32)
    elif reduction == "none":
        loss_val = loss_all
    else:
        loss_val = loss_all.sum(dtype=np.float32) / float(n_valid)

    if reduction == "mean":
        dx /= float(n_valid)

    if return_scalar:
        if reduction == "none":
            return float(loss_all.mean(dtype=np.float32)), dx
        else:
            return float(np.float32(loss_val).item()), dx
    else:
        return loss_val, dx


# --------- CuPy 경로 (ce_ops 없을 때 순수 CuPy 연산으로 계산) ---------
def _cupy_forward_pure(
    logits: "cp.ndarray",
    target_idx: "cp.ndarray",
    *,
    from_logits: bool,
    reduction: str,
    ignore_index: int,
    eps: float,
    ls_eps: float,
    return_scalar: bool,
):
    xp = cp
    x = logits.astype(xp.float32, copy=False)
    t = target_idx
    if not isinstance(t, xp.ndarray):
        t = xp.asarray(t)
    if t.dtype != xp.int32:
        t = t.astype(xp.int32, copy=False)

    N, C = x.shape
    valid = (t != ignore_index)
    n_valid = xp.maximum(valid.sum(dtype=xp.int32), 1)
    n_valid_f = n_valid.astype(xp.float32)  # ← CuPy 스칼라, host 전송 없음

    # one-hot + label smoothing
    oh = xp.zeros((N, C), dtype=xp.float32)
    t_clip = xp.clip(xp.where(valid, t, 0), 0, C - 1)
    oh[xp.arange(N, dtype=xp.int32), t_clip] = 1.0
    if ls_eps > 0.0:
        oh = (1.0 - ls_eps) * oh + (ls_eps / C)

    if from_logits:
        m = x.max(axis=1, keepdims=True)
        z = x - m
        exp = xp.exp(z)
        denom = exp.sum(axis=1, keepdims=True)
        p = exp / denom
        logp = z - xp.log(denom)
        loss_all = -(oh * logp).sum(axis=1).astype(xp.float32)
        dx = (p - oh).astype(xp.float32)
    else:
        p = xp.clip(x, eps, 1.0)
        loss_all = -(oh * xp.log(p)).sum(axis=1).astype(xp.float32)
        dx = (-(oh / p)).astype(xp.float32)

    loss_all = xp.where(valid, loss_all, xp.float32(0.0))
    dx = dx * valid[:, None].astype(xp.float32)

    if reduction == "sum":
        loss_val = loss_all.sum(dtype=xp.float32)
    elif reduction == "none":
        loss_val = loss_all
    else:  # mean over valid
        loss_val = loss_all.sum(dtype=xp.float32) / n_valid_f

    if reduction == "mean":
        dx = dx / n_valid_f

    if return_scalar:
        if reduction == "none":
            return float(loss_all.mean(dtype=xp.float32).get()), dx
        else:
            return float(loss_val.get().item()), dx
    else:
        return loss_val, dx



# --------- 공개 Loss 클래스 ---------
class SoftmaxCrossEntropy(Loss):
    """
    Softmax + Cross-Entropy (GPU가용 시 CUDA 커널 or 순수 CuPy, 불가 시 NumPy 폴백)

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

    def forward(
        self,
        logits: Array,
        target_idx: Array,
        *,
        return_scalar: bool = False,
    ):
        """
        Returns:
            (loss, dY)
            - return_scalar=False: loss는 텐서(디바이스/호스트) 그대로
              - reduction='mean'/'sum': 0-d 텐서
              - reduction='none': (N,) 텐서
            - return_scalar=True: 항상 Python float 반환
              - reduction='none'인 경우 mean(loss) 반환(표시용 규약)
        """
        # --- CuPy 텐서면 무조건 GPU 경로로 처리 ---
        if CUPY_AVAILABLE and isinstance(logits, cp.ndarray):  # type: ignore[attr-defined]
            if CE_OPS_AVAILABLE:
                # C++ 커널 경로
                x = logits.astype(cp.float32, copy=False)        # type: ignore[attr-defined]
                t = target_idx
                if not isinstance(t, cp.ndarray):                # type: ignore[attr-defined]
                    t = cp.asarray(t)                             # type: ignore[attr-defined]
                if t.dtype != cp.int32:                           # type: ignore[attr-defined]
                    t = t.astype(cp.int32, copy=False)            # type: ignore[attr-defined]

                loss_dev = ce_ops.forward(  # type: ignore[union-attr]
                    x, t,
                    from_logits=self.from_logits,
                    reduction=self.reduction,
                    ignore_index=self.ignore_index,
                    eps=self.eps,
                    ls_eps=self.ls_eps,
                )
                dx = ce_ops.backward(  # type: ignore[union-attr]
                    x, t,
                    from_logits=self.from_logits,
                    reduction=self.reduction,
                    ignore_index=self.ignore_index,
                    eps=self.eps,
                    ls_eps=self.ls_eps,
                )
                if return_scalar:
                    if self.reduction == "none":
                        return float(cp.mean(loss_dev).get()), dx  # type: ignore[attr-defined]
                    else:
                        return float(loss_dev.get().item()), dx    # type: ignore[attr-defined]
                else:
                    return loss_dev, dx
            else:
                # 순수 CuPy 연산 경로
                return _cupy_forward_pure(
                    logits, target_idx,
                    from_logits=self.from_logits,
                    reduction=self.reduction,
                    ignore_index=self.ignore_index,
                    eps=self.eps,
                    ls_eps=self.ls_eps,
                    return_scalar=return_scalar,
                )

        # --- 그 외는 NumPy 폴백 ---
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
            return_scalar=return_scalar,
        )


__all__ = ["SoftmaxCrossEntropy"]
