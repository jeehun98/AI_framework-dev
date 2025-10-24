# python/graph_executor_v2/losses/bce.py
from __future__ import annotations

from typing import Tuple, Optional
import numpy as np

from .base import Loss, Array

# CuPy가 있으면 GPU 경로 가속, 없으면 NumPy 폴백
try:
    import cupy as cp  # type: ignore
    _HAS_GPU = True
except Exception:
    cp = None  # type: ignore[assignment]
    _HAS_GPU = False


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    # overflow-safe sigmoid
    # sigmoid(x) = 1 / (1 + exp(-x))
    # 안정성 개선: 큰 음수/양수에 대해 np.exp의 overflow/underflow 고려
    out = np.empty_like(x, dtype=np.float32)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[neg])
    out[neg] = ex / (1.0 + ex)
    return out.astype(np.float32, copy=False)


def _log_sigmoid_np(x: np.ndarray) -> np.ndarray:
    # log(sigmoid(x)) = -softplus(-x)
    # softplus(z) = log(1 + exp(z))
    # 안정성: max(0, z) 트릭
    z = -x
    m = np.maximum(z, 0.0)
    return -(m + np.log1p(np.exp(z - m))).astype(np.float32, copy=False)


def _binary_cross_entropy_numpy(
    x: np.ndarray,
    t: np.ndarray,
    *,
    from_logits: bool,
    reduction: str,
    eps: float,
    weight: Optional[np.ndarray] = None,
    pos_weight: Optional[np.ndarray] = None,
) -> Tuple[float, np.ndarray]:
    """
    x, t: same shape, float32
    weight/pos_weight: optional, broadcastable to x.shape
    """
    x = x.astype(np.float32, copy=False)
    t = t.astype(np.float32, copy=False)

    if weight is not None:
        w = np.asarray(weight, dtype=np.float32)
    else:
        w = None

    if from_logits:
        # BCEWithLogits 스타일: -[ t * logσ(x) * pos_w + (1-t)*log(1-σ(x)) ]
        # logσ(x) = -softplus(-x), log(1-σ(x)) = -softplus(x)
        # 구현: -(t*logσ(x)*pos_w + (1-t)*log(1-σ(x)))
        log_sig = _log_sigmoid_np(x)
        log_one_minus_sig = _log_sigmoid_np(-x)  # = -softplus(x)

        if pos_weight is not None:
            pw = np.asarray(pos_weight, dtype=np.float32)
            pos_term = -t * log_sig * pw
        else:
            pos_term = -t * log_sig

        neg_term = -(1.0 - t) * log_one_minus_sig
        loss_elem = pos_term + neg_term

        # grad = (σ(x) - t) * scale
        sig = _sigmoid_np(x)
        grad = (sig - t)
        if pos_weight is not None:
            # positive 라벨에 대해서만 pos_weight 적용
            grad = grad * (1.0 + (pw - 1.0) * t)  # t∈{0,1}일 때 t=1이면 *pw, t=0이면 *1
    else:
        # 입력이 확률
        p = np.clip(x, eps, 1.0 - eps)
        loss_elem = -(t * np.log(p) + (1.0 - t) * np.log(1.0 - p))
        # grad = (p - t) / (p*(1-p))
        grad = (p - t) / np.clip(p * (1.0 - p), eps, None)

    if w is not None:
        loss_elem = loss_elem * w
        grad = grad * w

    loss_elem = loss_elem.astype(np.float32, copy=False)
    grad = grad.astype(np.float32, copy=False)

    numel = loss_elem.size
    if reduction == "sum":
        loss_scalar = float(loss_elem.sum())
    elif reduction == "none":
        loss_scalar = float(loss_elem.mean())  # 표시용 mean (스칼라는 항상 float)
    else:  # mean
        loss_scalar = float(loss_elem.mean())

    if reduction == "mean":
        # 전체 평균으로 맞추기 위해 grad도 numel로 스케일
        grad = grad / float(numel)
    # 'sum' / 'none'은 추가 스케일 없음

    return loss_scalar, grad


class BinaryCrossEntropy(Loss):
    """
    Binary Cross-Entropy (multi-label 포함) Loss

    Args:
      from_logits: True면 내부에서 sigmoid + 안정적 로그 적용 (권장)
      reduction: 'mean' | 'sum' | 'none'
      eps: 확률 입력 시 안정성 epsilon
      weight: 선택적 가중치 (broadcastable)
      pos_weight: 선택적 양성 클래스 가중치 (from_logits=True일 때 양성항에만 곱해짐)
    """
    def __init__(
        self,
        from_logits: bool = True,
        reduction: str = "mean",
        eps: float = 1e-7,
        weight: Optional[Array] = None,
        pos_weight: Optional[Array] = None,
    ):
        assert reduction in ("mean", "sum", "none")
        self.from_logits = bool(from_logits)
        self.reduction = reduction
        self.eps = float(eps)
        self.weight = weight  # Array 또는 None
        self.pos_weight = pos_weight  # Array 또는 None

    def forward(self, logits_or_prob: Array, target: Array) -> Tuple[float, Array]:
        # GPU 경로 (CuPy 배열인 경우)
        if _HAS_GPU:
            try:
                if isinstance(logits_or_prob, cp.ndarray):  # type: ignore[attr-defined]
                    x = logits_or_prob.astype(cp.float32, copy=False)  # type: ignore[attr-defined]
                    t = cp.asarray(target, dtype=cp.float32)  # type: ignore[attr-defined]

                    w = cp.asarray(self.weight, dtype=cp.float32) if self.weight is not None else None  # type: ignore[attr-defined]
                    pw = cp.asarray(self.pos_weight, dtype=cp.float32) if self.pos_weight is not None else None  # type: ignore[attr-defined]

                    if self.from_logits:
                        # logsigmoid
                        log_sig = -cp.logaddexp(0.0, -x)  # type: ignore[attr-defined]
                        log_one_minus_sig = -cp.logaddexp(0.0, x)  # = logσ(-x)

                        if pw is not None:
                            pos_term = -t * log_sig * pw
                        else:
                            pos_term = -t * log_sig
                        neg_term = -(1.0 - t) * log_one_minus_sig
                        loss_elem = pos_term + neg_term

                        sig = 1.0 / (1.0 + cp.exp(-x))  # type: ignore[attr-defined]
                        grad = (sig - t)
                        if pw is not None:
                            grad = grad * (1.0 + (pw - 1.0) * t)
                    else:
                        p = cp.clip(x, self.eps, 1.0 - self.eps)  # type: ignore[attr-defined]
                        loss_elem = -(t * cp.log(p) + (1.0 - t) * cp.log(1.0 - p))  # type: ignore[attr-defined]
                        grad = (p - t) / cp.clip(p * (1.0 - p), self.eps, None)  # type: ignore[attr-defined]

                    if w is not None:
                        loss_elem = loss_elem * w
                        grad = grad * w

                    numel = loss_elem.size
                    if self.reduction == "sum":
                        loss_scalar = float(loss_elem.sum().item())  # type: ignore[call-arg]
                    elif self.reduction == "none":
                        loss_scalar = float(loss_elem.mean().item())
                    else:
                        loss_scalar = float(loss_elem.mean().item())

                    if self.reduction == "mean":
                        grad = grad / float(numel)

                    return loss_scalar, grad  # type: ignore[return-value]
            except Exception:
                pass  # 실패 시 아래 CPU 폴백

        # CPU 폴백
        x_np = np.asarray(logits_or_prob, dtype=np.float32)
        t_np = np.asarray(target, dtype=np.float32)
        w_np = np.asarray(self.weight, dtype=np.float32) if self.weight is not None else None
        pw_np = np.asarray(self.pos_weight, dtype=np.float32) if self.pos_weight is not None else None

        return _binary_cross_entropy_numpy(
            x_np, t_np,
            from_logits=self.from_logits,
            reduction=self.reduction,
            eps=self.eps,
            weight=w_np,
            pos_weight=pw_np,
        )


__all__ = ["BinaryCrossEntropy"]
