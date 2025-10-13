# python/graph_executor_v2/layers/softmax_ce.py
from __future__ import annotations
from typing import Optional, Tuple
import cupy as cp

from .base import Layer
from graph_executor_v2.ops import cross_entropy as ce_ops


class SoftmaxCrossEntropy(Layer):
    """
    소프트맥스 크로스 엔트로피 손실 레이어.

    입력:
      - logits: (M, N) float32
      - targets: (M,) int32   (ignore_index 사용 가능)

    출력:
      - reduction='none' → (M,) per-sample loss
      - reduction='mean' / 'sum' → (1,) scalar

    backward:
      - dX(logits) shape (M, N) 반환
      - 외부 grad_output(스칼라 / (M,) / (M,1))로 스케일만 적용
    """
    def __init__(
        self,
        label_smoothing: float = 0.0,
        reduction: str = "none",     # 기본: per-sample loss를 쓰기 위해 'none'
        ignore_index: int = -1,
        from_logits: bool = True,
        name: Optional[str] = None,
        eps: float = 1e-7,
    ):
        super().__init__(name=name)
        red = reduction.lower()
        if red not in ("none", "mean", "sum"):
            raise ValueError("reduction must be 'none' | 'mean' | 'sum'")

        self.ls_eps = float(label_smoothing)
        self.reduction = red
        self.ignore_index = int(ignore_index)
        self.from_logits = bool(from_logits)
        self.eps = float(eps)

        # 캐시
        self.last_X: Optional[cp.ndarray] = None
        self.last_t: Optional[cp.ndarray] = None

    # 모델의 build는 보통 (M,N)만 전달 받음 (targets은 런타임 텐서)
    def build(self, input_shape):
        super().build(input_shape)
        if len(input_shape) != 2:
            raise ValueError(f"SoftmaxCrossEntropy expects 2D logits (M,N), got {input_shape}")
        M = int(input_shape[0])
        self.output_shape = (M,) if self.reduction == "none" else (1,)

    def call(self, inputs: Tuple[cp.ndarray, cp.ndarray]) -> cp.ndarray:
        logits, targets = inputs

        # 타입/디바이스 가드
        if logits.dtype != cp.float32:
            logits = logits.astype(cp.float32, copy=False)
        if targets.dtype != cp.int32:
            # 커널/바인딩은 int32 index를 기대
            targets = targets.astype(cp.int32, copy=False)

        self.last_X = logits
        self.last_t = targets

        loss = ce_ops.forward(
            logits,
            targets,
            from_logits=self.from_logits,
            reduction=self.reduction,
            ignore_index=self.ignore_index,
            eps=self.eps,
            ls_eps=self.ls_eps,
        )
        return loss

    def backward(self, grad_output: Optional[cp.ndarray]) -> cp.ndarray:
        """
        grad_output:
          - reduction='none' → (M,) 또는 (M,1) 혹은 스칼라(드묾)
          - reduction='mean'/'sum' → 보통 스칼라 또는 (1,)
        반환: dX (M,N)
        """
        if self.last_X is None or self.last_t is None:
            raise RuntimeError("SoftmaxCrossEntropy.backward called before forward")

        dX = ce_ops.backward(
            self.last_X,
            self.last_t,
            from_logits=self.from_logits,
            reduction=self.reduction,
            ignore_index=self.ignore_index,
            eps=self.eps,
            ls_eps=self.ls_eps,
        )
        # 외부 스케일 적용 (합 기반 dX에 스케일만 곱함)
        if grad_output is not None:
            if grad_output.ndim == 0:
                dX *= float(grad_output)
            elif grad_output.ndim == 1 and grad_output.shape[0] == dX.shape[0]:
                dX *= grad_output[:, None]
            elif grad_output.ndim == 2 and grad_output.shape == (dX.shape[0], 1):
                dX *= grad_output
            else:
                raise ValueError("grad_output must be scalar, (M,), or (M,1)")
        return dX

    def compute_output_shape(self, input_shape):
        M = int(input_shape[0])
        return (M,) if self.reduction == "none" else (1,)
