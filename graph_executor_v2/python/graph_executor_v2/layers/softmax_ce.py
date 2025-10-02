# python/graph_executor_v2/layers/softmax_ce.py
from __future__ import annotations
from typing import Optional, Tuple
import cupy as cp

from .base import Layer
from graph_executor_v2.ops import cross_entropy as ce_ops

class SoftmaxCrossEntropy(Layer):
    """
    로스 레이어:
      - 입력 logits(M,N) 과 targets(M,) → per-sample loss(M,) (reduction='none')
      - backward: dX(logits) 반환
    """
    def __init__(
        self,
        label_smoothing: float = 0.0,
        reduction: str = "none",     # 레이어 출력을 벡터로 쓰기 위해 기본 'none'
        ignore_index: int = -1,
        from_logits: bool = True,    # 일반적으로 True
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        if reduction.lower() not in ("none", "mean", "sum"):
            raise ValueError("reduction must be 'none' | 'mean' | 'sum'")
        self.ls_eps = float(label_smoothing)
        self.reduction = reduction.lower()
        self.ignore_index = int(ignore_index)
        self.from_logits = bool(from_logits)

        self.last_X: Optional[cp.ndarray] = None
        self.last_t: Optional[cp.ndarray] = None

    def build(self, input_shape):
        # input_shape는 (M,N)으로 가정
        super().build(input_shape)
        if len(input_shape) != 2:
            raise ValueError(f"SoftmaxCrossEntropy expects 2D logits (M,N), got {input_shape}")
        # 출력 shape: reduction에 따라 다름
        M = int(input_shape[0])
        if self.reduction == "none":
            self.output_shape = (M,)
        else:
            self.output_shape = (1,)

    def call(self, inputs: Tuple[cp.ndarray, cp.ndarray]) -> cp.ndarray:
        X, targets = inputs
        self.last_X = X
        self.last_t = targets
        loss = ce_ops.forward(
            X, targets,
            from_logits=self.from_logits,
            reduction=self.reduction,
            ignore_index=self.ignore_index,
            eps=1e-7,
            ls_eps=self.ls_eps
        )
        return loss

    def backward(self, grad_output: Optional[cp.ndarray]) -> cp.ndarray:
        """
        grad_output:
          - reduction='none'인 경우 (M,) 또는 (1,) scale 가능 (대부분 ones(M,))
          - reduction='mean'/'sum'인 경우 일반적으로 shape (1,) 혹은 스칼라
        반환값: dX (M,N)
        """
        if self.last_X is None or self.last_t is None:
            raise RuntimeError("SoftmaxCrossEntropy.backward called before forward")

        dX = ce_ops.backward(
            self.last_X, self.last_t,
            from_logits=self.from_logits,
            reduction=self.reduction,
            ignore_index=self.ignore_index,
            eps=1e-7,
            ls_eps=self.ls_eps
        )

        # 외부 grad_output 스케일 적용
        if grad_output is not None:
            if grad_output.ndim == 0:
                dX *= float(grad_output)  # 스칼라
            elif grad_output.ndim == 1 and grad_output.shape[0] == dX.shape[0]:
                dX *= grad_output[:, None]  # (M,) → (M,1) 브로드캐스트
            elif grad_output.ndim == 2 and grad_output.shape == (dX.shape[0], 1):
                dX *= grad_output
            else:
                raise ValueError("grad_output must be scalar, (M,), or (M,1)")
        return dX

    def compute_output_shape(self, input_shape):
        M = int(input_shape[0])
        return (M,) if self.reduction == "none" else (1,)
