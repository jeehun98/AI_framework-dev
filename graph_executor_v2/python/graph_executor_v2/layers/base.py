# python/graph_executor_v2/layers/base.py
from __future__ import annotations
from typing import Optional, Tuple, Any

class Layer:
    """
    미니멀 Layer 인터페이스:
      - build(input_shape)
      - __call__/call(x)
      - backward(grad_output)
      - compute_output_shape(input_shape)
    """
    def __init__(self, name: Optional[str] = None, **kwargs):
        self.name = name or f"{self.__class__.__name__.lower()}_{id(self)}"
        self.built: bool = False
        self.input_shape: Optional[Tuple[int, ...]] = None
        self.output_shape: Optional[Tuple[int, ...]] = None

    def build(self, input_shape: Tuple[int, ...]) -> None:
        self.input_shape = tuple(map(int, input_shape))
        self.built = True

    def __call__(self, x: Any):
        if not self.built:
            # (Tensor, target) 같은 튜플/리스트 입력도 지원
            shape_src = x
            if isinstance(x, (tuple, list)) and len(x) >= 1:
                shape_src = x[0]
            if hasattr(shape_src, "shape"):
                self.build(tuple(map(int, shape_src.shape)))
            else:
                raise ValueError(f"{self.name}: cannot infer shape from input")
        return self.call(x)

    # 하위 클래스가 구현
    def call(self, x: Any):
        raise NotImplementedError

    # 하위 클래스가 구현(미분 경로)
    def backward(self, grad_output: Any):
        raise NotImplementedError

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        raise NotImplementedError
