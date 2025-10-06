# python/graph_executor_v2/layers/flatten.py
from __future__ import annotations
from typing import Optional, Tuple
import cupy as cp
from .base import Layer

class Flatten(Layer):
    def __init__(self, name: Optional[str]=None):
        super().__init__(name=name)
        self.last_shape: Optional[Tuple[int,int,int,int]] = None

    def build(self, input_shape):
        super().build(input_shape)
        if len(input_shape)!=4:
            raise ValueError(f"Flatten expects 4D NCHW input, got {input_shape}")
        N,C,H,W = map(int, input_shape)
        self.output_shape = (N, C*H*W)

    def call(self, x: cp.ndarray) -> cp.ndarray:
        if x.ndim!=4: raise ValueError("Flatten input must be NCHW")
        self.last_shape = tuple(map(int, x.shape))
        N,C,H,W = self.last_shape
        return x.reshape(N, C*H*W)

    def backward(self, grad_output: cp.ndarray) -> cp.ndarray:
        if self.last_shape is None:
            raise RuntimeError("Flatten.backward called before forward")
        return grad_output.reshape(self.last_shape)
