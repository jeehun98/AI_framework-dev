# python/graph_executor_v2/layers/linear.py
from __future__ import annotations
from typing import Optional, Tuple, Dict
import cupy as cp
from .base import Layer
from graph_executor_v2.ops import gemm as gemm_ops

class Linear(Layer):
    """
    Fully-connected (GEMM) layer
      - input: (N, K)
      - W: (K, N_out)
      - b: (N_out,)
    """
    def __init__(self, out_features: int, use_bias: bool=True, initializer: str="xavier", name: Optional[str]=None):
        super().__init__(name=name)
        self.out_features = int(out_features)
        self.use_bias = bool(use_bias)
        self.initializer = initializer
        self.W: Optional[cp.ndarray] = None
        self.b: Optional[cp.ndarray] = None
        self.last_input: Optional[cp.ndarray] = None
        self.last_z: Optional[cp.ndarray] = None  # pre-activation
        self.dW: Optional[cp.ndarray] = None
        self.db: Optional[cp.ndarray] = None

    def _init_params(self, in_features: int):
        K = in_features; N = self.out_features
        if self.initializer == "xavier":
            lim = cp.sqrt(6.0 / (K + N))
            W = cp.random.uniform(-lim, lim, (K, N)).astype(cp.float32)
        elif self.initializer == "he":
            std = cp.sqrt(2.0 / K)
            W = cp.random.normal(0.0, std, (K, N)).astype(cp.float32)
        else:
            W = cp.random.normal(0.0, 0.05, (K, N)).astype(cp.float32)
        b = cp.zeros((N,), dtype=cp.float32) if self.use_bias else None
        return W, b

    def build(self, input_shape):
        super().build(input_shape)
        if len(input_shape)!=2: raise ValueError(f"Linear expects 2D input (N,K), got {input_shape}")
        N, K = map(int, input_shape)
        self.W, self.b = self._init_params(K)
        self.output_shape = (N, self.out_features)

    def call(self, x: cp.ndarray) -> cp.ndarray:
        # act='none' + save_z: pre-activation Z까지 함께 받아 저장
        self.last_input = x
        y, z = gemm_ops.forward(
            x, self.W, self.b if self.use_bias else None,
            act="none",
            with_bias=self.use_bias,
            save_z=True,
            return_z=True,   # 꼭 튜플로 받자
        )
        self.last_z = z
        return y

    def backward(self, grad_output: cp.ndarray) -> cp.ndarray:
        if self.last_input is None or self.last_z is None:
            raise RuntimeError("Linear.backward called before forward")
        
        outs = gemm_ops.backward(self.last_input, self.W, grad_output, self.last_z, act="none", with_bias=self.use_bias, want_gA=True, want_gB=True, want_gBias=self.use_bias)
        self.dW = outs.get("gB", None)      # (K, N_out)
        self.db = outs.get("gBias", None)   # (1, N_out)
        dX      = outs.get("gA", None)      # (N, K)
        if dX is None:
            raise RuntimeError("Linear backward did not return gA")
        return dX
