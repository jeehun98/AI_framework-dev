# python/graph_executor_v2/layers/dense_gemm.py
from __future__ import annotations
import cupy as cp
from typing import Optional, Tuple
from .base import Layer
from .activations import apply_activation_grad
from graph_executor_v2.ops import gemm as gemm_ops


class Dense(Layer):
    """
    GEMM(+bias+activation fused) 기반 Dense.
      - 가중치: (in_dim, units), bias: (1, units)
      - forward: A(M,in) * W(in,units) + b(1,units) -> Y(M,units)
      - pre-activation Z는 커널에서 save_z로 저장해 캐시한다.
    """
    def __init__(self, units: int, activation: Optional[str] = None,
                 initializer: str = "he", name: Optional[str] = None,
                 leaky_slope: float = 0.01, use_native_bwd: bool = False):
        super().__init__(name=name)
        self.units = int(units)
        self.activation = (activation or "none").lower()
        self.initializer = initializer
        self.leaky_slope = float(leaky_slope)
        self.use_native_bwd = bool(use_native_bwd)

        self.W: Optional[cp.ndarray] = None
        self.b: Optional[cp.ndarray] = None

        self.last_input: Optional[cp.ndarray]  = None  # x
        self.last_linear: Optional[cp.ndarray] = None  # Z(pre-activation)

        self.dW: Optional[cp.ndarray] = None
        self.db: Optional[cp.ndarray] = None

    # ----- init helpers -----
    def _init_weights(self, in_dim: int):
        if self.initializer == "zeros":
            W = cp.zeros((in_dim, self.units), dtype=cp.float32)
        elif self.initializer == "ones":
            W = cp.ones((in_dim, self.units), dtype=cp.float32)
        elif self.initializer == "uniform":
            lim = 0.05
            W = cp.random.uniform(-lim, lim, (in_dim, self.units)).astype(cp.float32)
        elif self.initializer == "normal":
            W = cp.random.normal(0.0, 0.05, (in_dim, self.units)).astype(cp.float32)
        elif self.initializer == "xavier":
            lim = cp.sqrt(6.0 / (in_dim + self.units))
            W = cp.random.uniform(-lim, lim, (in_dim, self.units)).astype(cp.float32)
        elif self.initializer == "he":
            std = cp.sqrt(2.0 / in_dim)
            W = cp.random.normal(0.0, std, (in_dim, self.units)).astype(cp.float32)
        elif self.initializer == "lecun":
            std = cp.sqrt(1.0 / in_dim)
            W = cp.random.normal(0.0, std, (in_dim, self.units)).astype(cp.float32)
        elif self.initializer == "small_uniform":
            W = cp.random.uniform(-1e-3, 1e-3, (in_dim, self.units)).astype(cp.float32)
        else:
            raise ValueError(f"Unknown initializer: {self.initializer}")
        b = cp.random.uniform(-1e-3, 1e-3, (1, self.units)).astype(cp.float32)
        return W, b

    # ----- Layer lifecycle -----
    def build(self, input_shape: Tuple[int, ...]) -> None:
        super().build(input_shape)
        if len(input_shape) != 2:
            raise ValueError(f"Dense expects 2D input (batch, in_dim), got {input_shape}")
        _, in_dim = map(int, input_shape)
        self.W, self.b = self._init_weights(in_dim)
        self.output_shape = (int(input_shape[0]), self.units)

    def call(self, x: cp.ndarray) -> cp.ndarray:
        """
        Forward (fused):
          - 커널에서 act를 적용하되, pre-activation Z를 save_z로 함께 저장
          - self.last_linear = Z(pre), self.last_input = x 캐시
        """
        if self.W is None or self.b is None:
            raise RuntimeError("Dense.call called before build")

        # fused forward + save Z(pre)
        Y, Z = gemm_ops.forward(
            x, self.W, self.b,
            act=self.activation, with_bias=True, leaky_slope=self.leaky_slope,
            save_z=True, return_z=True
        )

        self.last_input  = x
        self.last_linear = Z  # pre-activation
        return Y

    def backward(self, grad_output: cp.ndarray) -> cp.ndarray:
        """
        두 모드:
          - use_native_bwd=True  : 바인딩 backward 사용(Z=pre-act 필요)
          - use_native_bwd=False : 수동 미분 (CuPy로 dAct * dLinear)
        """
        if self.last_input is None or self.W is None or self.b is None:
            raise RuntimeError("Dense.backward called before forward/build")

        # ensure Z(pre) exists (방어적 재계산)
        if self.last_linear is None:
            _, Z = gemm_ops.forward(
                self.last_input, self.W, self.b,
                act="none", with_bias=True, save_z=True, return_z=True
            )
            self.last_linear = Z

        if self.use_native_bwd:
            outs = gemm_ops.backward(
                self.last_input, self.W, grad_output, self.last_linear,
                act=self.activation, with_bias=True, leaky_slope=self.leaky_slope,
                C=None, want_gA=True, want_gB=True, want_gBias=True
            )
            self.dW = outs.get("gB", None)       # shape: (in_dim, units)
            self.db = outs.get("gBias", None)    # shape: (1, units)
            dx = outs.get("gA", None)            # shape: (batch, in_dim)
            if dx is None:
                raise RuntimeError("native backward did not return gA")

            # --- 안전장치: db는 반드시 sum(dZ) == sum(dY_after_act) ---
            # 커널 구현/옵션에 따라 평균(/M)로 나올 가능성을 보정
            go_chk = apply_activation_grad(grad_output, self.last_linear, self.activation, self.leaky_slope)
            keep = (self.b is not None and self.b.ndim == 2)  # (1, U) 유지
            sum_go = go_chk.sum(axis=0, keepdims=keep)        # 정답: 합(sum)

            if self.db is not None:
                err = float(cp.max(cp.abs(self.db - sum_go)))
            else:
                err = float("inf")

            if err >= 1e-5:
                M = self.last_input.shape[0]
                err_scaled = float(cp.max(cp.abs(self.db * M - sum_go))) if self.db is not None else float("inf")
                if err_scaled < 1e-5:
                    self.db = self.db * M       # 평균으로 나온 경우 → 합으로 보정
                else:
                    self.db = sum_go            # 축/방향 오류 등 → 정답으로 교체

            return dx

        # -------- 수동 미분 경로 --------
        go = apply_activation_grad(grad_output, self.last_linear, self.activation, self.leaky_slope)  # dAct(Z) * gY
        self.dW = self.last_input.T @ go
        self.db = go.sum(axis=0, keepdims=True)   # PerN, shape (1, units)
        dx = go @ self.W.T
        return dx

    def compute_output_shape(self, input_shape):
        if len(input_shape) != 2:
            raise ValueError(f"Dense expects 2D input, got {input_shape}")
        return (int(input_shape[0]), self.units)
