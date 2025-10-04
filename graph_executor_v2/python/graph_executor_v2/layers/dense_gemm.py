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

        self.last_input: Optional[cp.ndarray] = None  # 캐시
        self.last_linear: Optional[cp.ndarray] = None # Z=pre-activation

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
        Forward:
          1) 선형 Z = xW + b (act=none)
          2) 필요 시 activation 적용
          3) fused path: act을 forward에서 함께 수행하려면 act 지정하고 호출
        """
        # 항상 Z(pre-act)를 캐시하기 위해 두 단계로 계산
        z = gemm_ops.forward(x, self.W, self.b, act="none", with_bias=True)
        self.last_linear = z
        self.last_input = x

        if self.activation == "none":
            return z
        # 후단 활성화를 바인딩에 맡기고 싶다면, 아래 한 줄로 대체 가능:
        # return gemm_ops.forward(x, self.W, self.b, act=self.activation, with_bias=True, leaky_slope=self.leaky_slope)
        # 하지만 backward에서 Z가 필요하므로 여기선 수동으로 활성화 적용 대신 바깥에서 처리 권장.
        # 간단히 CuPy로 활성화 적용:
        if self.activation == "relu":
            return z * (z > 0)
        if self.activation == "sigmoid":
            return 1 / (1 + cp.exp(-z))
        if self.activation == "tanh":
            return cp.tanh(z)
        if self.activation == "gelu":
            c = cp.sqrt(2.0 / cp.pi)
            return 0.5 * z * (1 + cp.tanh(c * (z + 0.044715 * z**3)))
        if self.activation in ("leakyrelu", "leaky_relu", "lrelu"):
            slope = self.leaky_slope
            return cp.where(z > 0, z, slope * z)
        raise ValueError(f"Unsupported activation: {self.activation}")

    def backward(self, grad_output: cp.ndarray) -> cp.ndarray:
        """
        두 모드:
          - use_native_bwd=True  : 바인딩 backward 사용(Z=pre-act 필요)
          - use_native_bwd=False : 수동 미분 (CuPy로 dAct * dLinear)
        """
        if self.last_input is None or self.W is None or self.b is None:
            raise RuntimeError("Dense.backward called before forward/build")

        if self.use_native_bwd:
            if self.last_linear is None:
                self.last_linear = gemm_ops.forward(self.last_input, self.W, self.b, act="none", with_bias=True)

            outs = gemm_ops.backward(
                self.last_input, self.W, grad_output, self.last_linear,
                act=self.activation, with_bias=True, leaky_slope=self.leaky_slope,
                C=None, want_gA=True, want_gB=True, want_gBias=True
            )            
            self.dW = outs.get("gB", None)
            self.db = outs.get("gBias", None)
            dx = outs.get("gA", None)
            if dx is None:
                raise RuntimeError("native backward did not return gA")

            # --- 안전장치: db는 반드시 sum(dY_after_act) ---
            go_chk = apply_activation_grad(grad_output, self.last_linear, self.activation, self.leaky_slope)
            keep = (self.b is not None and self.b.ndim == 2)  # b가 (1, U)면 keepdims 유지
            sum_go = go_chk.sum(axis=0, keepdims=keep)        # 정답

            # (A) 우선 같으면 통과
            if self.db is not None:
                err = float(cp.max(cp.abs(self.db - sum_go)))
            else:
                err = float('inf')

            if err >= 1e-5:
                # (B) 평균(/M) 케이스인지 체크
                M = self.last_input.shape[0]
                err_scaled = float(cp.max(cp.abs(self.db * M - sum_go))) if self.db is not None else float('inf')

                if err_scaled < 1e-5:
                    # 평균으로 나온 경우 → 합(sum)으로 보정
                    self.db = self.db * M
                else:
                    # (C) 축이 틀린(PerM) 등 다른 이유 → 정답으로 강제 교체
                    #   - 로깅만 남기고 sum_go로 교정해서 진행
                    #   - 필요시 print로 경고 노출(원하면 주석 해제)
                    # print(f"[warn] Dense native bwd: replacing db with sum(dY). "
                    #       f"mismatch={err:.3e}, scaled_mismatch={err_scaled:.3e}")
                    self.db = sum_go

            return dx
        
        # -------- 수동 미분 경로 --------
        if self.last_linear is None:
            self.last_linear = gemm_ops.forward(self.last_input, self.W, self.b, act="none", with_bias=True)
        go = apply_activation_grad(grad_output, self.last_linear, self.activation, self.leaky_slope)
        self.dW = self.last_input.T @ go
        # b 초기 shape가 (1, units)이므로 keepdims=True로 유지(네이티브와 브로드캐스트 호환)
        self.db = go.sum(axis=0, keepdims=True)
        dx = go @ self.W.T
        return dx

    def compute_output_shape(self, input_shape):
        if len(input_shape) != 2:
            raise ValueError(f"Dense expects 2D input, got {input_shape}")
        return (int(input_shape[0]), self.units)
