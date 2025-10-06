from __future__ import annotations
from typing import Optional, Tuple
import cupy as cp

from .base import Layer
from graph_executor_v2.ops import conv2d as conv_ops

class Conv2D(Layer):
    """
    Conv2D 레이어 (NCHW)
      - filters: out_channels
      - kernel_size/stride/padding/dilation: (h,w) 튜플
      - groups 지원
      - (본 레이어는 활성화를 포함하지 않음; 필요 시 별도 Activation 레이어를 추가하세요)
    """
    def __init__(
        self,
        filters: int,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0),
        dilation: Tuple[int, int] = (1, 1),
        groups: int = 1,
        use_bias: bool = True,
        initializer: str = "he",
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.filters = int(filters)
        self.kernel_size = tuple(map(int, kernel_size))
        self.stride = tuple(map(int, stride))
        self.padding = tuple(map(int, padding))
        self.dilation = tuple(map(int, dilation))
        self.groups = int(groups)
        self.use_bias = bool(use_bias)
        self.initializer = initializer

        self.W: Optional[cp.ndarray] = None  # (Cout, Cin/groups, KH, KW)
        self.b: Optional[cp.ndarray] = None  # (Cout,)

        # cache
        self.last_input: Optional[cp.ndarray] = None
        self.last_z: Optional[cp.ndarray] = None  # pre-activation (Y와 동일 shape; act=none이면 alias)

        # grads
        self.dW: Optional[cp.ndarray] = None
        self.db: Optional[cp.ndarray] = None

    def _init_weights(self, Cin: int):
        Cout = self.filters
        KH, KW = self.kernel_size
        fan_in = Cin * KH * KW / self.groups
        if self.initializer == "he":
            std = cp.sqrt(2.0 / fan_in)
            W = cp.random.normal(0.0, std, (Cout, Cin // self.groups, KH, KW)).astype(cp.float32)
        elif self.initializer == "xavier":
            lim = cp.sqrt(6.0 / (fan_in + Cout))
            W = cp.random.uniform(-lim, lim, (Cout, Cin // self.groups, KH, KW)).astype(cp.float32)
        else:
            W = cp.random.normal(0.0, 0.05, (Cout, Cin // self.groups, KH, KW)).astype(cp.float32)
        b = cp.zeros((Cout,), dtype=cp.float32) if self.use_bias else None
        return W, b

    def build(self, input_shape):
        super().build(input_shape)
        if len(input_shape) != 4:
            raise ValueError(f"Conv2D expects 4D NCHW, got {input_shape}")
        _, Cin, _, _ = map(int, input_shape)
        self.W, self.b = self._init_weights(Cin)
        # 출력 shape 계산
        N, _, H, W = map(int, input_shape)
        KH, KW = self.kernel_size; sH, sW = self.stride; pH, pW = self.padding; dH, dW = self.dilation
        outH = (H + 2*pH - dH*(KH-1) - 1)//sH + 1
        outW = (W + 2*pW - dW*(KW-1) - 1)//sW + 1
        self.output_shape = (N, self.filters, outH, outW)

    def call(self, x: cp.ndarray) -> cp.ndarray:
        """
        Forward (act='none'):
          커널에서 pre-activation Z를 저장(save_z=True).
          act='none'이므로 Y==Z이고, 파이썬 래퍼가 Z_saved를 out과 alias로 전달합니다.
        """
        self.last_input = x
        y = conv_ops.forward(
            x, self.W, self.b if self.use_bias else None,
            stride=self.stride, padding=self.padding,
            dilation=self.dilation, groups=self.groups,
            with_bias=self.use_bias,
            act="none",
            save_z=True,       # 항상 Z를 저장
            Z_saved=None,      # 래퍼가 act='none'이면 out과 alias로 자동 처리
        )
        # act='none'이라 y와 Z가 같음 → 캐시에 보관
        self.last_z = y
        return y

    def backward(self, grad_output: cp.ndarray) -> cp.ndarray:
        if self.last_input is None or self.last_z is None:
            raise RuntimeError("Conv2D.backward called before forward")
        outs = conv_ops.backward(
            self.last_input, self.W, grad_output, self.last_z,
            stride=self.stride, padding=self.padding,
            dilation=self.dilation, groups=self.groups,
            with_bias=self.use_bias,
            act="none",
            want_gX=True, want_gW=True, want_gB=self.use_bias,
        )
        self.dW = outs.get("gW", None)
        self.db = outs.get("gB", None)
        dx = outs.get("gX", None)
        if dx is None:
            raise RuntimeError("Conv2D backward did not return gX")
        return dx

    def compute_output_shape(self, input_shape):
        if len(input_shape) != 4:
            raise ValueError(f"Conv2D expects 4D NCHW, got {input_shape}")
        N, _, H, W = map(int, input_shape)
        KH, KW = self.kernel_size; sH, sW = self.stride; pH, pW = self.padding; dH, dW = self.dilation
        outH = (H + 2*pH - dH*(KH-1) - 1)//sH + 1
        outW = (W + 2*pW - dW*(KW-1) - 1)//sW + 1
        return (N, self.filters, outH, outW)
