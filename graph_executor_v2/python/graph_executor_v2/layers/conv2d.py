# python/graph_executor_v2/layers/conv2d.py
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
      - activation은 여기서는 생략(Conv2D+활성화는 보통 별도 레이어 권장)
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
        self.b: Optional[cp.ndarray] = None  # (Cout,) or (1,Cout,1,1)

        # cache
        self.last_input: Optional[cp.ndarray] = None

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
        KH, KW = self.kernel_size; sH,sW = self.stride; pH,pW = self.padding; dH,dW = self.dilation
        outH = (H + 2*pH - dH*(KH-1) - 1)//sH + 1
        outW = (W + 2*pW - dW*(KW-1) - 1)//sW + 1
        self.output_shape = (N, self.filters, outH, outW)

    def call(self, x: cp.ndarray) -> cp.ndarray:
        self.last_input = x
        if self.use_bias and self.b is not None:
            # 바인딩이 (Cout,) or (1,Cout,1,1) 모두 지원한다고 가정
            return conv_ops.forward(
                x, self.W, self.b,
                stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=self.groups
                
            )
        else:
            return conv_ops.forward(
                x, self.W, None,
                stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=self.groups
                
            )

    def backward(self, grad_output: cp.ndarray) -> cp.ndarray:
        outs = conv_ops.backward(
            self.last_input, self.W, grad_output,
            stride=self.stride, padding=self.padding,
            dilation=self.dilation, groups=self.groups,
            with_bias=self.use_bias,
            want_gX=True, want_gW=True, want_gB=self.use_bias
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
        KH, KW = self.kernel_size; sH,sW = self.stride; pH,pW = self.padding; dH,dW = self.dilation
        outH = (H + 2*pH - dH*(KH-1) - 1)//sH + 1
        outW = (W + 2*pW - dW*(KW-1) - 1)//sW + 1
        return (N, self.filters, outH, outW)
