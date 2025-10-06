# python/graph_executor_v2/layers/conv2d.py
from __future__ import annotations
from typing import Optional, Tuple
import cupy as cp

from .base import Layer
from graph_executor_v2.ops import conv2d as conv_ops

_USE_NATIVE_GROUPS = False  # 현재는 fallback 사용


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
        self.last_z: Optional[cp.ndarray] = None  # pre-activation (act=none이면 Y와 alias)

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
        _, Cin, H, W = map(int, input_shape)
        if Cin % self.groups != 0:
            raise ValueError(f"Cin({Cin}) must be divisible by groups({self.groups})")
        if self.filters % self.groups != 0:
            raise ValueError(f"filters/Cout({self.filters}) must be divisible by groups({self.groups})")

        self.W, self.b = self._init_weights(Cin)

        KH, KW = self.kernel_size; sH, sW = self.stride; pH, pW = self.padding; dH, dW = self.dilation
        outH = (H + 2*pH - dH*(KH-1) - 1)//sH + 1
        outW = (W + 2*pW - dW*(KW-1) - 1)//sW + 1
        self.output_shape = (input_shape[0], self.filters, outH, outW)

    def call(self, x: cp.ndarray) -> cp.ndarray:
        self.last_input = x
        G = self.groups

        # 빠른 경로: groups==1 또는 네이티브 groups 사용 허용 시
        if G == 1 or _USE_NATIVE_GROUPS:
            y = conv_ops.forward(
                x, self.W, self.b if self.use_bias else None,
                stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=self.groups,
                with_bias=self.use_bias,
                act="none",
                save_z=True,       # Z 저장(=out과 alias)
                Z_saved=None,
            )
            self.last_z = y
            return y

        # ---- 파이썬 fallback: 그룹 분할 실행 ----
        N, Cin, H, W = map(int, x.shape)
        KH, KW = self.kernel_size
        sH, sW = self.stride; pH, pW = self.padding; dH, dW = self.dilation
        Cout = self.filters

        Cin_g  = Cin  // G
        Cout_g = Cout // G

        outH = (H + 2*pH - dH*(KH-1) - 1)//sH + 1
        outW = (W + 2*pW - dW*(KW-1) - 1)//sW + 1

        y = cp.empty((N, Cout, outH, outW), dtype=cp.float32)
        Z = y  # act='none' → Z==Y

        for g in range(G):
            ci0, ci1 = g*Cin_g,  (g+1)*Cin_g
            co0, co1 = g*Cout_g, (g+1)*Cout_g

            x_g = x[:, ci0:ci1, :, :]
            W_g = self.W[co0:co1, :, :, :]             # (Cout_g, Cin_g, KH, KW)
            b_g = None if not self.use_bias else self.b[co0:co1]

            y_g = conv_ops.forward(
                x_g, W_g, b_g,
                stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=1,        # 그룹 내부는 1로 고정
                with_bias=self.use_bias,
                act="none",
                save_z=True,
                Z_saved=None,
            )
            y[:, co0:co1, :, :] = y_g
            # act='none' 이므로 Z도 동일
        self.last_z = Z
        return y

    def backward(self, grad_output: cp.ndarray) -> cp.ndarray:
        if self.last_input is None or self.last_z is None:
            raise RuntimeError("Conv2D.backward called before forward")

        G = self.groups

        # 빠른 경로: groups==1 또는 네이티브 groups 허용 시
        if G == 1 or _USE_NATIVE_GROUPS:
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

        # ---- 파이썬 fallback: 그룹 분할 실행 ----
        x = self.last_input
        z = self.last_z
        gy = grad_output

        N, Cin, H, W = map(int, x.shape)
        Cout = self.filters
        Cin_g  = Cin  // G
        Cout_g = Cout // G

        dx = cp.zeros_like(x)
        dW = cp.zeros_like(self.W)
        db = None if not self.use_bias else cp.zeros((Cout,), dtype=cp.float32)

        for g in range(G):
            ci0, ci1 = g*Cin_g,  (g+1)*Cin_g
            co0, co1 = g*Cout_g, (g+1)*Cout_g

            x_g  = x[:,  ci0:ci1, :, :]
            z_g  = z[:,  co0:co1, :, :]
            gy_g = gy[:, co0:co1, :, :]
            W_g  = self.W[co0:co1, :, :, :]
            b_g  = None if not self.use_bias else self.b[co0:co1]

            outs = conv_ops.backward(
                x_g, W_g, gy_g, z_g,
                stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=1,
                with_bias=self.use_bias,
                act="none",
                want_gX=True, want_gW=True, want_gB=self.use_bias,
            )
            dx[:, ci0:ci1, :, :] = outs["gX"]
            dW[co0:co1, :, :, :] = outs["gW"]
            if self.use_bias:
                db[co0:co1] = outs["gB"]

        self.dW = dW
        self.db = db
        return dx

    def compute_output_shape(self, input_shape):
        if len(input_shape) != 4:
            raise ValueError(f"Conv2D expects 4D NCHW, got {input_shape}")
        N, _, H, W = map(int, input_shape)
        KH, KW = self.kernel_size; sH, sW = self.stride; pH, pW = self.padding; dH, dW = self.dilation
        outH = (H + 2*pH - dH*(KH-1) - 1)//sH + 1
        outW = (W + 2*pW - dW*(KW-1) - 1)//sW + 1
        return (N, self.filters, outH, outW)
