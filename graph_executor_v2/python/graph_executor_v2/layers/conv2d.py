# python/graph_executor_v2/layers/conv2d.py

# 원하면 groups>1에도 캐시를 확장해줄 수 있고, 사용자 스트림(cudaStream_t) 포인터를 받을 수 있게 옵션도 여유롭게 추가해줄 수 있어요.
from __future__ import annotations
from typing import Optional, Tuple
import cupy as cp

from .base import Layer
from graph_executor_v2.ops import conv2d as conv_ops

# 내부 런처 직접 호출(워크스페이스 주입용)
from graph_executor_v2.ops import _ops_conv2d as _g

_USE_NATIVE_GROUPS = False  # 네이티브 groups 경로가 준비되면 True


class Conv2D(Layer):
    """
    Conv2D 레이어 (NCHW)
      - filters: out_channels
      - kernel_size/stride/padding/dilation: (h,w) 튜플
      - groups 지원 (groups==1 fast path는 내부 워크스페이스 캐시를 재사용)
      - 활성화는 포함하지 않음(필요 시 별도 Activation 레이어 추가)
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

        # ----- workspace caches (fast path: groups==1) -----
        self._ws_sig: Optional[tuple] = None
        self._ws_fwd: Optional[tuple] = None  # (dCol, W_KC, Y_tmp, Z_rows)
        self._ws_bwd: Optional[tuple] = None  # (dCol, dTmp, W_CK, dWpack, dY_HT, gy_rows, Z_rows)

    # ---------------------- init helpers ----------------------
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
        self.W = cp.ascontiguousarray(self.W)
        if self.use_bias and self.b is not None:
            self.b = cp.ascontiguousarray(self.b)

        KH, KW = self.kernel_size
        sH, sW = self.stride
        pH, pW = self.padding
        dH, dW = self.dilation
        outH = (H + 2 * pH - dH * (KH - 1) - 1) // sH + 1
        outW = (W + 2 * pW - dW * (KW - 1) - 1) // sW + 1
        self.output_shape = (input_shape[0], self.filters, outH, outW)

        # invalidate workspaces on build (shape may change later)
        self._ws_sig = None
        self._ws_fwd = None
        self._ws_bwd = None

    # ---------------------- workspace helpers ----------------------
    def _ensure_ws(self, x_shape: Tuple[int, int, int, int], save_z: bool) -> Tuple[int, int]:
        """
        Allocate (or reuse) forward/backward workspaces for groups==1 fast path.
        Signature: (HWo, K, Cout, save_z) to determine reuse.
        """
        N, Cin, H, W = map(int, x_shape)
        KH, KW = self.kernel_size
        sH, sW = self.stride
        pH, pW = self.padding
        dH, dW = self.dilation
        Cout = self.filters

        H_out = (H + 2 * pH - dH * (KH - 1) - 1) // sH + 1
        W_out = (W + 2 * pW - dW * (KW - 1) - 1) // sW + 1
        HWo = H_out * W_out
        # groups==1 fast path에서의 K
        K = Cin * KH * KW

        sig = (HWo, K, Cout, bool(save_z))
        if self._ws_sig == sig and self._ws_fwd is not None and self._ws_bwd is not None:
            return H_out, W_out

        # ----- allocate workspaces -----
        dCol = cp.empty((HWo, K), dtype=cp.float32)       # [HWo, K]
        W_KC = cp.empty((K,   Cout), dtype=cp.float32)    # [K, Cout]
        Y_tmp = cp.empty((HWo, Cout), dtype=cp.float32)   # [HWo, Cout]
        Z_rows = cp.empty((HWo, Cout), dtype=cp.float32) if save_z else None

        dTmp = cp.empty((max(Cout * K, HWo * K),), dtype=cp.float32)
        W_CK = cp.empty((Cout, K), dtype=cp.float32)      # gX path
        dWpack = cp.empty((Cout, K), dtype=cp.float32)    # gW path
        dY_HT = cp.empty((HWo, Cout), dtype=cp.float32)   # gX path
        gy_rows = cp.empty((Cout, HWo), dtype=cp.float32)
        Zr = cp.empty((Cout, HWo), dtype=cp.float32)

        self._ws_fwd = (dCol, W_KC, Y_tmp, Z_rows)
        self._ws_bwd = (dCol, dTmp, W_CK, dWpack, dY_HT, gy_rows, Zr)
        self._ws_sig = sig
        return H_out, W_out

    # ---------------------- forward ----------------------
    def call(self, x: cp.ndarray) -> cp.ndarray:
        # dtype/shape guard + contiguity
        if not isinstance(x, cp.ndarray) or x.dtype != cp.float32 or x.ndim != 4:
            raise TypeError(
                f"Conv2D.call: expected float32 NCHW, got {type(x)} {getattr(x,'shape',None)} {getattr(x,'dtype',None)}"
            )
        x = cp.ascontiguousarray(x)
        self.last_input = x

        # param contiguity
        self.W = cp.ascontiguousarray(self.W)
        if self.use_bias and self.b is not None:
            self.b = cp.ascontiguousarray(self.b)

        G = self.groups

        # ---------- fast path: groups==1 (native launcher + cached workspaces) ----------
        if G == 1 or _USE_NATIVE_GROUPS:
            save_z = True  # act='none' → Z==Y alias
            H_out, W_out = self._ensure_ws(x.shape, save_z=save_z)
            dCol, W_KC, Y_tmp, Z_rows = self._ws_fwd

            # output: pre-allocate and alias Z to Y
            y = cp.empty((x.shape[0], self.filters, H_out, W_out), dtype=cp.float32)

            # attrs
            attrs = _g.Conv2DAttrs()
            attrs.stride_h, attrs.stride_w = self.stride
            attrs.pad_h, attrs.pad_w = self.padding
            attrs.dil_h, attrs.dil_w = self.dilation
            attrs.groups = self.groups
            attrs.with_bias = self.use_bias
            attrs.act = getattr(_g.ActKind, "None")

            attrs.leaky_slope = 0.01
            attrs.save_z = True

            # pointers
            x_ptr = int(x.data.ptr)
            w_ptr = int(self.W.data.ptr)
            y_ptr = int(y.data.ptr)
            b_ptr = int(self.b.data.ptr) if (self.use_bias and self.b is not None) else None
            z_ptr = int(y.data.ptr)  # alias Z==Y (act=none)

            _g.forward(
                x_ptr, [int(v) for v in x.shape],
                w_ptr, [int(v) for v in self.W.shape],
                y_ptr, [int(v) for v in y.shape],
                b_ptr if b_ptr is not None else None,
                z_ptr,
                attrs,
                0,  # stream
                # workspaces
                int(dCol.data.ptr),
                int(W_KC.data.ptr),
                int(Y_tmp.data.ptr),
                int(Z_rows.data.ptr) if Z_rows is not None else 0
            )

            self.last_z = y
            return y

        # ---------- python fallback: groups>1 ----------
        N, Cin, H, W = map(int, x.shape)
        KH, KW = self.kernel_size
        sH, sW = self.stride
        pH, pW = self.padding
        dH, dW = self.dilation
        Cout = self.filters

        Cin_g = Cin // G
        Cout_g = Cout // G

        outH = (H + 2 * pH - dH * (KH - 1) - 1) // sH + 1
        outW = (W + 2 * pW - dW * (KW - 1) - 1) // sW + 1

        y = cp.empty((N, Cout, outH, outW), dtype=cp.float32)
        Z = y

        for g in range(G):
            ci0, ci1 = g * Cin_g, (g + 1) * Cin_g
            co0, co1 = g * Cout_g, (g + 1) * Cout_g

            x_g = cp.ascontiguousarray(x[:, ci0:ci1, :, :])
            W_g = cp.ascontiguousarray(self.W[co0:co1, :, :, :])
            b_g = None if not self.use_bias else cp.ascontiguousarray(self.b[co0:co1])

            y_g = conv_ops.forward(
                x_g, W_g, b_g,
                stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=1,
                with_bias=self.use_bias,
                act="none",
                save_z=True, Z_saved=None
            )
            y[:, co0:co1, :, :] = y_g

        self.last_z = Z
        return y

    # ---------------------- backward ----------------------
    def backward(self, grad_output: cp.ndarray) -> cp.ndarray:
        if self.last_input is None or self.last_z is None:
            raise RuntimeError("Conv2D.backward called before forward")
        if not isinstance(grad_output, cp.ndarray) or grad_output.dtype != cp.float32 or grad_output.ndim != 4:
            raise TypeError("Conv2D.backward: grad_output must be float32 NCHW")

        gy = cp.ascontiguousarray(grad_output)
        self.W = cp.ascontiguousarray(self.W)
        if self.use_bias and self.b is not None:
            self.b = cp.ascontiguousarray(self.b)

        G = self.groups

        # ---------- fast path: groups==1 (native launcher + cached workspaces) ----------
        if G == 1 or _USE_NATIVE_GROUPS:
            # ensure workspaces exist (save_z=True used in fwd)
            self._ensure_ws(self.last_input.shape, save_z=True)
            dCol, dTmp, W_CK, dWpack, dY_HT, gy_rows, Z_rows = self._ws_bwd

            X = self.last_input
            Z = self.last_z

            # outputs
            gW = cp.empty_like(self.W)
            gB = cp.empty((self.filters,), dtype=cp.float32) if self.use_bias else None
            gX = cp.empty_like(X)

            # attrs
            attrs = _g.Conv2DAttrs()
            attrs.stride_h, attrs.stride_w = self.stride
            attrs.pad_h, attrs.pad_w = self.padding
            attrs.dil_h, attrs.dil_w = self.dilation
            attrs.groups = self.groups
            attrs.with_bias = self.use_bias
            attrs.act = getattr(_g.ActKind, "None")

            attrs.leaky_slope = 0.01
            attrs.save_z = False

            _g.backward(
                int(X.data.ptr),  [int(v) for v in X.shape],
                int(self.W.data.ptr),  [int(v) for v in self.W.shape],
                int(gy.data.ptr), [int(v) for v in gy.shape],
                int(Z.data.ptr),  [int(v) for v in Z.shape],
                int(gW.data.ptr),                                     # dw_ptr
                int(gB.data.ptr) if gB is not None else None,         # db_ptr
                int(gX.data.ptr),                                     # dx_ptr
                attrs,
                0,  # stream
                # workspaces
                int(dCol.data.ptr),
                int(dTmp.data.ptr),
                int(W_CK.data.ptr),
                int(dWpack.data.ptr),
                int(dY_HT.data.ptr),
                int(gy_rows.data.ptr),
                int(Z_rows.data.ptr),
            )

            self.dW = gW
            self.db = gB
            return gX

        # ---------- python fallback: groups>1 ----------
        x = self.last_input
        z = self.last_z
        N, Cin, H, W = map(int, x.shape)
        Cout = self.filters
        Cin_g = Cin // G
        Cout_g = Cout // G

        dx = cp.zeros_like(x)
        dW = cp.zeros_like(self.W)
        db = None if not self.use_bias else cp.zeros((Cout,), dtype=cp.float32)

        for g in range(G):
            ci0, ci1 = g * Cin_g, (g + 1) * Cin_g
            co0, co1 = g * Cout_g, (g + 1) * Cout_g

            x_g = cp.ascontiguousarray(x[:, ci0:ci1, :, :])
            z_g = cp.ascontiguousarray(z[:, co0:co1, :, :])
            gy_g = cp.ascontiguousarray(gy[:, co0:co1, :, :])
            W_g = cp.ascontiguousarray(self.W[co0:co1, :, :, :])

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
                db[co0:co1] = cp.ascontiguousarray(outs["gB"])

        self.dW = dW
        self.db = db
        return dx

    # ---------------------- shape util ----------------------
    def compute_output_shape(self, input_shape):
        if len(input_shape) != 4:
            raise ValueError(f"Conv2D expects 4D NCHW, got {input_shape}")
        N, _, H, W = map(int, input_shape)
        KH, KW = self.kernel_size
        sH, sW = self.stride
        pH, pW = self.padding
        dH, dW = self.dilation
        outH = (H + 2 * pH - dH * (KH - 1) - 1) // sH + 1
        outW = (W + 2 * pW - dW * (KW - 1) - 1) // sW + 1
        return (N, self.filters, outH, outW)
