# python/graph_executor_v2/layers/dense_gemm.py
from __future__ import annotations
from typing import Optional, Tuple, List
from contextlib import nullcontext
import cupy as cp

from .base import Layer
from graph_executor_v2.ops import gemm as gemm_ops

try:
    from .activations import apply_activation_grad  # optional
except Exception:
    def apply_activation_grad(
        grad_output: cp.ndarray,
        z: cp.ndarray,
        act: Optional[str] = None,
        leaky_slope: float = 0.01,
    ) -> cp.ndarray:
        if act is None or act == "none":
            return grad_output
        a = act.lower()
        if a == "relu":
            return grad_output * (z > 0)
        if a in ("leakyrelu", "leaky_relu", "lrelu"):
            return grad_output * cp.where(z > 0, 1.0, leaky_slope)
        if a == "sigmoid":
            sig = 1.0 / (1.0 + cp.exp(-z))
            return grad_output * sig * (1.0 - sig)
        if a == "tanh":
            t = cp.tanh(z)
            return grad_output * (1.0 - t * t)
        if a == "gelu":
            c = cp.sqrt(2.0 / cp.pi, dtype=cp.float32)
            t = cp.tanh(c * (z + 0.044715 * z * z * z))
            dt = (1.0 - t * t) * c * (1.0 + 3.0 * 0.044715 * z * z)
            gelu_grad = 0.5 * (1.0 + t) + 0.5 * z * dt
            return grad_output * gelu_grad
        raise ValueError(f"Unknown activation grad: {act}")


class Dense(Layer):
    """
    GEMM(+bias+activation fused) 기반 Dense 레이어.
      - W: (in_dim, units), b: (1, units)
      - forward: A(M,in) * W(in,units) + b(1,units) -> Y(M,units)
      - 필요 시 커널에서 pre-activation Z를 save_z로 저장/사용.
    규약:
      - dtype=float32 고정
      - bias grad는 Per-N 합(sum), shape=(1, units)
    """

    def __init__(
        self,
        units: int,
        activation: Optional[str] = None,
        initializer: str = "he",
        name: Optional[str] = None,
        leaky_slope: float = 0.01,
        use_native_bwd: bool = False,
    ):
        super().__init__(name=name)
        self.units = int(units)
        self.activation = (activation or "none").lower()
        self.initializer = initializer
        self.leaky_slope = float(leaky_slope)
        self.use_native_bwd = bool(use_native_bwd)

        # 파라미터
        self.W: Optional[cp.ndarray] = None              # (in_dim, units) float32 C-contig
        self.b: Optional[cp.ndarray] = None              # (1, units)     float32 C-contig

        # 캐시(역전파용)
        self.last_input: Optional[cp.ndarray] = None     # x
        self.last_linear: Optional[cp.ndarray] = None    # Z(pre-activation)

        # 그라디언트
        self.dW: Optional[cp.ndarray] = None             # (in_dim, units)
        self.db: Optional[cp.ndarray] = None             # (1, units)

        # 학습/추론 스위치
        self.training: bool = True

    # ---------------- init helpers ----------------
    def _init_weights(self, in_dim: int) -> Tuple[cp.ndarray, cp.ndarray]:
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
        if not W.flags.c_contiguous:
            W = cp.ascontiguousarray(W)
        if not b.flags.c_contiguous:
            b = cp.ascontiguousarray(b)
        return W, b

    # ---------------- Layer lifecycle ----------------
    def build(self, input_shape: Tuple[int, ...]) -> None:
        super().build(input_shape)
        if len(input_shape) != 2:
            raise ValueError(f"Dense expects 2D input (batch, in_dim), got {input_shape}")
        _, in_dim = map(int, input_shape)
        self.W, self.b = self._init_weights(in_dim)

        # grad 버퍼 사전 준비
        self.dW = cp.zeros_like(self.W)
        self.db = cp.zeros_like(self.b)

        # 동적 배치
        self.output_shape = (None, self.units)

    # ---------------- Forward (alloc) ----------------
    def call(self, x: cp.ndarray, *, stream: Optional[int] = None) -> cp.ndarray:
        """
        Allocating forward.
        - 커널 경로와 수동(CuPy) 경로 모두 지정된 stream에서 실행되도록 강제.
        """
        if self.W is None or self.b is None:
            raise RuntimeError("Dense.call called before build")

        if x.dtype != cp.float32:
            x = x.astype(cp.float32, copy=False)

        save_z = bool(self.training) and (self.activation != "none")

        # 지정 스트림 콘텍스트
        ctx = cp.cuda.ExternalStream(stream) if stream is not None else nullcontext()
        with ctx:
            try:
                if save_z:
                    Y, Z = gemm_ops.forward(
                        x, self.W, self.b,
                        act=self.activation, with_bias=True,
                        leaky_slope=self.leaky_slope,
                        save_z=True, return_z=True,
                        stream=stream,
                    )
                    out = Y
                    self.last_linear = Z
                else:
                    out = gemm_ops.forward(
                        x, self.W, self.b,
                        act=self.activation, with_bias=True,
                        leaky_slope=self.leaky_slope,
                        save_z=False, return_z=False,
                        stream=stream,
                    )
                    self.last_linear = None
            except Exception:
                # --- 안전 폴백 (순수 CuPy 연산) ---
                z = x @ self.W + self.b  # (M,N) + (1,N)
                if self.activation == "none":
                    out = z
                    self.last_linear = None
                else:
                    self.last_linear = z
                    a = (self.activation or "none").lower()
                    if a == "relu":
                        out = cp.maximum(z, 0)
                    elif a in ("leakyrelu", "leaky_relu", "lrelu"):
                        out = cp.where(z > 0, z, self.leaky_slope * z)
                    elif a == "sigmoid":
                        out = 1.0 / (1.0 + cp.exp(-z))
                    elif a == "tanh":
                        out = cp.tanh(z)
                    elif a == "gelu":
                        c = cp.sqrt(2.0 / cp.pi, dtype=cp.float32)
                        t = cp.tanh(c * (z + 0.044715 * z * z * z))
                        out = 0.5 * z * (1.0 + t)
                    else:
                        raise ValueError(f"Unknown activation: {self.activation}")

        self.last_input = x
        if out is None:
            raise RuntimeError("[Dense.call] out is None; every branch must produce a tensor")
        return out

    # ---------------- Backward (alloc) ----------------
    def backward(self, grad_output: cp.ndarray, *, stream: Optional[int] = None) -> cp.ndarray:
        """
        Allocating backward: native 커널 또는 수동(CuPy) 경로.
        - 모든 경로 지정 스트림에서 실행.
        """
        if self.last_input is None or self.W is None or self.b is None:
            raise RuntimeError("Dense.backward called before forward/build")

        if grad_output.dtype != cp.float32:
            grad_output = grad_output.astype(cp.float32, copy=False)

        # 지정 스트림 콘텍스트
        ctx = cp.cuda.ExternalStream(stream) if stream is not None else nullcontext()
        with ctx:
            # Z(pre)가 없으면 방어적으로 재계산 (act='none'으로 Z만)
            if self.last_linear is None:
                _, Z = gemm_ops.forward(
                    self.last_input, self.W, self.b,
                    act="none", with_bias=True, save_z=True, return_z=True,
                    stream=stream,
                )
                self.last_linear = Z

            if self.use_native_bwd:
                outs = gemm_ops.backward(
                    self.last_input, self.W, grad_output, self.last_linear,
                    act=self.activation, with_bias=True, leaky_slope=self.leaky_slope,
                    C=None, want_gA=True, want_gB=True, want_gBias=True,
                    stream=stream,
                )
                gW_new = outs.get("gB", None)        # (in_dim, units)
                gB_new = outs.get("gBias", None)     # (1, units)
                if gW_new is None or gB_new is None:
                    raise RuntimeError("native backward did not return all required grads")
                if self.dW is None or self.dW.shape != gW_new.shape:
                    self.dW = gW_new
                else:
                    self.dW[...] = gW_new
                if self.db is None or self.db.shape != gB_new.shape:
                    self.db = gB_new
                else:
                    self.db[...] = gB_new

                dx = outs.get("gA", None)            # (batch, in_dim)
                if dx is None or self.dW is None or self.db is None:
                    raise RuntimeError("native backward did not return all required grads")

                # gBias 규약 확인(합 sum). 평균으로 온 경우 보정.
                if self.activation != "none":
                    go_chk = apply_activation_grad(grad_output, self.last_linear, self.activation, self.leaky_slope)
                else:
                    go_chk = grad_output
                sum_go = go_chk.sum(axis=0, keepdims=True)        # 정답: 합(sum)
                err = float(cp.max(cp.abs(self.db - sum_go)))
                if err >= 1e-5:
                    M = self.last_input.shape[0]
                    err_scaled = float(cp.max(cp.abs(self.db * M - sum_go)))
                    if err_scaled < 1e-5:
                        self.db = self.db * M       # 평균으로 나온 경우 → 합으로 보정
                    else:
                        self.db = sum_go            # 축/방향 오류 등 → 정답으로 교체
                return dx

            # -------- 수동 미분 경로 --------
            go = apply_activation_grad(grad_output, self.last_linear, self.activation, self.leaky_slope)
            if not go.flags.c_contiguous:
                go = cp.ascontiguousarray(go)

            gW_new = self.last_input.T @ go
            gB_new = go.sum(axis=0, keepdims=True)
            if self.dW is None or self.dW.shape != gW_new.shape:
                self.dW = gW_new
            else:
                self.dW[...] = gW_new
            if self.db is None or self.db.shape != gB_new.shape:
                self.db = gB_new
            else:
                self.db[...] = gB_new

            dx = go @ self.W.T
            return dx

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[Optional[int], int]:
        if len(input_shape) != 2:
            raise ValueError(f"Dense expects 2D input, got {input_shape}")
        return (None if input_shape[0] is None else int(input_shape[0]), self.units)

    # ---------------- Trainer-friendly helpers ----------------
    def parameters(self):
        lname = type(self).__name__
        if self.W is not None:
            yield (self.W, self.dW, f"{lname}.W")
        if self.b is not None:
            yield (self.b, self.db, f"{lname}.b")

    def grads(self) -> List[Optional[cp.ndarray]]:
        return [self.dW, self.db]

    def zero_grad(self) -> None:
        if self.dW is not None:
            self.dW.fill(0)
        if self.db is not None:
            self.db.fill(0)

    # ---------------- Capture-safe, NO-alloc ----------------
    def forward_into(
        self,
        x: cp.ndarray,
        out: cp.ndarray,
        z_out: Optional[cp.ndarray] = None,
        *,
        stream: Optional[int] = None,
        work: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        NO-alloc forward for CUDA Graph capture.
        - out, z_out(옵션)은 사전할당된 float32 C-contiguous 버퍼여야 함.
        - save_z가 필요한 경우 z_out은 반드시 제공되어야 함(자동 할당 금지).
        """
        if self.W is None or self.b is None:
            raise RuntimeError("Dense.forward_into called before build")
        if x.dtype != cp.float32:
            raise ValueError("[capture] x must be float32")
        if not (x.flags.c_contiguous and out.flags.c_contiguous and self.W.flags.c_contiguous and self.b.flags.c_contiguous):
            raise ValueError("[capture] inputs/params/out must be C-contiguous")
        M, in_dim = x.shape
        if out.shape != (M, self.units) or out.dtype != cp.float32:
            raise ValueError(f"[capture] out must be float32[{M},{U}] with C-contiguous layout")

        save_z = (self.activation != "none")

        # ❗ NO-alloc 정책: save_z가 필요한데 z_out이 없으면 실패
        if save_z and z_out is None:
            raise ValueError("[capture] save_z=True requires z_out preallocated")

        gemm_ops.forward_into(
            x, self.W,
            out=out, bias=self.b,
            act=self.activation, with_bias=True, leaky_slope=self.leaky_slope,
            save_z=save_z, z_out=z_out, stream=stream
        )
        self.last_input = x
        self.last_linear = (out if self.activation == "none" else (z_out if save_z else None))

    def backward_into(
        self,
        grad_output: cp.ndarray,
        gA_out: cp.ndarray,
        gW_out: cp.ndarray,
        gB_out: cp.ndarray,
        *,
        work_dZ: Optional[cp.ndarray],
        lt_workspace: Optional[cp.ndarray] = None,
        stream: Optional[int] = None,
        work: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        NO-alloc backward for CUDA Graph capture.
        - 모든 출력/워크스페이스 버퍼는 사전할당 & C-contiguous float32 (lt_workspace만 uint8).
        - gW_out: (in_dim, units), gB_out(=bias grad)는 (1, units)
        """
        if self.last_input is None or self.last_linear is None:
            raise RuntimeError("[capture] need forward_into (with save_z) before backward_into")
        if grad_output.dtype != cp.float32:
            raise ValueError("[capture] grad_output must be float32")

        M, in_dim = self.last_input.shape
        if grad_output.shape != (M, self.units):
            raise ValueError("[capture] grad_output shape mismatch")
        if gA_out.shape != (M, in_dim) or gA_out.dtype != cp.float32 or not gA_out.flags.c_contiguous:
            raise ValueError("[capture] gA_out must be C-contiguous float32[(M,in_dim)]")
        if gW_out.shape != (in_dim, self.units) or gW_out.dtype != cp.float32 or not gW_out.flags.c_contiguous:
            raise ValueError("[capture] gW_out must be C-contiguous float32[(in_dim,units)]")
        if gB_out.shape != (1, self.units) or gB_out.dtype != cp.float32 or not gB_out.flags.c_contiguous:
            raise ValueError("[capture] gB_out must be C-contiguous float32[(1,units)]")

        # ❗ NO-alloc 정책: work_dZ 미제공 시 실패
        if work_dZ is None:
            raise ValueError("[capture] work_dZ must be provided (preallocated)")

        gemm_ops.backward_into(
            self.last_input, self.W, grad_output, self.last_linear,
            act=self.activation, with_bias=True, leaky_slope=self.leaky_slope,
            C=None,
            gA_out=gA_out, gB_out=gW_out, gBias_out=gB_out, gC_out=None,
            work_dZ=work_dZ, lt_workspace=lt_workspace, stream=stream
        )
        # 내부 상태에 현재 grad 버퍼를 연결(옵션)
        self.dW = gW_out
        self.db = gB_out
