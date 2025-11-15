# python/graph_executor_v2/layers/dense_gemm.py
from __future__ import annotations
from typing import Optional, Tuple, List, Dict
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

        # 학습/추론 스위치(부모 Layer에 없을 수 있어 명시적으로 둠)
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
        # C-연속성 보장
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

        # grad 버퍼를 build 시점에 미리 준비 (0으로 초기화)
        self.dW = cp.zeros_like(self.W)
        self.db = cp.zeros_like(self.b)

        # 동적 배치
        self.output_shape = (None, self.units)

    def call(self, x: cp.ndarray) -> cp.ndarray:
        """
        Forward (fused):
        - 필요 시 pre-activation Z를 save_z로 저장
        - self.last_linear = Z(pre), self.last_input = x
        """
        if self.W is None or self.b is None:
            raise RuntimeError("Dense.call called before build")

        if x.dtype != cp.float32:
            x = x.astype(cp.float32, copy=False)

        # 학습 중 + 활성화가 있을 때만 Z 저장
        save_z = bool(self.training) and (self.activation != "none")

        try:
            if save_z:
                # 커널 경로: (Y, Z) 반환
                Y, Z = gemm_ops.forward(
                    x,
                    self.W,
                    self.b,                    # b는 (1, units)
                    act=self.activation,
                    with_bias=True,
                    leaky_slope=self.leaky_slope,
                    save_z=True,
                    return_z=True,
                )
                out = Y
                self.last_linear = Z
            else:
                # 커널 경로: Y만 반환
                out = gemm_ops.forward(
                    x,
                    self.W,
                    self.b,
                    act=self.activation,
                    with_bias=True,
                    leaky_slope=self.leaky_slope,
                    save_z=False,
                    return_z=False,
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

        # 공통 마무리
        self.last_input = x
        if out is None:
            raise RuntimeError("[Dense.call] out is None; every branch must produce a tensor")
        return out

    def backward(self, grad_output: cp.ndarray) -> cp.ndarray:
        """
        두 모드:
          - use_native_bwd=True  : 네이티브 커널 backward 사용(Z 필요)
          - use_native_bwd=False : 수동 미분 (CuPy 연산)
        """
        if self.last_input is None or self.W is None or self.b is None:
            raise RuntimeError("Dense.backward called before forward/build")

        if grad_output.dtype != cp.float32:
            grad_output = grad_output.astype(cp.float32, copy=False)

        # Z(pre)가 없으면 방어적으로 재계산 (act='none'으로 Z만)
        if self.last_linear is None:
            _, Z = gemm_ops.forward(
                self.last_input,
                self.W,
                self.b,
                act="none",
                with_bias=True,
                save_z=True,
                return_z=True,
            )
            self.last_linear = Z

        if self.use_native_bwd:
            outs: Dict[str, cp.ndarray] = gemm_ops.backward(
                self.last_input,
                self.W,
                grad_output,
                self.last_linear,
                act=self.activation,
                with_bias=True,
                leaky_slope=self.leaky_slope,
                C=None,
                want_gA=True,
                want_gB=True,
                want_gBias=True,
            )

            gW_new = outs.get("gB", None)        # (in_dim, units)
            gB_new = outs.get("gBias", None)     # (1, units)
            if gW_new is None or gB_new is None:
                raise RuntimeError("native backward did not return all required grads")

            # in-place 덮어쓰기 (기존 버퍼가 있으면 유지)
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

            # 안전장치: gBias 규약 확인(합 sum). 평균으로 온 경우 보정.
            if self.activation != "none":
                go_chk = apply_activation_grad(
                    grad_output,
                    self.last_linear,
                    self.activation,
                    self.leaky_slope,
                )
            else:
                go_chk = grad_output
            sum_go = go_chk.sum(axis=0, keepdims=True)        # 정답: 합(sum)
            err = float(cp.max(cp.abs(self.db - sum_go)))
            if err >= 1e-5:
                M = self.last_input.shape[0]
                err_scaled = float(cp.max(cp.abs(self.db * M - sum_go)))
                if err_scaled < 1e-5:
                    # 평균으로 나온 경우 → 합으로 보정
                    self.db = self.db * M
                else:
                    # 축/방향 오류 등 → 정답으로 교체
                    self.db = sum_go

            return dx

        # -------- 수동 미분 경로 --------
        go = apply_activation_grad(
            grad_output,
            self.last_linear,
            self.activation,
            self.leaky_slope,
        )  # dAct(Z) * gY
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

        dx = go @ self.W.T                                      # (batch, in_dim)
        return dx

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[Optional[int], int]:
        if len(input_shape) != 2:
            raise ValueError(f"Dense expects 2D input, got {input_shape}")
        return (None if input_shape[0] is None else int(input_shape[0]), self.units)

    # ---------------- Trainer-friendly helpers ----------------
    def parameters(self):
        """
        Optimizer/Sequential에서 바로 소비할 수 있도록 (param, grad, tag) 튜플을 yield.
        grad 버퍼는 build/backward에서 항상 준비되므로 그대로 노출.
        """
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

    # ---------------- (옵션) Capture-safe NO-alloc 경로 ----------------
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
        """
        if self.W is None or self.b is None:
            raise RuntimeError("Dense.forward_into called before build")
        if x.dtype != cp.float32:
            raise ValueError("[capture] x must be float32")
        if not (
            x.flags.c_contiguous
            and out.flags.c_contiguous
            and self.W.flags.c_contiguous
            and self.b.flags.c_contiguous
        ):
            raise ValueError("[capture] inputs/params/out must be C-contiguous")

        M, in_dim = x.shape
        if out.shape != (M, self.units) or out.dtype != cp.float32:
            raise ValueError(
                f"[capture] out must be float32[{M},{self.units}] with C-contiguous layout"
            )

        save_z = (self.activation != "none")

        # z_out 자동 관리: 필요하면 work 캐시에 배정 (없으면 즉석 할당)
        if save_z and z_out is None:
            key = ("dense_z", int(out.data.ptr))
            buf = None
            if work is not None:
                buf = work.get(key)
            if (
                buf is None
                or tuple(getattr(buf, "shape", ())) != (M, self.units)
                or getattr(buf, "dtype", None) != cp.float32
            ):
                # capture-safe를 엄격히 보려면 여기서도 사전할당을 요구해야 하지만,
                # 아직은 fallback으로 allocate 허용(work 캐시와 함께 사용).
                buf = cp.empty_like(out)
                if work is not None:
                    work[key] = buf
            z_out = buf

        gemm_ops.forward_into(
            x,
            self.W,
            out=out,
            bias=self.b,
            act=self.activation,
            with_bias=True,
            leaky_slope=self.leaky_slope,
            save_z=save_z,
            z_out=z_out,
            stream=stream,
        )
        self.last_input = x
        # act='none'이면 Z==Y이므로 last_linear를 out으로 alias
        if self.activation == "none":
            self.last_linear = out
        else:
            self.last_linear = z_out if save_z else None

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

        if grad_output.dtype != cp.float32 or not grad_output.flags.c_contiguous:
            raise ValueError("[capture] grad_output must be C-contiguous float32")

        M, in_dim = self.last_input.shape
        if grad_output.shape != (M, self.units):
            raise ValueError("[capture] grad_output shape mismatch")

        if (
            gA_out.shape != (M, in_dim)
            or gA_out.dtype != cp.float32
            or not gA_out.flags.c_contiguous
        ):
            raise ValueError("[capture] gA_out must be C-contiguous float32[(M,in_dim)]")

        if (
            gW_out.shape != (in_dim, self.units)
            or gW_out.dtype != cp.float32
            or not gW_out.flags.c_contiguous
        ):
            raise ValueError("[capture] gW_out must be C-contiguous float32[(in_dim,units)]")

        if (
            gB_out.shape != (1, self.units)
            or gB_out.dtype != cp.float32
            or not gB_out.flags.c_contiguous
        ):
            raise ValueError("[capture] gB_out must be C-contiguous float32[(1,units)]")

        # dZ(work_dZ) 자동 관리: 없으면 work 캐시 활용 (형상은 last_linear와 동일)
        if work_dZ is None:
            key = ("dense_dZ", int(self.last_linear.data.ptr))
            buf = None
            if work is not None:
                buf = work.get(key)
            if (
                buf is None
                or tuple(getattr(buf, "shape", ())) != tuple(self.last_linear.shape)
                or getattr(buf, "dtype", None) != cp.float32
            ):
                # 여기도 엄밀히는 사전할당이 맞지만, 아직은 fallback 허용
                buf = cp.empty_like(self.last_linear)
                if work is not None:
                    work[key] = buf
            work_dZ = buf

        gemm_ops.backward_into(
            self.last_input,
            self.W,
            grad_output,
            self.last_linear,
            act=self.activation,
            with_bias=True,
            leaky_slope=self.leaky_slope,
            C=None,
            gA_out=gA_out,
            gB_out=gW_out,   # W grad
            gBias_out=gB_out,  # bias grad
            gC_out=None,
            work_dZ=work_dZ,
            lt_workspace=lt_workspace,
            stream=stream,
        )
        # 내부 상태에 현재 grad 버퍼를 연결
        self.dW = gW_out
        self.db = gB_out
