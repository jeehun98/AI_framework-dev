# python/graph_executor_v2/layers/dense_gemm.py
from __future__ import annotations
from typing import Optional, Tuple, List
import cupy as cp

from .base import Layer
from .activations import apply_activation_grad
from graph_executor_v2.ops import gemm as gemm_ops


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
        # 성능 민감 시 주석 해제(모든 텐서 C-연속성 보장)
        # x = cp.ascontiguousarray(x)

        # 학습 중 + 활성화 있을 때만 Z 저장
        save_z = bool(self.training) and (self.activation != "none")

        if save_z:
            Y, Z = gemm_ops.forward(
                x, self.W, self.b,
                act=self.activation, with_bias=True, leaky_slope=self.leaky_slope,
                save_z=True, return_z=True
            )
            self.last_linear = Z
            out = Y
        else:
            out = gemm_ops.forward(
                x, self.W, self.b,
                act=self.activation, with_bias=True, leaky_slope=self.leaky_slope,
                save_z=False, return_z=False
            )
            self.last_linear = None

        self.last_input = x
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
            self.dW = outs.get("gB", None)       # (in_dim, units)
            self.db = outs.get("gBias", None)    # (1, units)
            dx = outs.get("gA", None)            # (batch, in_dim)
            if dx is None or self.dW is None or self.db is None:
                raise RuntimeError("native backward did not return all required grads")

            # 안전장치: gBias 규약 확인(합 sum). 평균으로 온 경우 보정.
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
        go = apply_activation_grad(grad_output, self.last_linear, self.activation, self.leaky_slope)  # dAct(Z) * gY
        # (contiguous 보장)
        if not go.flags.c_contiguous:
            go = cp.ascontiguousarray(go)

        self.dW = self.last_input.T @ go                        # (in_dim, units)
        self.db = go.sum(axis=0, keepdims=True)                 # (1, units) Per-N
        dx = go @ self.W.T                                      # (batch, in_dim)
        return dx

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[Optional[int], int]:
        if len(input_shape) != 2:
            raise ValueError(f"Dense expects 2D input, got {input_shape}")
        return (None if input_shape[0] is None else int(input_shape[0]), self.units)

    # ---------------- Trainer-friendly helpers ----------------
    def parameters(self) -> List[cp.ndarray]:
        return [p for p in (self.W, self.b) if p is not None]

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
    ) -> None:
        """
        NO-alloc forward for CUDA Graph capture.
        - out, z_out(옵션)은 사전할당된 float32 C-contiguous 버퍼여야 함.
        """
        if self.W is None or self.b is None:
            raise RuntimeError("Dense.forward_into called before build")
        if x.dtype != cp.float32:
            raise ValueError("[capture] x must be float32")
        if not (x.flags.c_contiguous and out.flags.c_contiguous and self.W.flags.c_contiguous and self.b.flags.c_contiguous):
            raise ValueError("[capture] inputs/params/out must be C-contiguous")
        M, in_dim = x.shape
        if out.shape != (M, self.units) or out.dtype != cp.float32:
            raise ValueError("[capture] out must be float32[{M},{U}] with C-contiguous layout")

        save_z = (self.activation != "none")
        if save_z:
            if z_out is None or z_out.shape != (M, self.units) or z_out.dtype != cp.float32 or not z_out.flags.c_contiguous:
                raise ValueError("[capture] z_out must be C-contiguous float32[(M, units)] when activation is used")

        # --- after ---
        gemm_ops.forward_into(
            x, self.W,
            out=out, bias=self.b,
            act=self.activation, with_bias=True, leaky_slope=self.leaky_slope,
            save_z=save_z, z_out=z_out, stream=stream
        )
        self.last_input = x
        # ⭐ act='none'이면 Z==Y이므로 last_linear를 out으로 alias
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
        work_dZ: cp.ndarray,
        lt_workspace: Optional[cp.ndarray] = None,
        stream: Optional[int] = None,
    ) -> None:
        """
        NO-alloc backward for CUDA Graph capture.
        - 모든 출력/워크스페이스 버퍼는 사전할당 & C-contiguous float32 (lt_workspace만 uint8).
        - gB_out: (in_dim, units), gB_out(=bias grad)는 (1, units)
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
