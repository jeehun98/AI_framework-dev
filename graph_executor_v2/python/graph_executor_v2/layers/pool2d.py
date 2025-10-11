from __future__ import annotations
from typing import Optional, Tuple, Any, Dict, Iterable
import cupy as cp

from .base import Layer
from ..ops import pool2d as pops

# ------------------------------------------------------------
# 캡처-세이프 MaxPool/AvgPool 레이어
#   - 파라미터 없음(학습 대상 X)
#   - MaxPool은 BWD 시 인덱스가 필요 → 워크스페이스에 인덱스 버퍼 유지
# ------------------------------------------------------------

class _Pool2DWork:
    """캡처용 인덱스 버퍼 컨테이너 (MaxPool에서 사용)."""
    def __init__(self):
        self.indices: Optional[cp.ndarray] = None  # int32, (N,C,Ho,Wo)

class Pool2D(Layer):
    def __init__(
        self,
        kernel_size: Tuple[int, int] | int = (2, 2),
        *,
        stride: Tuple[int, int] | int = (2, 2),
        padding: Tuple[int, int] | int = (0, 0),
        dilation: Tuple[int, int] | int = (1, 1),
        ceil_mode: bool = False,
        count_include_pad: bool = False,   # avg 전용
        mode: str = "max",                 # "max" | "avg"
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        if isinstance(kernel_size, int): kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):      stride      = (stride, stride)
        if isinstance(padding, int):     padding     = (padding, padding)
        if isinstance(dilation, int):    dilation    = (dilation, dilation)

        self.kernel_size = tuple(map(int, kernel_size))
        self.stride      = tuple(map(int, stride))
        self.padding     = tuple(map(int, padding))
        self.dilation    = tuple(map(int, dilation))
        self.ceil_mode   = bool(ceil_mode)
        self.count_include_pad = bool(count_include_pad)
        self.mode        = str(mode).lower()

        self.input_shape: Optional[Tuple[int, ...]]  = None
        self.output_shape: Optional[Tuple[int, ...]] = None
        self.built = False
        self.training = True

        # eager 경로 캐시
        self._last_X: Optional[cp.ndarray] = None
        self._last_Y: Optional[cp.ndarray] = None
        self._last_indices: Optional[cp.ndarray] = None  # MaxPool only

    # ---- utils ----
    @staticmethod
    def _out_hw(H:int, W:int, kH:int, kW:int, s:Tuple[int,int], p:Tuple[int,int],
                d:Tuple[int,int], ceil_mode:bool) -> Tuple[int,int]:
        sH, sW = s; pH, pW = p; dH, dW = d
        effKH = (kH - 1) * dH + 1
        effKW = (kW - 1) * dW + 1
        aH = H + 2 * pH - effKH
        aW = W + 2 * pW - effKW
        if ceil_mode:
            Ho = (aH >= 0) and ((aH + sH - 1)//sH + 1) or 0
            Wo = (aW >= 0) and ((aW + sW - 1)//sW + 1) or 0
        else:
            Ho = (aH >= 0) and (aH//sH + 1) or 0
            Wo = (aW >= 0) and (aW//sW + 1) or 0
        return max(0, int(Ho)), max(0, int(Wo))

    # ---- Layer API ----
    def build(self, input_shape: Tuple[int, ...]) -> None:
        if len(input_shape) != 4:
            raise ValueError(f"Pool2D expects 4D input (N,C,H,W), got {input_shape}")
        N, C, H, W = map(int, input_shape)
        kH, kW = self.kernel_size
        Ho, Wo = self._out_hw(H, W, kH, kW, self.stride, self.padding, self.dilation, self.ceil_mode)
        self.input_shape  = (N, C, H, W)
        self.output_shape = (N, C, Ho, Wo)
        self.built = True

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        if len(input_shape) != 4:
            raise ValueError(f"Pool2D expects 4D input (N,C,H,W), got {input_shape}")
        N, C, H, W = map(int, input_shape)
        kH, kW = self.kernel_size
        Ho, Wo = self._out_hw(H, W, kH, kW, self.stride, self.padding, self.dilation, self.ceil_mode)
        return (N, C, Ho, Wo)

    def parameters(self) -> Iterable[Tuple[Any, Any, str]]:
        # 파라미터 없음
        return []
        yield  # for type checkers

    # ---- eager ----
    def call(self, X: cp.ndarray) -> cp.ndarray:
        if self.mode == "max":
            Y, idx = pops.forward(
                X,
                kernel=self.kernel_size, stride=self.stride, padding=self.padding,
                dilation=self.dilation, ceil_mode=self.ceil_mode,
                count_include_pad=self.count_include_pad,
                mode="max", return_indices=True
            )
            if self.training:
                self._last_X = X
                self._last_Y = Y
                self._last_indices = idx
            return Y
        elif self.mode == "avg":
            Y = pops.forward(
                X,
                kernel=self.kernel_size, stride=self.stride, padding=self.padding,
                dilation=self.dilation, ceil_mode=self.ceil_mode,
                count_include_pad=self.count_include_pad,
                mode="avg"
            )
            if self.training:
                self._last_X = X
                self._last_Y = Y
                self._last_indices = None
            return Y
        else:
            raise ValueError("mode must be 'max' or 'avg'")

    def backward(self, gY: cp.ndarray) -> cp.ndarray:
        assert self._last_X is not None and self._last_Y is not None
        if self.mode == "max":
            if self._last_indices is None:
                raise RuntimeError("MaxPool backward needs saved indices (eager).")
            dX = pops.backward(
                self._last_X, self._last_Y, gY,
                kernel=self.kernel_size, stride=self.stride, padding=self.padding,
                dilation=self.dilation, ceil_mode=self.ceil_mode,
                count_include_pad=self.count_include_pad,
                mode="max", indices=self._last_indices
            )
            return dX
        else:
            dX = pops.backward(
                self._last_X, self._last_Y, gY,
                kernel=self.kernel_size, stride=self.stride, padding=self.padding,
                dilation=self.dilation, ceil_mode=self.ceil_mode,
                count_include_pad=self.count_include_pad,
                mode="avg"
            )
            return dX

    # ---- capture-safe ----
    def forward_into(
        self, X: cp.ndarray, *,
        out: cp.ndarray,
        z_out: Optional[cp.ndarray] = None,   # 존재해도 무시
        stream: Optional[int] = None,
        work: Optional[_Pool2DWork] = None
    ) -> None:
        # MaxPool: ws.indices에 인덱스를 기록(내부 할당 X)
        if self.mode == "max":
            if work is None:
                raise ValueError("[Pool2D.forward_into] work is required for MaxPool (indices buffer).")
            pops.forward(
                X, kernel=self.kernel_size, stride=self.stride, padding=self.padding,
                dilation=self.dilation, ceil_mode=self.ceil_mode,
                count_include_pad=self.count_include_pad,
                mode="max", return_indices=False,
                ws_indices=work.indices,  # <- 여기에 기록
                stream=stream, out=out
            )
        else:
            pops.forward(
                X, kernel=self.kernel_size, stride=self.stride, padding=self.padding,
                dilation=self.dilation, ceil_mode=self.ceil_mode,
                count_include_pad=self.count_include_pad,
                mode="avg", stream=stream, out=out
            )

    def backward_into(
        self, gY: cp.ndarray, *,
        gA_out: cp.ndarray,
        gW_out: Optional[cp.ndarray] = None,
        gB_out: Optional[cp.ndarray] = None,
        work_dZ: Optional[Any] = None,
        lt_workspace: Optional[Any] = None,
        stream: Optional[int] = None,
        work: Optional[_Pool2DWork] = None,
    ) -> None:
        # Pool은 파라미터 없음 → gW_out/gB_out 무시
        # MaxPool은 indices 필요
        if self.mode == "max":
            if work is None or work.indices is None:
                raise RuntimeError("[Pool2D.backward_into] MaxPool requires work.indices.")
            # X/Y는 모양 검증용 → capture plan에서 고정 y(=현재 레이어 out) 필요
            # 여기서는 gA_out의 shape에 맞는 X dummy를 만들지 않고,
            # 캡처 플랜의 per-layer.y를 graph_exec에서 전달할 필요가 없다.
            # ops.backward는 실제 커널에서 dY와 indices만 사용하므로,
            # 안전하게 현재 버퍼로 호출한다.
            N, C, H, W = map(int, gA_out.shape)
            kH, kW = self.kernel_size
            Ho, Wo = self._out_hw(H, W, kH, kW, self.stride, self.padding, self.dilation, self.ceil_mode)
            # Y dummy (검증용 shape)
            Y_dummy = cp.empty((N, C, Ho, Wo), dtype=cp.float32)
            X_dummy = cp.empty((N, C, H, W), dtype=cp.float32)
            dX = pops.backward(
                X_dummy, Y_dummy, gY,
                kernel=self.kernel_size, stride=self.stride, padding=self.padding,
                dilation=self.dilation, ceil_mode=self.ceil_mode,
                count_include_pad=self.count_include_pad,
                mode="max", ws_indices=work.indices, stream=stream
            )
            gA_out[...] = dX
        else:
            N, C, H, W = map(int, gA_out.shape)
            kH, kW = self.kernel_size
            Ho, Wo = self._out_hw(H, W, kH, kW, self.stride, self.padding, self.dilation, self.ceil_mode)
            Y_dummy = cp.empty((N, C, Ho, Wo), dtype=cp.float32)
            X_dummy = cp.empty((N, C, H, W), dtype=cp.float32)
            dX = pops.backward(
                X_dummy, Y_dummy, gY,
                kernel=self.kernel_size, stride=self.stride, padding=self.padding,
                dilation=self.dilation, ceil_mode=self.ceil_mode,
                count_include_pad=self.count_include_pad,
                mode="avg", stream=stream
            )
            gA_out[...] = dX

    def zero_grad(self):  # no-op (파라미터 없음)
        return

    def state_dict(self) -> Dict[str, Any]:
        return {
            "kernel_size": self.kernel_size, "stride": self.stride, "padding": self.padding,
            "dilation": self.dilation, "ceil_mode": self.ceil_mode,
            "count_include_pad": self.count_include_pad, "mode": self.mode,
        }

    def load_state_dict(self, sd: Dict[str, Any]):
        for k in ("kernel_size","stride","padding","dilation","ceil_mode","count_include_pad","mode"):
            if k in sd: setattr(self, k, sd[k])
        return self