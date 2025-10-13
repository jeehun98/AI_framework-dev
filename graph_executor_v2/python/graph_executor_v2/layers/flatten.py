# python/graph_executor_v2/layers/flatten.py
from __future__ import annotations
from typing import Tuple, Any, Optional
import math
import cupy as cp

from .base import Layer

class Flatten(Layer):
    """
    (N, C, H, W) -> (N, C*H*W) 처럼 지정 축부터 끝까지를 1D로 펴는 단순 View 레이어.
    - 캡처-세이프: forward_into / backward_into 제공
    - 파라미터 없음
    """
    def __init__(self, start_axis: int = 1, name: Optional[str] = None):
        super().__init__(name=name)
        self.start_axis = int(start_axis)
        self.input_shape: Optional[Tuple[int, ...]] = None
        self.output_shape: Optional[Tuple[int, ...]] = None
        self.built = False

    # ----- shape utils -----
    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        if not isinstance(input_shape, tuple):
            input_shape = tuple(map(int, input_shape))  # type: ignore
        n_dim = len(input_shape)
        s = self.start_axis if self.start_axis >= 0 else n_dim + self.start_axis
        if s < 0 or s >= n_dim:
            raise ValueError(f"start_axis out of range: {self.start_axis} for shape {input_shape}")
        head = input_shape[:s]
        tail = input_shape[s:]
        flat = 1
        for v in tail:
            flat *= int(v)
        return tuple(head + (flat,))

    def build(self, input_shape: Tuple[int, ...]) -> None:
        self.input_shape = tuple(map(int, input_shape))
        self.output_shape = self.compute_output_shape(self.input_shape)
        self.built = True

    # ----- eager path -----
    def call(self, x: cp.ndarray) -> cp.ndarray:
        if not self.built:
            # lazy build on first call
            self.build(tuple(map(int, x.shape)))
        return x.reshape(self.output_shape)

    def backward(self, grad_output: cp.ndarray) -> cp.ndarray:
        assert self.input_shape is not None, "Flatten.backward called before build"
        return grad_output.reshape(self.input_shape)

    # ----- capture-safe path -----
    def forward_into(
        self,
        x: cp.ndarray,
        *,
        out: cp.ndarray,
        z_out: Optional[cp.ndarray] = None,   # 미사용 (일관성 유지용)
        stream: Optional[int] = None,
    ) -> None:
        # 모든 텐서는 C-contiguous 가정(Sequential/capture_plan이 그렇게 할당)
        if not self.built:
            self.build(tuple(map(int, x.shape)))
        if tuple(out.shape) != tuple(self.output_shape):
            raise ValueError(f"[Flatten.forward_into] out shape mismatch: expect {self.output_shape}, got {out.shape}")
        out[...] = x.reshape(out.shape)

    def backward_into(
        self,
        gY: cp.ndarray,
        *,
        gA_out: cp.ndarray,
        gW_out: Optional[cp.ndarray] = None,
        gB_out: Optional[cp.ndarray] = None,
        work_dZ: Any = None,
        lt_workspace: Any = None,
        stream: Optional[int] = None,
    ) -> None:
        # gW/gB 없음
        if gW_out is not None or gB_out is not None:
            # 사용자가 주더라도 무시 (Flatten은 파라미터가 없음)
            pass
        assert self.input_shape is not None, "Flatten.backward_into called before build"
        if tuple(gA_out.shape) != tuple(self.input_shape):
            raise ValueError(f"[Flatten.backward_into] gA_out shape mismatch: expect {self.input_shape}, got {gA_out.shape}")
        gA_out[...] = gY.reshape(self.input_shape)

    # ----- parameters interface -----
    def parameters(self):
        # 파라미터 없음
        if False:
            yield  # generator 형식 유지
        return

    def state_dict(self) -> dict:
        return {
            "name": self.name,
            "start_axis": self.start_axis,
        }

    def load_state_dict(self, sd: dict):
        self.start_axis = int(sd.get("start_axis", self.start_axis))
        # shape는 런타임 build 시 재계산
        return self
