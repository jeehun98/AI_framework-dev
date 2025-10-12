# python/graph_executor_v2/layers/slice.py
from __future__ import annotations
from typing import Iterable, Optional, Tuple, Any, Dict, Sequence, List
import cupy as cp

from .base import Layer
from ..ops import slice as slops


def _to_i_list(x: Sequence[int], n: int, name: str) -> Tuple[int, ...]:
    if len(x) != n:
        raise ValueError(f"{name} must be len-{n}, got {len(x)}")
    return tuple(int(v) for v in x)


class Slice4D(Layer):
    """
    4D 슬라이스 레이어 (N,C,H,W; float32), capture-safe.

    Args:
        start: 길이 4 (포함 시작)
        end:   길이 4 (배타 종료)
        step:  길이 4 (0 불가, 음수 가능)
        clamp: 인덱스 클램프 여부(ops에 전달)
    특징:
      - 파라미터 없음.
      - forward_into/backward_into 모두 캡처-세이프.
      - 입력/출력은 반드시 float32, 4D.
    """
    def __init__(
        self,
        *,
        start: Sequence[int],
        end:   Sequence[int],
        step:  Sequence[int],
        clamp: bool = True,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.start = _to_i_list(start, 4, "start")
        self.end   = _to_i_list(end,   4, "end")
        self.step  = _to_i_list(step,  4, "step")
        self.clamp = bool(clamp)

        self.input_shape: Optional[Tuple[int, ...]] = None
        self.output_shape: Optional[Tuple[int, ...]] = None
        self.built: bool = False

    # ---------- shape helpers ----------
    @staticmethod
    def _infer_out_shape(start: Sequence[int], end: Sequence[int], step: Sequence[int], in_shape: Sequence[int]) -> Tuple[int, ...]:
        out = []
        for s, e, t, dim in zip(start, end, step, in_shape):
            t = int(t)
            if t == 0:
                raise ValueError("step cannot be zero")
            if t > 0:
                length = max(0, (int(e) - int(s) + (t - 1)) // t)
            else:
                length = max(0, (int(s) - int(e) - 1) // (-t) + 1)
            out.append(int(length))
        return tuple(out)

    # ---------- Layer API ----------
    def build(self, input_shape: Tuple[int, ...]) -> None:
        if len(input_shape) != 4:
            raise ValueError(f"Slice4D expects 4D input, got {input_shape}")
        out_shape = self._infer_out_shape(self.start, self.end, self.step, input_shape)
        self.input_shape  = tuple(map(int, input_shape))
        self.output_shape = tuple(map(int, out_shape))
        self.built = True

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        if len(input_shape) != 4:
            raise ValueError("Slice4D requires 4D input shape")
        return self._infer_out_shape(self.start, self.end, self.step, input_shape)

    def parameters(self) -> Iterable[Tuple[Any, Any, str]]:
        return
        yield  # for linters

    # ---------- eager ----------
    def call(self, X: cp.ndarray) -> cp.ndarray:
        assert self.built, "call() before build()"
        Y = slops.forward(
            X,
            start=self.start, end=self.end, step=self.step,
            out=None, clamp=self.clamp, stream=None
        )
        return Y

    def backward(self, gY: cp.ndarray) -> cp.ndarray:
        assert self.built and self.input_shape is not None, "backward() before build()"
        X_dummy = cp.empty(self.input_shape, dtype=cp.float32)
        gX = slops.backward(
            X_dummy,
            start=self.start, end=self.end, step=self.step,
            gy=gY, stream=None
        )
        return gX

    # ---------- capture-safe ----------
    def forward_into(
        self,
        X: cp.ndarray,
        *,
        out: cp.ndarray,
        z_out: Optional[cp.ndarray] = None,   # 미사용
        stream: Optional[int] = None,
        work: Optional[Any] = None,           # 시그니처 호환
    ) -> None:
        assert self.built, "forward_into() before build()"
        slops.forward(
            X,
            start=self.start, end=self.end, step=self.step,
            out=out, clamp=self.clamp, stream=stream
        )

    def backward_into(
        self,
        gY: cp.ndarray,
        *,
        gA_out: cp.ndarray,
        gW_out: Optional[cp.ndarray] = None,  # 미사용
        gB_out: Optional[cp.ndarray] = None,  # 미사용
        work_dZ: Optional[Any] = None,        # 미사용
        lt_workspace: Optional[Any] = None,   # 미사용
        stream: Optional[int] = None,
        work: Optional[Any] = None,           # 미사용
    ) -> None:
        assert self.built and self.input_shape is not None, "backward_into() before build()"
        X_dummy = cp.empty(self.input_shape, dtype=cp.float32)
        gX = slops.backward(
            X_dummy,
            start=self.start, end=self.end, step=self.step,
            gy=gY, stream=stream
        )
        gA_out[...] = gX

    # ---------- misc ----------
    def zero_grad(self):
        return

    def state_dict(self) -> Dict[str, Any]:
        return {
            "start": self.start,
            "end":   self.end,
            "step":  self.step,
            "clamp": self.clamp,
        }

    def load_state_dict(self, sd: Dict[str, Any]):
        if "start" in sd: self.start = _to_i_list(sd["start"], 4, "start")
        if "end"   in sd: self.end   = _to_i_list(sd["end"],   4, "end")
        if "step"  in sd: self.step  = _to_i_list(sd["step"],  4, "step")
        if "clamp" in sd: self.clamp = bool(sd["clamp"])
        return self
