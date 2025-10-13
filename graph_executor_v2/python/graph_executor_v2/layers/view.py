# python/graph_executor_v2/layers/view.py
from __future__ import annotations
from typing import Iterable, Optional, Tuple, Any, Dict, Sequence, List
import cupy as cp

from .base import Layer
from ..ops import view as vops


def _to_int_list(seq: Sequence[int]) -> List[int]:
    return [int(x) for x in seq]


class View(Layer):
    """
    Pointer-view (reshape/stride/offset) 레이어 — capture-safe.

    Args:
        shape: 출력 shape (예: (N, C*H*W)). contiguous(=stride=None)일 때 -1 1개 허용.
        stride: 요소 단위 strides. None이면 contiguous로 처리.
        offset: 요소 단위 오프셋(기본 0).

    특징:
      - 파라미터 없음.
      - forward_into / backward_into 모두 캡처-세이프.
      - dtype은 float32만 지원(ops.view 제약).
    """

    def __init__(
        self,
        shape: Sequence[int],
        *,
        stride: Optional[Sequence[int]] = None,
        offset: int = 0,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.req_shape = tuple(int(s) for s in shape)
        self.req_stride = tuple(int(s) for s in stride) if stride is not None else None
        self.offset = int(offset)

        # runtime state
        self.input_shape: Optional[Tuple[int, ...]] = None
        self.output_shape: Optional[Tuple[int, ...]] = None
        self.built: bool = False

    # ---------- shape helpers ----------
    @staticmethod
    def _numel(shape: Sequence[int]) -> int:
        n = 1
        for s in shape:
            n *= int(s)
        return int(n)

    def _infer_shape(self, in_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """contiguous(stride=None)일 때만 -1 1개 허용해 출력 shape 추론."""
        out = list(self.req_shape)
        if self.req_stride is not None:
            # stride 명시 시 -1 허용하지 않음 (명확성)
            if any(s == -1 for s in out):
                raise ValueError("View: stride is specified; shape cannot contain -1")
            return tuple(out)

        # contiguous: -1 한 개까지 허용
        neg_idx = [i for i, s in enumerate(out) if int(s) == -1]
        if len(neg_idx) == 0:
            return tuple(out)
        if len(neg_idx) > 1:
            raise ValueError("View: at most one -1 is allowed in shape for contiguous view")

        in_elems = self._numel(in_shape)
        if self.offset != 0:
            # 안전을 위해 오프셋이 있으면 -1 추론 막음 (필요시 확장 가능)
            raise ValueError("View: -1 shape inference is not supported when offset != 0")

        known = 1
        for s in out:
            if s != -1:
                known *= int(s)
        if in_elems % known != 0:
            raise ValueError(f"View: cannot infer dim (numel {in_elems} not divisible by {known})")
        out[neg_idx[0]] = in_elems // known
        return tuple(int(s) for s in out)

    # ---------- Layer API ----------
    def build(self, input_shape: Tuple[int, ...]) -> None:
        # float32 전제이므로 numel 기준으로만 체크(실제 dtype 검증은 ops에서 수행)
        out_shape = self._infer_shape(tuple(map(int, input_shape)))
        self.input_shape = tuple(map(int, input_shape))
        self.output_shape = tuple(map(int, out_shape))
        self.built = True

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        return self._infer_shape(tuple(map(int, input_shape)))

    def parameters(self) -> Iterable[Tuple[Any, Any, str]]:
        # 파라미터 없음
        return
        yield  # mypy/linters silence

    # ---------- eager path ----------
    def call(self, X: cp.ndarray) -> cp.ndarray:
        assert self.built, "View.call() before build()"
        # ops.view.forward 가 dtype/contiguity를 검증/정리
        Y = vops.forward(
            X,
            shape=self.output_shape,  # type: ignore[arg-type]
            stride=self.req_stride,
            offset=self.offset,
            out=None,
            stream=None,
        )
        return Y

    def backward(self, gY: cp.ndarray) -> cp.ndarray:
        assert self.built and self.input_shape is not None, "View.backward() before build()"
        # ops.backward는 입력 X의 shape가 필요하므로 더미 텐서를 생성해 전달
        X_dummy = cp.empty(self.input_shape, dtype=cp.float32)
        gX = vops.backward(
            X_dummy,
            shape=self.output_shape,  # type: ignore[arg-type]
            stride=self.req_stride,
            offset=self.offset,
            gy=gY,
            stream=None,
        )
        return gX

    # ---------- capture-safe path ----------
    def forward_into(
        self,
        X: cp.ndarray,
        *,
        out: cp.ndarray,
        z_out: Optional[cp.ndarray] = None,  # 미사용
        stream: Optional[int] = None,
        work: Optional[Any] = None,          # 시그니처 호환
    ) -> None:
        assert self.built, "View.forward_into() before build()"
        vops.forward(
            X,
            shape=self.output_shape,  # type: ignore[arg-type]
            stride=self.req_stride,
            offset=self.offset,
            out=out,
            stream=stream,
        )

    def backward_into(
        self,
        gY: cp.ndarray,
        *,
        gA_out: cp.ndarray,
        gW_out: Optional[cp.ndarray] = None,   # 미사용
        gB_out: Optional[cp.ndarray] = None,   # 미사용
        work_dZ: Optional[Any] = None,         # 미사용
        lt_workspace: Optional[Any] = None,    # 미사용
        stream: Optional[int] = None,
        work: Optional[Any] = None,            # 미사용
    ) -> None:
        assert self.built and self.input_shape is not None, "View.backward_into() before build()"
        X_dummy = cp.empty(self.input_shape, dtype=cp.float32)
        gX = vops.backward(
            X_dummy,
            shape=self.output_shape,  # type: ignore[arg-type]
            stride=self.req_stride,
            offset=self.offset,
            gy=gY,
            stream=stream,
        )
        gA_out[...] = gX

    # ---------- misc ----------
    def zero_grad(self):
        # 파라미터 없음
        return

    def state_dict(self) -> Dict[str, Any]:
        return {
            "shape": self.req_shape,
            "stride": self.req_stride,
            "offset": self.offset,
        }

    def load_state_dict(self, sd: Dict[str, Any]):
        if "shape" in sd:   self.req_shape = tuple(int(x) for x in sd["shape"])
        if "stride" in sd:  self.req_stride = tuple(int(x) for x in sd["stride"]) if sd["stride"] is not None else None
        if "offset" in sd:  self.offset = int(sd["offset"])
        # rebuild은 외부에서 호출
        return self
