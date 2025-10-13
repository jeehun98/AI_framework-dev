# python/graph_executor_v2/layers/concat.py
from __future__ import annotations
from typing import Iterable, List, Optional, Tuple, Any, Dict
import cupy as cp

from .base import Layer
from ..ops import concat as c_ops


def _norm_axis(axis: int, ndim: int) -> int:
    if axis < 0:
        axis += ndim
    if not (0 <= axis < ndim):
        raise ValueError(f"axis out of range: {axis} for ndim={ndim}")
    return axis


class Concat(Layer):
    """
    Concatenate N tensors along a given axis (float32 only), capture-safe.

    - 입력:  List[cp.ndarray], 모두 동일 dtype=float32, 동일 ndim/비연결 차원 크기
    - 파라미터: 없음
    - 출력:  단일 cp.ndarray (축 방향으로 이어붙인 결과)

    주의:
      * 다입력 연산이므로 Sequential(단일 텐서 체인)과 직접 호환되지 않습니다.
        그래프/모듈 컨텍스트에서 리스트 입력으로 사용하세요.
      * backward는 입력 개수만큼 gX 리스트를 돌려줍니다. capture 경로에서도 동일.
    """

    def __init__(self, axis: int, name: Optional[str] = None):
        super().__init__(name=name)
        self.axis = int(axis)
        self.input_shapes: Optional[List[Tuple[int, ...]]] = None
        self.output_shape: Optional[Tuple[int, ...]] = None
        self.built: bool = False

        # eager/capture 캐시 (역전파 시 분할용)
        self._last_input_shapes: Optional[List[Tuple[int, ...]]] = None
        self._cap_input_shapes: Optional[List[Tuple[int, ...]]] = None

    # --------- shape helpers ---------
    @staticmethod
    def _infer_out_shape(shapes: List[Tuple[int, ...]], axis: int) -> Tuple[int, ...]:
        if not shapes:
            raise ValueError("Concat requires at least one input shape")
        ndim = len(shapes[0])
        base = list(shapes[0])
        axis = _norm_axis(axis, ndim)

        cat = 0
        for s in shapes:
            if len(s) != ndim:
                raise ValueError("all inputs must have same ndim")
            for d in range(ndim):
                if d == axis: 
                    continue
                if s[d] != base[d]:
                    raise ValueError("non-concat dims must match")
            cat += int(s[axis])
        base[axis] = cat
        return tuple(base)

    # --------- Layer API ----------
    def build(self, input_shapes: List[Tuple[int, ...]]) -> None:  # 다입력
        if not isinstance(input_shapes, (list, tuple)) or len(input_shapes) == 0:
            raise ValueError("Concat.build expects a non-empty list of input shapes")
        shapes = [tuple(map(int, s)) for s in input_shapes]
        out_shape = self._infer_out_shape(shapes, self.axis)
        self.input_shapes = shapes
        self.output_shape = out_shape
        self.built = True

    def compute_output_shape(self, input_shapes: List[Tuple[int, ...]]) -> Tuple[int, ...]:
        shapes = [tuple(map(int, s)) for s in input_shapes]
        return self._infer_out_shape(shapes, self.axis)

    def parameters(self) -> Iterable[Tuple[Any, Any, str]]:
        return
        yield  # for linters

    # --------- eager ----------
    def call(self, xs: List[cp.ndarray]) -> cp.ndarray:
        assert self.built, "call() before build()"
        # 캐시: 역전파에서 split 용
        self._last_input_shapes = [tuple(map(int, x.shape)) for x in xs]
        y = c_ops.forward(xs, axis=self.axis, out=None, stream=None)
        return y

    def backward(self, gY: cp.ndarray) -> List[cp.ndarray]:
        assert self.built and self._last_input_shapes is not None, "backward() requires call() first"
        # dummy 텐서들로 shape만 맞춰 split (ops가 shape만 참조)
        xs_dummy = [cp.empty(s, dtype=cp.float32) for s in self._last_input_shapes]
        gxs = c_ops.backward(xs_dummy, axis=self.axis, gy=gY, stream=None)
        return gxs

    # --------- capture-safe ----------
    def forward_into(
        self,
        xs: List[cp.ndarray],
        *,
        out: cp.ndarray,
        z_out: Optional[cp.ndarray] = None,   # 미사용
        stream: Optional[int] = None,
        work: Optional[Any] = None,           # 시그니처 호환
    ) -> None:
        assert self.built, "forward_into() before build()"
        c_ops.forward(xs, axis=self.axis, out=out, stream=stream)
        # 캡처용 shape 저장 (bwd split)
        self._cap_input_shapes = [tuple(map(int, x.shape)) for x in xs]

    def backward_into(
        self,
        gY: cp.ndarray,
        *,
        gA_out: List[cp.ndarray],             # 입력 개수만큼의 grad 버퍼 리스트
        gW_out: Optional[cp.ndarray] = None,  # 미사용
        gB_out: Optional[cp.ndarray] = None,  # 미사용
        work_dZ: Optional[Any] = None,        # 미사용
        lt_workspace: Optional[Any] = None,   # 미사용
        stream: Optional[int] = None,
        work: Optional[Any] = None,           # 미사용
    ) -> None:
        """
        주의: capture 경로에서도 다입력이므로 gA_out은 List[ndarray]여야 합니다.
        프레임워크 실행기가 이를 지원해야 합니다.
        """
        assert self.built and self._cap_input_shapes is not None, "backward_into() requires forward_into() first"
        xs_dummy = [cp.empty(s, dtype=cp.float32) for s in self._cap_input_shapes]
        gxs = c_ops.backward(xs_dummy, axis=self.axis, gy=gY, stream=stream)
        if len(gxs) != len(gA_out):
            raise ValueError("gA_out length must match number of inputs")
        for dst, src in zip(gA_out, gxs):
            if tuple(dst.shape) != tuple(src.shape) or dst.dtype != cp.float32:
                raise ValueError("gA_out buffers must match input shapes and be float32")
            dst[...] = src

    # --------- misc ----------
    def zero_grad(self):  # no params
        return

    def state_dict(self) -> Dict[str, Any]:
        return {"axis": self.axis}

    def load_state_dict(self, sd: Dict[str, Any]):
        if "axis" in sd:
            self.axis = int(sd["axis"])
        return self
