from __future__ import annotations
from typing import Optional, Tuple, Sequence, Iterable, Any, Dict, Union, List
import cupy as cp

from .base import Layer
from ..ops import pad as padops

# --------------------- 내부 유틸 ---------------------
def _as_list_int(x: Union[int, Sequence[int]], rank: int, *, name: str) -> List[int]:
    if isinstance(x, int):
        return [int(x)] * rank
    try:
        xs = list(map(int, x))
    except Exception:
        raise TypeError(f"{name}: int 또는 길이 {rank}의 시퀀스여야 합니다.")
    return xs

def _norm_before_after(
    before: Union[int, Sequence[int]],
    after:  Union[int, Sequence[int]],
    in_shape: Tuple[int, ...],
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """
    사용 편의를 위해 다음을 허용:
      - int:   모든 차원에 동일 패딩
      - 2-tuple: NCHW 가정 시 (H,W)만 패딩, 나머지 0
      - rank-tuple: 각 차원별 패딩
    """
    rank = len(in_shape)

    # (H,W) 전용 축을 쓰고 싶을 때: 길이 2로 주면 뒤의 두 축(H,W)에만 적용
    def _expand(v, name):
        if isinstance(v, int):
            return (0,) * (rank - 2) + (int(v), int(v)) if rank >= 2 else (int(v),) * rank
        try:
            v = tuple(map(int, v))
        except Exception:
            raise TypeError(f"{name}: int 또는 길이 2/ rank({rank}) 시퀀스여야 합니다.")
        if len(v) == 2 and rank >= 2:
            return (0,) * (rank - 2) + (v[0], v[1])
        if len(v) == rank:
            return v
        raise ValueError(f"{name}: 길이 2 또는 {rank}이어야 합니다. got {len(v)}")

    b = _expand(before, "before")
    a = _expand(after,  "after")
    return tuple(b), tuple(a)

# --------------------- Pad 레이어 ---------------------
class Pad(Layer):
    """
    Constant Pad 레이어 (CUDA, capture-safe)

    Args:
      before: int | (H,W) | (rank, )  — 앞쪽 패딩 크기
      after:  int | (H,W) | (rank, )  — 뒤쪽 패딩 크기
      value:  float, 패딩 상수값
      name:   optional layer name

    특징:
      - 임의 랭크 입력 지원 (NCHW 권장)
      - 파라미터 없음 (gW/gB 없음), gA만 생성
      - eager/capture 경로 모두 지원 (forward_into/backward_into)
    """
    def __init__(
        self,
        before: Union[int, Sequence[int]],
        after:  Union[int, Sequence[int]],
        value: float = 0.0,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.before = before
        self.after  = after
        self.value  = float(value)

        self.input_shape: Optional[Tuple[int, ...]]  = None
        self.output_shape: Optional[Tuple[int, ...]] = None
        self.built: bool = False
        self.training: bool = True  # 의미적 영향 없음(일관성)

    # --------- Layer API ---------
    def build(self, input_shape: Tuple[int, ...]) -> None:
        in_shape = tuple(map(int, input_shape))
        b, a = _norm_before_after(self.before, self.after, in_shape)
        out_shape = padops.compute_padded_shape(in_shape, b, a)
        self.input_shape  = in_shape
        self.output_shape = out_shape
        self.built = True

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        in_shape = tuple(map(int, input_shape))
        b, a = _norm_before_after(self.before, self.after, in_shape)
        return padops.compute_padded_shape(in_shape, b, a)

    def parameters(self) -> Iterable[Tuple[Any, Any, str]]:
        # 파라미터 없음
        if False:
            yield  # type: ignore
        return

    # --------- eager ---------
    def call(self, X: cp.ndarray) -> cp.ndarray:
        in_shape = tuple(map(int, X.shape))
        b, a = _norm_before_after(self.before, self.after, in_shape)
        return padops.forward(X, before=b, after=a, value=self.value)

    def backward(self, gY: cp.ndarray) -> cp.ndarray:
        assert self.input_shape is not None, "build() 이후 호출"
        in_shape = self.input_shape
        b, a = _norm_before_after(self.before, self.after, in_shape)
        return padops.backward(gY, before=b, after=a, dX_shape=in_shape)

    # --------- capture-safe ---------
    def forward_into(
        self,
        X: cp.ndarray,
        *,
        out: cp.ndarray,
        z_out: Optional[cp.ndarray] = None,  # 호환성용(미사용)
        stream: Optional[int] = None,
    ) -> None:
        in_shape = tuple(map(int, X.shape))
        b, a = _norm_before_after(self.before, self.after, in_shape)
        padops.forward_into(X, before=b, after=a, value=self.value, out=out, stream=stream)

    def backward_into(
        self,
        gY: cp.ndarray,
        *,
        gA_out: cp.ndarray,
        gW_out: Optional[cp.ndarray] = None,  # 항상 None
        gB_out: Optional[cp.ndarray] = None,  # 항상 None
        work: Optional[Any] = None,           # 시그니처 호환용
        work_dZ: Optional[Any] = None,        # 시그니처 호환용
        lt_workspace: Optional[Any] = None,   # 시그니처 호환용
        stream: Optional[int] = None,
    ) -> None:
        assert self.input_shape is not None, "build() 이후 호출"
        b, a = _norm_before_after(self.before, self.after, self.input_shape)
        padops.backward_into(gY, before=b, after=a, dX_out=gA_out, stream=stream)

    # --------- 기타 ---------
    def zero_grad(self):
        # nothing to do
        return

    def state_dict(self) -> Dict[str, Any]:
        return {"before": self.before, "after": self.after, "value": self.value}

    def load_state_dict(self, sd: Dict[str, Any]):
        for k in ("before", "after", "value"):
            if k in sd:
                setattr(self, k, sd[k])
        return self
