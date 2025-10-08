# python/graph_executor_v2/ops/pad.py
from __future__ import annotations
from typing import Iterable, List, Optional, Sequence, Tuple, Union
import cupy as cp

from .common import ensure_cuda_dlls, get_stream_ptr
ensure_cuda_dlls()

try:
    from graph_executor_v2.ops import _ops_pad as _g
except Exception as e:
    raise ImportError(
        "[ops.pad] _ops_pad 바인딩을 찾을 수 없습니다. "
        "CMake 타겟(_ops_pad)을 빌드하여 python/graph_executor_v2/ops 에 배치하세요."
    ) from e


# ------------------------ 유틸 ------------------------
def _as_list_int(x: Union[int, Sequence[int]], rank: int, name: str) -> List[int]:
    if isinstance(x, int):
        return [int(x)] * rank
    try:
        xs = list(map(int, x))
    except Exception:
        raise TypeError(f"{name}: int 또는 길이 {rank}의 시퀀스여야 합니다.")
    if len(xs) != rank:
        raise ValueError(f"{name}: 길이가 {rank}이어야 합니다. got {len(xs)}")
    return xs

def _ensure_f32_nd(a: cp.ndarray, name: str):
    if not isinstance(a, cp.ndarray):
        raise TypeError(f"{name}: cupy.ndarray 필요, got {type(a)}")
    if a.dtype != cp.float32:
        raise TypeError(f"{name}: float32 필요, got {a.dtype}")
    if a.size == 0:
        raise ValueError(f"{name}: 빈 배열은 허용되지 않습니다.")

def _elem_strides(a: cp.ndarray) -> List[int]:
    """요소 단위(element) strides 리스트를 반환."""
    item = int(a.itemsize)
    return [int(s // item) for s in a.strides]

def _compute_out_shape(x_shape: Sequence[int],
                       before: Sequence[int],
                       after: Sequence[int]) -> Tuple[int, ...]:
    return tuple(int(x + b + a) for x, b, a in zip(x_shape, before, after))

def _make_spec(before: Sequence[int], after: Sequence[int], value: float) -> "_g.PadSpec":
    spec = _g.PadSpec()
    spec.before = list(map(int, before))
    spec.after  = list(map(int, after))
    spec.value  = float(value)
    return spec


# ------------------------ Public API ------------------------
def forward(
    X: cp.ndarray,
    *,
    before: Union[int, Sequence[int]],
    after:  Union[int, Sequence[int]],
    value: float = 0.0,
    out: Optional[cp.ndarray] = None,
    stream: Optional[Union[int, cp.cuda.Stream]] = None,
) -> cp.ndarray:
    """
    Y = pad(X, before/after, value)
      - X: float32, 임의 ND (NCHW 등 상관없음), 연속/비연속 모두 지원
      - before/after: 정수 또는 rank 길이 시퀀스
      - value: constant padding 값
      - out: 미리 할당한 출력 버퍼(선택). 없으면 새로 생성
    """
    _ensure_f32_nd(X, "X")
    rank = X.ndim
    before_l = _as_list_int(before, rank, "before")
    after_l  = _as_list_int(after,  rank, "after")
    y_shape  = _compute_out_shape(X.shape, before_l, after_l)

    if out is None:
        Y = cp.empty(y_shape, dtype=cp.float32)
    else:
        _ensure_f32_nd(out, "out")
        if tuple(out.shape) != y_shape:
            raise ValueError(f"out.shape mismatch: expected {y_shape}, got {tuple(out.shape)}")
        Y = out

    spec = _make_spec(before_l, after_l, value)

    x_ptr  = int(X.data.ptr)
    y_ptr  = int(Y.data.ptr)
    x_shape = list(map(int, X.shape))
    y_shape = list(map(int, Y.shape))

    # 요소 단위 strides (비연속 대응). C-연속이면 생략 가능하지만 항상 전달해도 안전.
    x_strides_elems = _elem_strides(X)
    y_strides_elems = _elem_strides(Y)

    sptr = int(get_stream_ptr(stream))
    _g.forward(
        x_ptr, x_shape,
        y_ptr, y_shape,
        spec,
        x_strides_elems,  # or None
        y_strides_elems,  # or None
        sptr
    )
    return Y


def forward_into(
    X: cp.ndarray,
    *,
    before: Union[int, Sequence[int]],
    after:  Union[int, Sequence[int]],
    value: float,
    out: cp.ndarray,
    stream: Optional[Union[int, cp.cuda.Stream]] = None,
) -> None:
    """미리 할당한 out 버퍼에 pad 결과를 씁니다."""
    _ = forward(X, before=before, after=after, value=value, out=out, stream=stream)
    return None


def backward(
    dY: cp.ndarray,
    *,
    before: Union[int, Sequence[int]],
    after:  Union[int, Sequence[int]],
    dX_shape: Sequence[int],   # 원래 입력 X의 shape
    out: Optional[cp.ndarray] = None,
    stream: Optional[Union[int, cp.cuda.Stream]] = None,
) -> cp.ndarray:
    """
    dX = slice(dY, spec)  — forward pad 의 역연산(패딩 영역 제외)
      - dY: float32 ND
      - before/after: forward 와 동일
      - dX_shape: 원래 X 모양
      - out: 미리 할당한 dX 버퍼(선택)
    """
    _ensure_f32_nd(dY, "dY")
    rank = dY.ndim
    if len(dX_shape) != rank:
        raise ValueError(f"dX_shape rank must be {rank}, got {len(dX_shape)}")

    before_l = _as_list_int(before, rank, "before")
    after_l  = _as_list_int(after,  rank, "after")

    # 검증: dY.shape == dX_shape + pads
    expect_y = _compute_out_shape(dX_shape, before_l, after_l)
    if tuple(dY.shape) != expect_y:
        raise ValueError(f"dY.shape mismatch: expected {expect_y}, got {tuple(dY.shape)}")

    if out is None:
        dX = cp.empty(tuple(map(int, dX_shape)), dtype=cp.float32)
    else:
        _ensure_f32_nd(out, "out(dX)")
        if tuple(out.shape) != tuple(map(int, dX_shape)):
            raise ValueError(f"dX(out).shape mismatch: expected {tuple(dX_shape)}, got {tuple(out.shape)}")
        dX = out

    spec = _make_spec(before_l, after_l, value=0.0)  # value는 bwd에서 사용되지 않음

    dy_ptr = int(dY.data.ptr)
    dx_ptr = int(dX.data.ptr)
    dy_shape = list(map(int, dY.shape))
    dx_shape = list(map(int, dX.shape))

    dy_strides_elems = _elem_strides(dY)
    dx_strides_elems = _elem_strides(dX)

    sptr = int(get_stream_ptr(stream))
    _g.backward(
        dy_ptr, dy_shape,
        dx_ptr, dx_shape,
        spec,
        dy_strides_elems,   # or None
        dx_strides_elems,   # or None
        sptr
    )
    return dX


def backward_into(
    dY: cp.ndarray,
    *,
    before: Union[int, Sequence[int]],
    after:  Union[int, Sequence[int]],
    dX_out: cp.ndarray,
    stream: Optional[Union[int, cp.cuda.Stream]] = None,
) -> None:
    """미리 할당한 dX_out에 slice(dY) 결과를 씁니다."""
    _ = backward(dY, before=before, after=after, dX_shape=dX_out.shape, out=dX_out, stream=stream)
    return None


# -------------------- 편의 함수 (테스트/유틸) --------------------
def compute_padded_shape(in_shape: Sequence[int],
                         before: Union[int, Sequence[int]],
                         after:  Union[int, Sequence[int]]) -> Tuple[int, ...]:
    """파이썬 측에서 출력 모양 계산(런처와 동일 공식)."""
    rank = len(in_shape)
    b = _as_list_int(before, rank, "before")
    a = _as_list_int(after,  rank, "after")
    return _compute_out_shape(in_shape, b, a)
