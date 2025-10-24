# python/graph_executor_v2/ops/memory.py
from __future__ import annotations

from typing import Iterable, Optional, Sequence, Dict, Any

import platform
import os
import importlib

import cupy as cp

# -------------------------------------------------------------
# DLL path bootstrap (Windows, Python 3.8+)
# -------------------------------------------------------------
def _maybe_add_dll_dirs(paths: Iterable[str]) -> None:
    if platform.system() != "Windows":
        return
    add = getattr(os, "add_dll_directory", None)
    if add is None:
        return
    for p in paths:
        if p and os.path.isdir(p):
            try:
                add(p)
            except Exception:
                pass

try:
    # 사용자의 프로젝트에서 제공하는 공용 부트스트랩
    from .common import ensure_cuda_dlls, get_stream_ptr
    ensure_cuda_dlls()
except Exception:
    # 최소한의 안전장치: CUDA_PATH/bin 추가 시도
    cuda_path = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
    if cuda_path:
        _maybe_add_dll_dirs([os.path.join(cuda_path, "bin")])

    def get_stream_ptr(stream: Optional[int | None]) -> int:
        # None이면 기본 스트림(0) 사용
        return int(stream or 0)

# 바인딩 로드
try:
    # graph_executor_v2.ops 패키지 내의 네이티브 모듈
    _g = importlib.import_module("graph_executor_v2.ops._ops_memory")
except Exception as e:
    raise ImportError(
        "[ops.memory] _ops_memory 바인딩을 찾을 수 없습니다. "
        "CMake 타겟(_ops_memory)을 빌드하여 python/graph_executor_v2/ops 에 배치하세요."
    ) from e


# -------------------------------------------------------------
# 유틸
# -------------------------------------------------------------
def _assert_contig_cuda(a: cp.ndarray, name: str) -> None:
    if not isinstance(a, cp.ndarray):
        raise TypeError(f"{name}: cupy.ndarray expected")
    if not a.flags.c_contiguous:
        raise ValueError(f"{name}: must be C-contiguous")
    if a.size <= 0:
        raise ValueError(f"{name}: empty array is not allowed")


def _as_shape64(a: cp.ndarray) -> list[int]:
    return [int(x) for x in a.shape]


# -------------------------------------------------------------
# 캡처-세이프 Fill 연산
# -------------------------------------------------------------
def fill_f32(dst: cp.ndarray, value: float, *, stream: Optional[int] = None) -> None:
    """
    dst(float32, contiguous)에 value를 씁니다. CUDA Graph 캡처 안전.
    """
    _assert_contig_cuda(dst, "dst")
    if dst.dtype != cp.float32:
        raise TypeError("dst: float32 required")
    sptr = int(get_stream_ptr(stream))
    _g.fill_f32(int(dst.data.ptr), _as_shape64(dst), float(value), sptr)


def fill_i32(dst: cp.ndarray, value: int, *, stream: Optional[int] = None) -> None:
    """
    dst(int32, contiguous)에 value를 씁니다. CUDA Graph 캡처 안전.
    """
    _assert_contig_cuda(dst, "dst")
    if dst.dtype != cp.int32:
        raise TypeError("dst: int32 required")
    sptr = int(get_stream_ptr(stream))
    _g.fill_i32(int(dst.data.ptr), _as_shape64(dst), int(value), sptr)


# -------------------------------------------------------------
# Capture-Safe Allocator (Arena) 제어
# -------------------------------------------------------------
def reserve_bytes(nbytes: int) -> None:
    """
    캡처 전에 워크스페이스 풀을 사전예약합니다. (추가 cudaMalloc 없이 재생되도록)
    """
    if nbytes <= 0:
        raise ValueError("nbytes must be > 0")
    _g.reserve_bytes(int(nbytes))


def reset_pool() -> None:
    """
    풀을 리셋합니다. (슬랩은 유지, bump/LIFO만 초기화)
    일반적으로 1 스텝/프레임 종료 시 호출을 권장.
    """
    _g.reset_pool()


def stats() -> Dict[str, Any]:
    """
    풀 통계를 반환합니다.
    keys: total_reserved, peak_in_use, curr_in_use, slabs
    """
    d = _g.stats()
    # pybind dict를 일반 dict로 복제 (타입 표준화)
    return dict(d)


# -------------------------------------------------------------
# 임시 워크스페이스 (토큰 기반)
# -------------------------------------------------------------
def alloc_temp(nbytes: int, align: int = 256, *, stream: Optional[int] = None) -> int:
    """
    임시 워크스페이스를 대여하고 토큰(uint64)을 반환합니다.
    캡처 중 부족하면 예외가 발생하도록 설계되어 있습니다.
    """
    if nbytes <= 0:
        raise ValueError("nbytes must be > 0")
    if align <= 0:
        raise ValueError("align must be > 0")
    sptr = int(get_stream_ptr(stream))
    token = int(_g.alloc_temp(int(nbytes), int(align), sptr))
    return token


def free_temp(token: int, *, stream: Optional[int] = None) -> None:
    """
    임시 워크스페이스 토큰을 반납합니다.
    (MVP 구현에서는 no-op일 수 있음. Arena가 재사용 free를 지원하면 내부에서 반납됩니다.)
    """
    sptr = int(get_stream_ptr(stream))
    _g.free_temp(int(token), sptr)


# (선택) 디버그 유틸
def token_bytes(token: int) -> int:
    """
    토큰에 인코딩된 바이트 수를 반환합니다. (디버깅용)
    """
    if hasattr(_g, "token_bytes"):
        return int(_g.token_bytes(int(token)))
    # 바인딩에 없으면 0 반환
    return 0


__all__ = [
    "fill_f32",
    "fill_i32",
    "reserve_bytes",
    "reset_pool",
    "stats",
    "alloc_temp",
    "free_temp",
    "token_bytes",
]
