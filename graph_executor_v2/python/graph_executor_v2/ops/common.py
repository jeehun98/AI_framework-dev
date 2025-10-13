# python/graph_executor_v2/ops/common.py
from __future__ import annotations
import os
from typing import Optional, Tuple
import cupy as cp

def ensure_cuda_dlls(paths: Optional[list[str]] = None) -> None:
    """(Windows 전용) CUDA DLL 경로 가드."""
    if not hasattr(os, "add_dll_directory"):
        return
    defaults = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin",
        r"C:\Windows\System32",
    ]
    for d in (paths or defaults):
        if os.path.isdir(d):
            try:
                os.add_dll_directory(d)
            except FileNotFoundError:
                pass

def assert_f32_2d(x: cp.ndarray, name: str = "array") -> Tuple[int, int]:
    if not isinstance(x, cp.ndarray):
        raise TypeError(f"{name}: expected cupy.ndarray, got {type(x)}")
    if x.dtype != cp.float32:
        raise TypeError(f"{name}: expected float32, got {x.dtype}")
    if x.ndim != 2:
        raise ValueError(f"{name}: expected 2D, got shape={x.shape}")
    return int(x.shape[0]), int(x.shape[1])


def get_stream_ptr(stream: Optional[int] = None) -> Optional[int]:
    """None -> 현재 CuPy 스트림 포인터 정수, int -> 그대로."""
    if stream is None:
        return int(cp.cuda.get_current_stream().ptr)
    return int(stream)

def to_voidp_capsule(ptr: Optional[int]) -> object:
    """
    pybind11에서 void* 인자로 받는 경우 Capsule로 감싸 전달해야 함.
    ptr가 None 또는 0이면 None을 리턴(= nullptr).
    """
    if not ptr:
        return None
    import ctypes
    PyCapsule_New = ctypes.pythonapi.PyCapsule_New
    PyCapsule_New.restype = ctypes.py_object
    PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    # name=None, destructor=None 로 단순 캡슐 생성
    return PyCapsule_New(ctypes.c_void_p(ptr), None, None)


def as_tensor_2d(x: cp.ndarray):
    """
    CuPy 2D float32 배열을 ai::Tensor로 래핑.
    _ops_common.make_tensor_2d(ptr, [M,N], dtype, device, device_index) 시그니처에 맞춤.
    비연속 배열이면 C-연속으로 복사해 안전 보장.
    """
    from graph_executor_v2.ops import _ops_gemm as g  # re-export된 타입/팩토리 사용
    M, N = assert_f32_2d(x, "as_tensor_2d(x)")

    # C-연속 보장 (row-major 전제)
    if not x.flags.c_contiguous:
        x = cp.ascontiguousarray(x)

    ptr = int(x.data.ptr)
    # DType/Device enum은 _ops_common에서 re-export됨
    return g.make_tensor_2d(ptr, [int(M), int(N)], g.DType.F32, g.Device.CUDA, 0)


def empty_like_2d(ref: cp.ndarray) -> cp.ndarray:
    assert_f32_2d(ref, "empty_like_2d(ref)")
    return cp.empty(ref.shape, dtype=cp.float32)

def empty_2d(m: int, n: int) -> cp.ndarray:
    return cp.empty((int(m), int(n)), dtype=cp.float32)
