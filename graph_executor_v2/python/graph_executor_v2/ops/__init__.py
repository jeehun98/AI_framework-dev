"""
graph_executor_v2.ops
- require(op): C++ 바인딩 모듈(_ops_<op>) 동적 import
- 하위 파이썬 래퍼(gemm, conv2d, ...)는 필요 시 lazy import (PEP 562)
- Windows: CUDA/NVTX/패키지-동반 DLL 검색 경로 부트스트랩
"""
from __future__ import annotations
import importlib
import os
import sys
import platform
import ctypes
from typing import Iterable, List

# -------------------------------------------------------------
# 1) 패키지 레벨 DLL 경로 부트스트랩 (존재 시)
# -------------------------------------------------------------
try:
    # 선택적: 별도 유틸이 있다면 우선 사용
    from .._dllpaths import ensure_runtime_paths  # type: ignore
    ensure_runtime_paths()
except Exception:
    pass

# -------------------------------------------------------------
# 2) 모듈 자체 부트스트랩 (Windows, Python 3.8+)
#    - GE_OPS_NO_DLL_BOOTSTRAP=1 로 끌 수 있음
#    - GE_OPS_DEBUG_DLLS=1 로 사전 진단 출력
# -------------------------------------------------------------
def _maybe_add_dll_dirs(paths: Iterable[str]) -> None:
    if platform.system() != "Windows":
        return
    add = getattr(os, "add_dll_directory", None)
    if add is None:
        return
    seen = set()
    for p in paths:
        if not p:
            continue
        p = os.path.normpath(p)
        if p in seen:
            continue
        if os.path.isdir(p):
            try:
                add(p)
            except Exception:
                # 권한/중복 등은 조용히 무시
                pass
            finally:
                seen.add(p)

def _collect_candidate_paths() -> List[str]:
    cands: List[str] = []

    # a) _ops_* .pyd가 위치한 폴더(동반 DLL 배치 시 필수)
    try:
        # __file__ = .../graph_executor_v2/ops/__init__.py
        ops_dir = os.path.abspath(os.path.dirname(__file__))
        cands.append(ops_dir)
    except Exception:
        pass

    # b) CUDA PATH/HOME
    for key in ("CUDA_PATH", "CUDA_HOME"):
        base = os.environ.get(key)
        if base:
            cands += [os.path.join(base, "bin"), os.path.join(base, "libnvvp")]

    # c) Conda/venv 내 DLL 경로
    for key in ("CONDA_PREFIX", "VIRTUAL_ENV"):
        base = os.environ.get(key)
        if base:
            cands += [
                os.path.join(base, "Library", "bin"),  # conda-forge on Windows
                os.path.join(base, "bin"),
            ]

    # d) 일반 설치 경로 힌트 (환경에 맞게 필요시 추가)
    cands += [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\libnvvp",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\libnvvp",
        r"C:\Program Files\NVIDIA Corporation\NVToolsExt\bin\x64",
        r"C:\Program Files\NVIDIA Corporation\NVToolsExt\bin",
    ]

    # 정리: 존재하는 경로만, 순서 유지, 중복 제거
    uniq: List[str] = []
    seen = set()
    for p in cands:
        if p and os.path.isdir(p) and p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq

if platform.system() == "Windows" and os.environ.get("GE_OPS_NO_DLL_BOOTSTRAP", "0") != "1":
    _maybe_add_dll_dirs(_collect_candidate_paths())
    if os.environ.get("GE_OPS_DEBUG_DLLS", "0") == "1":
        # 빠지기 쉬운 것들만 간단 체크(출력만)
        for dll in (
            "cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll",
            "nvToolsExt64_1.dll", "VCRUNTIME140.dll", "vcruntime140_1.dll",
            "MSVCP140.dll", "vcomp140.dll", "python312.dll"
        ):
            try:
                ctypes.WinDLL(dll)
                print(f"[GE-OPS-DLL] OK: {dll}", file=sys.stderr)
            except OSError as e:
                print(f"[GE-OPS-DLL] MISS: {dll} ({e})", file=sys.stderr)

# -------------------------------------------------------------
# Public API: require
# -------------------------------------------------------------
def require(op: str):
    """동적으로 C++ 바인딩 모듈(_ops_<op>)을 import.
    예) convops = require("conv2d")  -> graph_executor_v2.ops._ops_conv2d
    """
    mod_name = f"graph_executor_v2.ops._ops_{op}"
    try:
        return importlib.import_module(mod_name)
    except ImportError as e:
        hints = []
        if platform.system() == "Windows":
            hints.append(
                "- Windows에선 .pyd 의존 DLL 경로 누락으로 ImportError가 자주 발생합니다.\n"
                "  조치:\n"
                "    * CUDA_PATH/bin, NVToolsExt/bin 등을 os.add_dll_directory()에 추가\n"
                "    * (권장) CMake POST_BUILD로 cudart/cublas/cublasLt/nvToolsExt DLL을 .pyd 옆에 복사\n"
                "    * GE_OPS_DEBUG_DLLS=1 로 실행하면 누락 DLL을 표기합니다"
            )
        hints.append(
            "- 빌드한 CUDA/툴체인과 실행 환경의 CUDA/드라이버가 일치해야 합니다(CUDA 12.x 권장)."
        )
        msg = (
            f"Failed to import native module '{mod_name}'.\n"
            f"Original error: {e}\n\n"
            "Troubleshooting hints:\n" + "\n".join(hints)
        )
        raise ImportError(msg) from e

# -------------------------------------------------------------
# PEP 562 스타일 lazy import for submodules
# -------------------------------------------------------------
_lazy_submodules = {
    "gemm",
    "conv2d",
    "pool2d",
    "softmax",
    "cross_entropy",
    "layernorm",
    "pad",
    "dropout",
    "rnn",
    "optimizer",
    "batchnorm",
    "concat",
    "embedding",
    "memory",
    "rmsnorm",
    "slice",
    "view",
    "epilogue",
}

def __getattr__(name: str):
    if name in _lazy_submodules:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["require"] + sorted(_lazy_submodules)
