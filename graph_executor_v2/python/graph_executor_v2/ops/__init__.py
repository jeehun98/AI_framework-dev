"""
graph_executor_v2.ops
- require(op): C++ 바인딩 모듈(_ops_<op>) 동적 import
- 하위 파이썬 래퍼(gemm, conv2d, ...)는 필요 시 lazy import (PEP 562)
- Windows: CUDA/기타 DLL 검색 경로 부트스트랩
"""
from __future__ import annotations
import importlib
import os
import sys
import platform
from typing import Optional, Iterable

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
                # 조용히 무시 (권한/중복 등)
                pass

def _collect_candidate_cuda_paths() -> list[str]:
    cands: list[str] = []
    # 1) CUDA_PATH / CUDA_HOME
    for key in ("CUDA_PATH", "CUDA_HOME"):
        base = os.environ.get(key)
        if base:
            cands += [os.path.join(base, "bin"), os.path.join(base, "libnvvp")]

    # 2) Conda/venv 내 cuDNN/Library/bin
    for key in ("CONDA_PREFIX", "VIRTUAL_ENV"):
        base = os.environ.get(key)
        if base:
            cands += [
                os.path.join(base, "Library", "bin"),  # conda on Windows
                os.path.join(base, "bin"),
            ]

    # 3) 일반적인 설치 경로 힌트 (수정해서 쓰세요)
    cands += [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\libnvvp",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\libnvvp",
    ]
    # 중복 제거, 존재하는 경로만 유지
    uniq: list[str] = []
    for p in cands:
        if p and os.path.isdir(p) and p not in uniq:
            uniq.append(p)
    return uniq

# 환경변수 토글: 끄고 싶으면 GE_OPS_NO_DLL_BOOTSTRAP=1
if os.environ.get("GE_OPS_NO_DLL_BOOTSTRAP", "0") != "1":
    _maybe_add_dll_dirs(_collect_candidate_cuda_paths())

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
        # 친절한 진단 메시지
        hints = []
        if platform.system() == "Windows":
            hints.append(
                "- Windows에서는 .pyd가 의존하는 DLL 경로가 누락되면 ImportError가 납니다.\n"
                "  CUDA/ cuBLAS/ cuDNN DLL 경로가 검색되도록, 아래 방법 중 하나를 적용하세요:\n"
                "    * CUDA_PATH/bin, CUDA_PATH/libnvvp 등을 os.add_dll_directory()로 추가\n"
                "    * 본 패키지의 DLL bootstrap (_collect_candidate_cuda_paths) 경로를 실제 설치 경로로 수정\n"
                "    * CMake POST_BUILD로 필요한 DLL을 .pyd 옆으로 복사"
            )
        hints.append(
            "- PyTorch/CuPy와 시스템 CUDA 런타임 버전이 크게 어긋나면 로딩 실패 가능\n"
            "  (권장: 둘 다 CUDA 12.x 계열로 맞추고 최신 NVIDIA 드라이버 사용)"
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
    "view"
}

def __getattr__(name: str):
    if name in _lazy_submodules:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["require"] + sorted(_lazy_submodules)
