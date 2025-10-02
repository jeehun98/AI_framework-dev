# python/graph_executor_v2/ops/__init__.py
"""
graph_executor_v2.ops
- require(op): 특정 C++ 바인딩 모듈(_ops_*) 동적 import
- 하위 파이썬 래퍼(gemm, conv2d, pool2d, softmax, cross_entropy)는 필요할 때만 lazy import
"""

import importlib

def require(op: str):
    """동적으로 C++ 바인딩 모듈(_ops_<op>)을 import."""
    return importlib.import_module(f"graph_executor_v2.ops._ops_{op}")

# PEP 562 스타일 lazy import for submodules
_lazy_submodules = {
    "gemm",
    "conv2d",
    "pool2d",
    "softmax",
    "cross_entropy",
}

def __getattr__(name: str):
    if name in _lazy_submodules:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["require"] + sorted(_lazy_submodules)
