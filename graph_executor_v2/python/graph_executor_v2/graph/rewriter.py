# python/graph_executor_v2/graph/rewriter.py
from __future__ import annotations
from typing import Sequence, Any, List
from .pattern_registry import run_patterns as _run_patterns

# NVTX 태그 등 공통 디버깅 훅을 여기에 둡니다.
try:
    from graph_executor_v2.backends.cuda.ops.gemm.detail.nvtx_shim import nvtx_range
except Exception:
    class _Nvtx:  # no-op
        def __call__(self, *_a, **_k):
            class _Ctx:
                def __enter__(self): return None
                def __exit__(self, *args): return False
            return _Ctx()
    nvtx_range = _Nvtx()

def run(layers: Sequence[Any]) -> List[Any]:
    """Graph rewrite facade: 현재는 registry의 run_patterns만 위임.
    훗날 canonicalize/constant-fold 등 여러 패스를 여기서 오케스트레이션."""
    with nvtx_range("[REWRITER] run"):
        # 1) (optional) canonicalize(layers)
        # 2) local pattern fusion
        layers = _run_patterns(layers)
        # 3) (optional) constant_fold(layers)
        # 4) (optional) layout_rewrite(layers)
        return layers
