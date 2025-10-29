from __future__ import annotations
from typing import Sequence, Any, List

from .patterns import PatternPass, GraphPattern

# 현재는 빈 목록 → run_patterns는 항상 입력을 그대로 반환
DEFAULT_PATTERNS: List[GraphPattern] = []

# NVTX (선택)
try:
    from graph_executor_v2.backends.cuda.ops.gemm.detail.nvtx_shim import nvtx_range  # type: ignore
except Exception:
    class _DummyNvtx:
        def __call__(self, *_a, **_k):
            class _Ctx:
                def __enter__(self): return None
                def __exit__(self, *args): return False
            return _Ctx()
    nvtx_range = _DummyNvtx()  # type: ignore

def run_patterns(layers: Sequence[Any]) -> List[Any]:
    """패턴 패스 엔트리 포인트. 현재는 no-op."""
    if not DEFAULT_PATTERNS:
        # no-op 경로 (추후 패턴 추가 시 아래 패스 사용)
        return list(layers)
    with nvtx_range("[PATTERN] run"):
        return PatternPass(DEFAULT_PATTERNS).run(layers)
