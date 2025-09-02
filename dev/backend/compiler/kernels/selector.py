# -*- coding: utf-8 -*-
"""
커널 선택기
- 파이썬 레지스트리 + (있다면) 네이티브 capability 점수 합산으로 최종 선택
"""

from __future__ import annotations
from typing import Dict, List, Tuple
import importlib

from .registry import query as reg_query

# 네이티브 query_capability 사용 여부 감지
try:
    native = importlib.import_module("graph_executor")
    _HAS_QUERY = hasattr(native, "query_capability")
except Exception:
    native = None
    _HAS_QUERY = False


def _score_by_rules(op, km, caps: Dict[str, bool]) -> int:
    s = 0
    if km.get("flags", {}).get("tensor_core") and caps.get("tensor_core"):
        s += 30
    M, N, K = op.attrs.get("mnk", (0, 0, 0))
    mm = km.get("flags", {}).get("min_mnk", (0, 0, 0))
    if M >= mm[0] and N >= mm[1] and K >= mm[2]:
        s += 20
    # TODO: dtype/layout 호환성 체크
    return s


def pick_kernel(op, device_caps: Dict[str, bool]) -> str:
    candidates = reg_query(op.op_type)
    if not candidates:
        raise RuntimeError(f"No kernel registered for op_type={op.op_type}")

    scored: List[Tuple[int, str]] = []

    native_scores: Dict[str, int] = {}
    if _HAS_QUERY:
        try:
            ns = native.query_capability(op.op_type, {}, {})
            native_scores = {str(k): int(v) for k, v in ns.items()}
        except Exception:
            native_scores = {}

    for km in candidates:
        kname = km["name"]
        sc = _score_by_rules(op, km, device_caps) + native_scores.get(kname, 0)
        scored.append((sc, kname))

    scored.sort(reverse=True)
    return scored[0][1]
