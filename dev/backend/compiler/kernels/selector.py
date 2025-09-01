from __future__ import annotations
from typing import Dict
from .registry import query
import importlib

# graph_executor(.pyd)가 query_capability 제공하면 활용하고, 없으면 휴리스틱
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
    # dtype/레이아웃 체크는 스킵(스켈레톤)
    return s

def pick_kernel(op, device_caps: Dict[str, bool]) -> str:
    candidates = query(op.op_type)
    if not candidates:
        raise RuntimeError(f"No kernel registered for op_type={op.op_type}")
    # 네이티브가 점수 제공하면 합산
    scores = []
    for km in candidates:
        sc = _score_by_rules(op, km, device_caps)
        if _HAS_QUERY:
            try:
                # 선택적으로 네이티브 점수(없으면 무시)
                sc += int(native.query_capability(op.op_type, {}, {} ).get(km["name"], 0))  # 타입 단순화
            except Exception:
                pass
        scores.append((sc, km["name"]))
    scores.sort(reverse=True)
    return scores[0][1]
