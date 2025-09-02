# -*- coding: utf-8 -*-
"""
커널 선택기
- 레지스트리의 메타데이터(파이썬) + 네이티브 capability 점수(있다면)를 합산하여 최종 커널 이름을 선택
- 룰 기반 점수 예시: tensor_core 가용 여부, 최소 MNK 충족 등

주의:
- native 모듈은 executor 에서 별칭으로 'graph_executor'로 등록될 수 있으므로,
  여기서는 독립적으로 import 시도하되, 실패해도 휴리스틱만으로 동작 가능해야 함.
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
    """
    간단 룰:
      - tensor_core 플래그 + 디바이스 tensor_core True면 가산점
      - 최소 MNK(있는 경우) 만족하면 가산점
    dtype/layout 검증은 스켈레톤 단계에서는 생략(추후 확장)
    """
    s = 0
    if km.get("flags", {}).get("tensor_core") and caps.get("tensor_core"):
        s += 30
    M, N, K = op.attrs.get("mnk", (0, 0, 0))
    mm = km.get("flags", {}).get("min_mnk", (0, 0, 0))
    if M >= mm[0] and N >= mm[1] and K >= mm[2]:
        s += 20
    return s


def pick_kernel(op, device_caps: Dict[str, bool]) -> str:
    """
    @return 선택된 커널 이름 (launch_table 의 key 와 동일)
    @raises RuntimeError 후보가 없으면 예외
    """
    candidates = reg_query(op.op_type)
    if not candidates:
        raise RuntimeError(f"No kernel registered for op_type={op.op_type}")

    scored: List[Tuple[int, str]] = []

    # 네이티브 점수(있다면) + 휴리스틱 합산
    native_scores: Dict[str, int] = {}
    if _HAS_QUERY:
        try:
            # {"kernel_name": score, ...}
            ns = native.query_capability(op.op_type, {}, {})
            # pybind dict -> py dict 로 가정
            native_scores = {str(k): int(v) for k, v in ns.items()}
        except Exception:
            native_scores = {}

    for km in candidates:
        kname = km["name"]
        sc = _score_by_rules(op, km, device_caps) + native_scores.get(kname, 0)
        scored.append((sc, kname))

    scored.sort(reverse=True)
    return scored[0][1]
