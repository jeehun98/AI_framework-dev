# -*- coding: utf-8 -*-
"""
커널 메타데이터 전역 레지스트리 (Python-side)
- selector 가 사용할 후보군을 선언적으로 보관
- 네이티브 테이블(launch_table/capability)과 최종 매칭되어 동작
"""

from __future__ import annotations
from typing import Dict, List

KernelMeta = Dict[str, object]  # name, op_type, dtypes, flags, layouts ...
_REGISTRY: List[KernelMeta] = []


def register(meta: KernelMeta) -> None:
    """커널 메타 등록"""
    _REGISTRY.append(meta)


def query(op_type: str) -> List[KernelMeta]:
    """op_type 로 후보 필터링"""
    return [m for m in _REGISTRY if m.get("op_type") == op_type]


# 예시 등록: FP16 Tensor Core / FP32 일반 버전
register({
    "name": "gemm_bias_act_tc_f16",
    "op_type": "GEMM_BIAS_ACT",
    "dtypes": {"in": ["f16", "f16", "f16"], "out": "f16"},
    "flags": {"tensor_core": True, "min_mnk": (128, 128, 64)},
    "layouts": {"in": ["rowmajor", "rowmajor", "rowmajor"], "out": "rowmajor"},
})

register({
    "name": "gemm_bias_act_f32",
    "op_type": "GEMM_BIAS_ACT",
    "dtypes": {"in": ["f32", "f32", "f32"], "out": "f32"},
    "flags": {"tensor_core": False},
    "layouts": {"in": ["rowmajor", "rowmajor", "rowmajor"], "out": "rowmajor"},
})
