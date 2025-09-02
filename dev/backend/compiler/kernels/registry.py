# -*- coding: utf-8 -*-
"""
커널 메타데이터 전역 레지스트리 (Python-side)
- 목적: 파이썬 선택기가 사용할 후보 군(메타정보)을 선언적으로 보관
- 실제 런치 함수/성능 점수는 네이티브 테이블과 합쳐서 최종 결정을 내림

메타데이터 필드 예시:
- name:      네이티브 launch_table 의 key 와 정확히 일치해야 함
- op_type:   IR 상의 오퍼레이터 타입 (예: "GEMM_BIAS_ACT")
- dtypes:    {"in": [...], "out": "..."}
- flags:     임의의 힌트 (예: {"tensor_core": True, "min_mnk": (M,N,K)})
- layouts:   {"in": [...], "out": "..."}  (rowmajor/colmajor 등)

주의:
- 이 파일은 "선언" 성격이며, 실제 지원/검증은 런타임/네이티브에서 다시 확인해야 안전함.
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


# === 예시 등록: FP16 Tensor Core / FP32 일반 버전 =======================

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
