# 커널 메타데이터를 간단히 등록/조회하는 전역 레지스트리

from __future__ import annotations
from typing import Dict, List, Tuple

KernelMeta = Dict[str, object]  # name, op_type, dtypes, flags, layouts ...

_REGISTRY: List[KernelMeta] = []

def register(meta: KernelMeta) -> None:
    _REGISTRY.append(meta)

def query(op_type: str) -> List[KernelMeta]:
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
