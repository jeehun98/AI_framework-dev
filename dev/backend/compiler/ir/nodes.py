from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

# 아주 간단한 IR 노드들

@dataclass
class Tensor:
    name: str
    shape: Sequence[int]
    dtype: str = "f32"          # "f16" | "f32" | "i8" ...
    layout: str = "rowmajor"    # "NCHW" | "NHWC" | ...
    device: str = "cuda"        # "cuda" | "cpu"

@dataclass
class Op:
    op_type: str
    inputs: List[Tensor]
    outputs: List[Tensor]
    attrs: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Graph:
    ops: List[Op]

    def clone(self) -> "Graph":
        # 얕은 복사로 충분(스켈레톤)
        return Graph(ops=list(self.ops))
