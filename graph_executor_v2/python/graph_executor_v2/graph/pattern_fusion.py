from __future__ import annotations
from typing import List, Sequence
from .patterns import GraphPattern, PatternMatch

# 예시(비활성): Conv2D + BN (+ Act) → FusedConvBnAct
# 실제 구현을 넣기 전까지는 import 만 되고 사용은 안함.

class ConvBnActFusion(GraphPattern):
    priority = 10
    name = "conv-bn-(act)"

    def match(self, layers: Sequence):
        # TODO: 실제 매칭/치환 구현
        return []  # 아직은 매칭 없음

# 이후 DEFAULT_PATTERNS에 ConvBnActFusion() 등을 등록할 예정
