from __future__ import annotations
from typing import List, Sequence, Optional, Any

class PatternMatch:
    """하나의 매칭 결과: [start, end) 구간을 replacement로 치환."""
    __slots__ = ("start", "end", "replacement")
    def __init__(self, start: int, end: int, replacement: Any):
        self.start = int(start)
        self.end = int(end)
        self.replacement = replacement

class GraphPattern:
    """패턴 베이스 클래스. 우선순위가 낮을수록 먼저 적용."""
    priority: int = 100
    name: str = "base"

    def match(self, layers: Sequence[Any]) -> List[PatternMatch]:
        """layers[i:] 구간에서 첫 매칭을 반환하도록 구현 (없으면 [])."""
        return []

class PatternPass:
    """우선순위 순으로 좌→우 매칭/치환을 적용하는 간단한 패스."""
    def __init__(self, patterns: Sequence[GraphPattern]):
        self.patterns = sorted(patterns, key=lambda p: p.priority)

    def run(self, layers: Sequence[Any]) -> List[Any]:
        # 현재는 보수적으로: 겹침 방지, 한 번에 1개 치환, 좌→우 진행
        out = list(layers)
        i = 0
        while i < len(out):
            picked: Optional[PatternMatch] = None
            for p in self.patterns:
                matches = p.match(out[i:])
                if matches:
                    m = matches[0]
                    m.start += i
                    m.end += i
                    picked = m
                    break
            if picked is None:
                i += 1
                continue
            # 치환
            out[picked.start:picked.end] = [picked.replacement]
            i = picked.start + 1
        return out
