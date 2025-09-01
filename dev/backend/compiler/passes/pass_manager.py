from __future__ import annotations
from typing import Callable, List
from ..ir.nodes import Graph

Pass = Callable[[Graph], Graph]

class PassManager:
    def __init__(self, passes: List[Pass]) -> None:
        self.passes = passes

    def run(self, g: Graph) -> Graph:
        for p in self.passes:
            g = p(g)
        return g
