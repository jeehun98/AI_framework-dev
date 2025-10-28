# File: graph_executor_v2/graph/graph_executor.py (축소판)
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Tuple, Iterable, Hashable, Optional, Dict
import threading, time

__all__ = ["GraphSignature", "GraphKey", "MultiGraphPool", "graph_pool"]

@dataclass(frozen=True)
class GraphSignature:
    shape: Tuple[int, ...]
    dtype: str
    layout: str
    def __hash__(self): return hash((self.shape, self.dtype, self.layout))

@dataclass(frozen=True)
class GraphKey:
    signature: GraphSignature
    branch_id: str
    variant: Tuple[Tuple[str, Any], ...]
    def __hash__(self): return hash((self.signature, self.branch_id, self.variant))

class MultiGraphPool:
    def __init__(self, max_size: int = 16):
        self._max = max_size
        self._store: Dict[Hashable, Any] = {}
        self._ts: Dict[Hashable, float] = {}
        self._lock = threading.RLock()
    def get(self, k: Hashable) -> Optional[Any]:
        with self._lock:
            v = self._store.get(k)
            if v is not None: self._ts[k] = time.monotonic()
            return v
    def put(self, k: Hashable, v: Any) -> None:
        with self._lock:
            self._store[k] = v; self._ts[k] = time.monotonic()
            if len(self._store) > self._max:
                victim = min(self._ts.items(), key=lambda kv: kv[1])[0]
                self._store.pop(victim, None); self._ts.pop(victim, None)

graph_pool = MultiGraphPool(max_size=16)
