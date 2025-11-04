# File: graph_executor_v2/graph/graph_executor.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Tuple, Hashable, Optional, Dict
import threading, time
from collections import deque              # NEW
import hashlib                             # NEW

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
        # telemetry
        self._hits: int = 0
        self._misses: int = 0
        self._puts: int = 0
        self._evicts: int = 0
        self._last_evicted = deque(maxlen=16)           # keys only
        self._last_evicted_summaries = deque(maxlen=16) # NEW: summarized dicts

    # -------- NEW: key summarizer --------
    def _summarize_key(self, k: Hashable) -> Dict[str, Any]:
        """
        최근 evict 항목을 가볍게 요약.
        - key_hex: 안정적 짧은 식별자 (sha1 10자리)
        - sig: shape/dtype/layout
        - branch: branch_id (있다면)
        - variant_top: variant 중 자주 보는 몇 개만 (amp/unroll/opt 등) 추려서 표기
        """
        def _short_hash(obj: Any) -> str:
            try:
                b = repr(obj).encode("utf-8")
            except Exception:
                b = str(obj).encode("utf-8")
            return hashlib.sha1(b).hexdigest()[:10]

        summary = {"key_hex": _short_hash(k)}

        # GraphKey라면 풍부하게
        if isinstance(k, GraphKey):
            sig = k.signature
            summary.update({
                "shape": tuple(getattr(sig, "shape", ())),
                "dtype": getattr(sig, "dtype", ""),
                "layout": getattr(sig, "layout", ""),
                "branch": getattr(k, "branch_id", ""),
            })
            # variant에서 대표 키워드만 추려보기
            try:
                vd = dict(k.variant)
                top = {}
                for name in ("amp", "unroll", "optimizer", "opt", "loss_kind", "training"):
                    if name in vd:
                        top[name] = vd[name]
                # 없으면 앞에서 3~5개만
                if not top:
                    for i, (kk, vv) in enumerate(k.variant):
                        if i >= 5: break
                        top[kk] = vv
                summary["variant_top"] = top
            except Exception:
                pass
        else:
            # tuple 등의 fallback key
            summary["type"] = type(k).__name__

        return summary
    # -------------------------------------

    def get(self, k: Hashable) -> Optional[Any]:
        with self._lock:
            v = self._store.get(k)
            if v is not None:
                self._ts[k] = time.monotonic()
                self._hits += 1
            else:
                self._misses += 1
            return v

    def put(self, k: Hashable, v: Any) -> None:
        with self._lock:
            self._store[k] = v
            self._ts[k] = time.monotonic()
            self._puts += 1
            if len(self._store) > self._max:
                victim = min(self._ts.items(), key=lambda kv: kv[1])[0]
                self._store.pop(victim, None)
                self._ts.pop(victim, None)
                self._evicts += 1
                self._last_evicted.append(victim)
                # NEW: save summary
                try:
                    self._last_evicted_summaries.append(self._summarize_key(victim))
                except Exception:
                    self._last_evicted_summaries.append({"key_hex": "error"})

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "size": len(self._store),
                "max_size": self._max,
                "hits": self._hits,
                "misses": self._misses,
                "puts": self._puts,
                "evicts": self._evicts,
                "last_evicted_count": len(self._last_evicted),
                # NEW: include summaries (list of dicts)
                "last_evicted_summaries": list(self._last_evicted_summaries),
            }

    def reset_stats(self) -> None:
        with self._lock:
            self._hits = self._misses = self._puts = self._evicts = 0
            self._last_evicted.clear()
            self._last_evicted_summaries.clear()

    def dump_keys(self, limit: int = 32) -> Dict[str, Any]:
        with self._lock:
            out = []
            for i, k in enumerate(self._store.keys()):
                if i >= limit: break
                out.append(self._summarize_key(k))
            return {"size": len(self._store), "sample": out}


graph_pool = MultiGraphPool(max_size=16)
