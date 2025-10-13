# python/graph_executor_v2/utils/streams.py
from __future__ import annotations
import cupy as cp

class Timer:
    def __init__(self, stream: cp.cuda.Stream | None = None):
        self.stream = stream or cp.cuda.Stream.null
        self._start = cp.cuda.Event()
        self._end = cp.cuda.Event()

    def __enter__(self):
        self.stream.synchronize()
        self._start.record(self.stream)
        return self

    def __exit__(self, *exc):
        self._end.record(self.stream)
        self._end.synchronize()

    def ms(self) -> float:
        return cp.cuda.get_elapsed_time(self._start, self._end)
