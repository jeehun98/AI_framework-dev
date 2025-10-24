# python/graph_executor_v2/utils/streams.py
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, Literal
import cupy as cp
from collections import deque

Mode = Literal["train", "eval"]

# ----------------------------
# NVTX (optional, no-op fallback)
# ----------------------------
def _nvtx_push(msg: str) -> None:
    try:
        from cupy.cuda import nvtx as _nvtx  # type: ignore
        # Try common CuPy NVTX bindings
        for name in ("range_push", "RangePushA", "RangePush"):
            fn = getattr(_nvtx, name, None)
            if fn:
                fn(msg)
                return
    except Exception:
        pass  # no-op if unavailable

def _nvtx_pop() -> None:
    try:
        from cupy.cuda import nvtx as _nvtx  # type: ignore
        for name in ("range_pop", "RangePop"):
            fn = getattr(_nvtx, name, None)
            if fn:
                fn()
                return
    except Exception:
        pass  # no-op if unavailable

class NVTX:
    """Context manager for NVTX ranges (silently no-ops if NVTX unavailable)."""
    __slots__ = ("msg",)
    def __init__(self, msg: str) -> None:
        self.msg = msg
    def __enter__(self):
        _nvtx_push(self.msg)
        return self
    def __exit__(self, *exc):
        _nvtx_pop()


# ----------------------------
# Event Pool (reduce alloc churn)
# ----------------------------
class _EventPool:
    """Simple pool for cp.cuda.Event to reduce allocator overhead."""
    __slots__ = ("_pool",)
    def __init__(self) -> None:
        self._pool: deque[cp.cuda.Event] = deque()

    def get(self) -> cp.cuda.Event:
        try:
            return self._pool.popleft()
        except IndexError:
            # Default flags work well for timing & dependency
            return cp.cuda.Event()

    def put(self, evt: cp.cuda.Event) -> None:
        # Events are cheap, keep a modest pool
        if len(self._pool) < 64:
            self._pool.append(evt)


# ----------------------------
# Timer (capture-aware)
# ----------------------------
class Timer:
    """
    CUDA event-based timer.

    - Works on a given stream (defaults to Stream.null).
    - capture_mode=True avoids host syncs that are illegal during CUDA Graph capture.
      (It still records events; measuring ms() requires synchronization *outside* capture.)
    """
    __slots__ = ("stream", "_start", "_end", "capture_mode", "_pool")

    def __init__(self, stream: Optional[cp.cuda.Stream] = None, *, capture_mode: bool = False, _pool: Optional[_EventPool] = None):
        self.stream = stream or cp.cuda.Stream.null
        self.capture_mode = capture_mode
        self._pool = _pool or _GLOBAL_EVENT_POOL
        self._start: Optional[cp.cuda.Event] = None
        self._end: Optional[cp.cuda.Event] = None

    def __enter__(self):
        self._start = self._pool.get()
        self._end = self._pool.get()
        if not self.capture_mode:
            # Ensure kernels before this region are complete to get clean timing
            self.stream.synchronize()
        self._start.record(self.stream)
        return self

    def __exit__(self, *exc):
        assert self._end is not None
        self._end.record(self.stream)
        if not self.capture_mode:
            self._end.synchronize()
        # Keep events for reuse
        if self._start:
            self._pool.put(self._start)
        if self._end:
            self._pool.put(self._end)
        self._start = None
        self._end = None

    def ms(self) -> float:
        """Return elapsed ms between start and end. Requires end to be completed."""
        if self._start is None or self._end is None:
            raise RuntimeError("Timer not active or already released.")
        return cp.cuda.get_elapsed_time(self._start, self._end)


# ----------------------------
# Stream Manager
# ----------------------------
@dataclass
class _ActiveState:
    mode: Optional[Mode]
    stream: cp.cuda.Stream

class StreamManager:
    """
    Stream manager that separates train/eval streams and controls synchronization.

    Features:
    - Two non-blocking streams: train / eval
    - Active stream switching & scoped usage
    - Record/wait event for cross-stream dependencies
    - synchronize_all()
    - Capture-aware helpers (avoid host sync when capture_mode=True)
    - Optional NVTX annotations
    """

    __slots__ = ("train", "eval", "_active", "capture_mode", "_evt_pool")

    def __init__(self, *, capture_mode: bool = False) -> None:
        self.train = cp.cuda.Stream(non_blocking=True)
        self.eval = cp.cuda.Stream(non_blocking=True)
        self._active: _ActiveState = _ActiveState(mode=None, stream=cp.cuda.Stream.null)
        self.capture_mode = capture_mode
        self._evt_pool = _GLOBAL_EVENT_POOL

    # ---- Active stream control ----
    def use(self, mode: Mode) -> cp.cuda.Stream:
        if mode == "train":
            self._active = _ActiveState(mode="train", stream=self.train)
        elif mode == "eval":
            self._active = _ActiveState(mode="eval", stream=self.eval)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        return self._active.stream

    @property
    def active_stream(self) -> cp.cuda.Stream:
        return self._active.stream

    @property
    def active_mode(self) -> Optional[Mode]:
        return self._active.mode

    # ---- Synchronization helpers ----
    def synchronize_all(self) -> None:
        """Synchronize both train/eval streams (skips if capture_mode=True)."""
        if self.capture_mode:
            return  # host sync is illegal during capture
        self.train.synchronize()
        self.eval.synchronize()

    def record_event(self, stream: Optional[cp.cuda.Stream] = None) -> cp.cuda.Event:
        """Record an event on the given or active stream."""
        s = stream or self.active_stream
        evt = self._evt_pool.get()
        evt.record(s)
        return evt

    def wait_event(self, evt: cp.cuda.Event, stream: Optional[cp.cuda.Stream] = None) -> None:
        """Make the given or active stream wait until evt is completed."""
        s = stream or self.active_stream
        s.wait_event(evt)

    def barrier(self) -> None:
        """Cross-stream barrier: eval waits for train, then train waits for eval."""
        t_evt = self.record_event(self.train)
        self.eval.wait_event(t_evt)
        e_evt = self.record_event(self.eval)
        self.train.wait_event(e_evt)
        # return events to pool (safe after dependency is established)
        self._evt_pool.put(t_evt)
        self._evt_pool.put(e_evt)

    # ---- Context helpers ----
    @contextmanager
    def scope(self, mode: Mode, *, nvtx: Optional[str] = None):
        """
        Scoped switch to a mode's stream.
        Example:
            with streams.scope("train", nvtx="train-step"):
                launch_kernels()
        """
        prev = self._active
        s = self.use(mode)
        with (NVTX(nvtx) if nvtx else _nullcontext()):
            try:
                yield s
            finally:
                # restore previous active stream
                self._active = prev

    def timer(self, *, nvtx: Optional[str] = None) -> contextmanager:
        """
        Timer bound to the currently active stream; capture-aware.
        Example:
            with streams.scope("eval"):
                with streams.timer(nvtx="eval-pass") as t:
                    launch_eval()
            print("elapsed:", t.ms(), "ms")
        """
        @contextmanager
        def _timectx():
            with NVTX(nvtx) if nvtx else _nullcontext():
                with Timer(self.active_stream, capture_mode=self.capture_mode, _pool=self._evt_pool) as t:
                    yield t
        return _timectx()


# ----------------------------
# Utilities / globals
# ----------------------------
@contextmanager
def _nullcontext():
    yield

_GLOBAL_EVENT_POOL = _EventPool()

# A convenient global instance (optional)
STREAMS = StreamManager(capture_mode=False)

# Convenience top-level helpers (optional)
def train_stream() -> cp.cuda.Stream:
    return STREAMS.use("train")

def eval_stream() -> cp.cuda.Stream:
    return STREAMS.use("eval")
