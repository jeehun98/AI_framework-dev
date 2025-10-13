from __future__ import annotations
from typing import Callable, Dict, Optional, Tuple
import cupy as cp

# ----------------------------
# Public helpers (pinned host)
# ----------------------------
def alloc_pinned_like(a: cp.ndarray) -> cp.ndarray:
    """
    a와 동일한 shape/dtype의 pinned host 배열을 만든다(고정 주소).
    CuPy ndarray를 만들기 위해 UnownedMemory -> MemoryPointer 경유.
    """
    if not isinstance(a, cp.ndarray):
        raise TypeError("alloc_pinned_like: expected cupy.ndarray")
    pmem = cp.cuda.PinnedMemory(a.nbytes)                        # host-pinned block
    umem = cp.cuda.UnownedMemory(pmem.ptr, pmem.size, pmem)      # take ownership wrapper
    mptr = cp.cuda.MemoryPointer(umem, 0)
    return cp.ndarray(a.shape, dtype=a.dtype, memptr=mptr)       # host ndarray backed by pinned mem

def alloc_pinned(shape, dtype) -> cp.ndarray:
    """shape/dtype로 pinned host 배열 생성."""
    dummy = cp.ndarray(shape, dtype=dtype)  # device 배킹 아님, 단지 spec 용도
    return alloc_pinned_like(dummy)

# ----------------------------
# CaptureTrainer
# ----------------------------
class CaptureTrainer:
    """
    단일 그래프(1 step) 캡처 후 반복 재생하는 트레이너 유틸.
    - 입력(X_d, y_d), 파라미터(P...), 모멘텀(M/V), 그래디언트 버퍼 등 '고정 주소' 버퍼 사용
    - 캡처 본문:  H2D memcpy -> forward -> loss -> backward -> optimizer
    - 리플레이:   pinned host에 배치 데이터만 바꾸고 g.launch()
    """
    def __init__(
        self,
        *,
        # 고정 디바이스 버퍼들
        x_dev: cp.ndarray,         # [B,...] float32 등
        y_dev: cp.ndarray,         # [B]     (int32/float32 등)
        # host pinned 버퍼들 (고정 주소)
        x_host: cp.ndarray,        # same shape/dtype as x_dev (pinned)
        y_host: cp.ndarray,        # same shape/dtype as y_dev (pinned)
        # 모델 파라미터/그래드/옵티마 상태 버퍼 dict (고정 주소)
        params: Dict[str, cp.ndarray],
        grads:  Dict[str, cp.ndarray],
        optim_state: Dict[str, cp.ndarray],
        # 1-step 그래프를 구성하는 콜백들 (현재 stream 컨텍스트 내에서만 호출)
        build_step: Callable[[], Tuple[cp.ndarray, Optional[cp.ndarray]]],
        #  - build_step() -> (logits, loss_vec or None)
        backward_step: Callable[[cp.ndarray], None],
        #  - backward_step(dY_or_logits)
        optimizer_step: Callable[[], None],
        #  - optimizer_step(): params, grads, optim_state를 in-place 업데이트
        non_blocking_stream: bool = True,
    ):
        self.s = cp.cuda.Stream(non_blocking=non_blocking_stream)
        self.x_d, self.y_d = x_dev, y_dev
        self.x_h, self.y_h = x_host, y_host
        self.params, self.grads = params, grads
        self.optim_state = optim_state
        self.build_step = build_step
        self.backward_step = backward_step
        self.optimizer_step = optimizer_step
        self.g = None  # instantiated graph (exec graph)

        # 검증: 주소/shape 고정성
        assert self.x_d.flags.c_contiguous and self.x_h.flags.c_contiguous
        assert self.y_d.flags.c_contiguous and self.y_h.flags.c_contiguous
        assert self.x_d.dtype == self.x_h.dtype
        assert self.y_d.dtype == self.y_h.dtype
        assert self.x_d.shape == self.x_h.shape
        assert self.y_d.shape == self.y_h.shape

    def _h2d_copy(self):
        """고정 주소 host→device memcpy를 현재 스트림에서 실행."""
        cp.cuda.runtime.memcpyAsync(
            int(self.x_d.data.ptr), int(self.x_h.data.ptr), self.x_d.nbytes,
            cp.cuda.runtime.memcpyHostToDevice, self.s.ptr
        )
        cp.cuda.runtime.memcpyAsync(
            int(self.y_d.data.ptr), int(self.y_h.data.ptr), self.y_d.nbytes,
            cp.cuda.runtime.memcpyHostToDevice, self.s.ptr
        )

    def capture(self):
        """그래프 캡처: H2D→build→bwd→opt 의 고정 시퀀스를 한 번 캡처한다."""
        def body():
            self._h2d_copy()
            logits, loss_vec = self.build_step()
            # 여기서는 backward_step이 내부적으로 필요한 dY(or dlogits) 계산을 포함한다고 가정.
            # (필요하면 build_step에서 dlogits까지 만들어 tuple로 넘기고, 여기선 그걸 전달해도 됨)
            self.backward_step(logits)
            self.optimizer_step()

        # 워밍업(그래프 외부 1회)
        with self.s:
            body()
        self.s.synchronize()

        # 캡처
        with self.s:
            self.s.begin_capture()
            body()
            g = self.s.end_capture()

        # instantiate/launch 호환
        self.g = getattr(g, "instantiate", lambda: g)()
        with self.s:
            try:
                # 일부 CuPy/NVIDIA 드라이버 조합에서 제공
                if hasattr(self.g, "upload"):
                    self.g.upload()
            except Exception:
                pass
        self.s.synchronize()

    def load_batch(self, x_batch: cp.ndarray, y_batch: cp.ndarray):
        """고정 주소 host pinned 버퍼에 새 배치 내용을 복사(주소는 그대로, 내용만 교체)."""
        assert x_batch.shape == self.x_h.shape and x_batch.dtype == self.x_h.dtype
        assert y_batch.shape == self.y_h.shape and y_batch.dtype == self.y_h.dtype
        # host-pinned에 쓰기
        self.x_h[...] = x_batch
        self.y_h[...] = y_batch

    def step(self):
        """그래프 재생(1 step)."""
        assert self.g is not None, "call capture() first"
        with self.s:
            # CuPy/CUDA Graph API 버전 호환 런처
            if hasattr(self.g, "launch"):
                try:
                    self.g.launch(self.s)
                except TypeError:
                    self.g.launch(self.s.ptr)
            elif hasattr(cp.cuda.graph, "launch"):
                try:
                    cp.cuda.graph.launch(self.g, self.s)
                except TypeError:
                    cp.cuda.graph.launch(self.g, self.s.ptr)
            else:
                raise RuntimeError("CUDA Graph launch API not found")
        self.s.synchronize()
