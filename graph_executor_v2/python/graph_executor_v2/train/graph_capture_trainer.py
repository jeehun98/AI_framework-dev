# python/graph_executor_v2/train/capture_trainer.py
from __future__ import annotations
from typing import Callable, Optional, Tuple, Dict, Any
import numpy as np
import cupy as cp

# (필요시) 각종 ops 모듈 임포트 예시:
# from graph_executor_v2.ops import conv2d as conv_ops
# from graph_executor_v2.ops import gemm as gemm_ops
# from graph_executor_v2.ops import rmsnorm as rn_ops
# from graph_executor_v2.ops import softmax as sm_ops
# from graph_executor_v2.ops import cross_entropy as ce_ops
# from graph_executor_v2.ops import optimizer as opt_ops

def alloc_pinned_ndarray_like(a: cp.ndarray) -> tuple[np.ndarray, Any]:
    """
    CuPy 디바이스 배열 a와 동일한 shape/dtype의 **host pinned NumPy** 배열을 만든다.
    반환: (host_np, pinned_mem)
     - host_np: NumPy ndarray (버퍼가 pinned 메모리를 가리킴)
     - pinned_mem: keep-alive 용도로 반드시 참조 보관
    """
    nbytes = a.nbytes
    mem = cp.cuda.alloc_pinned_memory(nbytes)  # host pinned
    host = np.frombuffer(mem, dtype=a.dtype, count=a.size).reshape(a.shape)
    return host, mem


class CaptureTrainer:
    """
    단일 CUDA Graph(1 step)를 캡처하여 반복 재생하는 유틸.
    - 고정 주소 버퍼(디바이스/호스트 pinned)를 사용해야 그래프 재사용이 안전함.
    - 캡처 본문: H2D memcpy -> build_step(forward/optional loss) -> backward_step -> optimizer_step
    - 리플레이: host pinned에 배치만 교체하고 graph.launch()

    사용 예:
        x_d = cp.empty((B,C,H,W), cp.float32)
        y_d = cp.empty((B,), cp.int32)
        x_h, x_h_mem = alloc_pinned_ndarray_like(x_d)
        y_h, y_h_mem = alloc_pinned_ndarray_like(y_d)

        tr = CaptureTrainer(
            x_dev=x_d, y_dev=y_d,
            x_host=x_h, y_host=y_h,
            params=params, grads=grads, optim_state=optim_state,
            build_step=build_step, backward_step=backward_step, optimizer_step=optimizer_step,
            _x_host_mem=x_h_mem, _y_host_mem=y_h_mem,
        )
        tr.capture()

        for step in range(T):
            tr.load_batch(xb_np, yb_np)  # NumPy로 작성
            tr.step()
    """

    def __init__(
        self,
        *,
        # 고정 디바이스 버퍼
        x_dev: cp.ndarray,         # [B,...], float32 등
        y_dev: cp.ndarray,         # [B],    int32 등 (CE라면)
        # 고정 host pinned 버퍼 (NumPy)
        x_host: np.ndarray,
        y_host: np.ndarray,
        # 모델 파라미터/그래드/옵티마 상태 (디바이스, 고정 주소)
        params: Dict[str, cp.ndarray],
        grads: Dict[str, cp.ndarray],
        optim_state: Dict[str, cp.ndarray],
        # 그래프 구성 콜백들(현재 스트림 컨텍스트 내에서 호출됨)
        build_step: Callable[[], Tuple[cp.ndarray, Optional[cp.ndarray]]],
        #  - build_step() -> (logits, loss_vec or None)
        backward_step: Callable[[cp.ndarray], None],
        #  - backward_step(logits or dlogits): 그래디언트 채우기
        optimizer_step: Callable[[], None],
        # 옵션들
        non_blocking_stream: bool = True,
        # pinned 메모리 keep-alive
        _x_host_mem: Optional[Any] = None,
        _y_host_mem: Optional[Any] = None,
    ):
        # 스트림
        self.s = cp.cuda.Stream(non_blocking=non_blocking_stream)

        # 버퍼 레퍼런스
        self.x_d, self.y_d = x_dev, y_dev
        self.x_h, self.y_h = x_host, y_host
        self._x_host_mem = _x_host_mem
        self._y_host_mem = _y_host_mem

        self.params, self.grads = params, grads
        self.optim_state = optim_state

        # 콜백
        self.build_step = build_step
        self.backward_step = backward_step
        self.optimizer_step = optimizer_step

        # 그래프 핸들
        self.g = None

        # 기본 검증
        assert isinstance(self.x_d, cp.ndarray) and isinstance(self.y_d, cp.ndarray)
        assert isinstance(self.x_h, np.ndarray) and isinstance(self.y_h, np.ndarray)
        assert self.x_d.flags.c_contiguous and self.y_d.flags.c_contiguous
        assert self.x_d.shape == self.x_h.shape and self.x_d.dtype == self.x_h.dtype
        assert self.y_d.shape == self.y_h.shape and self.y_d.dtype == self.y_h.dtype

        # 파라미터/그래드/옵티마 상태 고정성(간단 체크)
        for d in (self.params, self.grads, self.optim_state):
            assert isinstance(d, dict)
            for k, v in d.items():
                assert isinstance(v, cp.ndarray) and v.flags.c_contiguous, f"{k} must be CuPy contiguous"

    # ---------------- internal helpers ----------------
    def _h2d_copy(self):
        """host pinned(NumPy) → device(CuPy) 비동기 memcpy (현재 스트림)."""
        # dst (device ptrs)
        dst_x = int(self.x_d.data.ptr)
        dst_y = int(self.y_d.data.ptr)
        # src (host pinned ptrs): NumPy __array_interface__ 이용
        src_x = int(self.x_h.__array_interface__['data'][0])
        src_y = int(self.y_h.__array_interface__['data'][0])

        cp.cuda.runtime.memcpyAsync(
            dst_x, src_x, self.x_d.nbytes,
            cp.cuda.runtime.memcpyHostToDevice, self.s.ptr
        )
        cp.cuda.runtime.memcpyAsync(
            dst_y, src_y, self.y_d.nbytes,
            cp.cuda.runtime.memcpyHostToDevice, self.s.ptr
        )

    # ---------------- public APIs ----------------
    def capture(self):
        """
        그래프 캡처:
          [H2D] -> [build_step (fwd/loss)] -> [backward_step] -> [optimizer_step]
        """
        def body():
            self._h2d_copy()
            logits, loss_vec = self.build_step()
            # (필요시) loss_vec을 사용해 dlogits를 만든 뒤 backward_step(dlogits)로 바꿔도 됨
            # 여기서는 logits만 전달한다고 가정
            self.backward_step(logits)
            self.optimizer_step()

        # 워밍업: 그래프 밖에서 1회 실행
        with self.s:
            body()
        cp.cuda.get_current_stream().synchronize()

        # 캡처
        with self.s:
            self.s.begin_capture()
            body()
            self.g = self.s.end_capture()

        # (옵션) 업로드 지원 시 업로드
        with self.s:
            try:
                if hasattr(self.g, "upload"):
                    self.g.upload()
            except Exception:
                pass
        self.s.synchronize()

    def load_batch(self, x_batch: np.ndarray, y_batch: np.ndarray):
        """host pinned 버퍼에 새 배치 복사(주소 유지)."""
        assert isinstance(x_batch, np.ndarray) and isinstance(y_batch, np.ndarray)
        assert x_batch.shape == self.x_h.shape and x_batch.dtype == self.x_h.dtype
        assert y_batch.shape == self.y_h.shape and y_batch.dtype == self.y_h.dtype
        self.x_h[...] = x_batch
        self.y_h[...] = y_batch

    def step(self):
        """캡처된 그래프 1회 실행."""
        assert self.g is not None, "capture()를 먼저 호출하세요."
        with self.s:
            # CuPy CUDA Graph 객체 호환 런처
            if hasattr(self.g, "launch"):
                try:
                    self.g.launch(self.s)
                except TypeError:
                    self.g.launch(self.s.ptr)
            else:
                # 구버전 fallback
                try:
                    cp.cuda.graph.launch(self.g, self.s)
                except TypeError:
                    cp.cuda.graph.launch(self.g, self.s.ptr)
        self.s.synchronize()
