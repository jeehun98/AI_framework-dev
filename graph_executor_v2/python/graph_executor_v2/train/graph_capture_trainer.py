from __future__ import annotations
from typing import Callable, Optional, Tuple, Dict, Any
import cupy as cp

from graph_executor_v2.ops import optimizer as opt
# 필요시: conv2d/gemm/rmsnorm/softmax/cross_entropy 등 import
# from graph_executor_v2.ops import conv2d as conv_ops
# from graph_executor_v2.ops import rmsnorm as rn
from graph_executor_v2.ops import cross_entropy as ce

class CaptureTrainer:
    """
    단일 그래프(1 step) 캡처 후 반복 재생하는 트레이너 유틸.
    - 입력(X_d, y_d), 파라미터(P...), 모멘텀(M/V), 그래디언트 버퍼 등 '고정 주소' 버퍼 사용
    - 캡처 본문:  H2D memcpy -> forward -> loss -> backward -> optimizer
    - 리플레이:   pinned host에 배치 데이터만 바꾸고 g.launch()

    사용 패턴(개념):
        tr = CaptureTrainer(build_step=..., optimizer_step=..., buffers=...)
        tr.capture()
        for step in range(T):
            tr.load_batch(X_host, y_host)  # 고정 주소 host pinned 버퍼에 내용만 교체
            tr.step()                      # 그래프 launch
    """
    def __init__(
        self,
        *,
        # 고정 디바이스 버퍼들
        x_dev: cp.ndarray,         # [B,...] float32
        y_dev: cp.ndarray,         # [B]     (int32 for CE)
        # host pinned 버퍼들 (고정 주소)
        x_host: cp.ndarray,        # same shape/dtype as x_dev (pinned)
        y_host: cp.ndarray,        # same shape/dtype as y_dev (pinned)
        # 모델 파라미터/그래드/옵티마 상태 버퍼 dict (고정 주소)
        params: Dict[str, cp.ndarray],  # float32 1D tensors (flatten되어 있어도 OK)
        grads:  Dict[str, cp.ndarray],
        # 옵티마 상태(예: SGD: V, AdamW: M/V)
        optim_state: Dict[str, cp.ndarray],
        # 1-step 그래프를 구성하는 콜백들 (현재 stream 컨텍스트 내에서만 호출)
        build_step: Callable[[], Tuple[cp.ndarray, cp.ndarray]],
        #  - build_step() -> (logits, loss_vec) 또는 (logits, None)
        #  - 내부에서 forward/CE 등 ops 호출로 그래프 노드를 생성
        backward_step: Callable[[cp.ndarray], None],
        #  - backward_step(dY): logits에 대한 dY(또는 dlogits)를 입력으로 파라미터 그래드 계산
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
        self.g = None

        # 검증: 주소/shape 고정성
        assert self.x_d.flags.c_contiguous and self.x_h.flags.c_contiguous
        assert self.y_d.flags.c_contiguous and self.y_h.flags.c_contiguous
        assert self.x_d.dtype == self.x_h.dtype
        assert self.y_d.dtype == self.y_h.dtype
        assert self.x_d.shape == self.x_h.shape
        assert self.y_d.shape == self.y_h.shape

    @staticmethod
    def alloc_pinned_like(a: cp.ndarray) -> cp.ndarray:
        """입력과 동일한 shape/dtype의 pinned host 배열을 만든다(고정 주소)."""
        mem = cp.cuda.alloc_pinned_memory(a.nbytes)
        return cp.ndarray(a.shape, dtype=a.dtype, memptr=mem)

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
            logits, loss_vec = self.build_step()  # forward (+loss fwd 가능)
            # loss 미리 계산 안했다면 여기서 CE fwd 호출 가능
            # dlogits는 CE.bwd에서 리턴되도록 설계하거나, 스스로 계산
            # 예시: CE fused bwd 없을 경우
            if loss_vec is not None:
                # dlogits 생성: CE 모듈이 제공하는 backward 사용 (예시)
                #   ce.backward는 (dlogits, dparams...) 형태일 수도 있으므로
                pass
            # 모델 bwd
            #   외부에서 제공된 backward_step이 dY(or dlogits)를 소비해 파라미터 그래드 채움
            #   여기서는 가령 softmax_ce 모듈이 dlogits을 돌려준다고 가정
            #   backward_step 내부에서 필요한 버퍼/워크스페이스는 고정 주소여야 함
            # → dY(or dlogits)는 build_step에서 생성한 것을 캡처 내에서 사용
            #   구체 shape는 사용자의 네트워크 구현에 따름
            #   (이 샘플에서는 backward_step이 내부에서 dY를 이미 알고 사용한다고 가정)
            self.backward_step(logits)  # 필요 시 logits 대신 dlogits 전달하도록 바꿔도 됨
            # optimizer
            self.optimizer_step()

        # 워밍업(그래프 외부)
        with self.s:
            body()
        cp.cuda.get_current_stream().synchronize()

        # 캡처
        with self.s:
            self.s.begin_capture()
            body()
            self.g = self.s.end_capture()

        # 업로드(가능한 경우)
        with self.s:
            try:
                if hasattr(self.g, "upload"):
                    self.g.upload()
            except Exception:
                pass
        self.s.synchronize()

    def load_batch(self, x_batch: cp.ndarray, y_batch: cp.ndarray):
        """고정 주소 host pinned 버퍼에 새 배치 내용을 복사(주소는 그대로, 내용만 교체)."""
        assert x_batch.shape == self.x_h.shape and x_batch.dtype == self.x_h.dtype
        assert y_batch.shape == self.y_h.shape and y_batch.dtype == self.y_h.dtype
        # host-pinned에 쓰기: numpy/cupy 복사 모두 가능 (cpu상에서 채워도 OK)
        self.x_h[...] = x_batch
        self.y_h[...] = y_batch

    def step(self):
        """그래프 재생(1 step)."""
        assert self.g is not None, "call capture() first"
        with self.s:
            self.g.launch()
        self.s.synchronize()
