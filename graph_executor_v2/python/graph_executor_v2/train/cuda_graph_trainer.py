# python/graph_executor_v2/train/cuda_graph_trainer.py
from __future__ import annotations
from typing import Optional, Tuple
import cupy as cp
from ..graph.capture_plan import make_plan_for_sequential
from ..graph.graph_exec import record_step_graph, TrainGraph
from ..optim.rebind import try_rebind_grads

class CudaGraphTrainer:
    def __init__(self, model, loss_fn, optimizer, *, lt_bytes: int = (8 << 20)):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lt_bytes = lt_bytes
        self.stream = cp.cuda.Stream(non_blocking=True)
        self._tg: Optional[TrainGraph] = None
        self._loss_buf: Optional[cp.ndarray] = None  # ✅ 그래프 내에서 갱신되는 손실값 버퍼

    def compile(self, input_shape: Tuple[int, ...]):
        """
        input_shape는 모델 입력 모양 그대로 전달. (예: (N,C,H,W) 또는 (N,D))
        """
        in_shape = tuple(map(int, input_shape))
        if not getattr(self.model, "built", False):
            self.model.build(in_shape)

        # 1) 캡처 플랜 생성
        plan = make_plan_for_sequential(
            self.model, in_shape, loss_kind="softmax_ce", lt_bytes=self.lt_bytes
        )

        # 2) 옵티마이저 grad 포인터 재바인딩(지원 시)
        try_rebind_grads(self.model, self.optimizer, plan)

        # 3) 고정 I/O 버퍼
        X_buf = cp.zeros(in_shape, dtype=cp.float32)              # 입력
        N = int(in_shape[0])
        y_buf = cp.zeros((N,), dtype=cp.int32)                    # 라벨(int32 사용 중이면 유지)
        loss_buf = cp.zeros((), dtype=cp.float32)                 # ✅ 손실 스칼라(디바이스)

        # 4) 그래프 녹화 (fwd+loss+bwd+step까지 포함, loss를 loss_buf에 write)
        gexec = record_step_graph(
            self.model,
            self.loss_fn,
            self.optimizer.step_into,         # step 호출 콜백
            plan,
            X_buf=X_buf,
            y_buf=y_buf,
            stream=self.stream,
            loss_out=loss_buf,                # ✅ 추가: graph 안에서 loss_buf[:] = loss
        )

        # 5) 출력 핸들 수집
        io = {"X": X_buf, "y": y_buf, "logits": plan.per_layer[-1].y}
        tg = TrainGraph(gexec, io, self.stream)

        self._tg = tg
        self._loss_buf = loss_buf

    def one_step(self, X, y) -> float:
        """
        고정 버퍼에 복사 -> 그래프 실행 -> loss_buf에서 스칼라만 읽어 반환
        (추가 forward 없음)
        """
        assert self._tg is not None, "call compile() first"
        assert self._loss_buf is not None, "loss buffer not initialized"
        # ✅ 모양/타입 가드(디버그 시 유용)
        xb, yb = self._tg.X_buf, self._tg.y_buf
        assert tuple(xb.shape) == tuple(cp.asarray(X).shape), f"X shape mismatch: {cp.asarray(X).shape} vs {xb.shape}"
        assert yb.shape == (xb.shape[0],), f"y shape must be (N,), got {yb.shape} vs N={xb.shape[0]}"
        assert yb.dtype == cp.int32, f"labels must be int32 for current CE kernel (got {yb.dtype})"        

        self._tg.set_batch(X, y)
        self._tg.launch()

        # D2H로 loss만 동기 복사 (필요 시 stream 연동)
        # 현재 stream에 그래프가 올라가 있으므로, 해당 stream에서 동기화 후 읽기
        
        loss = float(self._loss_buf.get())  # get()은 D2H copy
        return loss

    @property
    def tg(self) -> TrainGraph:
        assert self._tg is not None, "call compile() first"
        return self._tg
