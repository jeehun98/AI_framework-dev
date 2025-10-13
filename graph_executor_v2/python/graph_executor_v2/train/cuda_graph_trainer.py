# python/graph_executor_v2/train/cuda_graph_trainer.py
from __future__ import annotations
from typing import Optional, Tuple
import cupy as cp
from ..graph.capture_plan import make_plan_for_sequential
from ..graph.graph_exec import record_step_graph, TrainGraph
from ..optim.rebind import try_rebind_grads

# 사용자 입장에서 최소 호출로 그래프 캡처 학습을 돌릴 수 있는 E2E 트레이너.
class CudaGraphTrainer:
    def __init__(self, model, loss_fn, optimizer, *, lt_bytes: int = (8 << 20)):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lt_bytes = lt_bytes
        self.stream = cp.cuda.Stream(non_blocking=True)
        self._tg: Optional[TrainGraph] = None

    # 고정 X_buf, y_buf 생성 및 TrainGraph 구축.
    def compile(self, input_shape: Tuple[int, ...]):
        """
        input_shape는 모델 입력 모양 그대로 전달. (예: (N,C,H,W) 또는 (N,D))
        """
        in_shape = tuple(map(int, input_shape))
        if not getattr(self.model, "built", False):
            self.model.build(in_shape)

        # 캡처 플랜 생성
        plan = make_plan_for_sequential(
            self.model, in_shape, loss_kind="softmax_ce", lt_bytes=self.lt_bytes
        )

        # 옵티마이저 grad 포인터를 plan 버퍼(gW/gB)로 재바인딩(지원 시)
        try_rebind_grads(self.model, self.optimizer, plan)

        # ✅ 입력 버퍼: 입력 모양 그대로(Conv2D 등 4D 입력 지원)
        X_buf = cp.zeros(in_shape, dtype=cp.float32)

        # ✅ 라벨 버퍼: 배치 크기 = 첫 차원
        N = int(in_shape[0])
        y_buf = cp.zeros((N,), dtype=cp.int32)

        # 그래프 녹화
        gexec = record_step_graph(
            self.model, self.loss_fn, self.optimizer.step_into,
            plan, X_buf=X_buf, y_buf=y_buf, stream=self.stream
        )

        # 마지막 레이어 출력 버퍼(로짓) 노출
        io = {"X": X_buf, "y": y_buf, "logits": plan.per_layer[-1].y}
        self._tg = TrainGraph(gexec, io, self.stream)

    # 고정 버퍼에 복사 후 그래프 실행, 현재 모델 파라미터로 손실 재계산하여 반환.
    def one_step(self, X, y) -> float:
        assert self._tg is not None, "call compile() first"
        self._tg.set_batch(X, y)
        self._tg.launch()
        L, _ = self.loss_fn.forward(self.model(self._tg.X_buf), self._tg.y_buf)
        return float(L)

    @property
    def tg(self) -> TrainGraph:
        assert self._tg is not None, "call compile() first"
        return self._tg
