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

    def compile(self, input_shape: Tuple[int, int]):
        if not getattr(self.model, "built", False):
            self.model.build(tuple(map(int, input_shape)))

        plan = make_plan_for_sequential(
            self.model, tuple(map(int, input_shape)),
            loss_kind="softmax_ce", lt_bytes=self.lt_bytes
        )
        try_rebind_grads(self.model, self.optimizer, plan)

        bs, in_dim = int(input_shape[0]), int(input_shape[1])
        X_buf = cp.zeros((bs, in_dim), dtype=cp.float32)
        y_buf = cp.zeros((bs,), dtype=cp.int32)

        gexec = record_step_graph(
            self.model, self.loss_fn, self.optimizer.step_into,
            plan, X_buf=X_buf, y_buf=y_buf, stream=self.stream  # ✅ 전달
        )
        io = {"X": X_buf, "y": y_buf, "logits": plan.per_layer[-1].y}
        self._tg = TrainGraph(gexec, io, self.stream)

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
