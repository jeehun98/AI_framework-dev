# test/integration/test_embedding_trainer_smoke.py
import os, sys, math
THIS = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(THIS, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import cupy as cp

from graph_executor_v2.layers.sequential import Sequential
from graph_executor_v2.layers.flatten import Flatten
from graph_executor_v2.layers.dense_gemm import Dense
from graph_executor_v2.layers.activations import ActivationLayer
from graph_executor_v2.losses.softmax_ce import SoftmaxCrossEntropy
from graph_executor_v2.optim.adamw import AdamWOpt
from graph_executor_v2.graph.capture_plan import make_plan_for_sequential
from graph_executor_v2.graph.graph_exec import record_step_graph, TrainGraph
from graph_executor_v2.optim.rebind import try_rebind_grads
from graph_executor_v2.train.cuda_graph_trainer import CudaGraphTrainer

# Embedding 레이어
try:
    from graph_executor_v2.layers.embedding import Embedding
except Exception:
    Embedding = None


class EmbeddingGraphTrainer(CudaGraphTrainer):
    """입력 버퍼를 int32 (토큰 인덱스)로 생성하도록 오버라이드"""
    def compile(self, input_shape):
        if not getattr(self.model, "built", False):
            self.model.build(tuple(map(int, input_shape)))

        plan = make_plan_for_sequential(
            self.model, tuple(map(int, input_shape)),
            loss_kind="softmax_ce", lt_bytes=self.lt_bytes
        )
        try_rebind_grads(self.model, self.optimizer, plan)

        N, L = map(int, input_shape)
        X_buf = cp.zeros((N, L), dtype=cp.int32)  # ✅ int32 인덱스
        y_buf = cp.zeros((N,), dtype=cp.int32)

        gexec = record_step_graph(
            self.model, self.loss_fn, self.optimizer.step_into,
            plan, X_buf=X_buf, y_buf=y_buf, stream=self.stream
        )
        io = {"X": X_buf, "y": y_buf, "logits": plan.per_layer[-1].y}
        self._tg = TrainGraph(gexec, io, self.stream)


def make_model_for_embedding(*, vocab_size: int, emb_dim: int, seq_len: int, hidden: int, classes: int):
    if Embedding is None:
        raise RuntimeError("Embedding layer not available — skipping test.")
    return Sequential(
        Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim, padding_idx=-1),
        Flatten(),  # (N, L, D) -> (N, L*D)
        Dense(hidden,  activation="none",  initializer="he",     use_native_bwd=True),
        ActivationLayer(act="relu", save_y=True),
        Dense(classes, activation="none",  initializer="xavier", use_native_bwd=True),
    )


def run_smoke_for_embedding(name: str, *, V: int, D: int, L: int, C: int):
    if Embedding is None:
        print(f"[SKIP] {name}: Embedding layer is not available.")
        return

    N = 8
    cp.random.seed(2030)
    X = cp.random.randint(0, V, size=(N, L), dtype=cp.int32)  # ✅ int32
    y = cp.random.randint(0, C, size=(N,), dtype=cp.int32)

    model = make_model_for_embedding(vocab_size=V, emb_dim=D, seq_len=L, hidden=64, classes=C)
    model.build((N, L))  # 입력 shape은 (N, L)

    loss = SoftmaxCrossEntropy()
    opt = AdamWOpt([], lr=1e-3, wd=1e-4)
    if hasattr(opt, "ensure_initialized"):
        opt.ensure_initialized()

    trainer = EmbeddingGraphTrainer(model, loss, opt)
    trainer.compile((N, L))  # CUDA Graph capture

    last_L = None
    for t in range(3):
        Lval = trainer.one_step(X, y)
        print(f"[SMOKE][Embedding:{name}] step={t:02d} loss={Lval:.6f}")
        assert isinstance(Lval, float) and math.isfinite(Lval)
        last_L = Lval

    print(f"[OK] Integrated trainer smoke passed with Embedding({name}). Last loss={last_L:.6f}")


def main():
    print("== Integrated trainer smoke with Embedding inserted ==")
    cases = [
        ("V1k_D32_L16_C7", 1000, 32, 16, 7),
        ("V4k_D64_L20_C11", 4000, 64, 20, 11),
    ]
    for name, V, D, L, C in cases:
        run_smoke_for_embedding(name, V=V, D=D, L=L, C=C)


if __name__ == "__main__":
    main()
