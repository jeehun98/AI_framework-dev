# test/integration/test_embedding_trainer_smoke_extra.py
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

# ---- Optional layers (존재하면 사용) ----
try:
    from graph_executor_v2.layers.pad import Pad
except Exception:
    Pad = None

try:
    from graph_executor_v2.layers.view import View
except Exception:
    View = None

try:
    from graph_executor_v2.layers.slice import Slice
except Exception:
    Slice = None

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


def make_model_base(*, vocab_size: int, emb_dim: int, hidden: int, classes: int):
    """기본: Embedding → Flatten → Dense → Act → Dense"""
    if Embedding is None:
        raise RuntimeError("Embedding layer not available — skipping test.")
    return Sequential(
        Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim, padding_idx=-1),
        Flatten(),  # (N, L, D) -> (N, L*D)
        Dense(hidden,  activation="none",  initializer="he",     use_native_bwd=True),
        ActivationLayer(act="relu", save_y=True),
        Dense(classes, activation="none",  initializer="xavier", use_native_bwd=True),
    )


def make_model_with_pad(*, vocab_size: int, emb_dim: int, hidden: int, classes: int):
    """
    Embedding → (Pad) → Flatten → Dense → Act → Dense
    - Pad는 ND 지원, (N,L,D)에서 L축에 before/after 패딩을 줌 (예: 앞뒤 1 토큰 0패딩)
    """
    if Embedding is None:
        raise RuntimeError("Embedding layer not available — skipping test.")
    if Pad is None:
        # Pad 없으면 base 모델로 대체
        return make_model_base(vocab_size=vocab_size, emb_dim=emb_dim, hidden=hidden, classes=classes)

    return Sequential(
        Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim, padding_idx=-1),
        # (N,L,D) 기준: L축(두 번째 축)에 앞뒤 1씩 패딩
        Pad(before=(0, 1, 0), after=(0, 1, 0), value=0.0),
        Flatten(),
        Dense(hidden,  activation="none",  initializer="he",     use_native_bwd=True),
        ActivationLayer(act="relu", save_y=True),
        Dense(classes, activation="none",  initializer="xavier", use_native_bwd=True),
    )


def make_model_with_view_slice(*, vocab_size: int, emb_dim: int, hidden: int, classes: int):
    """
    Embedding → (View 3D→4D) → (Slice 4D) → (View 4D→3D) → Flatten → Dense → Act → Dense
    - View/Slice 레이어가 있는 경우에만 구성. 없으면 base 모델로 대체.
    - Slice는 H축(L 차원에 해당하도록 재해석)에서 앞뒤 1 토큰을 잘라 길이를 줄임.
    """
    if Embedding is None:
        raise RuntimeError("Embedding layer not available — skipping test.")
    if View is None or Slice is None:
        # 둘 중 하나라도 없으면 base로
        return make_model_base(vocab_size=vocab_size, emb_dim=emb_dim, hidden=hidden, classes=classes)

    # 가정: View 레이어가 입력 shape을 기준으로 target shape/stride를 연속형으로 설정 가능
    # (N,L,D) → (N,1,L,D) → Slice (H=L 축을 1..L-1로 자르기) → (N,L-2,D)
    return Sequential(
        Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim, padding_idx=-1),

        # 3D -> 4D: (N, L, D) 를 (N, 1, L, D)로 해석
        View(shape=("N", "1", "L", "D")),   # 구현체가 문자열 토큰을 허용하지 않으면 레이어 쪽에서 build 시 실제 크기로 결정

        # Slice: start=(0,0,1,0), end=(N,1,L-1,D), step=(1,1,1,1)
        # 구현에 따라 절대값/상대값 처리 방식이 다를 수 있으니, 레이어가 build에서 입력 shape로 end를 해석한다고 가정
        Slice(start=(0, 0, 1, 0), end=("N", "1", "H-1", "W"), step=(1, 1, 1, 1)),

        # 4D -> 3D: (N, 1, L-2, D) → (N, L-2, D)
        View(shape=("N", "H", "W")),        # 내부에서 연속형 stride로 3D 재해석

        Flatten(),
        Dense(hidden,  activation="none",  initializer="he",     use_native_bwd=True),
        ActivationLayer(act="relu", save_y=True),
        Dense(classes, activation="none",  initializer="xavier", use_native_bwd=True),
    )


def run_smoke_for_embedding(name: str, *, V: int, D: int, L: int, C: int, variant: str):
    if Embedding is None:
        print(f"[SKIP] {name}/{variant}: Embedding layer is not available.")
        return

    N = 8
    cp.random.seed(2031)
    X = cp.random.randint(0, V, size=(N, L), dtype=cp.int32)  # ✅ int32
    y = cp.random.randint(0, C, size=(N,), dtype=cp.int32)

    if variant == "base":
        model = make_model_base(vocab_size=V, emb_dim=D, hidden=64, classes=C)
    elif variant == "with_pad":
        model = make_model_with_pad(vocab_size=V, emb_dim=D, hidden=64, classes=C)
    elif variant == "with_view_slice":
        model = make_model_with_view_slice(vocab_size=V, emb_dim=D, hidden=64, classes=C)
    else:
        raise ValueError(f"unknown variant: {variant}")

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
        print(f"[SMOKE][Embedding:{name}/{variant}] step={t:02d} loss={Lval:.6f}")
        assert isinstance(Lval, float) and math.isfinite(Lval)
        last_L = Lval

    print(f"[OK] Integrated trainer smoke passed with Embedding({name}/{variant}). Last loss={last_L:.6f}")


def main():
    print("== Integrated trainer smoke with Embedding (extra variants) ==")
    cases = [
        ("V1k_D32_L16_C7",  1000, 32, 16, 7),
        ("V4k_D64_L20_C11", 4000, 64, 20, 11),
    ]
    variants = ["base", "with_pad", "with_view_slice"]  # concat 제외
    for name, V, D, L, C in cases:
        for v in variants:
            run_smoke_for_embedding(name, V=V, D=D, L=L, C=C, variant=v)


if __name__ == "__main__":
    main()
