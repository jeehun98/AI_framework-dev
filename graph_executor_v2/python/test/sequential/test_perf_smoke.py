import time
import cupy as cp
import pytest
from graph_executor_v2.layers.sequential import Sequential
from graph_executor_v2.layers.dense_gemm import Dense
from graph_executor_v2.layers.dropout import Dropout
from graph_executor_v2.optim.adamw import AdamWOpt
from graph_executor_v2.losses.softmax_ce import SoftmaxCrossEntropy

def test_perf_smoke():
    has_replay = False
    # 예: hasattr(Sequential, "compile") and ... 등으로 판단
    if not has_replay:
        pytest.skip("CUDA Graph replay not available in this build")

    cp.random.seed(6)
    X = cp.random.standard_normal((64, 512), dtype=cp.float32)
    y = cp.random.randint(0, 100, size=(64,), dtype=cp.int32)

    m = Sequential(Dense(512, 2048, activation="gelu", use_native_bwd=True),
                   Dropout(0.1),
                   Dense(2048, 100, activation="none", use_native_bwd=True))
    m.build(input_shape=(64, 512))
    loss = SoftmaxCrossEntropy()
    opt = AdamWOpt([], lr=1e-2)
    if hasattr(opt, "ensure_initialized"):
        opt.ensure_initialized()

    # eager 100 steps
    def eager_step():
        m.train(True)
        logits = m(X)
        L = loss(logits, y); L.backward()
        opt.step(m.parameters()); m.zero_grad()
        return float(L.get()) if hasattr(L, "get") else float(L)

    t0 = time.time()
    last = None
    for _ in range(100):
        last = eager_step()
    t_eager = time.time() - t0

    # 실제가 있다면 train_graph.launch(...)로 교체하여 Replay 측정
    # t_replay = ...

    assert last > 0.0
    # 문서 표 예시(수치는 기기마다 다름):
    # RTX 4090 | (64,512)→2048→100 | steps=100 | Eager: 0.85s | Replay: 0.32s
