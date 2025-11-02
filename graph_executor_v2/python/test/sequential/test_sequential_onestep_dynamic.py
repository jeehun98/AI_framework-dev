import cupy as cp
from graph_executor_v2.layers.sequential import Sequential
from graph_executor_v2.layers.dense_gemm import Dense
from graph_executor_v2.layers.dropout import Dropout
from graph_executor_v2.losses.softmax_ce import SoftmaxCrossEntropy
from graph_executor_v2.optim.adamw import AdamWOpt

def test_one_step_dynamic_smoke():
    cp.random.seed(4)
    N, Din, H, C = 32, 64, 128, 10
    X = cp.random.standard_normal((N, Din), dtype=cp.float32)
    y = cp.random.randint(0, C, size=(N,), dtype=cp.int32)

    m = Sequential(Dense(Din, H, activation="gelu", use_native_bwd=True),
                   Dropout(0.1),
                   Dense(H, C, activation="none", use_native_bwd=True))
    m.build(input_shape=(N, Din))
    m.train(True)

    loss = SoftmaxCrossEntropy()
    opt  = AdamWOpt([], lr=5e-2)
    if hasattr(opt, "ensure_initialized"):
        opt.ensure_initialized()

    ctx = {"variant": {"unroll": 1, "amp": "fp32"}}
    L = m.one_step_dynamic(X, y, loss=loss, optimizer=opt, ctx=ctx)
    assert isinstance(L, float) and cp.isfinite(cp.asarray(L))
