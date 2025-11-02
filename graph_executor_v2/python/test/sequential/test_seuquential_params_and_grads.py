import cupy as cp
from graph_executor_v2.layers.sequential import Sequential
from graph_executor_v2.layers.dense_gemm import Dense
from graph_executor_v2.losses.softmax_ce import SoftmaxCrossEntropy

def test_zero_grad_and_attach():
    cp.random.seed(3)
    N, Din, C = 8, 4, 5
    X = cp.random.standard_normal((N, Din), dtype=cp.float32)
    y = cp.random.randint(0, C, size=(N,), dtype=cp.int32)

    m = Sequential(Dense(Din, C, activation="none", use_native_bwd=True))
    m.build(input_shape=(N, Din))
    m.train(True)

    loss_fn = SoftmaxCrossEntropy()

    m.zero_grad()
    logits = m(X)
    loss = loss_fn(logits, y)
    loss.backward()

    # 파라미터 별 grad 확인
    for p in m.parameters():
        assert hasattr(p, "grad")
        g = p.grad
        # CuPy 배열 가정
        assert float(cp.linalg.norm(g)) > 0.0

    # zero_grad 후 grad=0
    m.zero_grad()
    for p in m.parameters():
        assert float(cp.max(cp.abs(p.grad))) == 0.0
