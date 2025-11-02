import cupy as cp
from graph_executor_v2.layers.sequential import Sequential
from graph_executor_v2.layers.dense_gemm import Dense
from graph_executor_v2.layers.dropout import Dropout

def test_smoke_minimal_forward():
    cp.random.seed(0)
    X = cp.random.standard_normal((8, 16), dtype=cp.float32)

    m = Sequential(
        Dense(16, 32, activation="relu", use_native_bwd=True),
        Dropout(0.1),
        Dense(32, 4, activation="none", use_native_bwd=True),
    )
    m.build(input_shape=(8, 16))
    m.train(True)

    Y = m(X)  # eager forward
    assert Y.shape == (8, 4)
