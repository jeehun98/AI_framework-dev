import cupy as cp
import numpy as np
from graph_executor_v2.layers.sequential import Sequential
from graph_executor_v2.layers.dense_gemm import Dense

def gelu_np(x):
    # NumPy 레퍼런스 (近似)
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

def test_forward_matches_numpy_reference():
    cp.random.seed(1)
    X = cp.random.standard_normal((4, 5), dtype=cp.float32)

    # 단일 Dense + GELU 비교
    m = Sequential(Dense(5, 3, activation="gelu", use_native_bwd=True))
    m.build(input_shape=(4, 5))
    m.eval()  # dropout 등 비활성

    dense = m.layers[0]
    W = cp.asnumpy(dense.weight)  # CuPy→NumPy
    b = cp.asnumpy(dense.bias)

    Y_py = m(X)                   # CuPy 텐서
    Y_np = gelu_np(cp.asnumpy(X) @ W + b)

    np.testing.assert_allclose(cp.asnumpy(Y_py), Y_np, rtol=1e-4, atol=1e-4)
