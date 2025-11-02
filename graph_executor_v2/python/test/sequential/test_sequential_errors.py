import cupy as cp
import pytest
from graph_executor_v2.layers.sequential import Sequential
from graph_executor_v2.layers.dense_gemm import Dense

def test_infer_shape_mismatch():
    m = Sequential(Dense(8, 4, activation="none", use_native_bwd=True))
    with pytest.raises(ValueError) as e:
        m.build(input_shape=(32, 7))  # 잘못된 dim
    assert "mismatch" in str(e.value).lower()

def test_forward_without_build():
    m = Sequential(Dense(2, 2, activation="none", use_native_bwd=True))
    X = cp.zeros((1, 2), dtype=cp.float32)
    with pytest.raises(RuntimeError) as e:
        m(X)
    assert "build" in str(e.value).lower()
