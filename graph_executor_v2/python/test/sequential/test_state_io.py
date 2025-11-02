import cupy as cp
import numpy as np
import pytest
from graph_executor_v2.layers.sequential import Sequential
from graph_executor_v2.layers.dense_gemm import Dense

def test_state_save_and_load(tmp_path):
    m1 = Sequential(Dense(4, 4, activation="none", use_native_bwd=True))
    m1.build(input_shape=(2, 4))

    if not hasattr(m1, "state_dict"):
        pytest.skip("state_dict/save/load API not available")

    state = m1.state_dict()
    p = tmp_path / "seq_state.npz"

    if hasattr(m1, "save"):
        m1.save(p)
        m2 = Sequential(Dense(4, 4, activation="none", use_native_bwd=True))
        m2.build(input_shape=(2, 4))
        if hasattr(m2, "load"):
            m2.load(p)
        elif hasattr(m2, "load_state_dict"):
            # 수동 로드 경로
            m2.load_state_dict(dict(np.load(p)))
    else:
        # 파일 API가 없다면 메모리 내 round-trip
        m2 = Sequential(Dense(4, 4, activation="none", use_native_bwd=True))
        m2.build(input_shape=(2, 4))
        if hasattr(m2, "load_state_dict"):
            m2.load_state_dict(state)
        else:
            pytest.skip("no load_state_dict; cannot test state IO")

    sd1 = m1.state_dict(); sd2 = m2.state_dict()
    assert sd1.keys() == sd2.keys()
    for k in sd1.keys():
        a, b = sd1[k], sd2[k]
        np.testing.assert_allclose(cp.asnumpy(a), cp.asnumpy(b))
