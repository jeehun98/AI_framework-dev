import cupy as cp
import numpy as np
from graph_executor_v2.layers.sequential import Sequential
from graph_executor_v2.layers.dense_gemm import Dense
from graph_executor_v2.layers.dropout import Dropout

def test_build_and_shapes_and_params():
    m = Sequential(
        Dense(16, 32, activation="gelu", use_native_bwd=True),
        Dropout(0.2),
        Dense(32, 8, activation="none", use_native_bwd=True),
    )
    report = m.build(input_shape=(None, 16))  # 배치 None 허용
    # build()가 리포트를 리턴하는 구현일 때
    if report is not None:
        assert report.input_shape == (None, 16)
        assert report.output_shape == (None, 8)
        assert report.param_count > 0

    params = m.parameters()
    expected = (16*32 + 32) + (32*8 + 8)  # W+b 합계
    total = 0
    for p in params:
        # 텐서 numel() 유틸이 있으면 사용, 없으면 cp.size로 대체
        if hasattr(p, "numel"):
            total += p.numel()
        else:
            total += int(cp.size(p.data))  # p.data: CuPy array 가정
    assert total == expected
