# dev/layers/core/tests/dense_test.py

import os
import sys
import numpy as np

# ✅ 프로젝트 루트 설정 (AI_framework-dev)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ✅ 공통 테스트 설정 적용
from dev.tests.test_setup import setup_paths, import_cuda_module
setup_paths()
import_cuda_module()

# ✅ 핵심 모듈 임포트
from dev.layers.dense_cuda import Dense
from dev.graph_engine.core_graph import Cal_graph


def test_dense_layer():
    print("===== [TEST] Dense Layer Forward Pass & Computation Graph =====")

    # ✅ 입력 및 초기 설정
    input_data = np.array([[1.0, 2.0], [3.0, 4.0]])  # (2, 2)
    units = 3

    dense_layer = Dense(units=units, activation=None, initializer="ones")
    dense_layer.build(input_shape=(2, 2))

    # ✅ 가중치 및 편향 강제 지정 (값 1로 고정)
    dense_layer.weights = np.ones((2, units))
    dense_layer.bias = np.ones((1, units))

    # ✅ Forward 수행
    output = dense_layer.call(input_data)

    # ✅ 기대값 계산: (input @ W) + b
    expected_output = np.array([
        [4.0, 4.0, 4.0],
        [8.0, 8.0, 8.0]
    ])

    print("\n✅ Dense Layer Output:")
    print(output)

    # ✅ 출력 검증
    assert np.allclose(output, expected_output), "❌ Forward Pass Output Mismatch!"

    # ✅ 계산 그래프 출력
    print("\n✅ Computation Graph:")
    dense_layer.cal_graph.print_graph()

    print("\n🎉 [TEST PASSED] Dense Layer and Computation Graph Successfully Validated!")


if __name__ == "__main__":
    test_dense_layer()