import sys
import os
import numpy as np

# ✅ 프로젝트 루트 경로 추가
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.insert(0, PROJECT_ROOT)

from dev.graph_engine.graph_compiler import GraphCompiler
from dev.layers.dense_mat import DenseMat
from dev.layers.activation_mat import ActivationMat


def test_graph_compiler_deep_model():
    # ✅ 모델: Dense(4→6)+ReLU → Dense(6→5)+Tanh → Dense(5→3)+Sigmoid
    dense1 = DenseMat(units=6, activation='sigmoid', input_dim=4)
    dense1.build(4)

    dense2 = DenseMat(units=5)
    dense2.build(6)
    act2 = ActivationMat('sigmoid')
    act2.build(5)

    dense3 = DenseMat(units=3)
    dense3.build(5)
    act3 = ActivationMat('sigmoid')
    act3.build(3)

    # ✅ GraphCompiler에 레이어 추가
    compiler = GraphCompiler()
    compiler.add_layer(dense1)
    compiler.add_layer(dense2)
    compiler.add_layer(act2)
    compiler.add_layer(dense3)
    compiler.add_layer(act3)

    # ✅ 그래프 컴파일
    compiler.build()
    matrices = compiler.get_matrices()

    # ✅ 출력 확인
    print("🧩 op_matrix:")
    print(matrices["op_matrix"])

    print("\n🧩 input_matrix:")
    print(matrices["input_matrix"])

    print("\n🧩 param_vector (길이):", len(matrices["param_vector"]))

    print("\n📊 visualize:")
    print(compiler.visualize())

    # ✅ 기본 검증
    assert matrices["op_matrix"].ndim == 1
    assert matrices["input_matrix"].ndim == 2
    assert isinstance(matrices["param_vector"], np.ndarray)
    assert len(matrices["op_matrix"]) == len(matrices["input_matrix"])
    print("\n✅ 2-은닉층 모델 테스트 통과")


if __name__ == "__main__":
    test_graph_compiler_deep_model()
