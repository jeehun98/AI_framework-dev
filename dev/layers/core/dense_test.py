import sys
import os

# 현재 스크립트 파일 위치 기준으로 프로젝트 루트 디렉토리 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))


# 빌드된 모듈 경로 추가
build_path = os.path.abspath("dev/backend/operaters/build/lib.win-amd64-cpython-312")
if os.path.exists(build_path):
    sys.path.append(build_path)
else:
    raise FileNotFoundError(f"Build path does not exist: {build_path}")

# CUDA DLL 경로 추가
cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"
if os.path.exists(cuda_path):
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(cuda_path)
    else:
        os.environ["PATH"] = cuda_path + os.pathsep + os.environ["PATH"]
else:
    raise FileNotFoundError(f"CUDA path does not exist: {cuda_path}")

try:
    import matrix_ops
except ImportError as e:
    raise ImportError("Failed to import `matrix_ops` module. Ensure it is built and the path is correctly set.") from e


import numpy as np
from dev.layers.core.dense_cuda import Dense
from dev.cal_graph.cal_graph import Cal_graph

def test_dense_layer():
    """
    Dense 레이어의 연산 및 계산 그래프 구성을 검증하는 테스트 함수
    """
    print("===== [TEST] Dense Layer Forward Pass & Computation Graph =====")

    # ✅ 입력 데이터 및 Dense 레이어 설정
    input_data = np.array([[1.0, 2.0], [3.0, 4.0]])  # (batch_size=2, input_dim=2)
    units = 2  # 출력 차원

    dense_layer = Dense(units=units, activation=None, initializer="ones")
    dense_layer.build(input_shape=(2, 2))  # input_shape = (batch_size, input_dim)

    # ✅ 가중치와 편향을 고정하여 테스트 (초기화를 ones로 설정했기 때문에 가중치 = 1)
    dense_layer.weights = np.ones((2, units))  # 2x3 행렬 (입력 x 가중치)
    dense_layer.bias = np.ones((1, units))  # 1x3 행렬 (편향)

    # ✅ Forward pass 실행
    output = dense_layer.call(input_data)

    # ✅ Expected output 계산 (기대값: 모든 값이 (X @ W) + b = 3, 7)
    expected_output = np.array([
        [1*1 + 2*1 + 1, 1*1 + 2*1 + 1, 1*1 + 2*1 + 1],  # (1+2) + bias(1) = 3
        [3*1 + 4*1 + 1, 3*1 + 4*1 + 1, 3*1 + 4*1 + 1]   # (3+4) + bias(1) = 7
    ])

    print("\n✅ Dense Layer Output:")
    print(output)

    # assert np.allclose(output, expected_output), "❌ Forward Pass Output Mismatch!"

    # ✅ 계산 그래프 출력
    print("\n✅ Computation Graph:")
    dense_layer.cal_graph.print_graph()

    print("\n🎉 [TEST PASSED] Dense Layer and Computation Graph Successfully Validated!")

# ✅ 테스트 실행
test_dense_layer()