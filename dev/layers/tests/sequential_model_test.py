import sys
import os
import ctypes
import numpy as np
import cupy as cp

# CUDA DLL 명시적 로드
ctypes.CDLL(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\cudart64_12.dll")

# Pybind11로 빌드된 .pyd 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), "build", "lib.win-amd64-cpython-312"))

# AI framework 루트 경로 추가
sys.path.insert(0, os.path.abspath("C:/Users/owner/Desktop/AI_framework-dev"))

sys.path.append("C:/Users/owner/Desktop/AI_framework-dev/dev/backend/graph_executor/test")
import graph_executor as ge  # Pybind11 모듈


# Graph Executor 모듈 임포트
from graph_executor import OpStruct, Shape, run_graph_cuda

# Sequential 모델 관련 임포트
from dev.models.sequential import Sequential
from dev.layers.dense import Dense
from dev.layers.activation_layer import Activation
from dev.layers.flatten import Flatten


def test_sequential_model_fit():
    print("\n=== [TEST] Sequential 모델 학습 테스트 ===")

    # 3. 입력 / 타겟 데이터 정의
    x = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)  # shape: (1, 2, 2)
    y = np.array([[0.7, 0.1]], dtype=np.float32)  # shape: (1, 2)

    # 1. 모델 구성
    model = Sequential(input_shape=(1, 2, 2))
    model.add(Flatten(input_shape=(1, 2, 2)))
    model.add(Dense(units=2, activation=None))
    model.add(Activation("sigmoid"))

    # ✅ Dense 강제 초기화 (여기에 삽입)
    for layer in model._layers:
        if isinstance(layer, Dense):
            layer.weights = cp.ones_like(layer.weights) * 0.5
            layer.bias = cp.ones_like(layer.bias) * 0.1
            layer.weights = cp.asarray(layer.weights)
            layer.bias = cp.asarray(layer.bias)
            print(f"[INFO] Dense layer `{layer.name}` 초기화 완료: weight=0.5, bias=0.1")

    # 2. 모델 컴파일
    model.compile(optimizer="sgd", loss="mse")

    # 4. 학습
    model.fit(x, y, epochs=3)

    print("\n✅ 학습 완료")


if __name__ == "__main__":
    test_sequential_model_fit()
