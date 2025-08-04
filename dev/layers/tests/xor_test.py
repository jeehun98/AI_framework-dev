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

# Pybind11 모듈
import graph_executor as ge

# AI Framework 임포트
from dev.models.sequential import Sequential
from dev.layers.dense import Dense
from dev.layers.activation_layer import Activation
from dev.layers.flatten import Flatten


def test_xor_classification():
    print("\n=== [TEST] XOR 분류 문제 테스트 ===")

    # 1. XOR 데이터셋 정의
    x = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ], dtype=np.float32)

    y = np.array([
        [0],
        [1],
        [1],
        [0]
    ], dtype=np.float32)

    # (B, C, H, W) 형태로 변형
    x = x.reshape(4, 1, 1, 2)

    # 2. 모델 구성
    model = Sequential(input_shape=(1, 1, 2))
    model.add(Flatten(input_shape=(1, 1, 2)))
    model.add(Dense(units=4, activation=None, initializer= 'he'))
    model.add(Activation("sigmoid"))
    model.add(Dense(units=1, activation=None, initializer= 'he'))
    model.add(Activation("sigmoid"))

    # 3. 컴파일
    model.compile(optimizer="adam", loss="mse", p_metrics="mse", learning_rate=0.00001)
    print(f"[DEBUG] learning_rate = {model.learning_rate}")

    # 4. 학습
    model.fit(x, y, epochs=3, batch_size=1)

    # 5. 평가
    metric = model.evaluate(x, y)
    print(f"\n📊 최종 평가 메트릭 (MSE): {metric:.6f}")

    # 6. 예측 확인
    y_pred = model.predict(x)
    print("🔍 예측 결과:")
    for i, (inp, pred) in enumerate(zip(x.reshape(4, 2), y_pred)):
        print(f"  입력 {inp.tolist()} → 예측: {pred[0]:.4f}")


if __name__ == "__main__":
    test_xor_classification()
