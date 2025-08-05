import sys
import os
import ctypes
import numpy as np

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


def test_xor_classification_equivalent_to_pytorch():
    print("\n=== [TEST] XOR - PyTorch 동일 구조 테스트 ===")

    # XOR 입력 및 정답
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

    # 모델 구성: 동일한 구조
    model = Sequential(input_shape=(1, 1, 2))
    model.add(Flatten(input_shape=(1, 1, 2)))
    model.add(Dense(units=4, activation=None))             # Linear(2, 4)
    model.add(Activation("sigmoid"))                       # Sigmoid
    model.add(Dense(units=1, activation=None))             # Linear(4, 1)
    model.add(Activation("sigmoid"))                       # Sigmoid

    # 손실함수 및 옵티마이저: BCE + SGD(lr=0.1)
    model.compile(optimizer="sgd", loss="bce", learning_rate=0.1)

    # 학습
    model.fit(x, y, epochs=10000, batch_size=4)  # 전체 배치 학습

    # 평가
    metric = model.evaluate(x, y)
    print(f"\n📊 최종 평가 메트릭 (BCE): {metric:.6f}")

    # 예측 출력
    y_pred = model.predict(x)

    print("\n🔍 XOR 예측 결과:")
    print("====================================")
    print("  입력         |  정답  |  예측값")
    print("---------------|--------|----------")
    for i in range(len(x)):
        input_vals = x[i].reshape(-1).tolist()
        label_val = y[i][0]
        pred_val = y_pred[i][0]
        print(f"  {input_vals}  |   {label_val:.1f}   |  {pred_val:.4f}")
    print("====================================")


if __name__ == "__main__":
    test_xor_classification_equivalent_to_pytorch()
