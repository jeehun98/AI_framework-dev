import sys
import os

# 경로 등록
sys.path.insert(0, os.path.abspath("C:/Users/owner/Desktop/AI_framework-dev"))

import numpy as np
from dev.layers.dense import Dense
from dev.layers.activation_layer import Activation
from dev.models.sequential import Sequential

def test_cuda_fit():
    print("\n🎯 [TEST] CUDA 기반 Sequential 모델 훈련 시작")

    # 🎯 입력 및 타겟 데이터 정의 (선형 관계 예시: y = x1 + x2)
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([[0], [1], [1], [2]], dtype=np.float32)

    # ✅ Sequential 모델 정의
    model = Sequential(input_shape=(1, 2))
    model.add(Dense(units=4, activation=None, input_shape=(1, 2)))
    model.add(Activation("relu"))
    model.add(Dense(units=1, activation=None))
    model.add(Activation("relu"))

    # ✅ 컴파일 및 학습
    model.compile(optimizer="sgd", loss="mse", p_metrics="mse", learning_rate=0.5)
    model.fit(x, y, epochs=2, batch_size=1)

    # ✅ 예측 확인
    print("\n🚀 예측 결과:")
    for i in range(len(x)):
        pred = model.run_forward(x[i:i+1])
        print(f"x={x[i]} => y_pred={pred[0][0]:.4f}, y_true={y[i][0]}")

if __name__ == "__main__":
    test_cuda_fit()
