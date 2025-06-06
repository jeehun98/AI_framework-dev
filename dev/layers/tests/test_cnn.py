import sys
import os

# 경로 등록
sys.path.insert(0, os.path.abspath("C:/Users/owner/Desktop/AI_framework-dev"))
sys.path.insert(0, os.path.abspath("C:/Users/owner/Desktop/AI_framework-dev/dev/backend/backend_ops/operaters"))

import cupy as cp
from dev.utils.activations import relu, sigmoid, tanh

from dev.models.sequential import Sequential
from dev.layers.conv2d import Conv2D
from dev.layers.pooling import MaxPooling2D
from dev.layers.flatten import Flatten
from dev.layers.dense import Dense

def test_cnn_model():
    # ✅ 입력 데이터 생성 (batch=2, 8x8, 채널=1)
    x = cp.random.rand(2, 8, 8, 1).astype(cp.float32)

    # ✅ 모델 구성
    model = Sequential()
    model.add(Conv2D(filters=4, kernel_size=3, activation=tanh, input_shape = (8, 8, 1 )))
    model.add(MaxPooling2D(pool_size=(2, 2), stride=2))
    model.add(Flatten())
    model.add(Dense(units=5, activation=sigmoid))
    model.add(Dense(units=1, activation=sigmoid))

    # ✅ 모델 컴파일
    model.compile(
        optimizer='sgd',
        loss='mse',
        p_metrics='mse',
        learning_rate=0.0005
    )

    # ✅ 순전파 수행
    out = model.forward_pass(x)
    print("최종 출력 shape:", out.shape)
    print("최종 출력 값:\n", out)

if __name__ == "__main__":
    test_cnn_model()
