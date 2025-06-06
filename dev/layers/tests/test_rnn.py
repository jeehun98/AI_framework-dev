import sys
import os

# 경로 등록
sys.path.insert(0, os.path.abspath("C:/Users/owner/Desktop/AI_framework-dev"))
sys.path.insert(0, os.path.abspath("C:/Users/owner/Desktop/AI_framework-dev/dev/backend/backend_ops/operaters"))

import cupy as cp
from dev.utils.activations import tanh, sigmoid
from dev.models.sequential import Sequential
from dev.layers.rnn import RNN
from dev.layers.dense import Dense

def test_rnn_model():
    # ✅ 입력 데이터 생성: (batch=2, time=5, input_dim=3)
    x = cp.random.rand(2, 5, 3).astype(cp.float32)

    # ✅ 모델 구성
    model = Sequential()
    model.add(RNN(units=4, activation=tanh, input_shape=(5, 3)))
    model.add(Dense(units=2, activation=sigmoid))
    model.add(Dense(units=1, activation=sigmoid))

    # ✅ 모델 컴파일
    model.compile(
        optimizer='sgd',
        loss='mse',
        p_metrics='mse',
        learning_rate=0.001
    )

    # ✅ 순전파 수행
    out = model.forward_pass(x)
    print("최종 출력 shape:", out.shape)
    print("최종 출력 값:\n", out)

if __name__ == "__main__":
    test_rnn_model()
