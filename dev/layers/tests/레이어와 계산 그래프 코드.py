import sys
import os

# 경로 등록
sys.path.insert(0, os.path.abspath("C:/Users/owner/Desktop/AI_framework-dev"))
sys.path.insert(0, os.path.abspath("C:/Users/owner/Desktop/AI_framework-dev/dev/backend/backend_ops/operaters"))

import numpy as np

# ✅ 모델 및 레이어 임포트
from dev.models.sequential import Sequential
from dev.layers.dense2 import Dense
from dev.layers.flatten import Flatten
from dev.layers.activations import Activation

# ✅ 랜덤 시드 고정
np.random.seed(42)

# ✅ 입력/출력 데이터 생성
x = np.random.rand(10, 4)
y = np.random.rand(10, 2)

# ✅ 모델 생성 및 레이어 추가
model = Sequential()
model.add(Flatten(input_shape=(4,)))
model.add(Dense(10))
model.add(Activation('sigmoid'))
model.add(Dense(2))
model.add(Activation('sigmoid'))

print("✅ 모델 레이어 추가 완료")

# ✅ 모델 컴파일
model.compile(
    optimizer='sgd',
    loss='mse',
    p_metrics='mse',
    learning_rate=0.001
)

print("✅ 컴파일 완료")

print(model.cal_graph.node_list, "계산 그래프 확인")
model.cal_graph.print_graph()

# ✅ 학습 실행
model.fit(x, y, epochs=1, batch_size=32)

print("✅ 끝")
