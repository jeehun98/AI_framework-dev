import sys
import os

# 경로 등록
sys.path.insert(0, os.path.abspath("C:/Users/owner/Desktop/AI_framework-dev"))
sys.path.insert(0, os.path.abspath("C:/Users/owner/Desktop/AI_framework-dev/dev/backend/backend_ops/operaters"))

import numpy as np

# ✅ 모델 및 레이어 임포트
from dev.models.sequential import Sequential
from dev.layers.dense import Dense
from dev.layers.flatten import Flatten
from dev.layers.activation_layer import Activation

# ✅ 랜덤 시드 고정
np.random.seed(47)

output_unit_count = 3


# ✅ 입력/출력 데이터 생성
x = np.random.rand(1, 4)
y = np.random.rand(1, output_unit_count)

# ✅ 모델 생성 및 레이어 추가
model = Sequential()
model.add(Flatten(input_shape=(4,)))
model.add(Dense(10, initializer="xavier"))
model.add(Activation('sigmoid'))
model.add(Dense(output_unit_count, initializer="xavier"))
model.add(Activation('sigmoid'))

# ✅ 모델 컴파일
model.compile(
    optimizer='sgd',
    loss='mse',
    p_metrics='mse',
    learning_rate=0.0005
)

model.compile_graph()
print("📊 컴파일된 그래프 연산 목록:")
for idx, op in enumerate(model.graph_ops):
    print(f"{idx:02d}: {op}")


# ✅ 학습 실행  
model.fit(x, y, epochs=100, batch_size=32)

print("✅ 끝")
