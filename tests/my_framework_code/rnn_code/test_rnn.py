import sys
# 경로를 절대 경로로 변환하여 추가
sys.path.insert(0, 'C:/Users/owner/Desktop/AI_framework-dev')

import numpy as np

from dev.models.sequential import Sequential
from dev.layers.Rnn import RNN
from dev.layers.dense import Dense
from dev.graph_engine.node import Node

np.random.seed(42)

# 데이터 생성
timesteps = 3  # 입력 데이터의 길이
features = 4    # 입력 데이터의 차원
samples = 10   # 샘플 수

# 무작위 데이터 생성 (10, 3, 4)
x_train = np.random.random((samples, timesteps, features)).astype(np.float64)
y_train = np.random.random((samples, 1)).astype(np.float64)

model = Sequential()

# 은닉 유닛의 개수...
model.add(RNN(5, activation="sigmoid", input_shape=(timesteps, features), use_bias=True))
model.add(Dense(1))

model.compile(optimizer='sgd', loss='mse', p_metrics='mse', learning_rate=0.001)

model.fit(x_train, y_train, epochs=1)

print(model.node_list[0])

model.node_list[0].print_tree()
