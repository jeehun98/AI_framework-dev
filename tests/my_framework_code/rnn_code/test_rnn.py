import sys
# 경로를 절대 경로로 변환하여 추가
sys.path.insert(0, 'C:/Users/owner/Desktop/AI_framework-dev')

import numpy as np

from dev.models.sequential import Sequential
from dev.layers.core.Rnn import RNN

np.random.seed(42)

# 데이터 생성
timesteps = 2  # 입력 데이터의 길이
features = 3    # 입력 데이터의 차원
samples = 10   # 샘플 수

# 무작위 데이터 생성
x_train = np.random.random((samples, timesteps, features))
y_train = np.random.random((samples, 1))

model = Sequential()

# 은닉 유닛의 개수...
model.add(RNN(4, activation="sigmoid", input_shape=(timesteps, features), use_bias=True))

model.fit(x_train, y_train, epochs=1)
print("완료")