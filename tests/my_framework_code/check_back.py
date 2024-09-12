import sys
# 경로를 절대 경로로 변환하여 추가
sys.path.insert(0, 'C:/Users/owner/Desktop/AI_framework-dev')

from dev.models.sequential import Sequential
from dev.layers.core.dense import Dense
from dev.layers.flatten import Flatten

import numpy as np

# 랜덤 시드값 고정
np.random.seed(42)

model = Sequential()

# input_shape 는 특성의 개수임
model.add(Flatten(input_shape=(4,)))
model.add(Dense(4, 'sigmoid'))
model.add(Dense(2, 'sigmoid'))

model.compile(optimizer='sgd',
              loss='mse',
              p_metrics='mse')

#print(model.get_weight())
# 연산을 수행해보자잇~

x = np.random.rand(10,4)
y = np.random.rand(10,2)

model.fit(x, y)
print("완료")