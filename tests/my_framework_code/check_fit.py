import sys
# 경로를 절대 경로로 변환하여 추가
sys.path.insert(0, 'C:/Users/owner/Desktop/AI_framework-dev')

from dev.models.sequential import Sequential
from dev.layers.core.dense import Dense
from dev.layers.flatten import Flatten

import numpy as np

model = Sequential()

# 특성의 개수를 입력받는 Flatten
model.add(Flatten(input_shape=(4,)))
model.add(Dense(2, activation = "sigmoid"))

model.compile(optimizer='sgd',
              loss='categoricalcrossentropy',
              p_metrics='accuracy')

# 연산을 수행해보자잇~

# 데이터 입력은 numpy array 로 받기로 하자 (2, 4) 의 입력, 4개의 특성을 가진 2 개의 데이터
x = np.array(
    [[0.0186703 , 0.64554151, 0.18496826, 0.43432135],
    [0.15490974, 0.57481778, 0.77393513, 0.03186789]]
)

y = model.fit(x)

print(y[0], "\n\n",y[1])