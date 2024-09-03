import sys
# 경로를 절대 경로로 변환하여 추가
sys.path.insert(0, 'C:/Users/owner/Desktop/AI_framework-dev')

from dev.models.sequential import Sequential
from dev.layers.core.dense import Dense
from dev.layers.flatten import Flatten

import numpy as np

model = Sequential()

model.add(Flatten(input_shape=(4,)))
model.add(Dense(2, activation = "sigmoid"))

model.compile(optimizer='sgd',
              loss='categoricalcrossentropy',
              p_metrics='accuracy')

# 연산을 수행해보자잇~

x = np.array(
    [[0.0186703 , 0.64554151, 0.18496826, 0.43432135],
    [0.15490974, 0.57481778, 0.77393513, 0.03186789]]
)

print(model.fit(x))