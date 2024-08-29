# 경로 지정...
import sys
sys.path.insert(0, 'C:/Users/owner/Desktop/AI_framework-dev')

from dev.models.sequential import Sequential
from dev.layers.core.dense import Dense
from dev.layers.flatten import Flatten

model = Sequential()

model.add(Flatten(input_shape=(784,)))
model.add(Dense(128, activation='relu'))

print(model.get_config())