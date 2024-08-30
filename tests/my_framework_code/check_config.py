# 경로 지정...
import sys
sys.path.insert(0, 'C:/Users/owner/Desktop/AI_framework-dev')

from dev.models.sequential import Sequential
from dev.layers.core.dense import Dense
from dev.layers.flatten import Flatten

model = Sequential()

model.add(Flatten(input_shape=(784,)))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='sgd',
              loss='categoricalcrossentropy',
              p_metrics='accuracy')

#print("컴파일 콘피그 확인")
print(model.get_compile_config())

#print("콘피그 확인")
print(model.get_config())

#print("빌드 콘피그 확인")
print(model.get_build_config())

print(model.get_weight()[0].shape)