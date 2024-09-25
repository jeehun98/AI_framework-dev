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

print("compile_config : ", model.get_compile_config())

print("\n\nget_config : ", model.get_config())

print("\n\nbuild_config : ", model.get_build_config())

print("\n\nweigh_shape : ", model.get_weight()[0].shape)

# model.fit(x_train, y_train, epochs=1, batch_size=32, validation_split=0.2)

# test_loss, test_acc = model.evaluate(x_test, y_test)

# predictions = model.predict(x_test)