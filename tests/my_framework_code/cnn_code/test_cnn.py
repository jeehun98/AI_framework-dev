import sys
# 경로를 절대 경로로 변환하여 추가
sys.path.insert(0, 'C:/Users/owner/Desktop/AI_framework-dev')

import numpy as np
from dev.models.sequential import Sequential
from dev.layers.core.dense import Dense
from dev.layers.core.Conv2D import Conv2D
from dev.layers.pooling import Pooling
from dev.layers.flatten import Flatten

from tensorflow.keras.utils import to_categorical

np.random.seed(42)

# 입력 데이터 생성 (10개의 샘플, 7x7 크기, 1 채널)
num_samples = 10
input_shape = (7, 7, 1)
input_data = np.random.rand(num_samples, *input_shape)

# 타겟 데이터 생성 (0~4 사이의 라벨)
num_classes = 5
target_data = np.random.randint(0, num_classes, size=(num_samples,))

# 타겟 데이터를 원-핫 인코딩
target_data_one_hot = to_categorical(target_data, num_classes=num_classes)

model = Sequential()

# input_shape 를 어떻게 입력해야할까    
model.add(Conv2D(7, (3,3), input_shape=(7, 7, 1)))
model.add(Pooling())
model.add(Conv2D(14, (3,3)))
model.add(Pooling())

# 이후 dense 층 추가하기
model.add(Flatten())
model.add(Dense(5, activation="sigmoid"))

model.compile(optimizer='sgd',
              loss='mse',
              p_metrics='mse',
              learning_rate=0.001)

model.fit(input_data, target_data_one_hot, epochs=1)
print("완료")