import sys
# 경로를 절대 경로로 변환하여 추가
sys.path.insert(0, 'C:/Users/owner/Desktop/AI_framework-dev')

import numpy as np
from sklearn.datasets import load_diabetes
from dev.models.sequential import Sequential
from dev.layers.core.dense import Dense
from dev.layers.flatten import Flatten


def load_diabetes_data():
    # 당뇨병 데이터셋 불러오기
    diabetes_data = load_diabetes()
    
    # 특성과 타겟을 넘파이 배열로 변환
    X = diabetes_data.data  # 특성 데이터 (442, 10)
    y = diabetes_data.target  # 타겟 데이터 (442,)
    
    return X, y

# 함수 호출 예시
X, y = load_diabetes_data()

X = X[:50]

# 랜덤 시드값 고정
np.random.seed(42)

model = Sequential()
# input_shape 는 특성의 개수임
model.add(Flatten(input_shape=(10,)))
model.add(Dense(6, 'sigmoid'))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(1))

model.compile(optimizer='sgd',
              loss='mse',
              p_metrics='mse')

model.fit(X, y, epochs=5)
print("완료")