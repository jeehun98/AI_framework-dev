import numpy as np
from keras.src import Sequential
from keras.src.layers import LSTM, Dense

# 데이터 생성
timesteps = 10  # 타임 스텝 수
features = 1    # 특성 수
samples = 100   # 샘플 수

# 무작위 데이터 생성
x_train = np.random.random((samples, timesteps, features))
y_train = np.random.random((samples, 1))

# RNN 모델 생성
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(timesteps, features)))
model.add(Dense(1))

# 모델 컴파일 (평가지표 mae 추가)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 모델 학습
model.fit(x_train, y_train, epochs=20, batch_size=32)

# 모델 평가
loss, mae = model.evaluate(x_train, y_train)
print(f"손실(MSE): {loss:.4f}")
print(f"평균 절대 오차(MAE): {mae:.4f}")

# 모델 요약 출력
model.summary()
