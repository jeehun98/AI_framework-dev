import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# 1. 데이터 로드 및 전처리
diabetes_data = load_diabetes()
X = diabetes_data.data
y = diabetes_data.target

# 데이터셋을 학습용과 테스트용으로 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 특성 스케일링 (표준화)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 2. MLP 모델 정의
model = Sequential([
    Dense(64, input_shape=(X_train.shape[1],), activation='relu'),  # 첫 번째 은닉층
    Dense(64, activation='relu'),  # 두 번째 은닉층
    Dense(1)  # 출력층
])

# 3. 모델 컴파일
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# 4. 모델 학습
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2, verbose=1)

# 5. 모델 평가
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error on Test Set: {mse:.4f}")

# 6. 예측 예시
sample_data = X_test[0].reshape(1, -1)  # 테스트 데이터셋의 첫 번째 샘플
predicted_value = model.predict(sample_data)
print(f"Predicted value for sample data: {predicted_value[0][0]:.4f}")
print(f"Actual value for sample data: {y_test[0]:.4f}")
