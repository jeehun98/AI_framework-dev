# 필요한 라이브러리 임포트
import numpy as np
from keras import layers, models

# 간단한 데이터 생성 (1000개의 샘플, 32개의 특징)
x_train = np.random.random((1000, 32))  # 랜덤 입력 데이터
y_train = np.random.randint(10, size=(1000,))  # 0~9 범위의 랜덤 라벨
y_train = np.eye(10)[y_train]  # one-hot 인코딩

# 입력 정의 (입력 크기: 32)
inputs = layers.Input(shape=(32,))

# 첫 번째 Dense 레이어 (유닛 64, 활성화 함수 ReLU)
x = layers.Dense(64, activation='relu')(inputs)

# 두 번째 Dense 레이어 (유닛 64, 활성화 함수 ReLU)
x = layers.Dense(64, activation='relu')(x)  
# 출력 Dense 레이어 (유닛 10, 활성화 함수 소프트맥스)
outputs = layers.Dense(10, activation='softmax')(x)

# 모델 정의 (입력과 출력 설정)
model = models.Model(inputs=inputs, outputs=outputs)

# 모델 컴파일 (Adam 옵티마이저, categorical_crossentropy 손실 함수, 정확도 평가 지표)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 모델 학습 (배치 크기: 32, 에포크: 10)
model.fit(x_train, y_train, batch_size=32, epochs=10)

# 모델 구조 출력
model.summary()
