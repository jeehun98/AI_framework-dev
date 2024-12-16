from keras.models import Sequential
from keras.layers import Dense, Flatten

# 2. Sequential 모델 생성
model = Sequential()

# 3. 모델에 레이어 추가
model.add(Flatten(input_shape=(784,)))  # 이미지를 1차원으로 평탄화
model.add(Dense(128, activation='relu'))  # 첫 번째 Dense 층
model.add(Dense(64, activation='relu'))   # 두 번째 Dense 층
model.add(Dense(10, activation='softmax'))  # 세 번째 Dense 층, 출력층

# input_shape 정보가 config에 저장되어 있는지 확인
a = model.get_config()
print("Model Configuration:\n", a)

# 4. 모델 컴파일
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

b = model.get_config()
print("Model Configuration after Compile:\n", b)

# 컴파일 설정 확인
compile_config = model.optimizer.get_config()
print("Compile Configuration:\n", compile_config)

# 빌드 정보는 Keras에서 직접 지원하지 않을 수 있음
# 대신 모델 요약이나 레이어 구성 정보를 사용
model.summary()
for layer in model.layers:
    print(f"Layer: {layer.name}, Config: {layer.get_config()}")
