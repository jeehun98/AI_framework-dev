from keras.src.models import Sequential
from keras.src.layers import Dense, Flatten

# 2. Sequential 모델 생성
model = Sequential()

# 3. 모델에 레이어 추가
model.add(Flatten(input_shape=(784,)))  # 이미지를 1차원으로 평탄화
model.add(Dense(128, activation='relu'))  # 첫 번째 Dense 층
model.add(Dense(64, activation='relu'))   # 두 번째 Dense 층
model.add(Dense(10, activation='softmax'))  # 세 번째 Dense 층, 출력층

# 4. 모델 컴파일
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#print(model.get_config())
#print(model.get_compile_config())
print(model.get_build_config())
