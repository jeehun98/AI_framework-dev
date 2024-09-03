from keras.src.models import Sequential
from keras.src.layers import Dense, Flatten
from keras.src.datasets import mnist
from keras.src.utils import to_categorical

# 1. 데이터셋 로드 및 전처리
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 이미지 데이터를 (28, 28)에서 (784,)로 변환
x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255

# 레이블을 원-핫 인코딩
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

print(x_train.shape, y_train.shape)

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

# 5. 모델 학습
model.fit(x_train, y_train, epochs=1, batch_size=32, validation_split=0.2)

# 6. 모델 평가
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc:.4f}')

# 7. 예측 수행
predictions = model.predict(x_test)

print(model.get_build_config())
model.get_weights