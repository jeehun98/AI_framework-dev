import tensorflow as tf
from tensorflow import keras
from keras import datasets, layers, models
from keras import to_categorical

print(tf.__version__, "버전확인")

# 1. 데이터 로드 및 전처리
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# 데이터 정규화 (0~255 -> 0~1)
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

# 레이블을 원-핫 인코딩으로 변환
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 2. CNN 모델 구성
model = models.Sequential()

# 첫 번째 Convolutional Layer
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))

# 두 번째 Convolutional Layer
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# 세 번째 Convolutional Layer
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Fully Connected Layer
model.add(layers.Flatten())  # Flatten the output to 1D vector
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))  # 10개의 클래스를 예측

# 3. 모델 컴파일
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 4. 모델 학습
model.fit(train_images, train_labels, epochs=1, batch_size=64, validation_split=0.2)

# 5. 모델 평가
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'\n테스트 정확도: {test_acc:.4f}')
