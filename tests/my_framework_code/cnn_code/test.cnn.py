import sys
# 경로를 절대 경로로 변환하여 추가
sys.path.insert(0, 'C:/Users/owner/Desktop/AI_framework-dev')

import tensorflow as tf
from tensorflow import keras
from keras import datasets
from keras import to_categorical

import numpy as np
from dev.models.sequential import Sequential
from dev.layers.core.dense import Dense
from dev.layers.core.Conv2D import Conv2D

# 1. 데이터 로드 및 전처리
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# 데이터 정규화 (0~255 -> 0~1)
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

# 레이블을 원-핫 인코딩으로 변환
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model = Sequential()

model.add(Conv2D(32, (3,3)))