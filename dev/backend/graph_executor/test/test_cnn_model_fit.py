import sys
import os
import ctypes
import numpy as np
import cupy as cp

# CUDA DLL 명시적 로드
ctypes.CDLL(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\cudart64_12.dll")

# Pybind11 빌드된 .pyd 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), "build", "lib.win-amd64-cpython-312"))

# 프로젝트 루트 등록
sys.path.insert(0, os.path.abspath("C:/Users/owner/Desktop/AI_framework-dev"))
sys.path.append("C:/Users/owner/Desktop/AI_framework-dev/dev/backend/graph_executor/test")

import numpy as np
from dev.models.sequential import Sequential
from dev.layers.Conv2D import Conv2D
from dev.layers.flatten import Flatten
from dev.layers.dense import Dense
from dev.layers.activation_layer import Activation

def test_cnn_model_stride_padding():
    model = Sequential()
    model.add(Conv2D(filters=2, kernel_size=(3, 3), stride=(2, 2), padding='same',
                     input_shape=(1, 4, 4, 1), use_bias=True))
    model.add(Flatten())
    model.add(Dense(1, use_bias=True))
    model.add(Activation('sigmoid'))

    model.compile(learning_rate=0.01)

    # ✅ 더 큰 입력 데이터: (B, H, W, C) = (1, 4, 4, 1)
    x_train = np.array([[[[1], [0], [1], [0]],
                         [[0], [1], [0], [1]],
                         [[1], [1], [0], [0]],
                         [[0], [0], [1], [1]]]], dtype=np.float32)

    y_train = np.array([[1.0]], dtype=np.float32)

    print("\n=== [TEST] CNN 모델 학습 테스트 (stride/padding 포함) ===")
    model.fit(x_train, y_train, epochs=3, batch_size=1)

if __name__ == "__main__":
    test_cnn_model_stride_padding()
