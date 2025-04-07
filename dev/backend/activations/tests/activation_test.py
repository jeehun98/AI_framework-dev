import os, sys
import numpy as np

build_path = r"C:\Users\owner\Desktop\AI_framework-dev\dev\backend\activations\build\lib.win-amd64-cpython-312"
sys.path.append(build_path)

os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin")

print("sys.path에 다음이 포함되어야 함:")
print(build_path)
print("파일 존재 여부:", os.path.exists(os.path.join(build_path, "activations_cuda.cp312-win_amd64.pyd")))

import activations_cuda

x = np.array([-1.0, 2.5, -3.0, 0.0, 4.2, -0.5, 1.3, -2.1, 3.6, -4.5], dtype=np.float32)
print("입력:", x)

relu_result = activations_cuda.apply_activation(x, "relu")
print("ReLU 결과:", relu_result)

sigmoid_result = activations_cuda.apply_activation(x, "sigmoid")
print("Sigmoid 결과:", sigmoid_result)

tanh_result = activations_cuda.apply_activation(x, "tanh")
print("Tanh 결과:", tanh_result)
