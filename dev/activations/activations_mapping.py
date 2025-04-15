# dev/activations/activations_mapping.py
# 활성화 함수의 종류 별 호출을 담당

import os
os.add_dll_directory("C:\\msys64\\mingw64\\bin")  # DLL 경로 등록

from dev.backend.backend_ops.activations import activations  # CUDA Pybind11 모듈 (복수 개 함수 포함)

# ✅ 계산 그래프 버전 (Node 리스트 반환 포함)
def relu(x, node_list=[]):
    return activations.relu(x, node_list)

def leaky_relu(x, alpha=0.01, node_list=[]):
    return activations.leaky_relu(x, alpha, node_list)

def sigmoid(x, node_list=[]):
    return activations.sigmoid(x, node_list)

def tanh(x, node_list=[]):
    return activations.tanh_activation(x, node_list)

def softmax(x, node_list=[]):
    return activations.softmax(x, node_list)
