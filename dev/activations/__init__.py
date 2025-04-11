# dev/activations/__init__.py

import numpy as np

import sys
import os
# ✅ 프로젝트 루트 (AI_framework-dev)를 sys.path에 삽입
cur = os.path.abspath(__file__)
while True:
    cur = os.path.dirname(cur)
    if os.path.basename(cur) == "AI_framework-dev":
        if cur not in sys.path:
            sys.path.insert(0, cur)
        break
    if cur == os.path.dirname(cur):
        raise RuntimeError("프로젝트 루트(AI_framework-dev)를 찾을 수 없습니다.")

# 이제 dev.tests.test_setup import가 가능해짐
from dev.tests.test_setup import setup_paths
setup_paths()

# ✅ activations_cuda 모듈 import
try:
    import activations_cuda
    print("✅ activations_cuda 모듈 로드 성공")
except ImportError as e:
    print("❌ activations_cuda import 실패:", e)
    sys.exit(1)

# ✅ CUDA 기반 래퍼 함수들
def relu(x):
    return activations_cuda.apply_activation(x.astype(np.float32), "relu")

def leaky_relu(x):
    return activations_cuda.apply_activation(x.astype(np.float32), "leaky_relu")

def sigmoid(x):
    return activations_cuda.apply_activation(x.astype(np.float32), "sigmoid")

def tanh(x):
    return activations_cuda.apply_activation(x.astype(np.float32), "tanh")

def softmax(x):
    return activations_cuda.apply_activation(x.astype(np.float32), "softmax")

# ✅ 이름 → 함수 딕셔너리
ALL_ACTIVATIONS_DICT = {
    "relu": relu,
    "leaky_relu": leaky_relu,
    "sigmoid": sigmoid,
    "tanh": tanh,
    "softmax": softmax,
}

def get(identifier):
    """
    활성화 함수 이름 또는 함수 객체를 받아 함수 반환
    """
    if isinstance(identifier, str):
        identifier = identifier.lower()
        activation_fn = ALL_ACTIVATIONS_DICT.get(identifier)
        if callable(activation_fn):
            return activation_fn

    if callable(identifier):
        return identifier

    raise ValueError(
        f"Invalid activation function identifier: '{identifier}'. "
        f"Available options: {', '.join(ALL_ACTIVATIONS_DICT.keys())}."
    )
