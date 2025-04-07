# dev/backend/activations/tests/activation_test.py 또는 tests/backend_test_code/activation_test.py 등에서 사용 가능

import os
import sys
import numpy as np

# ✅ 프로젝트 루트 경로 수동 등록 (dev 상위 루트가 sys.path 에 들어가야 함)
current_path = os.path.abspath(__file__)
while True:
    current_path = os.path.dirname(current_path)
    if os.path.basename(current_path) == "AI_framework-dev":
        if current_path not in sys.path:
            sys.path.insert(0, current_path)
        break
    if current_path == os.path.dirname(current_path):
        raise RuntimeError("AI_framework-dev 루트를 찾을 수 없습니다.")

# ✅ 설정 적용
from dev.tests.test_setup import setup_paths
setup_paths()

# ✅ 활성화 함수 모듈 임포트
from dev.backend.activations import activations
print("✅ activations 모듈 로드 성공")



# === 테스트 입력 ===
inputs = np.array([[-1.0, 0.5, 2.0], [1.0, -0.5, 0.0]])

# ReLU
print("\n🔹 ReLU")
result, nodes = activations.relu(inputs)
print("Result:", result)
for node in nodes:
    print(node.operation, node.output)
    for child in node.children:
        print(" └─", child.operation, child.output)

# Sigmoid
print("\n🔹 Sigmoid")
result, nodes = activations.sigmoid(inputs)
print("Result:", result)
for node in nodes:
    print(node.operation, node.output)
    for child in node.children:
        print(" └─", child.operation, child.output)

# Tanh
print("\n🔹 Tanh")
result, nodes = activations.tanh(inputs)
print("Result:", result)
for node in nodes:
    print(node.operation, node.output)
    for child in node.children:
        print(" └─", child.operation, child.output)

# Leaky ReLU
print("\n🔹 Leaky ReLU")
result, nodes = activations.leaky_relu(inputs, alpha=0.01)
print("Result:", result)
for node in nodes:
    print(node.operation, node.output)
    for child in node.children:
        print(" └─", child.operation, child.output)

# Softmax
print("\n🔹 Softmax")
result, nodes = activations.softmax(inputs)
print("Result:", result)
for node in nodes:
    print(node.operation, node.output)
    for child in node.children:
        print(" └─", child.operation, child.output)
