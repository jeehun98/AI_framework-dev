import os
import sys

# DLL 경로 추가
os.add_dll_directory("C:/msys64/mingw64/bin")
#os.add_dll_directory("C:/Users/as042/OneDrive/Desktop/AI_framework/AI_framework-dev/dev/backend/activations/")

# 경로 설정
sys.path.insert(0, 'C:/Users/as042/OneDrive/Desktop/AI_framework/AI_framework-dev')

# activations 모듈 임포트
try:
    from dev.backend.activations import activations
    print("activations 모듈 로드 성공")
except ImportError as e:
    print(f"ImportError 발생: {e}")

# 테스트 코드 실행
try:
    if 'activations' in locals():
        inputs = [0.5, -1.2, 3.0]  # 테스트용 입력값
        result, nodes = activations.relu(inputs)
        print("결과:", result)
        print("노드:", nodes)
    else:
        print("activations 모듈이 정의되지 않았습니다.")
except Exception as e:
    print(f"테스트 실행 중 에러 발생: {e}")


import numpy as np

# 예시 입력
inputs = np.array([[-1.0, 0.5, 2.0], [1.0, -0.5, 0.0]])

# ReLU 연산
result, nodes = activations.relu(inputs)
print("ReLU Result:", result)

for node in nodes:
    print(node.operation, node.output)
    for child_node in node.children:
        print(child_node.operation, node.output)

# Sigmoid 연산
result, nodes = activations.sigmoid(inputs)
print("Sigmoid Result:", result)

for node in nodes:
    print(node.operation, node.output)
    for child_node in node.children:
        print(child_node.operation, node.output)

# Tanh 연산
result, nodes = activations.tanh(inputs)
print("Tanh Result:", result)

for node in nodes:
    print(node.operation, node.output)
    for child_node in node.children:
        print(child_node.operation, node.output)

# Leaky ReLU 연산
result, nodes = activations.leaky_relu(inputs, alpha=0.01)
print("Leaky ReLU Result:", result)

for node in nodes:
    print(node.operation, node.output)
    for child_node in node.children:
        print(child_node.operation, node.output)

# Softmax 연산
result, nodes = activations.softmax(inputs)
print("Softmax Result:", result)

for node in nodes:
    print(node.operation, node.output)
    for child_node in node.children:
        print(child_node.operation, node.output)