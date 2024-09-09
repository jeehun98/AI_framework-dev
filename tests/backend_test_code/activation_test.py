# backend 변환
import os
os.add_dll_directory("C:\\msys64\\mingw64\\bin")

import sys
# 경로를 절대 경로로 변환하여 추가
sys.path.insert(0, 'C:/Users/owner/Desktop/AI_framework-dev')

from dev.backend.activations import activations

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