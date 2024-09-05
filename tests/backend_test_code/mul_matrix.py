# backend 변환
import os
os.add_dll_directory("C:\\msys64\\mingw64\\bin")

import sys
# 경로를 절대 경로로 변환하여 추가
sys.path.insert(0, 'C:/Users/owner/Desktop/AI_framework-dev')

from dev.backend.operaters import operations_matrix

import numpy as np

# 두 행렬 A와 B 정의
A = np.array([[1.0, 2.0], [3.0, 4.0]])
B = np.array([[5.0, 6.0], [7.0, 8.0]])

# matrix_multiply 호출
result, node_list = operations_matrix.matrix_multiply(A, B)

# 결과 행렬 출력
print("Result Matrix:")
print(result)

# 덧셈 노드와 자식 노드(곱셈 노드) 정보 출력
print("\nSum Node List and Child (Multiply) Nodes:")
for sum_node in node_list:
    print(f"Sum Node - Operation: {sum_node.operation}, Result: {sum_node.output}")
    for mul_node in sum_node.children:
        print(mul_node)
    