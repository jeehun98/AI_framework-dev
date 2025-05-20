import numpy as np
from dev.layers.dense_mat import DenseMat  # 실제 경로에 맞게 조정 필요

# 테스트 입력 (1, 3)
input_data = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)

# DenseMat 레이어 생성
dense = DenseMat(input_dim=3, output_dim=4, activation='sigmoid', initializer='he')

# forward 호출
output = dense.call(input_data)

# 결과 출력
print("입력 데이터:")
print(input_data)
print("\n출력값:")
print(output)
print("\n출력 shape:", output.shape)
