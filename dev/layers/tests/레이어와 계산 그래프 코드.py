import sys
import os

# 경로 등록
sys.path.insert(0, os.path.abspath("C:/Users/owner/Desktop/AI_framework-dev"))
sys.path.insert(0, os.path.abspath("C:/Users/owner/Desktop/AI_framework-dev/dev/backend/backend_ops/operaters"))

import numpy as np
from dev.models.sequential_mat import SequentialMat
from dev.layers.dense_mat import DenseMat
from dev.layers.activation_mat import ActivationMat

# ✅ 입력 차원 정의
input_dim = 4

# ✅ 테스트용 모델 생성
model = SequentialMat()

# ✅ 레이어 추가
model.add(DenseMat(units=5, input_dim=input_dim))  # 입력 → Dense(5)
model.add(ActivationMat("sigmoid"))                # → Sigmoid
model.add(DenseMat(units=3))                       # → Dense(3)
model.add(ActivationMat("relu"))                   # → ReLU

# ✅ 컴파일 (compile_model 내용 포함됨)
model.compile(
    optimizer='sgd',
    loss='mse',
    p_metrics='mse',
    learning_rate=0.01
)

# ✅ 컴파일 결과 출력
graph = model.graph_ir

print("\n📊 [Test] 컴파일된 그래프 정보:")
print(" - 총 노드 수:", graph["TotalNodes"])
print(" - 출력 노드 IDs:", graph["OutputIDs"])
print(" - 연산자별 노드 분포:", {k: len(v) for k, v in graph["OpTypeNodeMap"].items()})
print(" - Conn 행렬 shape:", graph["Conn"].shape)
print(" - OpType 벡터 shape:", graph["OpType"].shape)
