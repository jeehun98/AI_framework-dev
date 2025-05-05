import sys
import os
import numpy as np

# 경로 설정
sys.path.insert(0, os.path.abspath("C:/Users/owner/Desktop/AI_framework-dev"))
sys.path.insert(0, os.path.abspath("C:/Users/owner/Desktop/AI_framework-dev/dev/backend/backend_ops/operaters"))

# ✅ 모델 및 레이어 임포트
from dev.models.sequential_mat import SequentialMat
from dev.layers.dense_mat import DenseMat

# ✅ 랜덤 시드 고정
np.random.seed(42)

# ✅ 입력/출력 데이터 생성
x = np.random.rand(1, 4).astype(np.float32)
y = np.random.rand(1, 3).astype(np.float32)  # 출력 유닛 수 = 3

# ✅ 모델 생성 및 레이어 추가
model = SequentialMat()
model.add(DenseMat(input_dim=4, output_dim=10, activation="sigmoid"))
model.add(DenseMat(input_dim=10, output_dim=3, activation="sigmoid"))

# ✅ 모델 컴파일
model.compile(
    optimizer="sgd",
    loss="mse",
    p_metrics="mse",
    learning_rate=0.001
)

# ✅ 학습 실행
model.fit(x, y, epochs=1, batch_size=1)
