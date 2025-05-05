# 🧪 tests/sequential_mat_test.py

import sys, os
import numpy as np

# 프로젝트 루트 등록
sys.path.insert(0, os.path.abspath("C:/Users/owner/Desktop/AI_framework-dev"))
sys.path.insert(0, os.path.abspath("C:/Users/owner/Desktop/AI_framework-dev/dev/backend/backend_ops/operaters"))

from dev.models.sequential_mat import SequentialMat
from dev.layers.dense_mat import DenseMat
from dev.layers.activation_mat import ActivationMat

# ✅ 랜덤 시드 고정
np.random.seed(42)

# ✅ 입력/출력 데이터 생성
x = np.random.rand(1, 4).astype(np.float32)  # (1, 4)
y = np.random.rand(1, 3).astype(np.float32)  # (1, 3)

# ✅ 모델 생성
model = SequentialMat()
model.add(DenseMat(units=5, activation=None, input_dim=4))  # 첫 Dense (activation 없음)
model.add(ActivationMat("sigmoid"))                            # 별도 Activation
model.add(DenseMat(units=3))                                # 출력층
model.add(ActivationMat("sigmoid"))

# ✅ 모델 컴파일
model.compile(optimizer="sgd", loss="mse", p_metrics="mse", learning_rate=0.01)

# ✅ 순전파 테스트
y_pred = model.predict(x)

# ✅ 결과 출력
print("입력값:\n", x)
print("출력 예측값:\n", y_pred)
