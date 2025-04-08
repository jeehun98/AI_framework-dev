# 🧪 tests/backend_test_code/activation_test
import os, sys

import numpy as np

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

# ✅ 테스트 입력
x = np.array([-1.0, 2.5, -3.0, 0.0, 4.2, -0.5, 1.3, -2.1, 3.6, -4.5], dtype=np.float32)
print("\n🧪 입력:", x)

# ✅ ReLU
relu_result = activations_cuda.apply_activation(x, "relu")
print("🔹 ReLU 결과:", relu_result)

# ✅ Sigmoid
sigmoid_result = activations_cuda.apply_activation(x, "sigmoid")
print("🔹 Sigmoid 결과:", sigmoid_result)

# ✅ Tanh
tanh_result = activations_cuda.apply_activation(x, "tanh")
print("🔹 Tanh 결과:", tanh_result)
