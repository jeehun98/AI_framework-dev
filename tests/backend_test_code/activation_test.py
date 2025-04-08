# ✅ tests/backend_test_code/activation_test.py

import os
import sys
import numpy as np

# ✅ 프로젝트 루트(AI_framework-dev)를 sys.path에 명시적으로 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ✅ test_setup 경로 활성화 (dev/tests/test_setup.py)
from dev.tests.test_setup import setup_paths
setup_paths()

# ✅ 활성화 함수 모듈 임포트
from dev.backend.backend_ops.activations import activations
print("✅ activations 모듈 로드 성공")


# ✅ 입력 데이터
inputs = np.array([[-1.0, 0.5, 2.0], [1.0, -0.5, 0.0]], dtype=np.float32)

# ✅ 활성화 함수 테스트 루틴
def run_activation_test(name, func):
    print(f"\n🔹 {name}")
    result, nodes = func(inputs)
    print("Result:", result)
    for node in nodes:
        print(f"🧱 {node.operation} → {node.output}")
        for child in node.children:
            print(f"   └─ {child.operation} → {child.output}")

# ✅ 개별 테스트 실행
run_activation_test("ReLU", activations.relu)
run_activation_test("Sigmoid", activations.sigmoid)
run_activation_test("Tanh", activations.tanh)
run_activation_test("Leaky ReLU", lambda x: activations.leaky_relu(x, alpha=0.01))
run_activation_test("Softmax", activations.softmax)
