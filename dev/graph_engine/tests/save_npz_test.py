import sys, os
import numpy as np

# 프로젝트 루트 등록
sys.path.insert(0, os.path.abspath("C:/Users/owner/Desktop/AI_framework-dev"))
sys.path.insert(0, os.path.abspath("C:/Users/owner/Desktop/AI_framework-dev/dev/backend/backend_ops/operaters"))



import numpy as np
import cupy as cp
from dev.models.sequential import Sequential
from dev.layers.dense import Dense
from dev.layers.flatten import Flatten
from dev.layers.activation_layer import Activation
from dev.layers.Dropout import Dropout
from dev.layers.BatchNormalization import BatchNormalization
from dev.layers.conv2d import Conv2D

def test_compile_and_save_npz():
    print("🧪 테스트 시작: compile_and_save_npz")

    # ✅ 모델 구성
    model = Sequential()

    # 입력: (1, 4, 4, 1) 이미지 예시 → Flatten → Dense → Activation → Dropout → Dense
    model.add(Flatten(input_shape=(4, 4, 1)))
    model.add(Dense(units=8, input_shape=(1, 16), use_backend_init=True))
    model.add(Activation("relu", use_backend_init=True))
    model.add(Dropout(rate=0.3, use_backend_init=True))
    model.add(Dense(units=4, input_shape=(1, 8), use_backend_init=True))
    model.add(BatchNormalization(use_backend_init=True))
    model.add(Activation("sigmoid", use_backend_init=True))

    # ✅ 컴파일
    model.compile()
    compiled = model.compiler.compile_plan(use_backend_init=True)

    # ✅ 정보 확인
    print("📊 E 행렬 (연산 흐름):")
    print(compiled["E"])
    print("📐 W_shapes:", compiled.get("W_shapes", "N/A"))
    print("📐 b_shapes:", compiled.get("b_shapes", "N/A"))
    print("🔢 input_node:", compiled["input_node"])
    print("🔢 output_node:", compiled["output_node"])

    # ✅ 저장
    npz_path = "compiled_graph.npz"
    model.compiler.save_to_npz(npz_path, use_backend_init=True)
    print(f"✅ 저장 완료: {npz_path}")

if __name__ == "__main__":
    test_compile_and_save_npz()
