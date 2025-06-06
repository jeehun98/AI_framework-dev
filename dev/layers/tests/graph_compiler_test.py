
import sys, os
import numpy as np

# 프로젝트 루트 등록
sys.path.insert(0, os.path.abspath("C:/Users/owner/Desktop/AI_framework-dev"))
sys.path.insert(0, os.path.abspath("C:/Users/owner/Desktop/AI_framework-dev/dev/backend/backend_ops/operaters"))


import cupy as cp
from dev.models.sequential import Sequential
from dev.layers.dense import Dense
from dev.layers.activation_layer import Activation


def test_sequential_compile_graph():
    print("🧪 테스트 시작: Sequential.compile_graph()")

    model = Sequential()
    model.add(Dense(units=4, input_shape=(1, 5), use_backend_init=True))
    model.add(Dense(units=3, input_shape=(1, 4), use_backend_init=True))

    model.compile()  # 내부에서 self.compiler 생성됨

    compiled = model.compiler.compile_plan(use_backend_init=True)

    print("✅ 컴파일 완료! E 행렬:")
    print(compiled["E"])
    print("W_shapes:", compiled["W_shapes"])
    print("b_shapes:", compiled["b_shapes"])



if __name__ == "__main__":
    test_sequential_compile_graph()
