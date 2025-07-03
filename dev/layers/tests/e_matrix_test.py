import sys
import os

# 경로 등록
sys.path.insert(0, os.path.abspath("C:/Users/owner/Desktop/AI_framework-dev"))


import numpy as np
from dev.models.sequential import Sequential
from dev.layers.dense import Dense
from dev.layers.activation_layer import Activation
from dev.layers.flatten import Flatten

def test_e_matrix_generation():
    print("\n=== [TEST] E 행렬 생성 확인 ===")

    # ✅ 간단한 모델 정의
    model = Sequential(input_shape=(1, 2, 2))  # 예: 2x2 이미지 입력
    model.add(Flatten(input_shape=(1, 2, 2)))
    model.add(Dense(units=3, activation=None))
    model.add(Activation("relu"))

    # ✅ 모델 컴파일 (E 행렬 생성)
    model.compile(optimizer="sgd", loss="mse")

    # ✅ E 행렬, W, b 출력
    print("\n📐 [E 행렬]")
    for i, op in enumerate(model.E):
        print(f"{i+1:02d}: {op}")

    print("\n🧱 [Weights]")
    for k, v in model.weights.items():
        print(f"{k}: shape={v.shape}")

    print("\n🧈 [Biases]")
    for k, v in model.biases.items():
        print(f"{k}: shape={v.shape}")

    print("\n✅ 최종 출력 변수:", model.output_var)

def load_and_print_npz(filename="compiled_graph.npz"):
    print(f"\n=== [TEST] {filename} 내용 확인 ===")
    data = np.load(filename, allow_pickle=True)

    E = data['E']
    weights = data['weights'].item()
    biases = data['biases'].item()

    print("\n📐 [E 행렬 from npz]")
    for i, op in enumerate(E):
        print(f"{i+1:02d}: {op}")

    print("\n🧱 [Weights from npz]")
    for k, v in weights.items():
        print(f"{k}: shape={v.shape}, type={type(v)}")

    print("\n🧈 [Biases from npz]")
    for k, v in biases.items():
        print(f"{k}: shape={v.shape}, type={type(v)}")

if __name__ == "__main__":
    test_e_matrix_generation()
    print("다음")
    load_and_print_npz()


