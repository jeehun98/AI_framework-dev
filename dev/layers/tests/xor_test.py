import sys, os
import numpy as np

# CUDA DLL 경로 (임포트 전에 추가)
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin")

# Pybind11로 빌드된 .pyd 경로
sys.path.append(os.path.join(os.path.dirname(__file__), "build", "lib.win-amd64-cpython-312"))

# AI framework 루트 경로
sys.path.insert(0, os.path.abspath("C:/Users/owner/Desktop/AI_framework-dev"))
sys.path.append("C:/Users/owner/Desktop/AI_framework-dev/dev/backend/graph_executor/test")

# AI Framework
from dev.models.sequential import Sequential
from dev.layers.dense import Dense
from dev.layers.activation_layer import Activation
from dev.layers.flatten import Flatten


def test_xor():
    print("\n=== [TEST] XOR - BCE on probs ===")
    np.random.seed(42)

    # XOR 데이터 (NCHW: B,C,H,W)
    x = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32).reshape(4, 1, 1, 2)
    y = np.array([[0],[1],[1],[0]], dtype=np.float32)

    model = Sequential(input_shape=(1, 1, 2))
    model.add(Flatten(input_shape=(1, 1, 2)))
    model.add(Dense(units=4, activation=None, initializer="xavier"))
    model.add(Activation("tanh"))
    model.add(Dense(units=1, activation=None, initializer="xavier"))
    model.add(Activation("sigmoid"))

    model.compile(optimizer="sgd", loss="bce", learning_rate=0.3)

    print("\n=== [Graph E] ===")
    for i, op in enumerate(model.E):
        print(f"[{i}] type={op.op_type}, input={op.input_id}, output={op.output_id}")

    print("\n[BEFORE] evaluate on full batch")
    loss_before = model.evaluate(x, y)  # 현재 evaluate는 loss(float) 반환
    print(f"  BCE(before): {loss_before:.6f}")

    model.fit(x, y, epochs=5000, batch_size=4)

    print("\n[AFTER] evaluate on full batch")
    loss_after = model.evaluate(x, y)
    print(f"  BCE(after):  {loss_after:.6f}")

    # 예측 출력
    y_pred = model.predict(x)
    print("\n🔍 XOR 예측 결과:")
    print("====================================")
    print("  입력         |  정답  |  예측값")
    print("---------------|--------|----------")
    for i in range(len(x)):
        print(f"  {x[i].reshape(-1).tolist()}  |   {y[i][0]:.1f}   |  {float(y_pred[i][0]):.4f}")
    print("====================================")

if __name__ == "__main__":
    test_xor()
