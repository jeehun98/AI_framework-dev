import sys
import os
import ctypes
import numpy as np

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# CUDA DLL 명시적 로드
ctypes.CDLL(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\cudart64_12.dll")

# Pybind11로 빌드된 .pyd 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), "build", "lib.win-amd64-cpython-312"))

# AI framework 루트 경로 추가
sys.path.insert(0, os.path.abspath("C:/Users/owner/Desktop/AI_framework-dev"))
sys.path.append("C:/Users/owner/Desktop/AI_framework-dev/dev/backend/graph_executor/test")

# Pybind11 모듈
import graph_executor as ge

# AI Framework 임포트
from dev.models.sequential import Sequential
from dev.layers.dense import Dense
from dev.layers.activation_layer import Activation
from dev.layers.flatten import Flatten


def test_xor_classification_equivalent_to_pytorch():
    print("\n=== [TEST] XOR - Option A (BCE on probs) + Batch Mean ===")

    # 재현성
    np.random.seed(42)

    # XOR 입력/정답
    x = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
    y = np.array([[0],[1],[1],[0]], dtype=np.float32)

    # (B, C, H, W) → 프레임워크의 NCHW 규약 맞춤
    x = x.reshape(4, 1, 1, 2)

    # 모델 구성 (Option A): 마지막 Sigmoid 유지, Loss는 BCE(prob 기반)
    model = Sequential(input_shape=(1, 1, 2))
    model.add(Flatten(input_shape=(1, 1, 2)))
    model.add(Dense(units=4, activation=None, initializer="xavier"))   # Linear(2→4)
    model.add(Activation("sigmoid"))
    model.add(Dense(units=1, activation=None, initializer="xavier"))   # Linear(4→1)
    model.add(Activation("sigmoid"))                                   # Sigmoid

    # 옵티마이저/러닝레이트: 배치 평균 스케일에 맞춰 0.1 권장
    model.compile(optimizer="sgd", loss="bce", learning_rate=0.1)

    # 그래프 확인 (디버그 필요 시)
    print("\n=== [Graph E] 계산 그래프 ===")
    for i, op in enumerate(model.E):
        print(f"[{i}] type={op.op_type}, input={op.input_id}, output={op.output_id}")
        if op.op_type == 1:
            print(f"[ADD] input={op.input_id} + param={op.param_id} -> output={op.output_id}")

    # 학습 전·후 손실 비교
    print("\n[BEFORE] evaluate on full batch")
    metric_before = model.evaluate(x, y)
    print(f"  BCE(before): {metric_before:.6f}")

    # 학습 (배치 평균이 의도대로 적용되는지 확인: batch_size=4)
    model.fit(x, y, epochs=2000, batch_size=4)

    print("\n[AFTER] evaluate on full batch")
    metric_after = model.evaluate(x, y)
    print(f"  BCE(after): {metric_after:.6f}")

    # 예측 출력
    y_pred = model.predict(x)
    print("\n🔍 XOR 예측 결과:")
    print("====================================")
    print("  입력         |  정답  |  예측값")
    print("---------------|--------|----------")
    for i in range(len(x)):
        input_vals = x[i].reshape(-1).tolist()
        label_val = y[i][0]
        pred_val = float(y_pred[i][0])
        print(f"  {input_vals}  |   {label_val:.1f}   |  {pred_val:.4f}")
    print("====================================")

if __name__ == "__main__":
    test_xor_classification_equivalent_to_pytorch()
