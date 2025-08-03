import sys
import os
import ctypes
import numpy as np
import cupy as cp

# CUDA DLL 명시적 로드
ctypes.CDLL(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\cudart64_12.dll")

# Pybind11로 빌드된 .pyd 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), "build", "lib.win-amd64-cpython-312"))

# AI framework 루트 경로 추가
sys.path.insert(0, os.path.abspath("C:/Users/owner/Desktop/AI_framework-dev"))

sys.path.append("C:/Users/owner/Desktop/AI_framework-dev/dev/backend/graph_executor/test")
import graph_executor as ge  # Pybind11 모듈


# Graph Executor 모듈 임포트
from graph_executor import run_graph_forward_entry, run_graph_with_loss_entry, run_graph_backward_entry, OpStruct

# Sequential 모델 관련 임포트
from dev.models.sequential import Sequential
from dev.layers.dense import Dense
from dev.layers.activation_layer import Activation
from dev.layers.flatten import Flatten

def test_sequential_model_with_metrics():
    print("\n=== [TEST] Sequential 모델 학습 + 평가 (metrics + learning_rate 확인) ===")

    # 1. 입력 / 타겟 데이터 정의 (1개 샘플, shape: (1, 1, 2, 2))
    x = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)
    y = np.array([[0.7, 0.1]], dtype=np.float32)

    # 2. 모델 구성
    model = Sequential(input_shape=(1, 2, 2))
    model.add(Flatten(input_shape=(1, 2, 2)))
    model.add(Dense(units=2, activation=None))
    model.add(Activation("sigmoid"))

    # 3. Dense 초기화 강제 설정 (weight=0.5, bias=0.1)
    for layer in model._layers:
        if isinstance(layer, Dense):
            layer.weights = cp.ones_like(layer.weights) * 0.5
            layer.bias = cp.ones_like(layer.bias) * 0.1
            print(f"[INFO] Dense 초기화 완료: weights=0.5, bias=0.1")

    # 4. 컴파일 (MSE 손실, metric도 MSE)
    learning_rate = 0.001
    model.compile(optimizer="adam", loss="mse", p_metrics="mse", learning_rate=learning_rate)
    print(f"[DEBUG] compile() 후 learning_rate: {model.learning_rate}")

    # 5. fit() 내부에서 learning_rate 확인을 위해 monkey patch 삽입
    original_fit = model.fit

    def fit_with_lr_check(*args, **kwargs):
        print(f"[DEBUG] fit() 진입 시 learning_rate: {model.learning_rate}")
        return original_fit(*args, **kwargs)

    model.fit = fit_with_lr_check

    # 6. 학습
    model.fit(x, y, epochs=3)

    # 7. 평가 (손실 + metric)
    final_metric = model.evaluate(x, y)
    print(f"\n📊 최종 평가 메트릭 (MSE): {final_metric:.6f}")

    # 8. 예측 결과 확인
    y_pred = model.predict(x)
    print("🔍 예측 출력:\n", y_pred)



if __name__ == "__main__":
    test_sequential_model_with_metrics()