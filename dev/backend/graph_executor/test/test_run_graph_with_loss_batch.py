import sys
import os
import numpy as np
import cupy as cp
import ctypes

# CUDA DLL 명시적 로드
ctypes.CDLL(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\cudart64_12.dll")

# Pybind11 빌드된 .pyd 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), "../build/lib.win-amd64-cpython-312"))
import graph_executor as ge

def test_run_graph_with_loss_batch3():
    # ✅ 그래프 구성: Dense + Sigmoid + Loss
    E = [
        ge.OpStruct(ge.OpType.MATMUL, "x", "w", "z", ge.OpExtraParams()),
        ge.OpStruct(ge.OpType.ADD, "z", "b", "a", ge.OpExtraParams()),
        ge.OpStruct(ge.OpType.SIGMOID, "a", "", "y_pred", ge.OpExtraParams()),
        ge.OpStruct(ge.OpType.LOSS, "y_pred", "", "loss_out", ge.OpExtraParams())
    ]

    batch_size = 3
    input_dim = 2
    output_dim = 1

    # ✅ 입력: (3, 2)
    x = cp.array([
        [0.5, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ], dtype=cp.float32)

    # ✅ 가중치, 편향
    w = cp.array([[0.3], [0.7]], dtype=cp.float32)  # (2, 1)
    b = cp.array([[0.1]], dtype=cp.float32)         # (1, 1)

    # ✅ 정답: (3, 1)
    y_true = cp.array([[1.0], [0.0], [1.0]], dtype=cp.float32)

    # ✅ 출력 버퍼
    y_pred = cp.zeros((batch_size, output_dim), dtype=cp.float32)

    # ✅ 텐서 등록
    tensors = {
        "x": x,
        "w": w,
        "b": b,
        "y_true": y_true,
        "y_pred": y_pred,
    }

    shapes = {
        "x": ge.Shape(batch_size, input_dim),
        "w": ge.Shape(input_dim, output_dim),
        "b": ge.Shape(1, output_dim),
        "y_true": ge.Shape(batch_size, output_dim),
        "y_pred": ge.Shape(batch_size, output_dim),
    }

    tensor_ptrs = {k: tensor.data.ptr for k, tensor in tensors.items()}
    loss_type = "mse"

    # ✅ 실행
    loss_val = ge.run_graph_with_loss_entry(
        E=E,
        tensors=tensor_ptrs,
        shapes=shapes,
        final_output_id="y_pred",
        label_tensor_id="y_true",
        loss_type=loss_type,
        batch_size=batch_size
    )

    cp.cuda.Device().synchronize()

    print("=== [TEST] run_graph_with_loss_entry (batch=3) ===")
    print(f"Loss Type     : {loss_type}")
    print(f"Predicted y   : \n{cp.asnumpy(y_pred)}")
    print(f"Ground Truth  : \n{cp.asnumpy(y_true)}")
    print(f"Computed Loss : {loss_val:.6f}")

    # ✅ 수동 검산 출력 (optional)
    y_pred_cpu = cp.asnumpy(y_pred)
    y_true_cpu = cp.asnumpy(y_true)
    mse_manual = np.mean((y_true_cpu - y_pred_cpu) ** 2)
    print(f"Manual MSE    : {mse_manual:.6f}")

if __name__ == "__main__":
    test_run_graph_with_loss_batch3()
