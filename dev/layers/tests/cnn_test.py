# cnn_test.py
import os
import sys
import numpy as np

# CUDA DLL 경로 (Windows)
try:
    os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin")
except Exception:
    pass

# 프로젝트 경로
ROOT = "C:/Users/owner/Desktop/AI_framework-dev"
sys.path.insert(0, os.path.abspath(ROOT))
sys.path.append(os.path.join(ROOT, "dev", "backend", "graph_executor", "test"))

# 프레임워크 임포트
from dev.models.sequential import Sequential
from dev.layers.Conv2D import Conv2D
from dev.layers.activation_layer import Activation
from dev.layers.flatten import Flatten
from dev.layers.dense import Dense
import cupy as cp
import graph_executor as ge  # 그래프 확인용


def make_lines_dataset(B=64, H=8, W=8, C=1, seed=0):
    """8x8 입력에서 (세로선 or 가로선) 이진 분류 데이터셋 생성."""
    rng = np.random.default_rng(seed)
    x = np.zeros((B, H, W, C), dtype=np.float32)
    y = np.zeros((B, 1), dtype=np.float32)
    for i in range(B):
        if rng.random() < 0.5:
            r = rng.integers(0, H)    # 가로선 (label 0)
            x[i, r, :, 0] = 1.0
            y[i, 0] = 0.0
        else:
            c = rng.integers(0, W)    # 세로선 (label 1)
            x[i, :, c, 0] = 1.0
            y[i, 0] = 1.0
    return x, y


def dump_grad_ptr_keys(model, xb, yb, tag):
    """run_graph_backward_entry를 직접 호출하여 grads_ptrs의 키와 shape를 출력."""
    xb_cp = cp.asarray(xb, dtype=cp.float32)
    yb_cp = cp.asarray(yb, dtype=cp.float32)

    # y_true shape 등록(필요 시 1회만)
    model._ensure_label_shape(yb_cp)

    # 텐서 포인터 바인딩
    tensor_ptrs = {"input": xb_cp.data.ptr, "y_true": yb_cp.data.ptr}
    for name, arr in model.weights.items():
        tensor_ptrs[name] = arr.data.ptr
    for name, arr in model.biases.items():
        tensor_ptrs[name] = arr.data.ptr

    grads_ptrs = ge.run_graph_backward_entry(
        E=model.E,
        tensors=tensor_ptrs,
        shapes=model.shapes,
        gradients={},                     # C++에서 채워 반환
        final_output_id=model.output_var, # 손실 이전 출력
        batch_size=xb_cp.shape[0]
    )

    keys = sorted(grads_ptrs.keys())
    print(f"\n=== grads_ptrs keys ({tag}) ===")
    print("→", keys if keys else "∅ (none)")

    # 각 키의 shape도 같이 출력
    for k in keys:
        shp = model.shapes.get(k, None)
        if shp is None:
            print(f"  - {k}: shape = (unknown)")
        else:
            print(f"  - {k}: shape = ({shp.rows},{shp.cols})")

    # 레퍼런스: 현재 모델이 가진 파라미터 키
    print("\nweights keys:", sorted(model.weights.keys()))
    print("biases  keys:", sorted(model.biases.keys()))
    return grads_ptrs


def print_graph(model):
    """Graph E 노드들을 input/param/output 및 shape까지 보기 좋게 출력"""
    print("\n=== [Graph E] ===")
    OPNAME = {
        ge.OpType.MATMUL:   "MATMUL",
        ge.OpType.ADD:      "ADD",
        ge.OpType.RELU:     "RELU",
        ge.OpType.SIGMOID:  "SIGMOID",
        ge.OpType.TANH:     "TANH",
        ge.OpType.FLATTEN:  "FLATTEN",
        ge.OpType.CONV2D:   "CONV2D",
        ge.OpType.LOSS:     "LOSS",
        ge.OpType.LEAKY_RELU:"LEAKY_RELU",
        ge.OpType.ELU:      "ELU",
        ge.OpType.GELU:     "GELU",
        ge.OpType.SILU:     "SILU",
        ge.OpType.SOFTMAX:  "SOFTMAX",
    }

    def shape_str(shapes, tid):
        s = shapes.get(tid)
        return f"({s.rows},{s.cols})" if s else "(?)"

    for i, op in enumerate(model.E):
        tname = OPNAME.get(op.op_type, str(op.op_type))
        pin   = op.input_id
        ppar  = op.param_id or "-"
        pout  = op.output_id

        sin = shape_str(model.shapes, pin)
        spar= shape_str(model.shapes, ppar) if ppar != "-" else "-"
        sout= shape_str(model.shapes, pout)

        print(f"[{i}] {tname:<9} | in={pin:<16} {sin:<10} "
              f"| param={ppar:<12} {spar:<10} "
              f"| out={pout:<16} {sout}")

        if op.op_type == ge.OpType.CONV2D:
            ex = op.extra_params
            print(f"     ↳ B={ex.batch_size} Cin={ex.input_c} HxW={ex.input_h}x{ex.input_w} "
                  f"→ Cout={ex.output_c} KhxKw={ex.kernel_h}x{ex.kernel_w} "
                  f"stride={ex.stride_h}x{ex.stride_w} pad={ex.padding_h}x{ex.padding_w}")


def test_cnn_lines():
    print("\n=== [TEST] CNN - Lines (Horizontal vs Vertical) BCE ===")
    B, H, W, C = 64, 8, 8, 1

    # 데이터 생성
    x, y = make_lines_dataset(B=B, H=H, W=W, C=C, seed=42)

    # 모델 구성 — Conv2D → ReLU → Flatten → Dense(1) → Sigmoid
    model = Sequential(input_shape=(B, H, W, C))
    model.add(Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='same',
                     activation=None, input_shape=(B, H, W, C), initializer="he", name="conv"))
    model.add(Activation("relu", name="relu1"))
    model.add(Flatten(name="flat"))
    model.add(Dense(units=1, activation=None, initializer="xavier", name="fc"))
    model.add(Activation("sigmoid", name="sigm"))

    # 컴파일 (BCE + SGD)
    model.compile(optimizer="sgd", loss="bce", learning_rate=0.05)

    # 그래프 노드 출력
    print_graph(model)

    # 학습 전 손실
    print("\n[BEFORE] evaluate on full val set")
    loss_before = model.evaluate(x, y)
    print(f"  BCE(before): {loss_before:.6f}")

    # 🔎 학습 전 grads_ptrs 키 출력
    dump_grad_ptr_keys(model, x[:8], y[:8], tag="before training")

    # 학습
    model.fit(x, y, epochs=3000, batch_size=B, verbose=1)

    # 학습 후 손실
    print("\n[AFTER] evaluate on full val set")
    loss_after = model.evaluate(x, y)
    print(f"  BCE(after):  {loss_after:.6f}")

    # 🔎 학습 후 grads_ptrs 키 출력
    dump_grad_ptr_keys(model, x[:8], y[:8], tag="after training")

    # 예측 출력 (앞 8개)
    y_pred = model.predict(x)
    print("\n🔍 예측 결과 (앞 8개):")
    print("====================================")
    print("  정답  |  예측값(sigmoid)")
    print("--------|------------------")
    for i in range(min(8, B)):
        print(f"   {int(y[i,0])}   |   {float(y_pred[i,0]):.4f}")
    print("====================================")


if __name__ == "__main__":
    test_cnn_lines()
