# test_optimizer_with_wiring_debug.py
import os
import sys
import numpy as np
import cupy as cp

# ===== CUDA DLL 경로 (Windows, 필요 시 조정) =====
try:
    os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin")
except Exception:
    pass

# ===== 프로젝트/바인딩 모듈 경로 =====
# graph_executor .pyd 경로
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend", "graph_executor", "build", "lib.win-amd64-cpython-312"))
# 프로젝트 루트
sys.path.insert(0, os.path.abspath("C:/Users/owner/Desktop/AI_framework-dev"))

# ===== 프레임워크 임포트 =====
from dev.models.sequential import Sequential
from dev.layers.dense import Dense
from dev.layers.activation_layer import Activation
from dev.layers.flatten import Flatten
import graph_executor as ge  # wiring/grad/optimizer 점검에 사용


# ========== numeric vs backprop (W 그라디언트 부호 체크) ==========
def numeric_vs_backprop_grad_W(model, x, y, eps=1e-3):
    """
    단일 Dense(회귀) 가정에서 W의 dL/dW 부호를 수치미분 vs 역전파로 비교.
    - 기본은 (1,1) W 가정. (더 큰 경우 ih,iw 바꿔 반복)
    """
    wname = next((k for k in model.weights.keys() if k.endswith("_W")), None)
    if wname is None:
        raise RuntimeError("No weight param (*_W) found on model.")

    W = model.weights[wname]
    W_np = cp.asnumpy(W)
    ih, iw = 0, 0
    if W_np.ndim != 2:
        raise RuntimeError(f"{wname} must be 2D; got {W_np.shape}")

    # 수치미분(중심차분)
    orig = float(W_np[ih, iw])
    _ = float(model.evaluate(x, y))
    W[ih, iw] = orig + eps; loss_p = float(model.evaluate(x, y))
    W[ih, iw] = orig - eps; loss_m = float(model.evaluate(x, y))
    W[ih, iw] = orig
    num_grad = (loss_p - loss_m) / (2 * eps)

    # 역전파 grad 취득
    x_cp = cp.asarray(x, dtype=cp.float32); y_cp = cp.asarray(y, dtype=cp.float32)
    tensor_ptrs = {"input": x_cp.data.ptr, "y_true": y_cp.data.ptr}
    for name, arr in model.weights.items(): tensor_ptrs[name] = arr.data.ptr
    for name, arr in model.biases.items():  tensor_ptrs[name] = arr.data.ptr

    grads_ptrs = {}
    grads_dict = ge.run_graph_backward_entry(
        E=model.E, tensors=tensor_ptrs, shapes=model.shapes,
        gradients=grads_ptrs,
        final_output_id=(getattr(model, "loss_output_id", None) or model.output_var),
        batch_size=x.shape[0]
    )

    shp = model.shapes[wname]
    Kh, Kw = int(shp.rows), int(shp.cols)
    wgrad_ptr = int(grads_dict[wname])
    mem = cp.cuda.UnownedMemory(wgrad_ptr, Kh * Kw * 4, model)
    mp  = cp.cuda.MemoryPointer(mem, 0)
    wgrad_cp = cp.ndarray((Kh, Kw), dtype=cp.float32, memptr=mp)
    backprop_grad = float(cp.asnumpy(wgrad_cp)[ih, iw])

    same_sign = np.sign(num_grad) == np.sign(backprop_grad)
    print(f"[GradCheck W] numeric: {num_grad:.6e} | backprop: {backprop_grad:.6e} | same sign? {same_sign}")


# ========== numeric vs backprop (bias 그라디언트 부호 체크) ==========
def numeric_vs_backprop_grad_b(model, x, y, eps=1e-3):
    bname = next((k for k in model.biases.keys() if k.endswith("_b")), None)
    if bname is None:
        raise RuntimeError("No bias param (*_b) found on model.")
    b = model.biases[bname]

    # 수치미분(중심차분)
    b_np = cp.asnumpy(b)
    ih, iw = 0, 0  # (1,1) 가정
    orig = float(b_np[ih, iw])
    _ = float(model.evaluate(x, y))
    b[ih, iw] = orig + eps; loss_p = float(model.evaluate(x, y))
    b[ih, iw] = orig - eps; loss_m = float(model.evaluate(x, y))
    b[ih, iw] = orig
    num_grad = (loss_p - loss_m) / (2 * eps)

    # 역전파로 dB 얻기
    x_cp = cp.asarray(x, dtype=cp.float32); y_cp = cp.asarray(y, dtype=cp.float32)
    tensor_ptrs = {"input": x_cp.data.ptr, "y_true": y_cp.data.ptr}
    for name, arr in model.weights.items(): tensor_ptrs[name] = arr.data.ptr
    for name, arr in model.biases.items():  tensor_ptrs[name] = arr.data.ptr

    grads_ptrs = {}
    grads_dict = ge.run_graph_backward_entry(
        E=model.E, tensors=tensor_ptrs, shapes=model.shapes,
        gradients=grads_ptrs,
        final_output_id=(getattr(model, "loss_output_id", None) or model.output_var),
        batch_size=x.shape[0]
    )

    shp = model.shapes[bname]
    H, W = int(shp.rows), int(shp.cols)
    bgrad_ptr = int(grads_dict[bname])
    mem = cp.cuda.UnownedMemory(bgrad_ptr, H * W * 4, model)
    mp  = cp.cuda.MemoryPointer(mem, 0)
    bgrad_cp = cp.ndarray((H, W), dtype=cp.float32, memptr=mp)
    backprop_grad = float(cp.asnumpy(bgrad_cp)[ih, iw])

    same = np.sign(num_grad) == np.sign(backprop_grad)
    print(f"[GradCheck b] numeric: {num_grad:.6e} | backprop: {backprop_grad:.6e} | same sign? {same}")


# ========== SGD 한 스텝이 실제로 기대 업데이트와 일치하는지 ==========
def check_single_step_matches_sgd(model, x, y, lr):
    """
    같은 배치로
      1) 역전파로 grad 스냅샷 취득
      2) 그 grad로 CPU 기준 기대값: p_ref = p - lr*grad
      3) GPU에서 복사본에 ge.optimizer_update(SGD) 적용
      4) 두 결과 비교
    """
    x_cp = cp.asarray(x, dtype=cp.float32)
    y_cp = cp.asarray(y, dtype=cp.float32)
    tensor_ptrs = {"input": x_cp.data.ptr, "y_true": y_cp.data.ptr}
    for name, arr in model.weights.items(): tensor_ptrs[name] = arr.data.ptr
    for name, arr in model.biases.items():  tensor_ptrs[name] = arr.data.ptr

    grads_ptrs = {}
    grads_dict = ge.run_graph_backward_entry(
        E=model.E, tensors=tensor_ptrs, shapes=model.shapes,
        gradients=grads_ptrs,
        final_output_id=(getattr(model, "loss_output_id", None) or model.output_var),
        batch_size=x.shape[0]
    )

    def wrap_ptr(ptr, shape):
        H, W = int(shape.rows), int(shape.cols)
        mem = cp.cuda.UnownedMemory(int(ptr), H * W * 4, model)
        mp  = cp.cuda.MemoryPointer(mem, 0)
        return cp.ndarray((H, W), dtype=cp.float32, memptr=mp)

    ok_all = True
    for name, parr in {**model.weights, **model.biases}.items():
        if name not in grads_dict:
            print(f"  [warn] no grad for {name}, skip")
            continue
        grad_view = wrap_ptr(grads_dict[name], model.shapes[name])
        p0 = cp.asnumpy(parr).astype(np.float32)
        pref = p0 - lr * cp.asnumpy(grad_view)

        pcopy = parr.copy()
        # (교체) check_single_step_matches_sgd() 안의 ge.optimizer_update 호출
        ge.optimizer_update(
            param_ptr=int(pcopy.data.ptr),
            grad_ptr=int(grad_view.data.ptr),
            velocity_ptr=0,
            m_ptr=0,
            v_ptr=0,
            weight_decay=0.0,                       # 반드시 명시
            lr=float(lr),                           # 여기부터가 진짜 lr
            beta1=0.9,
            beta2=0.999,
            eps=1e-8,
            size=int(model.shapes[name].rows * model.shapes[name].cols),
            opt_type=ge.OptimizerType.SGD,
            timestep=1
        )

        pgpu = cp.asnumpy(pcopy)

        diff = np.max(np.abs(pref - pgpu))
        same = diff < 1e-6
        ok_all &= same
        print(f"[StepCheck] {name}: max|pref - pgpu| = {diff:.3e}  -> {'OK' if same else 'MISMATCH'}")
    print(f"[StepCheck] overall: {'OK ✅' if ok_all else 'MISMATCH ❌'}")


# ========== 유틸: CuPy/NumPy 안전 변환 ==========
def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    try:
        if isinstance(x, cp.ndarray):
            return cp.asnumpy(x)
    except Exception:
        pass
    if hasattr(x, "get"):
        try:
            return x.get()
        except Exception:
            pass
    if hasattr(x, "cpu"):
        try:
            return x.cpu().numpy()
        except Exception:
            pass
    return np.asarray(x)

def flatten_params(params_dict):
    flats = []
    for k in sorted(params_dict.keys()):
        arr_np = to_numpy(params_dict[k]).astype(np.float32, copy=False)
        flats.append(arr_np.ravel())
    if flats:
        return np.concatenate(flats)
    return np.zeros(0, dtype=np.float32)

def dot(delta, grad):
    delta = to_numpy(delta).astype(np.float32, copy=False).ravel()
    grad  = to_numpy(grad).astype(np.float32, copy=False).ravel()
    if delta.size == 0 or grad.size == 0:
        return 0.0
    return float(np.dot(delta, grad))

def l2norm(x):
    x = to_numpy(x).astype(np.float32, copy=False).ravel()
    return float(np.linalg.norm(x) if x.size else 0.0)

def get_params(model):
    params = {}
    params.update(getattr(model, "weights", {}))
    params.update(getattr(model, "biases", {}))
    return params

def dump_param_stats(model, tag=""):
    params = get_params(model)
    stats = {k: (l2norm(v), float(np.min(to_numpy(v))), float(np.max(to_numpy(v))))
             for k, v in params.items()}
    print(f"[{tag}] param stats (||·||2, min, max): {stats}")


# ========== 그래프↔파라미터 배선 점검 ==========
def debug_graph_bindings(model):
    uses_param = {ge.OpType.MATMUL, ge.OpType.ADD, ge.OpType.CONV2D}
    graph_param_ids = sorted({op.param_id for op in model.E if (op.op_type in uses_param) and op.param_id})
    weight_keys = sorted(getattr(model, "weights", {}).keys())
    bias_keys   = sorted(getattr(model, "biases", {}).keys())
    shape_keys  = sorted(getattr(model, "shapes", {}).keys())
    tensor_keys = sorted(set(weight_keys) | set(bias_keys))

    print("\n==== Graph/Param wiring check ====")
    print("Graph param_ids:", graph_param_ids)
    print("weights keys:", weight_keys)
    print("biases  keys:", bias_keys)
    print("shapes  keys (first 20):", shape_keys[:20])
    print("tensor keys:", tensor_keys)

    miss_tensors = [k for k in graph_param_ids if k not in tensor_keys]
    miss_shapes  = [k for k in graph_param_ids if k not in shape_keys]
    if miss_tensors:
        print("⚠️  MISSING in tensors:", miss_tensors)
    if miss_shapes:
        print("⚠️  MISSING in shapes:", miss_shapes)
    if not graph_param_ids:
        print("❌ Graph has NO trainable params (no MATMUL/ADD/CONV2D with param_id).")
    elif not tensor_keys:
        print("❌ Model has NO param buffers in weights/biases.")
    elif not miss_tensors and not miss_shapes:
        print("✅ Wiring looks good.")


# ========== 모델 빌더 ==========
def build_xor_model(lr, optimizer):
    model = Sequential(input_shape=(1, 1, 2))
    model.add(Flatten(input_shape=(1, 1, 2)))
    model.add(Dense(units=4, activation=None, initializer="xavier"))
    model.add(Activation("tanh"))
    model.add(Dense(units=1, activation=None, initializer="xavier"))
    model.add(Activation("sigmoid"))
    model.compile(optimizer=optimizer, loss="bce", learning_rate=lr)
    return model

def build_reg_model(lr, optimizer):
    model = Sequential(input_shape=(1, 1, 1))
    model.add(Flatten(input_shape=(1, 1, 1)))
    model.add(Dense(units=1, activation=None, initializer="xavier"))
    model.compile(optimizer=optimizer, loss="mse", learning_rate=lr)
    return model


# ========== 테스트 ==========
def test_optimizer_on_regression(opt_name, lr=0.0005, epochs=600):
    print(f"\n=== Linear Regression y=3x-1 / {opt_name} ===")
    np.random.seed(0)

    x = np.linspace(-2, 2, 64, dtype=np.float32).reshape(64,1,1,1)
    y = (3.0 * x.reshape(64,1) - 1.0).astype(np.float32)
    y += np.random.randn(*y.shape).astype(np.float32) * 0.05

    model = build_reg_model(lr=lr, optimizer=opt_name)
    debug_graph_bindings(model)

    # 🔎 학습 전, 그라디언트/업데이트 체크 (소배치)
    xb, yb = x[:16], y[:16]
    numeric_vs_backprop_grad_W(model, xb, yb, eps=1e-3)
    numeric_vs_backprop_grad_b(model, xb, yb, eps=1e-3)
    check_single_step_matches_sgd(model, xb, yb, lr)

    loss_before = float(model.evaluate(x, y))
    print(f"  MSE(before): {loss_before:.6f}")
    dump_param_stats(model, tag="init")

    # 학습
    remain = epochs
    for e in range(remain):
        model.fit(x, y, epochs=1, batch_size=16)
        if (e+1) % 50 == 0:
            mid = float(model.evaluate(x, y))
            print(f"  [epoch+{e+1}] MSE={mid:.6f}")
            if not np.isfinite(mid):
                raise RuntimeError(f"MSE became non-finite at epoch {e+1}")
            dump_param_stats(model, tag=f"epoch+{e+1}")

    loss_after = float(model.evaluate(x, y))
    print(f"  MSE(after):  {loss_after:.6f}")
    assert loss_after < loss_before, f"{opt_name}: loss did not decrease"

    # 최종 (W, b)
    wkey = next((k for k in model.weights.keys() if k.endswith("_W")), None)
    bkey = next((k for k in model.biases.keys() if k.endswith("_b")), None)
    if wkey and bkey:
        Wv = float(to_numpy(model.weights[wkey]).reshape(-1)[0])
        bv = float(to_numpy(model.biases[bkey]).reshape(-1)[0])
        print(f"  learned (W, b) ≈ ({Wv:.3f}, {bv:.3f})  target ≈ (3, -1)")


def test_optimizer_on_xor(opt_name, lr=0.3, epochs=1000):
    print(f"\n=== XOR / {opt_name} ===")
    np.random.seed(42)

    x = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32).reshape(4,1,1,2)
    y = np.array([[0],[1],[1],[0]], dtype=np.float32)

    model = build_xor_model(lr=lr, optimizer=opt_name)
    debug_graph_bindings(model)

    loss_before = float(model.evaluate(x, y))
    print(f"  BCE(before): {loss_before:.6f}")
    dump_param_stats(model, tag="init")

    remain = epochs
    for e in range(remain):
        model.fit(x, y, epochs=1, batch_size=len(x))
        if (e+1) % 100 == 0:
            mid = float(model.evaluate(x, y))
            print(f"  [epoch+{e+1}] BCE={mid:.6f}")
            if not np.isfinite(mid):
                raise RuntimeError(f"BCE became non-finite at epoch {e+1}")

    loss_after = float(model.evaluate(x, y))
    print(f"  BCE(after):  {loss_after:.6f}")
    assert loss_after < loss_before, f"{opt_name}: loss did not decrease"

    y_pred = to_numpy(model.predict(x)).reshape(-1)
    print("  preds:", np.round(y_pred, 3))


# ========== 실행 ==========
if __name__ == "__main__":
    # 옵티마이저별 기본 학습률 (회귀는 보수적으로)
    for opt in ["sgd", "momentum", "adam"]:
        if opt == "sgd":
            lr_xor, lr_reg = 0.3, 0.05
        elif opt == "momentum":
            lr_xor, lr_reg = 0.2, 0.02
        else:  # adam
            lr_xor, lr_reg = 0.01, 0.01

        # 필요 시 XOR도 테스트
        # test_optimizer_on_xor(opt_name=opt, lr=lr_xor, epochs=1000)
        test_optimizer_on_regression(opt_name=opt, lr=lr_reg, epochs=500)

    print("\nAll optimizer wiring & smoke tests passed ✅")
