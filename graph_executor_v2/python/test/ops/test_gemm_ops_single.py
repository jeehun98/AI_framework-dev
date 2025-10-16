# --- path bootstrap (put this at the very top) ---
import os, sys

# CUDA DLL 경로 (환경에 맞게 수정하세요)
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin")
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\libnvvp")
# 필요시 cuDNN 등 추가
# os.add_dll_directory(r"C:\tools\cudnn-9.x-windows\bin")

THIS = os.path.abspath(os.path.dirname(__file__))                       # .../python/test/ops
ROOT = os.path.abspath(os.path.join(THIS, "..", "..", ".."))            # .../python
PKG  = os.path.join(ROOT, "python")                                     # .../python

# 1) "<repo>/python" 을 sys.path 맨 앞에 넣어 패키지 import 가능하게
if PKG not in sys.path:
    sys.path.insert(0, PKG)

# 2) graph_executor_v2.ops._ops_gemm import
from graph_executor_v2.ops import _ops_gemm as gemmops

# --- test deps ---
import numpy as np
import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
try:
    #torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = False

except Exception:
    pass


def banner(msg):
    print("\n" + "="*80)
    print(msg)
    print("="*80)


def to_ai_tensor_2d(t: torch.Tensor):
    """
    torch float32 CUDA contiguous -> ai::Tensor
    make_tensor_2d(ptr_u64, shape=[M,N], dtype, device, device_index)
    """
    assert t.is_cuda and t.dtype == torch.float32 and t.is_contiguous()
    ptr = int(t.data_ptr())
    M, N = map(int, t.shape)
    dev_id = 0 if t.device.index is None else int(t.device.index)
    return gemmops.make_tensor_2d(
        ptr,
        [M, N],                      # ★ shape 시퀀스
        gemmops.DType.F32,
        gemmops.Device.CUDA,         # ★ device enum
        dev_id
    )


def run_forward_backward_once(act_name="relu", leaky=0.01, M=64, K=96, N=80, atol=1e-3, rtol=1e-3):
    device = "cuda"
    dtype = torch.float32

    A = torch.randn(M, K, device=device, dtype=dtype, requires_grad=True)
    B = torch.randn(K, N, device=device, dtype=dtype, requires_grad=True)
    bias = torch.randn(N,   device=device, dtype=dtype, requires_grad=True)  # PerN
    gY = torch.randn(M, N,  device=device, dtype=dtype)

    # --- PyTorch reference ---
    def apply_act_torch(x):
        if act_name == "none":
            return x
        if act_name == "relu":
            return torch.nn.functional.relu(x)
        if act_name in ("leakyrelu", "leaky_relu", "lrelu"):
            return torch.nn.functional.leaky_relu(x, negative_slope=leaky)
        if act_name == "gelu":
            # ★ 커널과 동일하게 tanh 근사 사용
            return torch.nn.functional.gelu(x, approximate="tanh")
        if act_name == "sigmoid":
            return torch.sigmoid(x)
        if act_name == "tanh":
            return torch.tanh(x)
        raise ValueError(act_name)


    Z_ref = A @ B + bias.view(1, N)
    Y_ref = apply_act_torch(Z_ref)
    (Y_ref * gY).sum().backward()
    gA_ref = A.grad.detach().clone()
    gB_ref = B.grad.detach().clone()
    gBias_ref = bias.grad.detach().clone().view(1, N)  # PerN

    # grad reset
    A.grad = None; B.grad = None; bias.grad = None

    # --- GEMM binding path (save_z 사용) ---
    A_ai = to_ai_tensor_2d(A)
    B_ai = to_ai_tensor_2d(B)
    Bias_ai = to_ai_tensor_2d(bias.view(1, N).contiguous())   # (1,N) PerN

    Y = torch.empty((M, N), device=device, dtype=dtype)
    Y_ai = to_ai_tensor_2d(Y)

    Z_saved = torch.empty((M, N), device=device, dtype=dtype)
    Z_ai    = to_ai_tensor_2d(Z_saved)

    attrs = gemmops.GemmAttrs()
    attrs.trans_a = False
    attrs.trans_b = False
    attrs.with_bias = True
    attrs.leaky_slope = float(leaky)
    attrs.save_z = True
    # Act enum 매핑
    if act_name == "none":
        attrs.act = getattr(gemmops.ActKind, "None")

    elif act_name == "relu":
        attrs.act = gemmops.ActKind.ReLU
    elif act_name in ("leakyrelu", "leaky_relu", "lrelu"):
        attrs.act = gemmops.ActKind.LeakyReLU
    elif act_name == "gelu":
        attrs.act = gemmops.ActKind.GELU
    elif act_name == "sigmoid":
        attrs.act = gemmops.ActKind.Sigmoid
    elif act_name == "tanh":
        attrs.act = gemmops.ActKind.Tanh
    else:
        raise ValueError(act_name)

    # Forward
    gemmops.forward(A_ai, B_ai, Bias_ai, Y_ai, attrs, Z_ai, None)
    torch.cuda.synchronize()

    # Forward check
    # GELU는 구현차(approx) 때문에 허용오차 완화
    if act_name == "gelu":
        rtol_fwd = max(rtol, 2e-3)
    else:
        rtol_fwd = rtol
    max_abs = (Y - Y_ref).abs().max().item()
    max_rel = ((Y - Y_ref).abs() / (Y_ref.abs() + 1e-6)).max().item()
    print(f"  [forward] act={act_name:>9s}, max_abs={max_abs:.3e}, max_rel={max_rel:.3e}")
    assert torch.allclose(Y, Y_ref, atol=atol, rtol=rtol_fwd), "forward mismatch"

    # Backward (gA/gB/gBias)
    gA = torch.empty_like(A)
    gB = torch.empty_like(B)
    gBias = torch.empty((1, N), device=device, dtype=dtype)   # PerN shape 권장

    gA_ai   = to_ai_tensor_2d(gA)
    gB_ai   = to_ai_tensor_2d(gB)
    gBias_ai= to_ai_tensor_2d(gBias)

    gY_ai   = to_ai_tensor_2d(gY)
    Z_ai    = to_ai_tensor_2d(Z_saved)

    gemmops.backward(A_ai, B_ai, None, gY_ai, Z_ai, gA_ai, gB_ai, None, gBias_ai, attrs, None)
    torch.cuda.synchronize()

    # Ref 다시 계산
    Y_ref2 = apply_act_torch(A @ B + bias.view(1, N))
    (Y_ref2 * gY).sum().backward()
    gA_ref2   = A.grad.detach().clone()
    gB_ref2   = B.grad.detach().clone()
    gBias_ref2= bias.grad.detach().clone().view(1, N)

    def check_close(name, got, ref, atol=1e-3, rtol=1e-3):
        ma = (got - ref).abs().max().item()
        mr = ((got - ref).abs() / (ref.abs() + 1e-6)).max().item()
        print(f"  [{name:>6s}] max_abs={ma:.3e}, max_rel={mr:.3e}")
        assert torch.allclose(got, ref, atol=atol, rtol=rtol), f"{name} mismatch"

    check_close("gA",    gA,    gA_ref2, atol=atol, rtol=rtol)
    check_close("gB",    gB,    gB_ref2, atol=atol, rtol=rtol)
    check_close("gBias", gBias, gBias_ref2, atol=atol, rtol=rtol)

    print("  [OK] forward/backward matched to PyTorch")



if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA is not available — skip.")
        raise SystemExit(0)

    banner("Import _ops_gemm")
    print("[OK] Loaded:", gemmops)

    banner("Single-case smoke")
    run_forward_backward_once(act_name="relu", leaky=0.01)

    banner("Sweep a few activations")
    for act in ["none", "relu", "leakyrelu", "sigmoid", "tanh", "gelu"]:
        run_forward_backward_once(act_name=act, leaky=0.1 if act=="leakyrelu" else 0.01)
    print("\n[PASS] All tests completed.")
