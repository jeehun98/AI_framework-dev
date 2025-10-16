# python/test/ops/test_gemm_ops_sweep.py

# --- path/bootstrap & determinism ---
import os, sys
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # cuBLAS deterministic

# (환경에 맞게 수정) CUDA DLL 경로
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin")
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\libnvvp")

THIS = os.path.abspath(os.path.dirname(__file__))                # .../python/test/ops
ROOT = os.path.abspath(os.path.join(THIS, "..", "..", ".."))     # .../python
PKG  = os.path.join(ROOT, "python")
if PKG not in sys.path:
    sys.path.insert(0, PKG)

from graph_executor_v2.ops import _ops_gemm as gemmops

import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
try:
    torch.use_deterministic_algorithms(True)
except Exception:
    pass


def banner(msg):
    print("\n" + "="*80)
    print(msg)
    print("="*80)


def to_ai_tensor_2d(t: torch.Tensor):
    """torch float32 CUDA contiguous -> ai::Tensor ([M,N], RowMajor assumed)"""
    assert t.is_cuda and t.dtype == torch.float32 and t.is_contiguous()
    ptr = int(t.data_ptr())
    M, N = map(int, t.shape)
    dev_id = 0 if t.device.index is None else int(t.device.index)
    return gemmops.make_tensor_2d(ptr, [M, N], gemmops.DType.F32, gemmops.Device.CUDA, dev_id)


def torch_apply_act(x, act, leaky):
    if act == "none":
        return x
    if act == "relu":
        return torch.nn.functional.relu(x)
    if act in ("leakyrelu", "leaky_relu", "lrelu"):
        return torch.nn.functional.leaky_relu(x, negative_slope=leaky)
    if act == "gelu":
        return torch.nn.functional.gelu(x, approximate="tanh")  # 커널과 동일화
    if act == "sigmoid":
        return torch.sigmoid(x)
    if act == "tanh":
        return torch.tanh(x)
    raise ValueError(act)


def act_to_enum(act: str):
    A = gemmops.ActKind
    return {
        "none": getattr(A, "None"),
        "relu": A.ReLU,
        "leakyrelu": A.LeakyReLU,
        "sigmoid": A.Sigmoid,
        "tanh": A.Tanh,
        "gelu": A.GELU,
    }[act]


def make_bias(kind: str, M, N, device, dtype):
    if kind == "none":
        return None
    if kind == "scalar":
        return torch.randn(1, device=device, dtype=dtype)     # (1,)
    if kind == "pern":
        return torch.randn(N, device=device, dtype=dtype)     # (N,)
    if kind == "perm":
        return torch.randn(M, device=device, dtype=dtype)     # (M,)
    raise ValueError(kind)


def add_bias_ref(Z, bias, kind, M, N):
    if bias is None:
        return Z
    if kind == "scalar":
        return Z + bias.view(1, 1)
    if kind == "pern":
        return Z + bias.view(1, N)
    if kind == "perm":
        return Z + bias.view(M, 1)
    return Z


def check_close(name, got, ref, act, atol=1e-3, rtol=1e-3):
    if act == "gelu":
        rtol = max(rtol, 2e-3)  # tanh-approx 미세 오차 허용
    ma = (got - ref).abs().max().item()
    mr = ((got - ref).abs() / (ref.abs() + 1e-6)).max().item()
    print(f"  [{name:>7s}] max_abs={ma:.3e}  max_rel={mr:.3e}")
    assert torch.allclose(got, ref, atol=atol, rtol=rtol), f"{name} mismatch"


def run_case(M, K, N, act, bias_kind, save_z, z_alias, use_backward_into, seed=0):
    torch.manual_seed(seed)
    device, dtype = "cuda", torch.float32

    A = torch.randn(M, K, device=device, dtype=dtype, requires_grad=True)
    B = torch.randn(K, N, device=device, dtype=dtype, requires_grad=True)
    bias = make_bias(bias_kind, M, N, device, dtype)
    if bias is not None:
        bias.requires_grad_(True)

    gY = torch.randn(M, N, device=device, dtype=dtype)

    # ----- reference (PyTorch) -----
    Z_ref = add_bias_ref(A @ B, bias, bias_kind, M, N)
    Y_ref = torch_apply_act(Z_ref, act, 0.1)
    (Y_ref * gY).sum().backward()
    gA_ref = A.grad.detach().clone()
    gB_ref = B.grad.detach().clone()
    gBias_ref = bias.grad.detach().clone() if bias is not None else None
    A.grad = None; B.grad = None
    if bias is not None: bias.grad = None

    # ----- binding forward -----
    A_ai = to_ai_tensor_2d(A)
    B_ai = to_ai_tensor_2d(B)
    Y = torch.empty((M, N), device=device, dtype=dtype)
    Y_ai = to_ai_tensor_2d(Y)

    Bias_ai = None
    if bias is not None:
        if bias_kind == "scalar":
            Bias_ai = to_ai_tensor_2d(bias.view(1, 1).contiguous())
        elif bias_kind == "pern":
            Bias_ai = to_ai_tensor_2d(bias.view(1, N).contiguous())
        elif bias_kind == "perm":
            Bias_ai = to_ai_tensor_2d(bias.view(M, 1).contiguous())

    attrs = gemmops.GemmAttrs()
    attrs.trans_a = False
    attrs.trans_b = False
    attrs.with_bias = (bias is not None)
    attrs.leaky_slope = 0.1
    attrs.act = act_to_enum(act)
    attrs.save_z = bool(save_z)

    Z_saved = None
    Z_ai = None
    if save_z:
        Z_saved = Y if z_alias else torch.empty((M, N), device=device, dtype=dtype)
        Z_ai = to_ai_tensor_2d(Z_saved)

    gemmops.forward(A_ai, B_ai, Bias_ai, Y_ai, attrs, Z_ai, None)
    torch.cuda.synchronize()

    # forward check
    check_close("forward", Y, Y_ref, act)

    # ----- prepare Z for backward (always pre-activation) -----
    if save_z:
        if z_alias and act != "none":
            # alias + non-linear act -> pre 가 덮여 사라졌으므로 재계산해서 전달
            with torch.no_grad():
                Z_for_bwd = add_bias_ref(A @ B, bias, bias_kind, M, N)
        else:
            Z_for_bwd = Z_saved  # pre 가 온전히 남아있음 (act=="none"이거나 별도 버퍼)
    else:
        with torch.no_grad():
            Z_for_bwd = add_bias_ref(A @ B, bias, bias_kind, M, N)

    Z_ai2 = to_ai_tensor_2d(Z_for_bwd.contiguous())

    # ----- binding backward -----
    gA = torch.empty_like(A)
    gB = torch.empty_like(B)

    if bias is None:
        gBias = None
        gBias_ai = None
    else:
        if bias_kind == "scalar":
            gBias = torch.empty((1, 1), device=device, dtype=dtype)
            gBias_ai = to_ai_tensor_2d(gBias)
        elif bias_kind == "pern":
            gBias = torch.empty((1, N), device=device, dtype=dtype)
            gBias_ai = to_ai_tensor_2d(gBias)
        elif bias_kind == "perm":
            # PerM gBias는 바인딩이 금지 → 제출하지 않음
            gBias = None
            gBias_ai = None

    gA_ai = to_ai_tensor_2d(gA)
    gB_ai = to_ai_tensor_2d(gB)
    gY_ai = to_ai_tensor_2d(gY)

    if use_backward_into:
        dZ = torch.empty((M, N), device=device, dtype=dtype)
        gemmops.backward_into(
            A_ai, B_ai, None, gY_ai, Z_ai2,
            gA_ai, gB_ai, None, gBias_ai,
            attrs, None,
            int(dZ.data_ptr()), 0, 0
        )
    else:
        gemmops.backward(
            A_ai, B_ai, None, gY_ai, Z_ai2,
            gA_ai, gB_ai, None, gBias_ai,
            attrs, None
        )
    torch.cuda.synchronize()

    # ----- reference backward (fresh) -----
    Z_ref2 = add_bias_ref(A @ B, bias, bias_kind, M, N)
    Y_ref2 = torch_apply_act(Z_ref2, act, 0.1)
    (Y_ref2 * gY).sum().backward()
    gA_ref2 = A.grad.detach().clone()
    gB_ref2 = B.grad.detach().clone()
    gBias_ref2 = bias.grad.detach().clone() if bias is not None else None

    # compare
    check_close("gA", gA, gA_ref2, act)
    check_close("gB", gB, gB_ref2, act)
    if bias is not None and bias_kind in ("scalar", "pern"):
        got = gBias.view(-1)
        ref = gBias_ref2.view(-1)
        check_close("gBias", got, ref, act)


def run_sweep():
    banner("Import _ops_gemm")
    print("[OK] Loaded:", gemmops)

    sizes = [
        (4, 7, 5),     # 아주 작은 비정형
        (16, 16, 16),  # 정사각
        (32, 48, 40),  # 직사각
        (64, 96, 80),  # 중간
    ]
    acts = ["none", "relu", "leakyrelu", "sigmoid", "tanh", "gelu"]
    bias_kinds = ["none", "scalar", "perm", "pern"]
    seeds = [0, 1]

    for M, K, N in sizes:
        for act in acts:
            for bk in bias_kinds:
                for save_z in [False, True]:
                    for z_alias in ([False, True] if save_z else [False]):
                        for use_bwd_into in [False, True]:
                            banner(f"Case: M={M},K={K},N={N}, act={act}, bias={bk}, "
                                   f"save_z={save_z}, z_alias={z_alias}, bwd_into={use_bwd_into}")
                            for sd in seeds:
                                run_case(M, K, N, act, bk, save_z, z_alias, use_bwd_into, seed=sd)
    print("\n[PASS] All sweep cases completed.")


def run_error_cases():
    banner("Error-path checks")
    device, dtype = "cuda", torch.float32
    M, K, N = 8, 9, 7
    A = torch.randn(M, K, device=device, dtype=dtype)
    B = torch.randn(K, N, device=device, dtype=dtype)
    Y = torch.empty(M, N, device=device, dtype=dtype)
    Z = torch.empty(M, N, device=device, dtype=dtype)
    gY = torch.randn(M, N, device=device, dtype=dtype)

    A_ai = to_ai_tensor_2d(A); B_ai = to_ai_tensor_2d(B)
    Y_ai = to_ai_tensor_2d(Y); Z_ai = to_ai_tensor_2d(Z)
    gY_ai = to_ai_tensor_2d(gY)

    attrs = gemmops.GemmAttrs()
    attrs.act = getattr(gemmops.ActKind, "None")

    attrs.with_bias = False
    attrs.save_z = True

    # 1) save_z=True 인데 Z_saved=None -> 예외/에러 유도
    try:
        gemmops.forward(A_ai, B_ai, None, Y_ai, attrs, None, None)
        raise AssertionError("Expected error not raised for save_z=True && Z_saved=None")
    except Exception as e:
        print("  [OK] raised as expected:", type(e).__name__, str(e)[:120], "...")

    # 2) backward에서 gBias 모양을 PerM로 주는 금지 경로 (바인딩 가드)
    gA = torch.empty_like(A); gB = torch.empty_like(B)
    gBias_bad = torch.empty((M,1), device=device, dtype=dtype)
    gA_ai = to_ai_tensor_2d(gA); gB_ai = to_ai_tensor_2d(gB); gBias_bad_ai = to_ai_tensor_2d(gBias_bad)
    try:
        gemmops.backward(A_ai, B_ai, None, gY_ai, Z_ai, gA_ai, gB_ai, None, gBias_bad_ai, attrs, None)
        raise AssertionError("Expected error not raised for PerM gBias in backward()")
    except Exception as e:
        print("  [OK] raised as expected (PerM gBias rejected):", type(e).__name__)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available — skip.")
        raise SystemExit(0)

    run_sweep()
    run_error_cases()
