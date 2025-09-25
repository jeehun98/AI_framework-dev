import os, torch, sys, numpy as np

# === Import path & DLL 경로 설정 ===
THIS = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(THIS, "..", ".."))
PKG  = os.path.join(ROOT, "python")
if PKG not in sys.path:
    sys.path.insert(0, PKG)

cuda_bins = [
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin",
]
if hasattr(os, "add_dll_directory"):
    for d in cuda_bins:
        if os.path.isdir(d):
            os.add_dll_directory(d)

from graph_executor_v2 import _core as ge

torch.manual_seed(0)

# Shapes
B, H, M, N, D = 2, 3, 4, 5, 8
scale  = 0.0   # 0 => use 1/sqrt(D)
causal = True

# Base tensors (CPU로 테스트; 필요하면 device="cuda" 사용 가능)
q = torch.randn(B, H, M, D, dtype=torch.float32, requires_grad=True, device="cpu")
k = torch.randn(B, H, N, D, dtype=torch.float32, requires_grad=True, device="cpu")
v = torch.randn(B, H, N, D, dtype=torch.float32, requires_grad=True, device="cpu")

# --- PyTorch reference (uses scaled_dot_product_attention) ---
# grad 추적용 별도 복제
q_t = q.detach().clone().requires_grad_(True)
k_t = k.detach().clone().requires_grad_(True)
v_t = v.detach().clone().requires_grad_(True)

y_ref = torch.nn.functional.scaled_dot_product_attention(
    q_t.reshape(B * H, M, D),
    k_t.reshape(B * H, N, D),
    v_t.reshape(B * H, N, D),
    attn_mask=None,
    dropout_p=0.0,
    is_causal=causal
).reshape(B, H, M, D)

# 랜덤 loss로 backward
gy = torch.randn_like(y_ref)
(y_ref * gy).sum().backward()

# --- Our path (ge: NumPy 입력 기대) ---
# 주의: .numpy() 호출 전 반드시 detach().cpu() 필요
q_np  = q.detach().cpu().numpy()
k_np  = k.detach().cpu().numpy()
v_np  = v.detach().cpu().numpy()
gy_np = gy.detach().cpu().numpy()

# fwd 더미 (대칭성 위해 호출하되 결과는 사용 안 함)
_ = ge.sdpa(q_np, k_np, v_np, mask=None,
            scale=scale, causal=causal,
            dropout_p=0.0, scale_in_train=True, seed=0)

# backward
dQ, dK, dV = ge.sdpa_backward(q_np, k_np, v_np, gy_np,
                              scale=scale, causal=causal)

def allclose(a, b, atol=2e-3, rtol=1e-3):
    return np.allclose(a, b, atol=atol, rtol=rtol)

print("dQ close:", allclose(dQ, q_t.grad.detach().cpu().numpy()))
print("dK close:", allclose(dK, k_t.grad.detach().cpu().numpy()))
print("dV close:", allclose(dV, v_t.grad.detach().cpu().numpy()))
