import os, torch, sys, numpy as np

# === Import path & DLL 경로 설정 ===
THIS = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(THIS, "..", ".."))
PKG  = os.path.join(ROOT, "python")
if PKG not in sys.path:
    sys.path.insert(0, PKG)


from graph_executor_v2 import _core as ge

torch.manual_seed(0)
np.random.seed(0)

# Shapes
B, H, M, N, D = 2, 3, 4, 5, 8
scale  = 0.0   # 0 => use 1/sqrt(D)
causal = True

# Dropout 비교는 난수 동기화가 어려우니 기본 0으로 테스트(필요시 값/seed 맞춰 실험)
dropout_p = 0.0
scale_in_train = True
seed = 1234

# Base tensors (CPU로 테스트; 필요시 device="cuda")
q = torch.randn(B, H, M, D, dtype=torch.float32, requires_grad=True, device="cpu")
k = torch.randn(B, H, N, D, dtype=torch.float32, requires_grad=True, device="cpu")
v = torch.randn(B, H, N, D, dtype=torch.float32, requires_grad=True, device="cpu")

# --- PyTorch reference ---
q_t = q.detach().clone().requires_grad_(True)
k_t = k.detach().clone().requires_grad_(True)
v_t = v.detach().clone().requires_grad_(True)

y_ref = torch.nn.functional.scaled_dot_product_attention(
    q_t.reshape(B * H, M, D),
    k_t.reshape(B * H, N, D),
    v_t.reshape(B * H, N, D),
    attn_mask=None,
    dropout_p=dropout_p,
    is_causal=causal
).reshape(B, H, M, D)

gy = torch.randn_like(y_ref)
(y_ref * gy).sum().backward()

# --- Our path (ge: NumPy 입력) ---
q_np  = q.detach().cpu().numpy()
k_np  = k.detach().cpu().numpy()
v_np  = v.detach().cpu().numpy()
gy_np = gy.detach().cpu().numpy()

# fwd (비교엔 필요 없지만 호출해 누락 이슈 없게)
_ = ge.sdpa(
    q_np, k_np, v_np,
    mask=None,
    scale=scale, causal=causal,
    dropout_p=dropout_p, scale_in_train=scale_in_train, seed=seed
)

# bwd (바인딩 시그니처에 맞춰 mask, dropout 인자 모두 전달)
dQ, dK, dV = ge.sdpa_backward(
    q_np, k_np, v_np, gy_np,
    mask=None,
    scale=scale, causal=causal,
    dropout_p=dropout_p, scale_in_train=scale_in_train, seed=seed
)

def allclose(a, b, atol=2e-3, rtol=1e-3):
    return np.allclose(a, b, atol=atol, rtol=rtol)

dq_ok = allclose(dQ, q_t.grad.detach().cpu().numpy())
dk_ok = allclose(dK, k_t.grad.detach().cpu().numpy())
dv_ok = allclose(dV, v_t.grad.detach().cpu().numpy())

def max_absdiff(a, b):
    return float(np.max(np.abs(a - b)))

print("dQ close:", dq_ok, " max_abs:", max_absdiff(dQ, q_t.grad.detach().cpu().numpy()))
print("dK close:", dk_ok, " max_abs:", max_absdiff(dK, k_t.grad.detach().cpu().numpy()))
print("dV close:", dv_ok, " max_abs:", max_absdiff(dV, v_t.grad.detach().cpu().numpy()))
