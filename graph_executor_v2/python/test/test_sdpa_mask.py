# test_sdpa_mask.py
import os, sys, numpy as np, torch
import torch.nn.functional as F

# === Import path & CUDA DLL 경로 (환경에 맞게 조정) ===
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
np.random.seed(0)

# ---- 테스트 파라미터 ----
B, H, M, N, D = 2, 3, 4, 5, 8
dtype_torch = torch.float32
scale = 0.0  # 0.0 => 1/sqrt(D) 자동 (우리/torch 동일 규약)

# ---- 마스크(-inf vs -1e9) 통일 스위치 ----
# PyTorch 참조 경로와 ge 경로가 같은 상수를 쓰면 수치 차이가 줄어듭니다.
USE_NEG_INF = False             # ← False 로
NEG_INF_VAL = -1e9              # ← -1e9 고정


def make_random_qkv(device="cpu"):
    q = torch.randn(B, H, M, D, dtype=dtype_torch, device=device)
    k = torch.randn(B, H, N, D, dtype=dtype_torch, device=device)
    v = torch.randn(B, H, N, D, dtype=dtype_torch, device=device)
    return q, k, v


def make_mask(kind="i8", p_mask=0.3):
    """
    kind: "none" | "i8" | "i32" | "f32"
    mask shape: [B, 1, M, N]
      - i8/i32: 1=mask, 0=keep
      - f32: additive mask; masked 위치에 큰 음수(-inf 또는 -1e9), keep=0.0
    """
    if kind == "none":
        return None
    base = (np.random.rand(B, 1, M, N) < p_mask)  # True=mask, False=keep
    if kind == "i8":
        return base.astype(np.int8)
    if kind == "i32":
        return base.astype(np.int32)
    if kind == "f32":
        m = np.zeros((B, 1, M, N), dtype=np.float32)
        m[base] = NEG_INF_VAL  # 일관된 마스킹 값
        return m
    raise ValueError("unknown kind")


def to_additive_mask_for_torch(mask_np, causal: bool, device, dtype):
    """
    우리 API의 mask([B,1,M,N])와 causal(True/False)를
    파이토치 additive attn_mask([B*H, M, N], float)로 결합.
    - i8/i32: 1=mask → -inf/ -1e9, 0=keep → 0.0
    - f32: 값 그대로 사용 (대개 0 또는 -inf/-1e9)
    - None: 0으로 시작
    """
    if mask_np is None:
        add = torch.zeros((B, 1, M, N), dtype=dtype, device=device)
    else:
        if mask_np.dtype in (np.int8, np.int32, np.int64):
            add = np.where(mask_np != 0, NEG_INF_VAL, 0.0).astype(np.float32)
            add = torch.from_numpy(add).to(device=device, dtype=dtype)
        elif mask_np.dtype == np.float32:
            add = torch.from_numpy(mask_np).to(device=device, dtype=dtype)
        else:
            raise ValueError(f"unsupported mask dtype: {mask_np.dtype}")
    # causal을 upper-tri(-inf/-1e9)로 합치기
    if causal:
        causal_bool = torch.ones((M, N), dtype=torch.bool, device=device).triu(1)  # (n>m) True
        causal_add  = causal_bool.to(dtype=dtype) * NEG_INF_VAL
        add = add + causal_add.view(1, 1, M, N)
    # [B,1,M,N] -> [B,H,M,N] (H broadcast) -> [B*H,M,N]
    add = add.repeat(1, H, 1, 1).reshape(B*H, M, N)
    return add


def run_one(kind="i8", causal=False, atol=2e-3, rtol=1e-3, device="cpu"):
    # --- 데이터 준비 ---
    q, k, v = make_random_qkv(device=device)
    mask_np = make_mask(kind)  # numpy or None

    # --- PyTorch reference (additive mask로 causal까지 합쳐서 is_causal=False) ---
    q_ref = q.reshape(B*H, M, D)
    k_ref = k.reshape(B*H, N, D)
    v_ref = v.reshape(B*H, N, D)
    attn_mask = to_additive_mask_for_torch(mask_np, causal=causal, device=device, dtype=torch.float32)

    y_ref = F.scaled_dot_product_attention(
        q_ref, k_ref, v_ref,
        attn_mask=attn_mask,  # [B*H, M, N]
        dropout_p=0.0,
        is_causal=False,      # causal은 attn_mask에 합쳐서 전달
        # scale=None  # PyTorch는 내부적으로 1/sqrt(D); 우리는 scale=0.0로 맞춤
    ).reshape(B, H, M, D).detach().cpu().numpy()

    # --- ours (ge.sdpa): mask 그대로 + causal 플래그 별도 전달 ---
    q_np = q.detach().cpu().numpy()
    k_np = k.detach().cpu().numpy()
    v_np = v.detach().cpu().numpy()

    y_ge = ge.sdpa(
        q_np, k_np, v_np,
        mask=mask_np,         # None 또는 [B,1,M,N] with i8/i32/f32
        scale=scale,          # 0.0 => auto 1/sqrt(D)
        causal=causal,
        dropout_p=0.0,
        scale_in_train=True,
        seed=0
    )

    ok = np.allclose(y_ge, y_ref, atol=atol, rtol=rtol)
    max_abs = float(np.max(np.abs(y_ge - y_ref)))
    print(f"[mask={kind:>4}, causal={str(causal):>5}] allclose={ok}  max_abs={max_abs:.4e}")
    if not ok:
        idx = np.unravel_index(np.argmax(np.abs(y_ge - y_ref)), y_ge.shape)
        print("  worst idx:", idx, " ge:", y_ge[idx], " ref:", y_ref[idx])


def main():
    print("=== SDPA mask forward consistency test ===")
    device = "cpu"  # 필요하면 "cuda"로 바꿔도 됨(양쪽 동일 디바이스 권장)
    for kind in ["none", "i8", "i32", "f32"]:
        for causal in [False, True]:
            run_one(kind=kind, causal=causal, device=device)


if __name__ == "__main__":
    main()
