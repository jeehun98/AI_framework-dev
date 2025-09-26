# test_sdpa_dropout_consistency.py (revised)
import os, sys, numpy as np, torch

# === Import path (환경에 맞게 수정) ===
THIS = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(THIS, "..", ".."))
PKG  = os.path.join(ROOT, "python")
if PKG not in sys.path:
    sys.path.insert(0, PKG)

from graph_executor_v2 import _core as ge

torch.manual_seed(0)
np.random.seed(0)

def allclose(a, b, atol=2e-3, rtol=1e-3):
    return np.allclose(a, b, atol=atol, rtol=rtol)

def run_once(q_np, k_np, v_np, gy_np, *, seed, dropout_p=0.3, causal=True):
    scale = 0.0  # 0 => use 1/sqrt(D)
    # forward
    y = ge.sdpa(q_np, k_np, v_np,
                mask=None,
                scale=scale, causal=causal,
                dropout_p=dropout_p, scale_in_train=True,
                seed=seed)
    # backward
    
    dQ, dK, dV = ge.sdpa_backward(q_np, k_np, v_np, gy_np,
                              mask=None,
                              scale=scale, causal=causal,
                              dropout_p=dropout_p,       # ← 추가
                              scale_in_train=True,       # ← 추가
                              seed=seed)                 # ← 추가

    return y, dQ, dK, dV

def main():
    print("=== SDPA + Dropout reproducibility test (fixed) ===")

    # Shapes (고정)
    B, H, M, N, D = 2, 3, 4, 5, 8

    # 고정 입력 1세트만 생성 (모든 run에서 동일 입력 사용)
    torch.manual_seed(123)  # 입력 고정을 위한 별도 시드
    q = torch.randn(B, H, M, D, dtype=torch.float32)
    k = torch.randn(B, H, N, D, dtype=torch.float32)
    v = torch.randn(B, H, N, D, dtype=torch.float32)
    gy = torch.randn(B, H, M, D, dtype=torch.float32)

    q_np, k_np, v_np, gy_np = (x.detach().cpu().numpy() for x in (q, k, v, gy))

    # 1) 같은 seed + 같은 입력 + dropout ON => forward 동일 (backward는 API상 동일 보장 어려움)
    y1, dQ1, dK1, dV1 = run_once(q_np, k_np, v_np, gy_np, seed=1234, dropout_p=0.3, causal=True)
    y2, dQ2, dK2, dV2 = run_once(q_np, k_np, v_np, gy_np, seed=1234, dropout_p=0.3, causal=True)
    print("[same seed, dropout>0]  y equal:",  allclose(y1, y2))
    # 아래 backward는 dropout mask 재현을 못하므로 False일 수 있음(현재 API 기준)
    print("[same seed, dropout>0] dQ equal:", allclose(dQ1, dQ2))
    print("[same seed, dropout>0] dK equal:", allclose(dK1, dK2))
    print("[same seed, dropout>0] dV equal:", allclose(dV1, dV2))

    # 2) 다른 seed + 같은 입력 + dropout ON => forward 달라야 함
    y3, dQ3, dK3, dV3 = run_once(q_np, k_np, v_np, gy_np, seed=5678, dropout_p=0.3, causal=True)
    diff = float(np.max(np.abs(y1 - y3)))
    print("[diff seed, dropout>0]  max|y1-y3|:", f"{diff:.4e}", "(should be > 0)")

    # 3) dropout_p=0 => seed 무관하게 fwd/bwd 모두 동일해야 함 (입력 동일 가정)
    y4, dQ4, dK4, dV4 = run_once(q_np, k_np, v_np, gy_np, seed=1111, dropout_p=0.0, causal=True)
    y5, dQ5, dK5, dV5 = run_once(q_np, k_np, v_np, gy_np, seed=2222, dropout_p=0.0, causal=True)
    print("[p=0]       y equal:",  allclose(y4, y5))
    print("[p=0]      dQ equal:", allclose(dQ4, dQ5))
    print("[p=0]      dK equal:", allclose(dK4, dK5))
    print("[p=0]      dV equal:", allclose(dV4, dV5))

    # 4) 비-causal 경로도 동일 입력 기준으로 확인
    y6, dQ6, dK6, dV6 = run_once(q_np, k_np, v_np, gy_np, seed=1234, dropout_p=0.3, causal=False)
    y7, dQ7, dK7, dV7 = run_once(q_np, k_np, v_np, gy_np, seed=1234, dropout_p=0.3, causal=False)
    print("[same seed, non-causal, dropout>0]  y equal:",  allclose(y6, y7))
    print("[same seed, non-causal, dropout>0] dQ equal:", allclose(dQ6, dQ7))

    # === 참고: dropout ON에서 backward 재현성까지 확인하려면 ===
    #  - ge.sdpa(...)가 dropout mask를 반환하도록 바꾸고,
    #  - ge.sdpa_backward(..., mask=that_mask)로 전달하거나
    #  - ge.sdpa_backward에 (seed, dropout_p, counter_base) 같은 stateless RNG 인자를 추가하세요.
    # 그 후 동일 seed/입력에서 dQ/dK/dV도 allclose가 True여야 합니다.

    _, dQ_a, dK_a, dV_a = run_once(q_np, k_np, v_np, gy_np, seed=1234, dropout_p=0.3, causal=True)
    _, dQ_b, dK_b, dV_b = run_once(q_np, k_np, v_np, gy_np, seed=5678, dropout_p=0.3, causal=True)
    print("[diff seed, dropout>0] dQ equal:", allclose(dQ_a, dQ_b))
    print("[diff seed, dropout>0] dK equal:", allclose(dK_a, dK_b))
    print("[diff seed, dropout>0] dV equal:", allclose(dV_a, dV_b))

if __name__ == "__main__":
    main()
