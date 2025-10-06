# --- add project root to sys.path (Windows/any) ---
import os, sys
THIS = os.path.abspath(os.path.dirname(__file__))                      # .../graph_executor_v2/python/test/layers
ROOT = os.path.abspath(os.path.join(THIS, "..", ".."))                 # .../graph_executor_v2 (package root)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# --------------------------------------------------

import cupy as cp
from graph_executor_v2.layers.conv2d import Conv2D

def stats(arr, name):
    m = float(cp.max(cp.abs(arr)))
    n = float(cp.linalg.norm(arr).astype(cp.float32))
    print(f"  {name}: max={m:.3e}, norm={n:.3e}, shape={tuple(arr.shape)}")

def run_case(act_none=True, repeat=2):
    cp.random.seed(1)
    x = cp.random.randn(8, 3, 32, 32).astype(cp.float32)  # NCHW

    conv = Conv2D(
        filters=16,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),   # same-ish
        dilation=(1, 1),
        groups=1,
        use_bias=True,
        initializer="he",
    )

    for step in range(repeat):
        print(f"\n--- pass {step+1}/{repeat} ---")

        # Forward
        y = conv(x)
        print(f"[Conv2D] forward OK: {y.shape}")

        # Z(pre-act) 저장 확인 (act=none이면 y==Z)
        assert hasattr(conv, "last_z") and conv.last_z is not None, "last_z was not saved in forward"
        z = conv.last_z
        diff = float(cp.max(cp.abs(y - z)))
        print(f"[Conv2D] check Z_saved: max|y - Z| = {diff:.3e}")
        if act_none:
            assert diff < 1e-6, f"Z_saved mismatch too large: {diff}"

        # Backward (바인딩 backward는 Z 필요)
        cp.random.seed(123 + step)  # 패스별 고정된 gY
        gy = cp.random.randn(*y.shape).astype(cp.float32)
        dx = conv.backward(gy)

        # 기본 검증
        assert dx.shape == x.shape, f"dx.shape={dx.shape}"
        assert conv.dW is not None, "dW was not returned"
        if conv.use_bias:
            assert conv.db is not None, "db was not returned"
        print("[Conv2D] backward OK")
        stats(dx,  "dx")
        stats(conv.dW, "dW")
        if conv.db is not None:
            stats(conv.db, "db")

if __name__ == "__main__":
    # 1) 기본 케이스: act=none → Y == Z 검증 + 2회 반복(워크스페이스 재사용 경로 체킹)
    run_case(act_none=True, repeat=2)

    # 2) (선택) 활성화 케이스: 만약 레이어에서 act를 지원한다면 별도 레이어로 적용하고
    #    Z는 pre-act, Y는 post-act이므로 Y != Z가 나와야 함.
    # run_case(act_none=False, repeat=1)

    print("\n[Conv2D] all good ✅")
