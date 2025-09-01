# --- add project paths ---
import os, sys
DEV_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, DEV_DIR)  # 이제 dev/가 sys.path에 올라감
# --------------------------

from utils.load_cuda import ensure_cuda_dlls
ensure_cuda_dlls()

import graph_executor_v2 as v2
print("cap:", v2.query_capability("GEMM_BIAS_ACT", {}, {}))
v2.launch_kernel("gemm_bias_act_tc_f16", [], {}, 0)
print("OK")
