# --- add paths (project + build) ---
import os, sys
HERE = os.path.dirname(__file__)  # .../dev/backend/graph_executor_v2
DEV_DIR = os.path.abspath(os.path.join(HERE, "..", ".."))            # .../dev
BUILD_DIR = os.path.join(HERE, "build")                               # Ninja 기본 출력
if not os.path.isdir(BUILD_DIR):                                      # 혹시 Release 하위일 때
    BUILD_DIR = os.path.join(HERE, "build", "Release")

sys.path[:0] = [DEV_DIR, BUILD_DIR]   # utils용 DEV_DIR, .pyd용 BUILD_DIR
# -----------------------------------

from utils.load_cuda import ensure_cuda_dlls
ensure_cuda_dlls()

import graph_executor_v2 as v2
print("cap:", v2.query_capability("GEMM_BIAS_ACT", {}, {}))
v2.launch_kernel("gemm_bias_act_tc_f16", [], {}, 0)
print("OK")
