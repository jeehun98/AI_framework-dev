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

# dev/backend/graph_executor_v2/smoke_test_v2_f16.py
import cupy as cp
import numpy as np
import graph_executor_v2 as gev2
import ctypes as ct

M, N, K = 64, 96, 80
A = (cp.random.randn(M, K) * 0.25).astype(cp.float16)
B = (cp.random.randn(K, N) * 0.25).astype(cp.float16)
bias = (cp.random.randn(N) * 0.1).astype(cp.float32)   # FP32 bias
C_out = cp.empty((M, N), dtype=cp.float16)

class Params(ct.Structure):
    _fields_ = [("M", ct.c_int), ("N", ct.c_int), ("K", ct.c_int),
                ("has_bias", ct.c_int), ("act", ct.c_int)]
# 조합 바꿔가며 테스트 가능: (0,0), (0,1), (1,0), (1,1)
params = Params(M, N, K, 1, 1)  # bias+ReLU
bufs = [int(A.data.ptr), int(B.data.ptr), int(bias.data.ptr), int(C_out.data.ptr), ct.addressof(params)]

gev2.launch_kernel("gemm_bias_act_tc_f16", bufs, {"buffers": []}, 0)

# 참값 대비 확인 (fp32 accumulate → fp16 cast)
ref = (cp.asnumpy(A).astype(np.float32) @ cp.asnumpy(B).astype(np.float32))
if params.has_bias: ref = ref + cp.asnumpy(bias).astype(np.float32)
if params.act == 1: ref = np.maximum(ref, 0.0)

mx = float(np.max(np.abs(ref.astype(np.float16).astype(np.float32) - cp.asnumpy(C_out).astype(np.float32))))
print("f16 max_err:", mx)
assert mx < 1.0
print("OK: f16 TC cuBLASLt smoke passed", bufs[0])
