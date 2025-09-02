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

import cupy as cp
import numpy as np
import graph_executor_v2 as gev2
import ctypes as ct  # <-- 별칭은 ct로

# 문제 크기
M, N, K = 32, 48, 16

# 데이터 준비 (row-major)
A = cp.random.randn(M, K).astype(cp.float32)
B = cp.random.randn(K, N).astype(cp.float32)
bias = cp.random.randn(N).astype(cp.float32)
C_out = cp.empty((M, N), dtype=cp.float32)   # <-- 출력 이름을 C_out으로

# Host 파라미터 블록 정의 (네이티브와 동일 레이아웃)
class Params(ct.Structure):
    _fields_ = [
        ("M", ct.c_int),
        ("N", ct.c_int),
        ("K", ct.c_int),
        ("has_bias", ct.c_int),
        ("act", ct.c_int),  # 0:none, 1:ReLU
    ]

params = Params(M, N, K, 1, 1)             # bias 사용, ReLU
params_ptr = ct.addressof(params)          # Host 포인터 정수값

# buffers = [A, B, bias, C_out, params(host)]
bufs = [
    int(A.data.ptr),
    int(B.data.ptr),
    int(bias.data.ptr),
    int(C_out.data.ptr),
    params_ptr,
]

# descs는 지금은 네이티브에서 사용하지 않으므로 빈 형식이면 충분
gev2.launch_kernel("gemm_bias_act_f32", bufs, {"buffers": []}, 0)

# CPU 참값과 비교
ref = (cp.asnumpy(A) @ cp.asnumpy(B)) + cp.asnumpy(bias)
ref = np.maximum(ref, 0.0)  # ReLU
mx_err = float(np.max(np.abs(ref - cp.asnumpy(C_out))))
print("gemm_bias_act_f32 max_err:", mx_err)
assert mx_err < 1e-2
print("OK: GEMM+Bias+ReLU smoke passed")
