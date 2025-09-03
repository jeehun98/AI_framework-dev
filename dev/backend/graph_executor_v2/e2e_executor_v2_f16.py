# --- add paths (project + build) ---
import os, sys
HERE = os.path.dirname(__file__)  # .../dev/backend/graph_executor_v2
DEV_DIR = os.path.abspath(os.path.join(HERE, "..", "..", ".."))            # .../dev
BUILD_DIR = os.path.join(HERE, "build")                               # Ninja 기본 출력
if not os.path.isdir(BUILD_DIR):                                      # 혹시 Release 하위일 때
    BUILD_DIR = os.path.join(HERE, "build", "Release")

sys.path[:0] = [DEV_DIR, BUILD_DIR]   # utils용 DEV_DIR, .pyd용 BUILD_DIR
# -----------------------------------

from dev.utils.load_cuda import ensure_cuda_dlls
ensure_cuda_dlls()


# -*- coding: utf-8 -*-
"""
IR(Graph/Op/Tensor) → Pass → Selector → ExecutorV2(launch_kernel) e2e 샘플
- A: [M,K] fp16
- B: [K,N] fp16
- bias: [N] fp32
- C: [M,N] fp16
- op_type: "GEMM_BIAS_ACT", attrs: {"mnk": (M,N,K), "act": "relu"}
필요: CuPy, CUDA GPU, graph_executor_v2(.pyd) 빌드 완료
"""

import os
import ctypes as ct
import numpy as np
import cupy as cp

# 네이티브 모듈 강제(선택): 환경에 따라 자동 로딩되지만 확실히
os.environ.setdefault("GE_NATIVE", "graph_executor_v2")

# 컴파일러 런타임/IR
from dev.backend.compiler.runtime.executor import ExecutorV2
from dev.backend.compiler.ir.nodes import Graph, Tensor, Op

# ---- 문제 크기/데이터 준비 ----
M, N, K = 64, 96, 80

A_dev   = (cp.random.randn(M, K) * 0.25).astype(cp.float16)
B_dev   = (cp.random.randn(K, N) * 0.25).astype(cp.float16)
bias_dev= (cp.random.randn(N) * 0.10).astype(cp.float32)   # FP32 bias
C_dev   = cp.empty((M, N), dtype=cp.float16)

# ---- IR 텐서 생성
# Tensor 클래스에 't'(실제 디바이스 배열) 속성을 달아 포인터를 추출할 수 있게 한다.
A = Tensor(name="A", shape=(M, K), dtype="f16", layout="rowmajor", device="cuda"); A.t = A_dev
B = Tensor(name="B", shape=(K, N), dtype="f16", layout="rowmajor", device="cuda"); B.t = B_dev
Bias = Tensor(name="Bias", shape=(N,),   dtype="f32", layout="rowmajor", device="cuda"); Bias.t = bias_dev
C = Tensor(name="C", shape=(M, N), dtype="f16", layout="rowmajor", device="cuda"); C.t = C_dev

# ---- Op 구성: GEMM + Bias + ReLU (이미 fused된 op라고 가정)
op = Op(
    op_type="GEMM_BIAS_ACT",
    inputs=[A, B, Bias],         # 입력: A,B,(bias)
    outputs=[C],                 # 출력: C
    attrs={"mnk": (M, N, K), "act": "relu"},
)

# ---- Graph 구성 (단일 op)
g = Graph()
# 보통 Graph에 ops 리스트가 있습니다. 없으면 add_op 같은 메서드가 있을 수 있음.
# 가장 단순한 형태로 .ops를 직접 지정(프로젝트의 Graph 구현에 맞게 조정)
g.ops = [op]

# ---- ExecutorV2로 실행
ex = ExecutorV2(dry_run=False)   # 실제 launch 호출
ex.run(g)

# ---- 정답 검증 (fp32 accumulate → fp16 cast)
ref = (cp.asnumpy(A_dev).astype(np.float32) @ cp.asnumpy(B_dev).astype(np.float32))
ref = ref + cp.asnumpy(bias_dev).astype(np.float32)
ref = np.maximum(ref, 0.0)  # ReLU
mx_err = float(np.max(np.abs(ref.astype(np.float16).astype(np.float32)
                             - cp.asnumpy(C_dev).astype(np.float32))))
print("e2e f16 GEMM+Bias+ReLU max_err:", mx_err)
# FP16 오차는 여유 있게 1.0 이하 허용 (환경에 따라 0.2~0.6 정도)
assert mx_err < 1.0
print("OK: ExecutorV2 e2e (f16 Lt) passed")
