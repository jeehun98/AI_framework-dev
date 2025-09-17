import numpy as np, cupy as cp, graph_executor_v2 as ge2

M,N,K = 128,128,128
rng = np.random.default_rng(0)
hA = rng.standard_normal((M,K), np.float32)
hB = rng.standard_normal((K,N), np.float32)
hC = rng.standard_normal((M,N), np.float32)
hBiasN = np.full((N,), 0.1, np.float32)

A = cp.asarray(hA); B = cp.asarray(hB); C = cp.asarray(hC)
D = cp.empty((M,N), np.float32); biasN = cp.asarray(hBiasN)

px = ge2.GemmBiasActParamsEx()
px.M,px.N,px.K = M,N,K
px.lda,px.ldb,px.ldc,px.ldd = K,N,N,N   # row-major
px.alpha = 1.0
px.beta  = 1.0
px.use_C = 1
px.has_bias = 1
px.bias_kind = ge2.BiasKind.PerN
px.act_kind  = ge2.ActKind.GELU

ge2.gemm_bias_act_f32_ex(int(A.data.ptr), int(B.data.ptr), int(C.data.ptr),
                         int(D.data.ptr), int(biasN.data.ptr), px, None)

# 레퍼런스
def gelu(x):  # tanh approximation
    return 0.5*x*(1.0 + np.tanh(np.sqrt(2/np.pi)*(x + 0.044715*np.power(x,3))))
hRef = hA @ hB
hRef = hRef + hC          # beta=1.0
hRef = hRef + hBiasN      # Per-N
hRef = gelu(hRef)

hD = cp.asnumpy(D)
print("max|diff|=", np.max(np.abs(hD-hRef)))
