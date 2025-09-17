import numpy as np, cupy as cp, graph_executor_v2 as ge2
M,N,K = 64,64,64
hA = np.random.randn(M,K).astype(np.float32)
hB = np.random.randn(K,N).astype(np.float32)
hBiasM = np.linspace(-0.2, 0.2, M, dtype=np.float32)  # 행별(M) bias

A = cp.asarray(hA); B = cp.asarray(hB)
D = cp.empty((M,N), np.float32); biasM = cp.asarray(hBiasM)

px = ge2.GemmBiasActParamsEx()
px.M,px.N,px.K = M,N,K
px.lda,px.ldb,px.ldc,px.ldd = K,N,N,N
px.alpha = 1.0
px.beta  = 0.0
px.use_C = 0
px.has_bias = 1
px.bias_kind = ge2.BiasKind.PerM
px.act_kind  = ge2.ActKind.Sigmoid

ge2.gemm_bias_act_f32_ex(int(A.data.ptr), int(B.data.ptr), None,
                         int(D.data.ptr), int(biasM.data.ptr), px, None)

# 레퍼런스
hRef = hA @ hB + hBiasM[:,None]
hRef = 1.0/(1.0 + np.exp(-hRef))
hD = cp.asnumpy(D)
print("max|diff|=", np.max(np.abs(hD-hRef)))
