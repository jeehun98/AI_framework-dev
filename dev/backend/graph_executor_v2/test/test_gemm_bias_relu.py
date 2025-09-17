import numpy as np
import cupy as cp
import graph_executor_v2 as ge2

# --- 문제 크기 (작게 시작해도 됨) ---
M, N, K = 128, 128, 128

# --- 입력 생성 (row-major; float32) ---
rng = np.random.default_rng(123)
hA = rng.uniform(-1, 1, size=(M, K)).astype(np.float32)
hB = rng.uniform(-1, 1, size=(K, N)).astype(np.float32)
hBiasN = np.full((N,), 0.10, dtype=np.float32)   # Per-N bias (브로드캐스트)

# --- GPU로 전송 ---
A = cp.asarray(hA)
B = cp.asarray(hB)
D = cp.empty((M, N), dtype=cp.float32)           # 출력
biasN = cp.asarray(hBiasN)

# --- 파라미터 (구 API: has_bias / act=1 -> ReLU) ---
p = ge2.GemmBiasActParams()
p.M, p.N, p.K = M, N, K
p.has_bias = 1
p.act = 1    # 0: None, 1: ReLU (현재 바인딩은 이 2개만)

# --- 커널 실행 (중요: 디바이스 포인터 정수 주소를 넘김) ---
ge2.gemm_bias_act_f32(
    int(A.data.ptr),
    int(B.data.ptr),
    int(D.data.ptr),
    int(biasN.data.ptr),
    p,
    None  # cudaStream (None이면 기본 스트림)
)

# --- 결과 확인: CPU에서 레퍼런스 계산 ---
# 주의: 현재 어댑터는 alpha=1, beta=0, C=None 로 고정 → D = ReLU(A@B + biasN)
hRef = hA @ hB + hBiasN  # biasN 브로드캐스트 (행마다 동일한 열 바이어스)
hRef = np.maximum(hRef, 0.0)

hD = cp.asnumpy(D)

# --- 정밀도 체크 ---
diff = np.abs(hD - hRef)
print("max|diff| =", diff.max(), "  mean|diff| =", diff.mean())
print("samples GPU vs REF:", hD[0, :6], "|", hRef[0, :6])
