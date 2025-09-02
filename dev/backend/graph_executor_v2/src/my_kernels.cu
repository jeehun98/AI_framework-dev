#include "ge_v2_api.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>

/**
 * 커널별 파라미터-블록(Host 메모리)에 대한 정의
 * - Python에서 ctypes로 동일한 레이아웃의 구조체를 만들어 마지막 buffers 항목으로 전달
 * - 이 블록은 Host 메모리이므로, 여기서 "런처 함수"가 읽어 grid/block/shape 결정에 사용
 */
struct GemmBiasActParams {
  int M;         // A: MxK, C: MxN
  int N;         // B: KxN, C: MxN
  int K;         // A: MxK, B: KxN
  int has_bias;  // 0 or 1
  int act;       // 0:none, 1:ReLU (확장 가능: 2:GELU 등)
};

// ======== 간단한 naive GEMM(+Bias+ReLU) f32 =========

__global__ void gemm_bias_act_f32_kernel(
    float* __restrict__ C,
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,  // nullable
    int M, int N, int K,
    int has_bias, int act) {
  int row = blockIdx.y * blockDim.y + threadIdx.y; // i
  int col = blockIdx.x * blockDim.x + threadIdx.x; // j
  if (row >= M || col >= N) return;

  // Row-major: A[i,k] = A[i*K + k], B[k,j] = B[k*N + j], C[i,j] = C[i*N + j]
  float acc = 0.f;
  for (int k = 0; k < K; ++k) {
    acc = fmaf(A[row * K + k], B[k * N + col], acc);
  }
  if (has_bias && bias) acc += bias[col];

  // Activation
  if (act == 1) { // ReLU
    acc = acc < 0.f ? 0.f : acc;
  }
  C[row * N + col] = acc;
}

static int launch_gemm_bias_act_f32(
    const ge2_uintptr* bufs, int n, cudaStream_t stream) {
  // 규약: buffers 순서 = [A, B, (bias?), C, params_host_ptr]
  if (n < 4) return -1; // 최소 A,B,C,params

  // 마지막은 항상 Host params
  auto* params = reinterpret_cast<const GemmBiasActParams*>(bufs[n - 1]);
  if (!params) return -1;

  const int has_bias = params->has_bias ? 1 : 0;
  const int act      = params->act;
  const int M        = params->M;
  const int N        = params->N;
  const int K        = params->K;

  const float* A = reinterpret_cast<const float*>(bufs[0]);
  const float* B = reinterpret_cast<const float*>(bufs[1]);

  const float* bias = nullptr;
  int idx_C;
  if (has_bias) {
    if (n < 5) return -1; // A,B,bias,C,params
    bias = reinterpret_cast<const float*>(bufs[2]);
    idx_C = 3;
  } else {
    idx_C = 2;
  }
  float* C = reinterpret_cast<float*>(bufs[idx_C]);

  // grid/block
  dim3 block(16, 16);
  dim3 grid((N + block.x - 1) / block.x,
            (M + block.y - 1) / block.y);

  gemm_bias_act_f32_kernel<<<grid, block, 0, stream>>>(C, A, B, bias, M, N, K, has_bias, act);
  auto st = cudaGetLastError();
  return (st == cudaSuccess) ? 0 : -2;
}

// ======== (스텁) f16 버전: 아직 미구현 → -3 반환 =========
// 필요 시 f16 read → f32 accumulate → f16 cast로 구현 가능
extern "C" int ge2_launch_gemm_bias_act_tc_f16(const ge2_uintptr*, int, void*) {
  return -3; // not implemented (향후 WMMA/cutlass로 교체 예정)
}

// ======== 외부 진입점(f32) =========
extern "C" int ge2_launch_gemm_bias_act_f32(const ge2_uintptr* bufs, int n, void* stream) {
  if (n < 4) return -1;
  return launch_gemm_bias_act_f32(bufs, n, reinterpret_cast<cudaStream_t>(stream));
}
