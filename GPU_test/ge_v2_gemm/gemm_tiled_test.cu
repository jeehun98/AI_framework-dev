// gemm_tiled_test.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// ================================
// 간단한 CUDA 에러 체크 매크로
// ================================
#ifndef CHECK_CUDA
#define CHECK_CUDA(call)                                                   \
  do {                                                                     \
    cudaError_t _e = (call);                                               \
    if (_e != cudaSuccess) {                                               \
      fprintf(stderr, "CUDA error %s:%d: %s\n",                            \
              __FILE__, __LINE__, cudaGetErrorString(_e));                 \
      std::exit(EXIT_FAILURE);                                             \
    }                                                                      \
  } while (0)
#endif

// ================================
// 타일 / 스레드 파라미터
// (네 커널 구조랑 동일한 방식으로 설정)
// ================================

constexpr int BM  = 64;   // block이 담당하는 M 타일 크기
constexpr int BN  = 64;   // block이 담당하는 N 타일 크기
constexpr int BK  = 16;   // K 타일 크기 (K 블록)

constexpr int TDX = 16;   // threadIdx.x
constexpr int TDY = 4;    // threadIdx.y

constexpr int THR_M = BM / TDY; // = 16
constexpr int THR_N = BN / TDX; // = 4

static_assert(TDX * THR_N == BN, "BN must equal TDX*THR_N");
static_assert(TDY * THR_M == BM, "BM must equal TDY*THR_M");

// shared padding은 일단 끈 버전 (필요하면 1로 바꿔서 실험)
constexpr int PADK = 0;
constexpr int PADN = 0;

// ================================
// Tiled GEMM 커널
//   C[M, N] = A[M, K] * B[K, N]
//   row-major, alpha=1, beta=0
// ================================
__global__ void gemm_tiled_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__       C,
    int M, int N, int K)
{
  // 이 block이 담당하는 C 타일의 좌상단 (global index)
  const int m0 = blockIdx.y * BM;
  const int n0 = blockIdx.x * BN;

  // block 내 스레드 인덱스
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  // 이 스레드가 담당하는 출력 미니 타일의 시작 위치
  const int tm0 = m0 + ty * THR_M;
  const int tn0 = n0 + tx * THR_N;

  // shared 타일: A(BM x BK), B(BK x BN)
  __shared__ float As[BM][BK + PADK];
  __shared__ float Bs[BK][BN + PADN];

  // 레지스터 누산기: 스레드당 THR_M x THR_N
  float acc[THR_M][THR_N];
  #pragma unroll
  for (int i = 0; i < THR_M; ++i) {
    #pragma unroll
    for (int j = 0; j < THR_N; ++j) {
      acc[i][j] = 0.f;
    }
  }

  // K축을 BK씩 잘라가며 타일 순회
  for (int k0 = 0; k0 < K; k0 += BK) {

    // --------------------------------
    // 1) A/B 타일을 shared로 로드
    // --------------------------------
    const int tid   = ty * TDX + tx;      // block 내 1D 스레드 인덱스
    const int nthreads = TDX * TDY;

    // A 타일 (BM x BK)
    {
      const int elems = BM * BK;
      for (int e = tid; e < elems; e += nthreads) {
        const int r = e / BK;      // [0, BM)
        const int c = e % BK;      // [0, BK)
        const int gm = m0 + r;     // global m
        const int gk = k0 + c;     // global k

        float v = 0.f;
        if (gm < M && gk < K) {
          v = A[gm * K + gk];      // row-major: lda = K
        }
        As[r][c] = v;
      }
    }

    // B 타일 (BK x BN)
    {
      const int elems = BK * BN;
      for (int e = tid; e < elems; e += nthreads) {
        const int r = e / BN;      // [0, BK)
        const int c = e % BN;      // [0, BN)
        const int gk = k0 + r;     // global k
        const int gn = n0 + c;     // global n

        float v = 0.f;
        if (gk < K && gn < N) {
          v = B[gk * N + gn];      // row-major: ldb = N
        }
        Bs[r][c] = v;
      }
    }

    __syncthreads();

    // --------------------------------
    // 2) shared 타일 기반 마이크로커널
    //    - kk = 0..BK-1
    //    - A[:,kk]와 B[kk,:] outer-product
    // --------------------------------
    #pragma unroll
    for (int kk = 0; kk < BK; ++kk) {
      float a_vec[THR_M];
      float b_vec[THR_N];

      // 이 스레드가 담당하는 THR_M개의 row에 대한 A 값들
      #pragma unroll
      for (int i = 0; i < THR_M; ++i) {
        const int rm = tm0 + i;     // global row index
        float v = 0.f;
        if (rm < M) {
          const int tile_r = rm - m0;  // 타일 내부 row
          v = As[tile_r][kk];
        }
        a_vec[i] = v;
      }

      // 이 스레드가 담당하는 THR_N개의 col에 대한 B 값들
      #pragma unroll
      for (int j = 0; j < THR_N; ++j) {
        const int cn = tn0 + j;     // global col index
        float v = 0.f;
        if (cn < N) {
          const int tile_c = cn - n0;  // 타일 내부 col
          v = Bs[kk][tile_c];
        }
        b_vec[j] = v;
      }

      // rank-1 업데이트: acc += a_vec * b_vec^T
      #pragma unroll
      for (int i = 0; i < THR_M; ++i) {
        #pragma unroll
        for (int j = 0; j < THR_N; ++j) {
          acc[i][j] = fmaf(a_vec[i], b_vec[j], acc[i][j]);
        }
      }
    }

    __syncthreads();
  }

  // --------------------------------
  // 3) 결과를 global C로 저장
  // --------------------------------
  #pragma unroll
  for (int i = 0; i < THR_M; ++i) {
    const int m = tm0 + i;
    if (m >= M) continue;

    #pragma unroll
    for (int j = 0; j < THR_N; ++j) {
      const int n = tn0 + j;
      if (n >= N) continue;

      C[m * N + n] = acc[i][j];
    }
  }
}

// ================================
// CPU reference GEMM (naive)
// C = A * B
// ================================
void gemm_cpu_ref(const float* A, const float* B, float* C,
                  int M, int N, int K)
{
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float acc = 0.f;
      for (int k = 0; k < K; ++k) {
        acc += A[m * K + k] * B[k * N + n];
      }
      C[m * N + n] = acc;
    }
  }
}

// ================================
// 메인: 간단한 테스트 & 벤치
// ================================
int main(int argc, char** argv)
{
  int M = 1024;
  int N = 1024;
  int K = 1024;
  int iters = 50;

  if (argc >= 4) {
    M = std::atoi(argv[1]);
    N = std::atoi(argv[2]);
    K = std::atoi(argv[3]);
  }
  if (argc >= 5) {
    iters = std::atoi(argv[4]);
  }

  printf("==== Tiled GEMM test ====\n");
  printf("  M=%d, N=%d, K=%d\n", M, N, K);
  printf("  Tile: BM=%d, BN=%d, BK=%d\n", BM, BN, BK);
  printf("  Block: (%d x %d), Thread tile: (%d x %d)\n",
         TDX, TDY, THR_M, THR_N);

  size_t bytesA = size_t(M) * K * sizeof(float);
  size_t bytesB = size_t(K) * N * sizeof(float);
  size_t bytesC = size_t(M) * N * sizeof(float);

  float *hA = (float*)std::malloc(bytesA);
  float *hB = (float*)std::malloc(bytesB);
  float *hC = (float*)std::malloc(bytesC);
  float *hC_ref = (float*)std::malloc(bytesC);

  if (!hA || !hB || !hC || !hC_ref) {
    fprintf(stderr, "Host malloc failed\n");
    return 1;
  }

  // 호스트 데이터 초기화
  for (int i = 0; i < M * K; ++i) {
    hA[i] = 0.001f * (i % 1000);
  }
  for (int i = 0; i < K * N; ++i) {
    hB[i] = 0.002f * (i % 1000);
  }

  // 레퍼런스 계산
  /*
  printf("[CPU] computing reference...\n");
  gemm_cpu_ref(hA, hB, hC_ref, M, N, K);

  */
 
  // 디바이스 할당
  float *dA = nullptr, *dB = nullptr, *dC = nullptr;
  CHECK_CUDA(cudaMalloc(&dA, bytesA));
  CHECK_CUDA(cudaMalloc(&dB, bytesB));
  CHECK_CUDA(cudaMalloc(&dC, bytesC));

  CHECK_CUDA(cudaMemcpy(dA, hA, bytesA, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dB, hB, bytesB, cudaMemcpyHostToDevice));

  dim3 block(TDX, TDY);
  dim3 grid((N + BN - 1) / BN,
            (M + BM - 1) / BM);

  // 워밍업
  gemm_tiled_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
  CHECK_CUDA(cudaDeviceSynchronize());

  // 타이밍
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  CHECK_CUDA(cudaEventRecord(start));
  for (int it = 0; it < iters; ++it) {
    gemm_tiled_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
  }
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float ms = 0.f;
  CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
  ms /= iters;

  // 결과 가져와서 검증
  CHECK_CUDA(cudaMemcpy(hC, dC, bytesC, cudaMemcpyDeviceToHost));
 /*
  double max_diff = 0.0;
  for (int i = 0; i < M * N; ++i) {
    double diff = std::fabs((double)hC[i] - (double)hC_ref[i]);
    if (diff > max_diff) max_diff = diff;
  }
  printf("[Check] max |diff| = %.6e\n", max_diff);
  */

  // 성능 계산 (GFLOP/s)
  double flops = 2.0 * (double)M * (double)N * (double)K;
  double gflops = flops / (ms * 1e6);

  printf("[Perf]  avg time = %.3f ms,  GFLOP/s = %.2f\n", ms, gflops);

  // 정리
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  CHECK_CUDA(cudaFree(dA));
  CHECK_CUDA(cudaFree(dB));
  CHECK_CUDA(cudaFree(dC));
  std::free(hA);
  std::free(hB);
  std::free(hC);
  std::free(hC_ref);

  return 0;
}

// nvcc -O3 gemm_tiled_test.cu -o gemm_tiled_test

// ./gemm_tiled_test

// ./gemm_tiled_test 2048 2048 2048 100
