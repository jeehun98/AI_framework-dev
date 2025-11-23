// coalesced_shared_bench.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#ifndef CHECK_CUDA
#define CHECK_CUDA(call)                                                     \
  do {                                                                       \
    cudaError_t _e = (call);                                                 \
    if (_e != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,          \
              cudaGetErrorString(_e));                                       \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)
#endif

// ===============================
// 설정값
// ===============================
static const int BLOCK_DIM = 32;      // 32x32 block
static const int REPEAT    = 128;     // kernel 안에서 반복해서 load/compute

// grid 전체에서 다루는 행렬 크기 (N x N)
struct Config {
  int N;
  int iters;
};

// ===============================
// 커널: coalesced load
// ===============================
// global -> shared 로 읽을 때,
// 같은 warp 내에서 threadIdx.x 가 연속된 주소를 읽도록 설계
__global__
void kernel_coalesced(const float* __restrict__ gmem,
                      float* __restrict__ out,
                      int N)
{
  __shared__ float tile[BLOCK_DIM][BLOCK_DIM];

  int gx = blockIdx.x * BLOCK_DIM + threadIdx.x;
  int gy = blockIdx.y * BLOCK_DIM + threadIdx.y;

  // 블록 범위를 벗어나는 스레드는 바로 리턴
  if (gx >= N || gy >= N) return;

  // thread-private accumulator (register)
  float acc = 0.0f;

  // REPEAT 번 반복해서 global -> shared load + compute
  for (int r = 0; r < REPEAT; ++r) {
    // coalesced: row-major index
    int idx = gy * N + gx;  // 연속: threadIdx.x 가 0..31 이면 주소도 +1씩 증가

    // global -> shared
    tile[threadIdx.y][threadIdx.x] = gmem[idx];

    __syncthreads();

    // shared 에서 간단한 연산 수행 (계산 시간 조금 넣기)
    float v = tile[threadIdx.y][threadIdx.x];
    acc += v * 1.000001f;  // 뻔한 최적화 제거용 미세 연산

    __syncthreads();
  }

  // 결과를 global 에 적당히 써줌 (최적화 방지용)
  int out_idx = gy * N + gx;
  out[out_idx] = acc;
}

// ===============================
// 커널: non-coalesced load
// ===============================
// 같은 shared tile 을 채우지만,
// global 접근을 column-major 식 / stride 접근으로 비틀어서
// warp 내 주소가 크게 벌어지도록 설계
__global__
void kernel_non_coalesced(const float* __restrict__ gmem,
                          float* __restrict__ out,
                          int N)
{
  __shared__ float tile[BLOCK_DIM][BLOCK_DIM];

  int gx = blockIdx.x * BLOCK_DIM + threadIdx.x;
  int gy = blockIdx.y * BLOCK_DIM + threadIdx.y;

  if (gx >= N || gy >= N) return;

  float acc = 0.0f;

  for (int r = 0; r < REPEAT; ++r) {
    // non-coalesced:
    //   row-major 가 아니라 column-major 처럼 인덱싱
    //   같은 warp 내에서 threadIdx.x 가 바뀔 때 주소가 N 단위로 뜀
    int idx = gx * N + gy;  // stride 가 N

    tile[threadIdx.y][threadIdx.x] = gmem[idx];

    __syncthreads();

    float v = tile[threadIdx.y][threadIdx.x];
    acc += v * 1.000001f;

    __syncthreads();
  }

  int out_idx = gy * N + gx;
  out[out_idx] = acc;
}

// ===============================
// 타이머 유틸
// ===============================
float bench_kernel(void (*kernel)(const float*, float*, int),
                   const float* d_in,
                   float* d_out,
                   int N,
                   dim3 grid,
                   dim3 block,
                   int iters,
                   const char* name)
{
  // 함수 포인터를 <<<>>> 로 부를 수 없어서 템플릿으로 감싸는 대신
  // 여기서는 switch 형식으로 분리한다.
  // (간단하게 두 커널을 각각 직접 호출하는 래퍼를 쓰자.)
  // 이 함수는 실제론 안 쓰고 아래에서 직접 호출할 거라 이름만 남김.
  (void)kernel;
  (void)d_in;
  (void)d_out;
  (void)N;
  (void)grid;
  (void)block;
  (void)iters;
  (void)name;
  return 0.0f;
}

// coalesced / non-coalesced 각각에 대해 별도 벤치 함수를 만들자
float run_coalesced(const float* d_in, float* d_out,
                    int N, dim3 grid, dim3 block, int iters)
{
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  // warm-up
  kernel_coalesced<<<grid, block>>>(d_in, d_out, N);
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) {
    kernel_coalesced<<<grid, block>>>(d_in, d_out, N);
  }
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));

  return ms / iters;
}

float run_non_coalesced(const float* d_in, float* d_out,
                        int N, dim3 grid, dim3 block, int iters)
{
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  // warm-up
  kernel_non_coalesced<<<grid, block>>>(d_in, d_out, N);
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) {
    kernel_non_coalesced<<<grid, block>>>(d_in, d_out, N);
  }
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));

  return ms / iters;
}

// ===============================
// 메인
// ===============================
int main(int argc, char** argv)
{
  Config cfg;
  cfg.N     = 4096;
  cfg.iters = 50;

  if (argc >= 2) {
    cfg.N = std::atoi(argv[1]);
  }
  if (argc >= 3) {
    cfg.iters = std::atoi(argv[2]);
  }

  int N = cfg.N;
  int iters = cfg.iters;

  printf("==== Coalesced vs Non-coalesced shared load benchmark ====\n");
  printf("  N       = %d (matrix N x N)\n", N);
  printf("  REPEAT  = %d (per-kernel inner repeats)\n", REPEAT);
  printf("  iters   = %d (outer kernel launches)\n", iters);
  printf("  block   = (%d, %d)\n", BLOCK_DIM, BLOCK_DIM);

  size_t bytes = static_cast<size_t>(N) * N * sizeof(float);
  printf("  buffer  = %.2f MB\n", bytes / (1024.0 * 1024.0));

  float* h_in  = (float*)std::malloc(bytes);
  float* h_out = (float*)std::malloc(bytes);
  if (!h_in || !h_out) {
    fprintf(stderr, "Host malloc failed\n");
    return EXIT_FAILURE;
  }

  // 입력 초기화
  for (int i = 0; i < N * N; ++i) {
    h_in[i] = (float)(i % 100) * 0.01f;
  }

  float* d_in  = nullptr;
  float* d_out = nullptr;
  CHECK_CUDA(cudaMalloc(&d_in, bytes));
  CHECK_CUDA(cudaMalloc(&d_out, bytes));

  CHECK_CUDA(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

  dim3 block(BLOCK_DIM, BLOCK_DIM);
  dim3 grid((N + BLOCK_DIM - 1) / BLOCK_DIM,
            (N + BLOCK_DIM - 1) / BLOCK_DIM);

  // coalesced
  float t_coal = run_coalesced(d_in, d_out, N, grid, block, iters);

  // 결과를 한 번 가져와서 최적화 방지
  CHECK_CUDA(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

  double checksum_coal = 0.0;
  for (int i = 0; i < N * N; ++i) {
    checksum_coal += h_out[i];
  }

  // non-coalesced
  float t_ncoal = run_non_coalesced(d_in, d_out, N, grid, block, iters);
  CHECK_CUDA(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

  double checksum_ncoal = 0.0;
  for (int i = 0; i < N * N; ++i) {
    checksum_ncoal += h_out[i];
  }

  printf("\n[Result]\n");
  printf("  avg time (coalesced)     : %8.4f ms\n", t_coal);
  printf("  avg time (non-coalesced) : %8.4f ms\n", t_ncoal);
  printf("  ratio (non / coal)       : %8.4f x\n", t_ncoal / t_coal);
  printf("  checksum (coalesced)     : %.6e\n", checksum_coal);
  printf("  checksum (non-coalesced) : %.6e\n", checksum_ncoal);

  CHECK_CUDA(cudaFree(d_in));
  CHECK_CUDA(cudaFree(d_out));
  std::free(h_in);
  std::free(h_out);

  return 0;
}

// nvcc -O3 -arch=sm_86 coalesced_shared_bench.cu -o coalesced_shared_bench

// 기본 (N=4096, iters=50)
// /coalesced_shared_bench

// 크기/반복 수정
// ./coalesced_shared_bench 2048 100
