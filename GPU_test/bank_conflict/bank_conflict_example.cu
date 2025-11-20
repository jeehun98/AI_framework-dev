// bank_conflict_padded.cu
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

constexpr int WARP_SIZE    = 32;
constexpr int NUM_THREADS  = 32;     // 1 warp per block
constexpr int NUM_BLOCKS   = 256;    // SM 활용도 조금 올리기
constexpr int ROWS         = NUM_THREADS;
constexpr int COLS         = 32;     // 실제 연산에 쓰는 열 개수
constexpr int COLS_PAD     = 33;     // padding 열 (conflict-free용)
constexpr int REPEAT_INNER = 4096;   // 커널 내부 반복 (워크로드 키우기)
constexpr int NUM_LAUNCH   = 50;     // 커널 런치 반복 (평균 시간용)

#define CUDA_CHECK(expr)                                      \
  do {                                                        \
    cudaError_t _err = (expr);                                \
    if (_err != cudaSuccess) {                                \
      fprintf(stderr, "CUDA error %s at %s:%d\n",             \
              cudaGetErrorString(_err), __FILE__, __LINE__);  \
      std::exit(1);                                           \
    }                                                         \
  } while (0)


// -----------------------------------------------------------------------------
// 1) Padding을 사용한 conflict-free 커널
//    - __shared__ volatile float buf[ROWS][COLS_PAD];
//    - row stride = 33
//    - 접근: buf[row][col], col=0..31
//    - bank = (row*33 + col) % 32 = (row + col) % 32
//      → 고정된 col 에 대해 row=0..31 이 bank 0..31 을 1:1로 커버 → conflict 없음
// -----------------------------------------------------------------------------
__global__ void shared_conflict_free_padded(float *out) {
    __shared__ volatile float buf[ROWS][COLS_PAD];

    int row = threadIdx.x;  // 0..31
    if (row >= ROWS) return;

    // 공통 초기화: 논리값은 row*COLS + col 로 동일
    for (int col = 0; col < COLS; ++col) {
        buf[row][col] = static_cast<float>(row * COLS + col);
    }
    __syncthreads();

    float acc = 0.0f;

    for (int rep = 0; rep < REPEAT_INNER; ++rep) {
        #pragma unroll
        for (int col = 0; col < COLS; ++col) {
            // padding 열(col=32)은 사용하지 않음
            acc += buf[row][col];
        }
    }

    // 각 thread는 동일 row에 대해 동일 연산 → block마다 같은 결과가 나옴
    out[blockIdx.x * blockDim.x + row] = acc;
}


// -----------------------------------------------------------------------------
// 2) Padding 없는 conflict-heavy 커널
//    - __shared__ volatile float buf[ROWS][COLS];
//    - row stride = 32
//    - 접근: buf[row][col], col=0..31
//    - bank = (row*32 + col) % 32 = col
//      → 같은 col 에서 warp 내 모든 thread가 동일 bank 로 몰림 → 최악 conflict
// -----------------------------------------------------------------------------
__global__ void shared_conflict_heavy_unpadded(float *out) {
    __shared__ volatile float buf[ROWS][COLS];

    int row = threadIdx.x;  // 0..31
    if (row >= ROWS) return;

    // 초기화는 free 커널과 논리적으로 동일 (row*COLS + col)
    for (int col = 0; col < COLS; ++col) {
        buf[row][col] = static_cast<float>(row * COLS + col);
    }
    __syncthreads();

    float acc = 0.0f;

    for (int rep = 0; rep < REPEAT_INNER; ++rep) {
        #pragma unroll
        for (int col = 0; col < COLS; ++col) {
            acc += buf[row][col];
        }
    }

    out[blockIdx.x * blockDim.x + row] = acc;
}


// -----------------------------------------------------------------------------
// 공통 타이밍 함수: 커널을 NUM_LAUNCH 번 실행하고 평균 시간(ms) 리턴
// -----------------------------------------------------------------------------
float run_kernel_and_time(void (*kernel)(float*), const char* name) {
    const int total_threads = NUM_BLOCKS * NUM_THREADS;

    float *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, total_threads * sizeof(float)));

    dim3 grid(NUM_BLOCKS);
    dim3 block(NUM_THREADS);

    // warmup
    kernel<<<grid, block>>>(d_out);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < NUM_LAUNCH; ++i) {
        kernel<<<grid, block>>>(d_out);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    ms /= NUM_LAUNCH;

    // 결과 sanity check (block 0, thread 0 값만 확인)
    float h_out_first[NUM_THREADS];
    CUDA_CHECK(cudaMemcpy(h_out_first, d_out,
                          NUM_THREADS * sizeof(float),
                          cudaMemcpyDeviceToHost));

    printf("[%s] avg time = %.3f ms, sample out[0] = %f\n",
           name, ms, h_out_first[0]);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_out));

    return ms;
}


int main() {
    CUDA_CHECK(cudaSetDevice(0));

    float t_free  = run_kernel_and_time(shared_conflict_free_padded,
                                        "conflict_free_padded");
    float t_heavy = run_kernel_and_time(shared_conflict_heavy_unpadded,
                                        "conflict_heavy_unpadded");

    printf("ratio (heavy / free) = %.3fx\n", t_heavy / t_free);

    CUDA_CHECK(cudaDeviceReset());
    return 0;
}
