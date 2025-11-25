// 02_micro_gemm_4x4.cu

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// ---------------------------------------------------------
// 타일 파라미터 (원하면 여기만 바꿔서 튜닝)
// ---------------------------------------------------------
constexpr int BLOCK_M = 64;  // C 블록 타일 M
constexpr int BLOCK_N = 64;  // C 블록 타일 N
constexpr int BLOCK_K = 8;   // K 타일 (언롤 대상)

constexpr int THREAD_TILE_M = 4; // 스레드당 마이크로 타일 M
constexpr int THREAD_TILE_N = 4; // 스레드당 마이크로 타일 N

// 블록 내 스레드 배치
constexpr int THREADS_PER_BLOCK_M = BLOCK_M / THREAD_TILE_M; // 64 / 4 = 16
constexpr int THREADS_PER_BLOCK_N = BLOCK_N / THREAD_TILE_N; // 64 / 4 = 16

// ---------------------------------------------------------
// 마이크로 GEMM 커널: C = alpha * A * B + beta * C
// A: (M x K), B: (K x N), C: (M x N)
// Row-major 가정
// ---------------------------------------------------------
__global__ void micro_gemm_4x4_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    float alpha, float beta)
{
    // 블록이 담당하는 C 타일의 시작 좌표
    int block_row = blockIdx.y * BLOCK_M;
    int block_col = blockIdx.x * BLOCK_N;

    // 스레드가 담당하는 마이크로 타일 좌표 (타일 내부 인덱스)
    int tid_row = threadIdx.y; // [0, THREADS_PER_BLOCK_M)
    int tid_col = threadIdx.x; // [0, THREADS_PER_BLOCK_N)

    // 이 스레드가 계산하는 4x4 타일의 글로벌 시작 위치
    int thread_row = block_row + tid_row * THREAD_TILE_M;
    int thread_col = block_col + tid_col * THREAD_TILE_N;

    // -----------------------------------------------------
    // 레지스터에 마이크로 타일 accumulator (FMA 대상)
    // -----------------------------------------------------
    float acc[THREAD_TILE_M][THREAD_TILE_N] = {0.0f};

    // -----------------------------------------------------
    // Double-buffered shared memory
    //   As[buf][BLOCK_M x BLOCK_K]
    //   Bs[buf][BLOCK_K x BLOCK_N]
    // -----------------------------------------------------
    extern __shared__ float shared_mem[];
    float* As = shared_mem;                                        // [2][BLOCK_M * BLOCK_K]
    float* Bs = As + 2 * BLOCK_M * BLOCK_K;                        // [2][BLOCK_K * BLOCK_N]

    // 버퍼 인덱스
    int buf = 0;

    // -----------------------------------------------------
    // 헬퍼 lambda: A/B 타일을 shared 로 로드
    //  - 각 스레드가 여러 원소를 나눠서 로드하는 패턴 (coalescing 고려)
    // -----------------------------------------------------
    auto load_AB_tile = [&] (int k0, int buf_idx, bool valid) {
        if (!valid) return;

        // A 타일 로드: 크기 BLOCK_M x BLOCK_K
        for (int i = tid_row; i < BLOCK_M; i += THREADS_PER_BLOCK_M) {
            for (int kk = tid_col; kk < BLOCK_K; kk += THREADS_PER_BLOCK_N) {
                int global_row = block_row + i;
                int global_k   = k0 + kk;

                float val = 0.0f;
                if (global_row < M && global_k < K) {
                    val = A[global_row * K + global_k];  // row-major
                }
                As[buf_idx * BLOCK_M * BLOCK_K + i * BLOCK_K + kk] = val;
            }
        }

        // B 타일 로드: 크기 BLOCK_K x BLOCK_N
        for (int kk = tid_row; kk < BLOCK_K; kk += THREADS_PER_BLOCK_M) {
            for (int j = tid_col; j < BLOCK_N; j += THREADS_PER_BLOCK_N) {
                int global_k = k0 + kk;
                int global_col = block_col + j;

                float val = 0.0f;
                if (global_k < K && global_col < N) {
                    val = B[global_k * N + global_col]; // row-major
                }
                Bs[buf_idx * BLOCK_K * BLOCK_N + kk * BLOCK_N + j] = val;
            }
        }
    };

    // -----------------------------------------------------
    // 첫 번째 K 타일 pre-load
    // -----------------------------------------------------
    int k0 = 0;
    load_AB_tile(k0, buf, (k0 < K));
    __syncthreads();

    // -----------------------------------------------------
    // K 방향으로 BLOCK_K 씩 전진
    //   - 다음 타일을 백그라운드에서 로드 (pipeline)
    //   - 현재 타일로 FMA (FMA 패턴 + unroll)
    // -----------------------------------------------------
    for (; k0 < K; k0 += BLOCK_K) {
        int next_k0 = k0 + BLOCK_K;
        int next_buf = buf ^ 1;

        // 다음 타일 로드(파이프라인) - cp.async 쓰면 여기에 넣으면 됨
        if (next_k0 < K) {
            load_AB_tile(next_k0, next_buf, true);
        }

        // -------------------------------------------------
        // 현재 타일로 FMA (BLOCK_K 방향으로 완전 언롤)
        // -------------------------------------------------
        float a_frag[THREAD_TILE_M];
        float b_frag[THREAD_TILE_N];

        // BLOCK_K 만큼 K 축 진행
        #pragma unroll
        for (int kk = 0; kk < BLOCK_K; ++kk) {
            // 이 스레드가 담당하는 4개의 row 에 대해 A 값을 레지스터로
            #pragma unroll
            for (int i = 0; i < THREAD_TILE_M; ++i) {
                int row_in_block = tid_row * THREAD_TILE_M + i;
                a_frag[i] = As[buf * BLOCK_M * BLOCK_K +
                              row_in_block * BLOCK_K + kk];
            }

            // 이 스레드가 담당하는 4개의 col 에 대해 B 값을 레지스터로
            #pragma unroll
            for (int j = 0; j < THREAD_TILE_N; ++j) {
                int col_in_block = tid_col * THREAD_TILE_N + j;
                b_frag[j] = Bs[buf * BLOCK_K * BLOCK_N +
                              kk * BLOCK_N + col_in_block];
            }

            // -----------------------------
            // FMA 패턴 최적화 구간
            // acc[i][j] = a_frag[i] * b_frag[j] + acc[i][j]
            // i, j 둘 다 언롤해서 ILP 확보
            // -----------------------------
            #pragma unroll
            for (int i = 0; i < THREAD_TILE_M; ++i) {
                #pragma unroll
                for (int j = 0; j < THREAD_TILE_N; ++j) {
                    acc[i][j] = fmaf(a_frag[i], b_frag[j], acc[i][j]);
                }
            }
        }

        __syncthreads();  // 다음 buf 로딩 끝났는지 보장
        buf = next_buf;
    }

    // -----------------------------------------------------
    // 최종 C에 쓰기: C = alpha * acc + beta * C
    // out-of-bounds 체크 포함
    // -----------------------------------------------------
    #pragma unroll
    for (int i = 0; i < THREAD_TILE_M; ++i) {
        int global_row = thread_row + i;
        if (global_row >= M) continue;

        #pragma unroll
        for (int j = 0; j < THREAD_TILE_N; ++j) {
            int global_col = thread_col + j;
            if (global_col >= N) continue;

            float c_val = acc[i][j] * alpha;

            if (beta != 0.0f) {
                c_val += beta * C[global_row * N + global_col];
            }

            C[global_row * N + global_col] = c_val;
        }
    }
}

// ---------------------------------------------------------
// 간단한 런처
// ---------------------------------------------------------
void launch_micro_gemm_4x4(
    const float* dA, const float* dB, float* dC,
    int M, int N, int K,
    float alpha = 1.0f, float beta = 0.0f,
    cudaStream_t stream = 0)
{
    dim3 blockDim(THREADS_PER_BLOCK_N, THREADS_PER_BLOCK_M, 1); // (16,16)
    dim3 gridDim((N + BLOCK_N - 1) / BLOCK_N,
                 (M + BLOCK_M - 1) / BLOCK_M,
                 1);

    size_t shared_bytes =
        2 * BLOCK_M * BLOCK_K * sizeof(float) +
        2 * BLOCK_K * BLOCK_N * sizeof(float);

    micro_gemm_4x4_kernel<<<gridDim, blockDim, shared_bytes, stream>>>(
        dA, dB, dC,
        M, N, K,
        alpha, beta
    );
}

// ---------------------------------------------------------
// 테스트용 main
// ---------------------------------------------------------
int main()
{
    // 테스트 사이즈 (64 x 64 x 64)
    int M = 64;
    int N = 64;
    int K = 64;

    size_t bytesA = static_cast<size_t>(M) * K * sizeof(float);
    size_t bytesB = static_cast<size_t>(K) * N * sizeof(float);
    size_t bytesC = static_cast<size_t>(M) * N * sizeof(float);

    // host 메모리
    float* hA = (float*)std::malloc(bytesA);
    float* hB = (float*)std::malloc(bytesB);
    float* hC = (float*)std::malloc(bytesC);

    if (!hA || !hB || !hC) {
        std::printf("Host malloc failed\n");
        return 1;
    }

    // 간단 초기화: A(i,k) = 1, B(k,j) = 1
    for (int i = 0; i < M * K; ++i) hA[i] = 1.0f;
    for (int i = 0; i < K * N; ++i) hB[i] = 1.0f;

    // device 메모리
    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    cudaMalloc(&dA, bytesA);
    cudaMalloc(&dB, bytesB);
    cudaMalloc(&dC, bytesC);

    cudaMemcpy(dA, hA, bytesA, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, bytesB, cudaMemcpyHostToDevice);
    cudaMemset(dC, 0, bytesC);

    // 커널 호출
    launch_micro_gemm_4x4(dA, dB, dC, M, N, K);

    cudaMemcpy(hC, dC, bytesC, cudaMemcpyDeviceToHost);

    // 결과 일부 찍기
    std::printf("C[0..3]: ");
    for (int i = 0; i < 4; ++i) {
        std::printf("%f ", hC[i]);
    }
    std::printf("\n");

    // 이 경우 A, B 전부 1이니까
    // C(i,j) = sum_{k=0..K-1} 1*1 = K = 64 근처 값 나와야 정상

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    std::free(hA);
    std::free(hB);
    std::free(hC);

    return 0;
}

/*
빌드 예시 (Windows, nvcc):

nvcc 02_micro_gemm_4x4.cu -o 02_micro_gemm_4x4.exe

실행:

.\02_micro_gemm_4x4.exe
*/
