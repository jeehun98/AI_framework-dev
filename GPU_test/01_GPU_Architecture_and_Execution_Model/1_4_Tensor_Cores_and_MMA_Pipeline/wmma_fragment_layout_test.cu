// wmma_fragment_layout_test.cu
#include <cstdio>
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

constexpr int M = 16;
constexpr int N = 16;
constexpr int K = 16;
constexpr int ELEMS = M * N;

// -----------------------------
// 1) accumulator fragment: 16x16 float tile
// -----------------------------
__global__ void wmma_acc_frag_layout_kernel(float* out) {
    int lane = threadIdx.x % 32;   // 0 ~ 31

#if __CUDA_ARCH__ >= 700
    wmma::fragment<
        wmma::accumulator,
        M, N, K,
        float
    > c_frag;

    wmma::fill_fragment(c_frag, -1.0f);

    #pragma unroll
    for (int i = 0; i < c_frag.num_elements; ++i) {
        c_frag.x[i] = static_cast<float>(lane);
    }

    // row-major 로 16x16 타일 저장
    wmma::store_matrix_sync(out, c_frag, N, wmma::mem_row_major);
#else
    (void)out;
#endif
}

// -----------------------------
// 2) matrix_a fragment 레이아웃 시각화
//    - A fragment: lane id 로 채움
//    - B fragment: identity matrix 로 로드
//    - C = A * I  => C 에 lane id 패턴이 나타남
// -----------------------------
__global__ void wmma_a_frag_layout_kernel(float* out, const __half* B_identity) {
    int lane = threadIdx.x % 32;   // 0 ~ 31

#if __CUDA_ARCH__ >= 700
    wmma::fragment<
        wmma::matrix_a,
        M, N, K,
        __half,
        wmma::row_major
    > a_frag;

    wmma::fragment<
        wmma::matrix_b,
        M, N, K,
        __half,
        wmma::row_major
    > b_frag;

    wmma::fragment<
        wmma::accumulator,
        M, N, K,
        float
    > c_frag;

    // A fragment: 각 lane 이 자기 fragment 요소를 lane id 로 채움
    #pragma unroll
    for (int i = 0; i < a_frag.num_elements; ++i) {
        a_frag.x[i] = __float2half(static_cast<float>(lane));
    }

    // B fragment: 글로벌 메모리의 identity matrix 로딩
    wmma::load_matrix_sync(b_frag, B_identity, N);

    // C = 0 초기화 후 C = A * B
    wmma::fill_fragment(c_frag, 0.0f);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // C를 row-major 로 저장
    wmma::store_matrix_sync(out, c_frag, N, wmma::mem_row_major);
#else
    (void)out;
    (void)B_identity;
#endif
}

// -----------------------------
// 3) matrix_b fragment 레이아웃 시각화
//    - B fragment: lane id 로 채움
//    - A fragment: identity matrix 로 로드
//    - C = I * B  => C 에 lane id 패턴이 나타남
// -----------------------------
__global__ void wmma_b_frag_layout_kernel(float* out, const __half* A_identity) {
    int lane = threadIdx.x % 32;   // 0 ~ 31

#if __CUDA_ARCH__ >= 700
    wmma::fragment<
        wmma::matrix_a,
        M, N, K,
        __half,
        wmma::row_major
    > a_frag;

    wmma::fragment<
        wmma::matrix_b,
        M, N, K,
        __half,
        wmma::row_major
    > b_frag;

    wmma::fragment<
        wmma::accumulator,
        M, N, K,
        float
    > c_frag;

    // A fragment: identity matrix 로딩
    wmma::load_matrix_sync(a_frag, A_identity, N);

    // B fragment: lane id 로 채움
    #pragma unroll
    for (int i = 0; i < b_frag.num_elements; ++i) {
        b_frag.x[i] = __float2half(static_cast<float>(lane));
    }

    // C = 0 초기화 후 C = A * B
    wmma::fill_fragment(c_frag, 0.0f);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // C를 row-major 로 저장
    wmma::store_matrix_sync(out, c_frag, N, wmma::mem_row_major);
#else
    (void)out;
    (void)A_identity;
#endif
}

// -----------------------------
// host code
// -----------------------------
int main() {
    // device buffers
    float*  d_acc = nullptr;
    float*  d_a_vis = nullptr;
    float*  d_b_vis = nullptr;
    __half* d_I = nullptr;     // 16x16 identity (half)

    cudaMalloc(&d_acc,   ELEMS * sizeof(float));
    cudaMalloc(&d_a_vis, ELEMS * sizeof(float));
    cudaMalloc(&d_b_vis, ELEMS * sizeof(float));
    cudaMalloc(&d_I,     ELEMS * sizeof(__half));

    cudaMemset(d_acc,   0, ELEMS * sizeof(float));
    cudaMemset(d_a_vis, 0, ELEMS * sizeof(float));
    cudaMemset(d_b_vis, 0, ELEMS * sizeof(float));

    // host side identity matrix (16x16)
    __half h_I[ELEMS];
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float v = (i == j) ? 1.0f : 0.0f;
            h_I[i * N + j] = __float2half(v);
        }
    }
    cudaMemcpy(d_I, h_I, ELEMS * sizeof(__half), cudaMemcpyHostToDevice);

    dim3 block(32, 1, 1);
    dim3 grid (1, 1, 1);

    printf("== WMMA fragment lane mapping (16x16 tile) ==\n\n");

    // 1) accumulator
    printf("[accumulator fragment]\n");
    wmma_acc_frag_layout_kernel<<<grid, block>>>(d_acc);
    cudaDeviceSynchronize();

    float h_acc[ELEMS];
    cudaMemcpy(h_acc, d_acc, ELEMS * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            int lane = static_cast<int>(h_acc[i * N + j]);
            printf("%2d ", lane);
        }
        printf("\n");
    }
    printf("\n");

    // 2) matrix_a
    printf("[matrix_a fragment (row_major) → via A * I]\n");
    wmma_a_frag_layout_kernel<<<grid, block>>>(d_a_vis, d_I);
    cudaDeviceSynchronize();

    float h_a_vis[ELEMS];
    cudaMemcpy(h_a_vis, d_a_vis, ELEMS * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            int lane = static_cast<int>(h_a_vis[i * N + j]);
            printf("%2d ", lane);
        }
        printf("\n");
    }
    printf("\n");

    // 3) matrix_b
    printf("[matrix_b fragment (row_major) -> via I * B]\n");
    wmma_b_frag_layout_kernel<<<grid, block>>>(d_b_vis, d_I);
    cudaDeviceSynchronize();

    float h_b_vis[ELEMS];
    cudaMemcpy(h_b_vis, d_b_vis, ELEMS * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            int lane = static_cast<int>(h_b_vis[i * N + j]);
            printf("%2d ", lane);
        }
        printf("\n");
    }
    printf("\n");

    cudaFree(d_acc);
    cudaFree(d_a_vis);
    cudaFree(d_b_vis);
    cudaFree(d_I);
    return 0;
}

/*
빌드 & 실행 예시:

nvcc -arch=sm_80 -O3 wmma_fragment_layout_test.cu -o wmma_fragment_layout_test.exe
./wmma_fragment_layout_test.exe

*/

