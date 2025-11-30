// wmma_fragment_layout_test.cu
// 목적:
//  - WMMA A/B/C fragment(m16n16k16)에 대해
//    (row, col) <-> (lane_id, frag_idx) 매핑을 복원하는 테스트
//
//  - Test A: matrix_a (row_major)
//  - Test B: matrix_b (col_major)
//  - Test C: accumulator
//
// 빌드 예:
//   nvcc -O0 -G -std=c++17 -arch=sm_86 -lineinfo \
//        -o wmma_fragment_layout_test.exe wmma_fragment_layout_test.cu

#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

#ifndef CHECK_CUDA
#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                          \
                    __FILE__, __LINE__, cudaGetErrorString(err__));            \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)
#endif

constexpr int WM = 16;
constexpr int WN = 16;
constexpr int TILE_ELEMS = WM * WN;

// -----------------------------------------------------------------------------
// Kernel A: matrix_a fragment (m16n16k16, row_major)
//   - global A[r,c] = row*100 + col (half)
//   - load_matrix_sync(a_frag, A)
//   - 각 lane이 fragment.x[i]에서 (row, col)를 복원 후
//       mapA[row, col] = lane_id*100 + i
// -----------------------------------------------------------------------------
__global__ void test_matrix_a_kernel(const half* __restrict__ A,
                                     int* __restrict__ mapA) {
    int lane_id = threadIdx.x & 31;

    wmma::fragment<wmma::matrix_a, WM, WN, 16, half, wmma::row_major> a_frag;
    wmma::load_matrix_sync(a_frag, A, WN);

    // lane 0 fragment 내용 간단 print
    if (lane_id == 0) {
        printf("=== [matrix_a] lane 0 fragment dump (num_elements=%d) ===\n",
               a_frag.num_elements);
        for (int i = 0; i < a_frag.num_elements; ++i) {
            float code = __half2float(a_frag.x[i]);
            printf("  frag_idx %2d -> raw_code %.1f\n", i, code);
        }
    }

    for (int i = 0; i < a_frag.num_elements; ++i) {
        float code_f = __half2float(a_frag.x[i]);
        int code = static_cast<int>(code_f + 0.5f);
        int r = code / 100;
        int c = code % 100;

        int map_code = lane_id * 100 + i;
        mapA[r * WN + c] = map_code;
    }
}

// -----------------------------------------------------------------------------
// Kernel B: matrix_b fragment (m16n16k16, col_major)
//   - global B[r,c] = row*100 + col (half)
//   - load_matrix_sync(b_frag, B) with col_major
//   - mapB[row, col] = lane_id*100 + i
// -----------------------------------------------------------------------------
__global__ void test_matrix_b_kernel(const half* __restrict__ B,
                                     int* __restrict__ mapB) {
    int lane_id = threadIdx.x & 31;

    wmma::fragment<wmma::matrix_b, WM, WN, 16, half, wmma::col_major> b_frag;
    wmma::load_matrix_sync(b_frag, B, WM);  // col_major: leading dim = rows

    if (lane_id == 0) {
        printf("=== [matrix_b] lane 0 fragment dump (num_elements=%d) ===\n",
               b_frag.num_elements);
        for (int i = 0; i < b_frag.num_elements; ++i) {
            float code = __half2float(b_frag.x[i]);
            printf("  frag_idx %2d -> raw_code %.1f\n", i, code);
        }
    }

    for (int i = 0; i < b_frag.num_elements; ++i) {
        float code_f = __half2float(b_frag.x[i]);
        int code = static_cast<int>(code_f + 0.5f);
        int r = code / 100;
        int c = code % 100;

        int map_code = lane_id * 100 + i;
        mapB[r * WN + c] = map_code;
    }
}

// -----------------------------------------------------------------------------
// Kernel C: accumulator fragment (m16n16k16)
//   - fragment 내부에 c_frag.x[i] = lane_id*100 + i
//   - store_matrix_sync(C)
//   - host에서 C[row,col]을 decode해서 (lane, frag_idx) 복원
// -----------------------------------------------------------------------------
__global__ void test_accumulator_kernel(float* __restrict__ C, int ldc) {
    int lane_id = threadIdx.x & 31;

    wmma::fragment<wmma::accumulator, WM, WN, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (int i = 0; i < c_frag.num_elements; ++i) {
        int code = lane_id * 100 + i;
        c_frag.x[i] = static_cast<float>(code);
    }

    if (lane_id == 0) {
        printf("=== [accumulator] lane 0 fragment dump (num_elements=%d) ===\n",
               c_frag.num_elements);
        for (int i = 0; i < c_frag.num_elements; ++i) {
            printf("  frag_idx %2d -> code %.1f\n", i, c_frag.x[i]);
        }
    }

    // warp 전체가 16x16 타일 전체를 store
    wmma::store_matrix_sync(C, c_frag, ldc, wmma::mem_row_major);
}

// -----------------------------------------------------------------------------
// Host helpers
// -----------------------------------------------------------------------------
void fill_encoded_half_matrix(half* hM, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            int code = r * 100 + c;       // (row, col) encode
            float f = static_cast<float>(code);
            hM[r * cols + c] = __float2half(f);
        }
    }
}

void print_mapping_int(const char* title, const int* map, int rows, int cols) {
    printf("\n=== %s: raw mapping (value = lane*100 + frag_idx) ===\n", title);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            printf("%4d ", map[r * cols + c]);
        }
        printf("\n");
    }

    printf("\n=== %s: decoded mapping (row,col,lane,frag_idx) ===\n", title);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            int code = map[r * cols + c];
            int lane_id  = code / 100;
            int frag_idx = code % 100;
            printf("(%2d,%2d): lane %2d, frag_idx %2d\n",
                   r, c, lane_id, frag_idx);
        }
    }
}

void print_mapping_from_C(const char* title, const float* C, int rows, int cols) {
    printf("\n=== %s: raw C (value = lane*100 + frag_idx) ===\n", title);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            int code = static_cast<int>(C[r * cols + c] + 0.5f);
            printf("%4d ", code);
        }
        printf("\n");
    }

    printf("\n=== %s: decoded mapping (row,col,lane,frag_idx) ===\n", title);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            int code = static_cast<int>(C[r * cols + c] + 0.5f);
            int lane_id  = code / 100;
            int frag_idx = code % 100;
            printf("(%2d,%2d): lane %2d, frag_idx %2d\n",
                   r, c, lane_id, frag_idx);
        }
    }
}

// -----------------------------------------------------------------------------
// main: A / B / C fragment mapping을 한 번에 실행
// -----------------------------------------------------------------------------
int main(int argc, char** argv) {
    printf("WMMA fragment & lane mapping test (m16n16k16)\n");

    // 공통 차원
    int M = WM;
    int N = WN;

    // ---- A: matrix_a (row_major) ----
    {
        size_t bytesA = sizeof(half) * M * N;
        size_t bytesMap = sizeof(int) * M * N;

        half* hA = (half*)std::malloc(bytesA);
        int*  hMapA = (int*)std::malloc(bytesMap);

        fill_encoded_half_matrix(hA, M, N);

        half* dA = nullptr;
        int* dMapA = nullptr;
        CHECK_CUDA(cudaMalloc(&dA, bytesA));
        CHECK_CUDA(cudaMalloc(&dMapA, bytesMap));
        CHECK_CUDA(cudaMemcpy(dA, hA, bytesA, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(dMapA, 0, bytesMap));

        dim3 block(32, 1, 1);
        dim3 grid(1, 1, 1);
        test_matrix_a_kernel<<<grid, block>>>(dA, dMapA);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(hMapA, dMapA, bytesMap, cudaMemcpyDeviceToHost));

        print_mapping_int("[matrix_a row_major]", hMapA, M, N);

        std::free(hA);
        std::free(hMapA);
        CHECK_CUDA(cudaFree(dA));
        CHECK_CUDA(cudaFree(dMapA));
    }

    // ---- B: matrix_b (col_major) ----
    {
        size_t bytesB = sizeof(half) * M * N;
        size_t bytesMap = sizeof(int) * M * N;

        half* hB = (half*)std::malloc(bytesB);
        int*  hMapB = (int*)std::malloc(bytesMap);

        fill_encoded_half_matrix(hB, M, N);

        half* dB = nullptr;
        int* dMapB = nullptr;
        CHECK_CUDA(cudaMalloc(&dB, bytesB));
        CHECK_CUDA(cudaMalloc(&dMapB, bytesMap));
        CHECK_CUDA(cudaMemcpy(dB, hB, bytesB, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(dMapB, 0, bytesMap));

        dim3 block(32, 1, 1);
        dim3 grid(1, 1, 1);
        test_matrix_b_kernel<<<grid, block>>>(dB, dMapB);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(hMapB, dMapB, bytesMap, cudaMemcpyDeviceToHost));

        print_mapping_int("[matrix_b col_major]", hMapB, M, N);

        std::free(hB);
        std::free(hMapB);
        CHECK_CUDA(cudaFree(dB));
        CHECK_CUDA(cudaFree(dMapB));
    }

    // ---- C: accumulator ----
    {
        size_t bytesC = sizeof(float) * M * N;
        float* hC = (float*)std::malloc(bytesC);
        float* dC = nullptr;
        CHECK_CUDA(cudaMalloc(&dC, bytesC));
        CHECK_CUDA(cudaMemset(dC, 0, bytesC));

        dim3 block(32, 1, 1);
        dim3 grid(1, 1, 1);
        test_accumulator_kernel<<<grid, block>>>(dC, N);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(hC, dC, bytesC, cudaMemcpyDeviceToHost));

        print_mapping_from_C("[accumulator]", hC, M, N);

        std::free(hC);
        CHECK_CUDA(cudaFree(dC));
    }

    return 0;
}
/*
nvcc -O0 -G -std=c++17 -arch=sm_86 -lineinfo   -o wmma_fragment_layout_test.exe wmma_fragment_layout_test.cu

*/