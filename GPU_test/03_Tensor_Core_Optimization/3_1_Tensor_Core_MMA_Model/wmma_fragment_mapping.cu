// wmma_fragment_mapping.cu
// 목적:
//  - WMMA accumulator fragment(m16n16k16)의 warp-level layout 조사
//  - 각 (row, col)이 어떤 lane / fragment index에 매핑되는지 눈으로 복원하기

#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <mma.h>

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

// ---------------------------------------------------------------------
// Kernel: m16n16k16 accumulator fragment layout
//  - 한 warp (32 threads)이 16x16 타일 하나를 담당
//  - 각 thread(lane)의 fragment에 code = lane_id * 100 + frag_idx 저장
//  - store_matrix_sync 로 16x16 타일에 기록
//  - inspect_lane == lane_id 인 thread는 자기 fragment 내용 printf
// ---------------------------------------------------------------------
__global__ void wmma_fragment_m16n16_kernel(
    float* __restrict__ C,
    int ldc,
    int inspect_lane)
{
    // 한 블록 = 1 warp (32 threads) 가정
    int lane_id = threadIdx.x & 31;

    // accumulator fragment: 16x16x16
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    // fragment 내부에 lane별로 고유 코드 작성
    // code = lane_id * 100 + frag_idx
    for (int i = 0; i < c_frag.num_elements; ++i) {
        int code = lane_id * 100 + i;
        c_frag.x[i] = static_cast<float>(code);
    }

    // 선택한 lane만 fragment 내용 printf
    if (lane_id == inspect_lane) {
        printf("=== [lane %d] fragment dump (m16n16k16, accumulator) ===\n",
               lane_id);
        printf("num_elements = %d\n", c_frag.num_elements);
        for (int i = 0; i < c_frag.num_elements; ++i) {
            printf("  frag_idx %2d -> code %.1f\n", i, c_frag.x[i]);
        }
    }

    // warp 전체가 협력해서 16x16 타일을 C[0:16, 0:16]에 저장
    // (row-major, leading dimension = ldc)
    wmma::store_matrix_sync(C, c_frag, ldc, wmma::mem_row_major);
}

// (옵션) 다른 shape용 커널 스켈레톤: 나중에 비교 실험용으로 확장 가능
// m8n32k16, m32n8k16 등도 같은 패턴으로 작성하면 된다.
/*
__global__ void wmma_fragment_m8n32_kernel(float* C, int ldc, int inspect_lane) {
    int lane_id = threadIdx.x & 31;
    wmma::fragment<wmma::accumulator, 8, 32, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);
    for (int i = 0; i < c_frag.num_elements; ++i) {
        int code = lane_id * 100 + i;
        c_frag.x[i] = static_cast<float>(code);
    }
    if (lane_id == inspect_lane) {
        printf("=== [lane %d] fragment dump (m8n32k16, accumulator) ===\n",
               lane_id);
        printf("num_elements = %d\n", c_frag.num_elements);
        for (int i = 0; i < c_frag.num_elements; ++i) {
            printf("  frag_idx %2d -> code %.1f\n", i, c_frag.x[i]);
        }
    }
    wmma::store_matrix_sync(C, c_frag, ldc, wmma::mem_row_major);
}
*/

// ---------------------------------------------------------------------
// Host side: 결과 matrix를 다시 읽어서
//   C[row, col] = lane_id * 100 + frag_idx
//   => lane_id = code / 100, frag_idx = code % 100
// 형태로 출력해서 mapping 복원
// ---------------------------------------------------------------------
void dump_mapping_m16n16(const float* hC, int ldc)
{
    printf("\n=== Stored 16x16 tile (values = lane_id*100 + frag_idx) ===\n");
    for (int r = 0; r < 16; ++r) {
        for (int c = 0; c < 16; ++c) {
            int code = static_cast<int>(hC[r * ldc + c] + 0.5f);
            printf("%4d ", code);
        }
        printf("\n");
    }

    printf("\n=== Decoded mapping (row, col, lane_id, frag_idx) ===\n");
    for (int r = 0; r < 16; ++r) {
        for (int c = 0; c < 16; ++c) {
            int code = static_cast<int>(hC[r * ldc + c] + 0.5f);
            int lane_id  = code / 100;
            int frag_idx = code % 100;
            printf("C[%2d,%2d] = lane %2d, frag_idx %2d\n",
                   r, c, lane_id, frag_idx);
        }
    }
}

int main(int argc, char** argv)
{
    // 인자로 inspect_lane, tile shape 선택 가능하게
    //   argv[1] = inspect_lane (optional, default = 0)
    //   argv[2] = shape 문자열 ("16x16"만 사용, 나중에 확장)
    int inspect_lane = 0;
    if (argc >= 2) {
        inspect_lane = std::atoi(argv[1]);
    }

    const char* shape = "16x16";
    if (argc >= 3) {
        shape = argv[2];
    }

    printf("WMMA fragment mapping test\n");
    printf("  inspect_lane = %d\n", inspect_lane);
    printf("  shape        = %s\n", shape);

    // 이번 테스트에선 m16n16k16만 사용
    // C는 최소 16x16 타일 저장용
    int M = 16;
    int N = 16;
    int ldc = N;

    size_t bytesC = sizeof(float) * M * N;
    float* hC = (float*)std::malloc(bytesC);
    float* dC = nullptr;
    CHECK_CUDA(cudaMalloc(&dC, bytesC));
    CHECK_CUDA(cudaMemset(dC, 0, bytesC));

    dim3 block(32, 1, 1); // 한 warp
    dim3 grid(1, 1, 1);   // 타일 1개만

    if (strcmp(shape, "16x16") == 0) {
        wmma_fragment_m16n16_kernel<<<grid, block>>>(dC, ldc, inspect_lane);
    } else {
        printf("Unsupported shape for now. Use \"16x16\".\n");
        return 1;
    }

    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(hC, dC, bytesC, cudaMemcpyDeviceToHost));

    dump_mapping_m16n16(hC, ldc);

    std::free(hC);
    CHECK_CUDA(cudaFree(dC));

    return 0;
}
/*
nvcc -O0 -G -std=c++17 -arch=sm_86   -o wmma_fragment_mapping.exe wmma_fragment_mapping.cu

# lane 0 fragment 내용 + 전체 매핑
.\wmma_fragment_mapping.exe 0

# lane 5 fragment 내용 + 전체 매핑
.\wmma_fragment_mapping.exe 5

*/