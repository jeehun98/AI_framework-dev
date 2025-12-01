// wmma_ldmatrix_load_correctness_test.cu
// Test 2. ldmatrix load correctness test
//
// 목적:
//  - 16x16 타일을 shared memory에 배치하고
//  - warp 단위로 ldmatrix.x4 로 로드했을 때
//  - 각 lane 이 들고 있는 8개 요소가 논리적 (row, col) 위치와 어떻게 연결되는지 확인
//
// 전략:
//  - A 타일(16x16)에 대해 value = row * 16 + col (0 ~ 255)로 채움
//  - shared memory 에 그대로 row-major 로 저장 (no-swizzle 기준)
//  - ldmatrix.x4.trans.shared.b16 으로 warp 전체가 16x16 타일을 로드
//  - 각 lane 이 들고 있는 8개 16-bit 값(= row*16+col)을 디코딩해서
//      row = v / 16
//      col = v % 16
//    로 논리 좌표 복원
//  - lane별 (row, col) 8개를 출력
//  - 동시에 map[row][col] = lane*100 + frag_idx 로 채워서 전체 16x16 매핑 테이블도 시각화
//
// 이 코드는 "no-swizzle + ldmatrix" 기준의 정합성 테스트다.
// shared swizzle를 넣고 싶으면 SMEM 인덱싱 부분에 XOR 패턴을 추가하고,
// 동일한 방식으로 값을 디코딩하면 된다.

#include <cstdio>
#include <vector>
#include <cuda.h>
#include <cuda_fp16.h>

#define CUDA_CHECK(stmt)                                                     \
    do {                                                                     \
        cudaError_t err = (stmt);                                            \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA Error %s:%d: %s\n",                        \
                    __FILE__, __LINE__, cudaGetErrorString(err));            \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

// warp 단위 테스트: 16x16 타일 하나만 사용
constexpr int M = 16;
constexpr int N = 16;

// ldmatrix.x4 를 위한 shared memory row-stride
// 여기서는 swizzle 없이 깔끔한 16 stride (row-major)
constexpr int SMEM_STRIDE = 16;

// ldmatrix.x4 (m8n8.x4) warp load를 위한 헬퍼
// - 32 lanes가 협업해서 16x16 타일을 가져옴
// - 샘플 패턴: lane별 base addr = s + (lane % 16) * 16 + (lane / 16) * 8
//   (知乎 등의 예제 코드에서 자주 나오는 패턴)
__device__ inline uint32_t cvta_to_shared(const void* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__global__ void ldmatrix_load_correctness_kernel(
    const unsigned short* __restrict__ A_global, // row*16 + col 값 저장 (uint16)
    int* __restrict__ lane_row_out,              // [32 * 8] lane별 8개 row
    int* __restrict__ lane_col_out,              // [32 * 8] lane별 8개 col
    int* __restrict__ logical_owner_map          // [16 * 16] map[row][col] = lane*100 + frag_idx
)
{
    // 단일 warp 가정 (blockDim.x == 32)
    int lane_id = threadIdx.x & 31;

    // shared memory: 16x16 half
    __shared__ unsigned short sA[M * SMEM_STRIDE];

    // 1) global -> shared (no-swizzle, row-major)
    //    16x16 = 256 요소, 32 lanes 이면 lane당 8개씩
    for (int i = 0; i < 8; ++i) {
        int linear = lane_id + i * 32; // 0..255
        if (linear < M * N) {
            int row = linear / N;
            int col = linear % N;
            unsigned short v = A_global[linear];
            sA[row * SMEM_STRIDE + col] = v;
        }
    }

    __syncthreads();

    // 2) ldmatrix.x4 로 shared memory -> 레지스터 로드
    //    m8n8.x4.trans.shared.b16 : 8x8 행렬 4개를 로드 (총 16x16)
    //    base addr는 lane별 다르게 주는 패턴 (row-major no-swizzle 기준)
    uint32_t a_regs[4];

    // lane별 base address 계산
    //  - row_base = lane_id % 16
    //  - col_block = (lane_id / 16) * 8  (0 or 8)
    int row_base = lane_id % 16;
    int col_block = (lane_id / 16) * 8;
    unsigned short* sA_ptr = &sA[row_base * SMEM_STRIDE + col_block];
    uint32_t sA_addr = cvta_to_shared(sA_ptr);

    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
        "{%0,%1,%2,%3}, [%4];\n"
        : "=r"(a_regs[0]), "=r"(a_regs[1]), "=r"(a_regs[2]), "=r"(a_regs[3])
        : "r"(sA_addr));

    // 3) 레지스터 안에 들어있는 16-bit 값 8개를 추출
    //    각 32-bit 레지스터에 half 2개 (16-bit * 2)
    unsigned short vals[8];
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        uint32_t reg = a_regs[r];
        vals[2 * r + 0] = static_cast<unsigned short>(reg & 0xFFFF);
        vals[2 * r + 1] = static_cast<unsigned short>((reg >> 16) & 0xFFFF);
    }

    // 4) 각 값은 row*16 + col 로 인코딩되어 있으므로,
    //    row, col을 복원해서 lane별 결과와 전체 16x16 map에 기록
    for (int e = 0; e < 8; ++e) {
        int v = static_cast<int>(vals[e]); // 0..255 (row*16 + col)
        int row = v / N;
        int col = v % N;

        // lane별 row/col 기록
        lane_row_out[lane_id * 8 + e] = row;
        lane_col_out[lane_id * 8 + e] = col;

        // 논리 타일의 (row,col)가 어느 lane / frag_idx 에 의해 소유되는지 기록
        // (이전 C fragment 테스트와 동일한 스타일: lane*100 + frag_idx)
        if (row >= 0 && row < M && col >= 0 && col < N) {
            logical_owner_map[row * N + col] = lane_id * 100 + e;
        }
    }
}

// ---- host side ----

int main() {
    printf("WMMA ldmatrix load correctness test (m16n16, no-swizzle baseline)\n\n");

    // 1) host A 초기화: A[row, col] = row*16 + col
    std::vector<unsigned short> hA(M * N);
    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < N; ++c) {
            hA[r * N + c] = static_cast<unsigned short>(r * N + c);
        }
    }

    // 2) device 메모리 할당
    unsigned short* dA = nullptr;
    int* d_lane_row = nullptr;
    int* d_lane_col = nullptr;
    int* d_owner_map = nullptr;

    CUDA_CHECK(cudaMalloc(&dA, M * N * sizeof(unsigned short)));
    CUDA_CHECK(cudaMalloc(&d_lane_row, 32 * 8 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_lane_col, 32 * 8 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_owner_map, M * N * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(dA, hA.data(), M * N * sizeof(unsigned short), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_lane_row, 0, 32 * 8 * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_lane_col, 0, 32 * 8 * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_owner_map, -1, M * N * sizeof(int)));

    // 3) kernel 실행 (단일 warp)
    dim3 block(32, 1, 1);
    dim3 grid(1, 1, 1);
    ldmatrix_load_correctness_kernel<<<grid, block>>>(dA, d_lane_row, d_lane_col, d_owner_map);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 4) 결과를 host로 복사
    std::vector<int> h_lane_row(32 * 8);
    std::vector<int> h_lane_col(32 * 8);
    std::vector<int> h_owner_map(M * N);

    CUDA_CHECK(cudaMemcpy(h_lane_row.data(), d_lane_row, 32 * 8 * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_lane_col.data(), d_lane_col, 32 * 8 * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_owner_map.data(), d_owner_map, M * N * sizeof(int), cudaMemcpyDeviceToHost));

    // 5) lane별 매핑 출력
    printf("=== [ldmatrix.x4 per-lane 8-element logical mapping] ===\n");
    for (int lane = 0; lane < 32; ++lane) {
        printf("lane %2d:", lane);
        for (int e = 0; e < 8; ++e) {
            int row = h_lane_row[lane * 8 + e];
            int col = h_lane_col[lane * 8 + e];
            printf(" f%d=(%2d,%2d)", e, row, col);
        }
        printf("\n");
    }
    printf("\n");

    // 6) 전체 16x16 타일에서 (row, col) 이 어느 lane/frag_idx 에 매핑되는지 출력
    printf("=== [logical 16x16 tile owner map (value = lane*100 + frag_idx)] ===\n");
    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < N; ++c) {
            int v = h_owner_map[r * N + c];
            if (v < 0) {
                printf("   - ");
            } else {
                printf("%4d ", v);
            }
        }
        printf("\n");
    }
    printf("\n");

    // 정리
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(d_lane_row));
    CUDA_CHECK(cudaFree(d_lane_col));
    CUDA_CHECK(cudaFree(d_owner_map));

    return 0;
}
/*
 nvcc -std=c++17 -O3   -arch=sm_86   wmma_ldmatrix_load_correctness_test.cu   -o wmma_ldmatrix_load_correctness_test.exe
*/