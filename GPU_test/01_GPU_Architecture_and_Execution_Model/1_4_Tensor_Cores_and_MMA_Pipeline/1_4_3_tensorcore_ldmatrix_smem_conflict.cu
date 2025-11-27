// 1_4_3_tensorcore_ldmatrix_smem_conflict.cu
// 
// 1.4.3 Test — Shared Memory Layout & ldmatrix / bank conflict
//
// - 단일 warp(32 threads)가 16x16 half 타일을 shared mem에 올려놓고
//   wmma::load_matrix_sync() -> 내부 ldmatrix 사용 패턴을 유도.
// - 두 가지 케이스 비교:
//   (1) naive row-major shared layout
//   (2) XOR swizzled shared layout  (bank conflict 감소 목적)
//
// Nsight Compute에서 비교해서 볼 metric 예:
//   - shared_load_transactions_per_request
//   - smsp__inst_executed.shared_<...>
//   - l1tex__data_bank_conflicts_pipe_lsu_mem_shared_bank_conflict.sum
//   - l1tex__t_sectors_pipe_lsu_mem_shared_op_ld.sum
//
// 빌드 예:
//   nvcc -arch=sm_80 -O3 1_4_3_tensorcore_ldmatrix_smem_conflict.cu -o tensorcore_ldmatrix_smem_conflict.exe
//
// 실행 후 ncu 사용 예:
//   ncu --set full --kernel-name regex:.*naive.*   ./tensorcore_ldmatrix_smem_conflict.exe
//   ncu --set full --kernel-name regex:.*swizzled.* ./tensorcore_ldmatrix_smem_conflict.exe

#include <cstdio>
#include <cuda.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                         \
                    cudaGetErrorString(err__), __FILE__, __LINE__);             \
            std::abort();                                                       \
        }                                                                       \
    } while (0)

// 반복 횟수: ldmatrix / mma.sync 를 충분히 실행해서 shared 메트릭이 의미 있게 나오도록
constexpr int ITER = 4096;

// 16x16 타일 하나만 반복해서 사용하는 구조
// block: 1 warp (32 threads), grid: 여러 block

// ---------------------------------------------------------
// 1) Naive row-major shared layout
// ---------------------------------------------------------
__global__ void wmma_naive_shared_kernel(float* out_dummy)
{
#if __CUDA_ARCH__ >= 800
    // 16x16 A, 16x16 B (half)
    __shared__ __half shmemA[16 * 16];
    __shared__ __half shmemB[16 * 16];

    const int lane = threadIdx.x % 32;

    // row-major 그대로 채우기 (값은 크게 중요하지 않아서 1.0f 사용)
    for (int i = lane; i < 16 * 16; i += 32) {
        shmemA[i] = __float2half(1.0f);
        shmemB[i] = __float2half(1.0f);
    }
    __syncthreads();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float>                c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    // 반복적으로 shared -> register (ldmatrix) + mma.sync 실행
    for (int it = 0; it < ITER; ++it) {
        // 여기서 내부적으로 ldmatrix.sync.*.shared.b16 가 생성됨
        wmma::load_matrix_sync(a_frag, shmemA, 16);
        wmma::load_matrix_sync(b_frag, shmemB, 16);
        // Tensor Core MMA
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // 뭔가 하나라도 쓰게 해서 컴파일러가 완전히 죽여버리지 않도록
    if (lane == 0) {
        out_dummy[blockIdx.x] = c_frag.x[0];
    }
#else
    (void)out_dummy;
#endif
}

// ---------------------------------------------------------
// 2) Swizzled shared layout
//
//  - 같은 wmma load / mma 패턴이지만,
//    shared 메모리 인덱스를 XOR-swizzle 해서 bank conflict 완화 시도
//
//  - 여기 swizzle은 "수학적으로 정확한 GEMM 결과"를 위한 것이 아니라
//    "shared bank conflict 패턴 변화"를 보기 위한 용도임.
// ---------------------------------------------------------
__global__ void wmma_swizzled_shared_kernel(float* out_dummy)
{
#if __CUDA_ARCH__ >= 800
    __shared__ __half shmemA[16 * 16];
    __shared__ __half shmemB[16 * 16];

    const int lane = threadIdx.x % 32;

    // 간단 XOR swizzle:
    //   - 원래 index: i = row * 16 + col
    //   - row, col로 나눈 뒤,
    //   - col_group(하위 3비트)와 row LSB를 XOR 해서 column 재배치
    //
    //   swizzled_col = (col & 0x7) ^ ((row & 0x1) << 3);
    //   최종 index = row * 16 + (swizzled_col | (col & 0x8));
    //
    //   * 실제 최적 swizzle은 아키텍처/타일크기에 따라 다를 수 있고,
    //     여기선 데모 목적.
    for (int i = lane; i < 16 * 16; i += 32) {
        int row = i / 16;
        int col = i % 16;

        int col_group = col & 0x7;           // 하위 3비트 (0~7)
        int col_block = col & 0x8;           // 상위 비트 (8 또는 0)
        int swizzled_group = col_group ^ ((row & 0x1) << 3); // row LSB와 XOR
        int swizzled_col   = (swizzled_group & 0x7) | col_block;

        int sidx = row * 16 + swizzled_col;

        shmemA[sidx] = __float2half(1.0f);
        shmemB[sidx] = __float2half(1.0f);
    }
    __syncthreads();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float>                c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    // 똑같이 load + mma 반복
    for (int it = 0; it < ITER; ++it) {
        wmma::load_matrix_sync(a_frag, shmemA, 16);
        wmma::load_matrix_sync(b_frag, shmemB, 16);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    if (lane == 0) {
        out_dummy[blockIdx.x] = c_frag.x[0];
    }
#else
    (void)out_dummy;
#endif
}

// ---------------------------------------------------------
// Host side: 두 커널을 한 번씩 실행해서 ncu가 캡처할 수 있도록
// ---------------------------------------------------------
int main()
{
    printf("== 1.4.3 Test — Shared Memory Layout & ldmatrix / bank conflict ==\n");

    int device = 0;
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("Device %d: %s (SM %d.%d)\n",
           device, prop.name, prop.major, prop.minor);

    // grid/block 설정: 1 warp per block, 여러 block으로 약간 채움
    dim3 block(32, 1, 1);
    dim3 grid(64, 1, 1);  // SM 수보다 조금 많은 정도

    float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, grid.x * sizeof(float)));

    // ---- Naive layout ----
    printf("\n[Naive row-major shared layout]\n");
    wmma_naive_shared_kernel<<<grid, block>>>(d_out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- Swizzled layout ----
    printf("\n[Swizzled shared layout (XOR) ]\n");
    wmma_swizzled_shared_kernel<<<grid, block>>>(d_out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(d_out));

    printf("\nDone. Use Nsight Compute (ncu) to compare shared memory metrics.\n");
    printf("Example:\n");
    printf("  ncu --set full --kernel-name regex:.*naive.*     ./tensorcore_ldmatrix_smem_conflict.exe\n");
    printf("  ncu --set full --kernel-name regex:.*swizzled.* ./tensorcore_ldmatrix_smem_conflict.exe\n");

    return 0;
}
