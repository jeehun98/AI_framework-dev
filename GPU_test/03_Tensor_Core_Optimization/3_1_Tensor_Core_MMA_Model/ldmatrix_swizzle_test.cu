// ldmatrix_swizzle_test.cu
// Test 3. ldmatrix + shared memory swizzling 효과 실험용
//
// - Kernel 1: ldmatrix_noswizzle_kernel
//   shared memory에 row-major로 그대로 배치 후 WMMA load
// - Kernel 2: ldmatrix_swizzle_kernel
//   shared memory에 간단한 swizzle 배치 후 WMMA load
//
// Nsight Compute에서 다음을 비교:
//   - l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum
//   - smsp__inst_executed_op_ldmatrix.sum

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

// 16x16 타일 기준 (m16n16k16)
constexpr int WM = 16;
constexpr int WN = 16;
constexpr int TILE_ELEMS = WM * WN;
constexpr int ITER = 64; // load를 여러 번 반복해서 통계 의미 있게

// -------------------------
// Kernel 1: no swizzle
// -------------------------
__global__ void ldmatrix_noswizzle_kernel(const half* __restrict__ A,
                                          float* __restrict__ out)
{
    // 한 block = 한 warp (32 threads)
    int lane_id = threadIdx.x & 31;

    extern __shared__ half shmem[]; // 크기: TILE_ELEMS

    // 1) 글로벌에서 16x16 타일을 row-major 그대로 shared에 복사
    for (int i = lane_id; i < TILE_ELEMS; i += 32) {
        int r = i / WN;
        int c = i % WN;
        shmem[r * WN + c] = A[r * WN + c];
    }
    __syncthreads();

    // 2) WMMA + ldmatrix (A fragment load)
    wmma::fragment<wmma::matrix_a, WM, WN, 16, half, wmma::row_major> a_frag;
    float acc = 0.0f;

    #pragma unroll
    for (int it = 0; it < ITER; ++it) {
        // ldmatrix는 내부에서 발생 (wmma::load_matrix_sync → PTX ldmatrix)
        wmma::load_matrix_sync(a_frag, shmem, WN);
        // 아무 연산이나 해서 최적화로 지워지지 않게
        acc += __half2float(a_frag.x[0]);
    }

    // warp 내 한 thread만 결과 기록 (side-effect용)
    if (lane_id == 0) {
        out[0] = acc;
    }
}

// -------------------------
// Kernel 2: swizzle 적용
//   - shared memory에 쓸 때, column에 간단한 XOR swizzle 적용
//   - 목적: 같은 ldmatrix load라도 bank conflict 패턴이 달라지게
// -------------------------
__global__ void ldmatrix_swizzle_kernel(const half* __restrict__ A,
                                        float* __restrict__ out)
{
    int lane_id = threadIdx.x & 31;
    extern __shared__ half shmem[]; // TILE_ELEMS

    // 1) 글로벌에서 16x16 타일을 shared로 복사하되, column swizzle 적용
    for (int i = lane_id; i < TILE_ELEMS; i += 32) {
        int r = i / WN;
        int c = i % WN;

        // 기본값: row-major
        half val = A[r * WN + c];

        // 간단한 swizzle 예시:
        //   - 하위 8열(0~7)에 대해, row parity에 따라 상위 비트 XOR
        //   - 목적: 같은 warp lane들이 서로 다른 bank를 더 잘 치도록 퍼뜨리는 것
        int base = c & 0x7;            // 하위 3bit (0~7)
        int high = (c & ~0x7);         // 나머지 상위 비트 (8 or 0)
        int swz  = base ^ ((r & 0x1) << 3); // row 짝/홀에 따라 0/8 XOR
        int c_swizzled = high | swz;        // 최종 col (0~15)

        shmem[r * WN + c_swizzled] = val;
    }
    __syncthreads();

    // 2) WMMA + ldmatrix load (배치는 swizzled지만, 여기선 값 의미는 신경 안 씀)
    wmma::fragment<wmma::matrix_a, WM, WN, 16, half, wmma::row_major> a_frag;
    float acc = 0.0f;

    #pragma unroll
    for (int it = 0; it < ITER; ++it) {
        wmma::load_matrix_sync(a_frag, shmem, WN);
        acc += __half2float(a_frag.x[0]);
    }

    if (lane_id == 0) {
        out[0] = acc;
    }
}

// -------------------------
// Host side
// -------------------------
int main(int argc, char** argv)
{
    // 단일 16x16 타일만 사용
    int M = WM;
    int N = WN;

    size_t bytesA = sizeof(half) * M * N;
    size_t bytesOut = sizeof(float);

    half* hA = (half*)std::malloc(bytesA);
    for (int i = 0; i < M * N; ++i) {
        float v = (std::rand() / (float)RAND_MAX) * 2.f - 1.f;
        hA[i] = __float2half(v);
    }

    half* dA = nullptr;
    float* dOut = nullptr;
    CHECK_CUDA(cudaMalloc(&dA, bytesA));
    CHECK_CUDA(cudaMalloc(&dOut, bytesOut));
    CHECK_CUDA(cudaMemcpy(dA, hA, bytesA, cudaMemcpyHostToDevice));

    dim3 block(32, 1, 1); // 한 warp
    dim3 grid(1, 1, 1);
    size_t shmem_size = sizeof(half) * TILE_ELEMS;

    printf("ldmatrix + swizzle test\n");

    // 1) no swizzle kernel 한 번 실행 (warm-up)
    ldmatrix_noswizzle_kernel<<<grid, block, shmem_size>>>(dA, dOut);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 2) swizzle kernel 한 번 실행 (warm-up)
    ldmatrix_swizzle_kernel<<<grid, block, shmem_size>>>(dA, dOut);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 결과를 굳이 볼 필요는 없음 (bank conflict / inst count가 목적)
    float hOut = 0.0f;
    CHECK_CUDA(cudaMemcpy(&hOut, dOut, bytesOut, cudaMemcpyDeviceToHost));
    printf("dummy out = %f\n", hOut);

    std::free(hA);
    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dOut));
    return 0;
}
/*
nvcc -O3 -std=c++17 -arch=sm_86 -lineinfo   -o ldmatrix_swizzle_test.exe ldmatrix_swizzle_test.cu

ncu --kernel-name regex:.*ldmatrix_noswizzle_kernel.*     --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,smsp__inst_executed_pipe_tensor,smsp__inst_executed_pipe_tensor_op_hmma,smsp__pipe_tensor_cycles_active     --launch-skip 0 --launch-count 1     --set full     .\ldmatrix_swizzle_test.exe
ncu --kernel-name regex:.*ldmatrix_swizzle_kernel.*     --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,smsp__inst_executed_pipe_tensor,smsp__inst_executed_pipe_tensor_op_hmma,smsp__pipe_tensor_cycles_active     --launch-skip 0 --launch-count 1     --set full     .\ldmatrix_swizzle_test.exe


*/