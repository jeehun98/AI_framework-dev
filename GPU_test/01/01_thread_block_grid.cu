#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <iomanip>

#define CUDA_CHECK(call)                                      \
    do {                                                      \
        cudaError_t err__ = (call);                           \
        if (err__ != cudaSuccess) {                           \
            std::cerr << "CUDA error: "                       \
                      << cudaGetErrorString(err__)            \
                      << " at " << __FILE__ << ":" << __LINE__\
                      << std::endl;                           \
            std::exit(1);                                     \
        }                                                     \
    } while (0)

__global__ void fill_indices(
    int* out_global,   // global thread id
    int* out_block,    // blockIdx.x
    int* out_thread,   // threadIdx.x
    int* out_warp,     // warp id within block
    int   N)
{
    int tid_in_block = threadIdx.x;
    int bid          = blockIdx.x;
    int bdim         = blockDim.x;
    int gid          = bid * bdim + tid_in_block;  // global thread id

    if (gid >= N) return;

    int warp_id_in_block = tid_in_block / 32;      // warp size = 32 가정

    out_global[gid] = gid;
    out_block[gid]  = bid;
    out_thread[gid] = tid_in_block;
    out_warp[gid]   = warp_id_in_block;
}

int main()
{
    // ===== 실험 파라미터 =====
    const int N = 128;          // 전체 “스레드 수” 느낌으로 생각
    const int block_size = 32;  // 여기 값을 32, 64, 128 등으로 바꿔볼 예정
    int grid_size = (N + block_size - 1) / block_size;

    std::cout << "N = " << N
              << ", block_size = " << block_size
              << ", grid_size = " << grid_size << std::endl;

    // ===== host 메모리 =====
    std::vector<int> h_global(N), h_block(N), h_thread(N), h_warp(N);

    // ===== device 메모리 =====
    int *d_global = nullptr, *d_block = nullptr, *d_thread = nullptr, *d_warp = nullptr;
    CUDA_CHECK(cudaMalloc(&d_global, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_block,  N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_thread, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_warp,   N * sizeof(int)));

    // ===== 커널 런치 =====
    fill_indices<<<grid_size, block_size>>>(d_global, d_block, d_thread, d_warp, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ===== 결과 복사 =====
    CUDA_CHECK(cudaMemcpy(h_global.data(), d_global, N * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_block.data(),  d_block,  N * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_thread.data(), d_thread, N * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_warp.data(),   d_warp,   N * sizeof(int), cudaMemcpyDeviceToHost));

    // ===== 앞부분만 출력 =====
    std::cout << "idx | global | block | thread | warp_in_block\n";
    std::cout << "----+--------+-------+--------+--------------\n";

    int print_N = std::min(N, 64);  // 앞 64개만 보기
    for (int i = 0; i < print_N; ++i) {
        std::cout << std::setw(3) << i << " | "
                  << std::setw(6) << h_global[i] << " | "
                  << std::setw(5) << h_block[i]  << " | "
                  << std::setw(6) << h_thread[i] << " | "
                  << std::setw(12) << h_warp[i]  << "\n";
    }

    // ===== 정리 =====
    CUDA_CHECK(cudaFree(d_global));
    CUDA_CHECK(cudaFree(d_block));
    CUDA_CHECK(cudaFree(d_thread));
    CUDA_CHECK(cudaFree(d_warp));

    return 0;
}

// nvcc 01_thread_block_grid.cu -o 01_thread_block_grid.exe
