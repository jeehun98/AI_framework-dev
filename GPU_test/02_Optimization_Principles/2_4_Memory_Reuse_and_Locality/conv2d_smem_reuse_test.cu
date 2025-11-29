#include <cstdio>
#include <cstdlib>
#include <vector>
#include <iostream>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA Error %s:%d: %s\n",                    \
                        __FILE__, __LINE__, cudaGetErrorString(err));         \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

constexpr int K = 3;              // 3x3 conv
constexpr int R = K / 2;          // radius = 1
constexpr int BLOCK_W = 16;
constexpr int BLOCK_H = 16;

// ----------------------------
// Naive Conv2D kernel
// ----------------------------
__global__
void conv2d_naive_kernel(const float* __restrict__ input,
                         const float* __restrict__ kernel,
                         float* __restrict__ output,
                         int H, int W)
{
    int ox = blockIdx.x * blockDim.x + threadIdx.x; // output x
    int oy = blockIdx.y * blockDim.y + threadIdx.y; // output y

    int outH = H - 2 * R;
    int outW = W - 2 * R;

    if (ox >= outW || oy >= outH) return;

    float acc = 0.0f;

    int in_y = oy;      // input top-left y (with valid conv)
    int in_x = ox;      // input top-left x

    // naive: 매 output 픽셀마다 주변 3x3을 그대로 global에서 읽음
    for (int ky = 0; ky < K; ++ky) {
        for (int kx = 0; kx < K; ++kx) {
            int iy = in_y + ky;
            int ix = in_x + kx;
            float v = input[iy * W + ix];
            float w = kernel[ky * K + kx];
            acc += v * w;
        }
    }

    output[oy * outW + ox] = acc;
}

// ----------------------------
// Tiled Conv2D kernel (SMEM reuse)
// ----------------------------
//
// 각 블록:
// - BLOCK_H x BLOCK_W output 영역 담당
// - 그 주변 입력을 (BLOCK_H+2R) x (BLOCK_W+2R) tile로 shared에 올려서
//   여러 thread가 재사용
//
__global__
void conv2d_tiled_kernel(const float* __restrict__ input,
                         const float* __restrict__ kernel,
                         float* __restrict__ output,
                         int H, int W)
{
    extern __shared__ float shmem[]; // 크기: (BLOCK_H + 2R) * (BLOCK_W + 2R)

    int outH = H - 2 * R;
    int outW = W - 2 * R;

    int ox = blockIdx.x * blockDim.x + threadIdx.x; // output coord
    int oy = blockIdx.y * blockDim.y + threadIdx.y;

    // 이 블록이 담당하는 입력 tile의 좌상단 (input space)
    int tile_x0 = blockIdx.x * blockDim.x;
    int tile_y0 = blockIdx.y * blockDim.y;

    int tile_W = BLOCK_W + 2 * R;
    int tile_H = BLOCK_H + 2 * R;

    // shared memory 2D index: [ty][tx]
    // 전체 thread가 협동해서 tile 전체를 load
    int tIdx = threadIdx.y * blockDim.x + threadIdx.x;
    int tCount = blockDim.x * blockDim.y;
    int tileSize = tile_W * tile_H;

    for (int idx = tIdx; idx < tileSize; idx += tCount) {
        int ty = idx / tile_W;
        int tx = idx % tile_W;

        int gx = tile_x0 + tx; // global x
        int gy = tile_y0 + ty; // global y

        float val = 0.0f;
        if (gx >= 0 && gx < W && gy >= 0 && gy < H) {
            val = input[gy * W + gx];
        }
        shmem[ty * tile_W + tx] = val;
    }

    __syncthreads();

    if (ox >= outW || oy >= outH) return;

    // 이 thread의 output 위치에 대한 tile 내부 좌표
    int local_x = threadIdx.x + R;
    int local_y = threadIdx.y + R;

    float acc = 0.0f;

    // kernel은 그대로 global에서 읽어도 되고, const mem으로 옮겨도 됨
    for (int ky = 0; ky < K; ++ky) {
        for (int kx = 0; kx < K; ++kx) {
            int sy = local_y + ky - R;
            int sx = local_x + kx - R;
            float v = shmem[sy * tile_W + sx];
            float w = kernel[ky * K + kx];
            acc += v * w;
        }
    }

    output[oy * outW + ox] = acc;
}

// ----------------------------
// Host reference conv2d (valid)
// ----------------------------
void conv2d_host(const std::vector<float>& input,
                 const std::vector<float>& kernel,
                 std::vector<float>& output,
                 int H, int W)
{
    int outH = H - 2 * R;
    int outW = W - 2 * R;
    output.assign(outH * outW, 0.0f);

    for (int oy = 0; oy < outH; ++oy) {
        for (int ox = 0; ox < outW; ++ox) {
            float acc = 0.0f;
            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    int iy = oy + ky;
                    int ix = ox + kx;
                    float v = input[iy * W + ix];
                    float w = kernel[ky * K + kx];
                    acc += v * w;
                }
            }
            output[oy * outW + ox] = acc;
        }
    }
}

float max_abs_diff(const std::vector<float>& a,
                   const std::vector<float>& b)
{
    if (a.size() != b.size()) {
        return -1.0f;
    }
    float m = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float d = std::fabs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

int main()
{
    const int H = 1024;
    const int W = 1024;
    const int outH = H - 2 * R;
    const int outW = W - 2 * R;

    std::printf("=== Spatial Locality Test 2: Conv2D naive vs local-tile reuse ===\n");
    std::printf("Input:  H=%d, W=%d (3x3 valid conv -> outH=%d, outW=%d)\n",
                H, W, outH, outW);
    std::printf("Block:  %dx%d, tile in SMEM: %dx%d (with halo)\n",
                BLOCK_H, BLOCK_W,
                BLOCK_H + 2 * R, BLOCK_W + 2 * R);

    // Host buffers
    std::vector<float> h_input(H * W);
    std::vector<float> h_kernel(K * K);
    std::vector<float> h_out_ref(outH * outW);
    std::vector<float> h_out_naive(outH * outW);
    std::vector<float> h_out_tiled(outH * outW);

    // Init input & kernel
    for (int i = 0; i < H * W; ++i) {
        h_input[i] = static_cast<float>((i % 256) / 255.0f);
    }
    for (int i = 0; i < K * K; ++i) {
        h_kernel[i] = 1.0f / (K * K); // 간단하게 평균 필터
    }

    // Host reference
    std::printf("Computing host reference...\n");
    conv2d_host(h_input, h_kernel, h_out_ref, H, W);

    // Device buffers
    float *d_input = nullptr, *d_kernel = nullptr;
    float *d_out_naive = nullptr, *d_out_tiled = nullptr;

    CHECK_CUDA(cudaMalloc(&d_input, H * W * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_kernel, K * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out_naive, outH * outW * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out_tiled, outH * outW * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_input, h_input.data(),
                          H * W * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_kernel, h_kernel.data(),
                          K * K * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(BLOCK_W, BLOCK_H);
    dim3 grid_naive((outW + BLOCK_W - 1) / BLOCK_W,
                    (outH + BLOCK_H - 1) / BLOCK_H);
    dim3 grid_tiled = grid_naive; // 같은 output partition

    // Timing용 이벤트
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // ----------------------------
    // Run naive kernel
    // ----------------------------
    std::printf("\n[Naive Conv2D] running...\n");
    // warm-up
    conv2d_naive_kernel<<<grid_naive, block>>>(d_input, d_kernel,
                                               d_out_naive, H, W);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start));
    conv2d_naive_kernel<<<grid_naive, block>>>(d_input, d_kernel,
                                               d_out_naive, H, W);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_naive = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms_naive, start, stop));

    CHECK_CUDA(cudaMemcpy(h_out_naive.data(), d_out_naive,
                          outH * outW * sizeof(float), cudaMemcpyDeviceToHost));
    float max_diff_naive = max_abs_diff(h_out_naive, h_out_ref);

    // ----------------------------
    // Run tiled kernel
    // ----------------------------
    std::printf("\n[Tiled Conv2D (SMEM reuse)] running...\n");
    size_t shmem_bytes = (BLOCK_W + 2 * R) * (BLOCK_H + 2 * R) * sizeof(float);

    // warm-up
    conv2d_tiled_kernel<<<grid_tiled, block, shmem_bytes>>>(
        d_input, d_kernel, d_out_tiled, H, W);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start));
    conv2d_tiled_kernel<<<grid_tiled, block, shmem_bytes>>>(
        d_input, d_kernel, d_out_tiled, H, W);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_tiled = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms_tiled, start, stop));

    CHECK_CUDA(cudaMemcpy(h_out_tiled.data(), d_out_tiled,
                          outH * outW * sizeof(float), cudaMemcpyDeviceToHost));
    float max_diff_tiled = max_abs_diff(h_out_tiled, h_out_ref);

    // ----------------------------
    // Theoretical global load / SMEM reuse 계산
    // ----------------------------
    // naive:
    //  - output 픽셀 수: outH * outW
    //  - 각 output: 3x3 입력 = 9개 global load
    // => input global load 이론값
    double naive_global_loads = static_cast<double>(outH) *
                                static_cast<double>(outW) * (K * K);

    // tiled:
    //  - 각 block: (BLOCK_W+2R) x (BLOCK_H+2R) 입력 tile을 한 번 global에서 읽음
    //  - block 수: grid_tiled.x * grid_tiled.y
    double num_blocks = static_cast<double>(grid_tiled.x) *
                        static_cast<double>(grid_tiled.y);
    double tile_elems = static_cast<double>(BLOCK_W + 2 * R) *
                        static_cast<double>(BLOCK_H + 2 * R);
    double tiled_global_loads = num_blocks * tile_elems;

    double reuse_factor = naive_global_loads / tiled_global_loads;

    std::printf("\n=== Results ===\n");
    std::printf("Naive Conv2D time     : %8.3f ms, max diff vs ref = %e\n",
                ms_naive, max_diff_naive);
    std::printf("Tiled Conv2D time     : %8.3f ms, max diff vs ref = %e\n",
                ms_tiled, max_diff_tiled);
    std::printf("Speedup (naive / tiled) = %.2fx\n", ms_naive / ms_tiled);

    std::printf("\n--- Theoretical global load / reuse ---\n");
    std::printf("Naive  global loads (input elements): %.0f\n", naive_global_loads);
    std::printf("Tiled  global loads (input elements): %.0f\n", tiled_global_loads);
    std::printf("Reuse factor (naive / tiled)        : %.2fx\n", reuse_factor);
    std::printf(" (이론적으로 global load를 약 %.1fx 줄이고, 그만큼 SMEM에서 재사용)\n",
                reuse_factor);

    // cleanup
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_kernel));
    CHECK_CUDA(cudaFree(d_out_naive));
    CHECK_CUDA(cudaFree(d_out_tiled));

    return 0;
}

/*
nvcc -O3 -arch=sm_86 conv2d_smem_reuse_test.cu -o conv2d_smem_reuse_test.exe

ncu --kernel-name regex:conv2d_naive_kernel.*     --metrics dram__bytes_read.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum     ./conv2d_smem_reuse_test.exe
ncu --kernel-name regex:conv2d_tiled_kernel.*     --metrics dram__bytes_read.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum     ./conv2d_smem_reuse_test.exe

*/
