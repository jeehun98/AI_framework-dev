#include <cstdio>
#include <cuda_runtime.h>

constexpr int N = 1 << 24;  // 16M elements
constexpr int REPEAT = 32;

__global__
void seq_reuse_kernel(const float* __restrict__ in, float* out)
{
    float sum = 0.f;
    // 동일한 인덱스로 여러 번 접근 → cache reuse 기대
    for (int r = 0; r < REPEAT; ++r) {
        sum += in[threadIdx.x];
    }
    out[threadIdx.x] = sum;
}

__global__
void stride_large_kernel(const float* __restrict__ in, float* out)
{
    float sum = 0.f;
    int stride = 1024 * 32; // 32KB stride → 캐시 라인 건너뛰기
    for (int r = 0; r < REPEAT; ++r) {
        int idx = (threadIdx.x * stride + r) & (N - 1);
        sum += in[idx];
    }
    out[threadIdx.x] = sum;
}

__global__
void random_kernel(const float* __restrict__ in, const int* idxs, float* out)
{
    float sum = 0.f;
    for (int r = 0; r < REPEAT; ++r) {
        int idx = idxs[(threadIdx.x + r) & (N - 1)];
        sum += in[idx];
    }
    out[threadIdx.x] = sum;
}

int main() {
    float *d_in, *d_out;
    int *d_rand;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, 1024 * sizeof(float));
    cudaMalloc(&d_rand, N * sizeof(int));

    // dummy initialization
    float* h_in = new float[N];
    int* h_rand = new int[N];
    for (int i = 0; i < N; ++i) {
        h_in[i] = float(i);
        h_rand[i] = rand() % N;
    }

    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rand, h_rand, N * sizeof(int), cudaMemcpyHostToDevice);

    printf("== L1 / L2 Cache Test ==\n");

    printf("[1] Sequential Reuse\n");
    seq_reuse_kernel<<<1, 1024>>>(d_in, d_out);
    cudaDeviceSynchronize();

    printf("[2] Stride-Large Access\n");
    stride_large_kernel<<<1, 1024>>>(d_in, d_out);
    cudaDeviceSynchronize();

    printf("[3] Random Access Pattern\n");
    random_kernel<<<1, 1024>>>(d_in, d_rand, d_out);
    cudaDeviceSynchronize();

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_rand);
    delete[] h_in;
    delete[] h_rand;
    return 0;
}
// nvcc -O3 -arch=sm_86 l2_cache_test.cu -o l2_cache_test.exe

/*

# reuse
ncu --metrics lts__t_sectors_hit_rate.pct,lts__t_sectors_miss_rate.pct,lts__t_requests_global_op_read.sum,smsp__cycles_active.avg,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum --kernel-name regex:.*reuse.* ./l2_cache_test.exe

# stride
ncu --metrics lts__t_sectors_hit_rate.pct,lts__t_sectors_miss_rate.pct,lts__t_requests_global_op_read.sum,smsp__cycles_active.avg,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum --kernel-name regex:.*stride.* ./l2_cache_test.exe

# random
ncu --metrics lts__t_sectors_hit_rate.pct,lts__t_sectors_miss_rate.pct,lts__t_requests_global_op_read.sum,smsp__cycles_active.avg,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum --kernel-name regex:.*random.* ./l2_cache_test.exe




*/