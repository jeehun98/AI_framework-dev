#include <cstdio>
#include <cuda_runtime.h>

constexpr int THREADS = 128;
constexpr int BLOCKS  = 1;

__global__
void smem_broadcast(int* out) {
    __shared__ int sm[32];

    int tid = threadIdx.x;

    // broadcast: 모든 thread가 동일 인덱스를 읽음
    int v = sm[0];

    out[tid] = v;
}

__global__
void smem_noconflict(int* out) {
    __shared__ int sm[128];

    int tid = threadIdx.x;

    // conflict-free: 연속 index 접근
    int v = sm[tid];

    out[tid] = v;
}

__global__
void smem_full_conflict(int* out) {
    __shared__ int sm[128];

    int tid = threadIdx.x;

    // worst conflict: stride = 32 → 모든 warp thread가 같은 bank로 들어감
    int v = sm[(tid * 32) & 127];

    out[tid] = v;
}

int main() {
    int* d_out;
    cudaMalloc(&d_out, THREADS * sizeof(int));

    printf("== smem bank conflict test ==\n");

    printf("\n[Broadcast] conflict-free broadcast\n");
    smem_broadcast<<<BLOCKS, THREADS>>>(d_out);
    cudaDeviceSynchronize();

    printf("\n[Stride-1] fully conflict-free\n");
    smem_noconflict<<<BLOCKS, THREADS>>>(d_out);
    cudaDeviceSynchronize();

    printf("\n[Stride-32] worst conflict expected\n");
    smem_full_conflict<<<BLOCKS, THREADS>>>(d_out);
    cudaDeviceSynchronize();

    cudaFree(d_out);
    return 0;
}


// nvcc -O3 -arch=sm_86 smem_bank_conflict_test.cu -o smem_bank_conflict_test.exe

// ncu --set full --kernel-name regex:smem_.* .\smem_bank_conflict_test.exe