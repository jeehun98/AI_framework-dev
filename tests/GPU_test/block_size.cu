#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024 * 1024 // 데이터 크기

__global__ void kernel(float *data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        data[idx] += 1.0f; // 간단한 작업
    }
}

int main() {
    float *d_data;
    cudaMalloc(&d_data, N * sizeof(float));

    int blockSizes[] = {32, 64, 128, 256, 512, 1024};
    for (int blockSize : blockSizes) {
        int gridSize = (N + blockSize - 1) / blockSize;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        kernel<<<gridSize, blockSize>>>(d_data);
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("Block Size: %d, Time: %.3f ms\n", blockSize, milliseconds);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    cudaFree(d_data);
    return 0;
}
