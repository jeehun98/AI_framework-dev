#include <cuda_runtime.h>
#include <iostream>

__global__ void multiplyByTwo(int *a, int size) {
    int idx = threadIdx.x;
    if (idx < size) {
        a[idx] *= 2;
    }
}

int main() {
    const int size = 5;
    int h_a[size] = {1, 2, 3, 4, 5};
    int *d_a;

    cudaMalloc(&d_a, size * sizeof(int));
    cudaMemcpy(d_a, h_a, size * sizeof(int), cudaMemcpyHostToDevice);

    multiplyByTwo<<<1, size>>>(d_a, size);

    cudaMemcpy(h_a, d_a, size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_a);

    for (int i = 0; i < size; i++) {
        std::cout << h_a[i] << " ";
    }

    return 0;
}
