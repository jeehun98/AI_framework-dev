#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);  // Use the first GPU

    size_t totalGlobalMem = prop.totalGlobalMem;
    size_t freeMem = totalGlobalMem * 0.8;  // Use about 80% of memory
    int maxMatrixSize = static_cast<int>(sqrt(freeMem / (3 * sizeof(float))));

    std::cout << "Total Global Memory: " << totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Recommended maximum matrix size: " << maxMatrixSize << " x " << maxMatrixSize << std::endl;

    return 0;
}
