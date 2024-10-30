#include <cuda_runtime.h>
#include <iostream>

void printDeviceProperties() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    for (int device = 0; device < deviceCount; device++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);

        std::cout << "Device " << device << ": " << prop.name << std::endl;
        std::cout << "  Total Global Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  Number of Multiprocessors: " << prop.multiProcessorCount << std::endl;
        std::cout << "  Max Threads Per Block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "  Max Thread Dimensions (Per Block): (" 
                  << prop.maxThreadsDim[0] << ", " 
                  << prop.maxThreadsDim[1] << ", " 
                  << prop.maxThreadsDim[2] << ")" << std::endl;
        std::cout << "  Max Grid Dimensions: (" 
                  << prop.maxGridSize[0] << ", " 
                  << prop.maxGridSize[1] << ", " 
                  << prop.maxGridSize[2] << ")" << std::endl;
        std::cout << "  Memory Clock Rate: " << prop.memoryClockRate / 1000 << " MHz" << std::endl;
        std::cout << "  Memory Bus Width: " << prop.memoryBusWidth << " bits" << std::endl;
        std::cout << "  Warp Size: " << prop.warpSize << std::endl;
        std::cout << "--------------------------------------------" << std::endl;
    }
}

int main() {
    printDeviceProperties();
    return 0;
}
