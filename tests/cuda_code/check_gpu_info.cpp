#include <iostream>
#include <cuda_runtime.h>

void printGPUInfo() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int device = 0; device < deviceCount; ++device) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);

        std::cout << "Device Number: " << device << std::endl;
        std::cout << "Device Name: " << prop.name << std::endl;
        std::cout << "Total Global Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "Shared Memory per Block: " << prop.sharedMemPerBlock << " bytes" << std::endl;
        std::cout << "Registers per Block: " << prop.regsPerBlock << std::endl;
        std::cout << "Warp Size: " << prop.warpSize << std::endl;
        std::cout << "Maximum Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "Number of Multiprocessors: " << prop.multiProcessorCount << std::endl;
        std::cout << "Clock Rate: " << prop.clockRate / 1000 << " MHz" << std::endl;
        std::cout << "-------------------------------" << std::endl;
    }
}

int main() {
    printGPUInfo();
    return 0;
}
