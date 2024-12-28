#include <iostream>
#include <cuda_runtime.h>

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0); // 0번 GPU 사용

    std::cout << "Device Name: " << prop.name << "\n";
    std::cout << "Max Threads Per Block: " << prop.maxThreadsPerBlock << "\n";
    std::cout << "Max Threads Dim: (" 
              << prop.maxThreadsDim[0] << ", " 
              << prop.maxThreadsDim[1] << ", " 
              << prop.maxThreadsDim[2] << ")\n";
    std::cout << "Max Grid Size: (" 
              << prop.maxGridSize[0] << ", " 
              << prop.maxGridSize[1] << ", " 
              << prop.maxGridSize[2] << ")\n";
    return 0;
}
