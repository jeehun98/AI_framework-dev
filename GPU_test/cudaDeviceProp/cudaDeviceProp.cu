#include <iostream>
#include <cuda_runtime.h>

int main() {
    cudaDeviceProp prop;
    int device_id = 0;

    cudaGetDeviceProperties(&prop, device_id);

    std::cout << "===== GPU Device Info =====" << std::endl;
    std::cout << "Name            : " << prop.name << std::endl;
    std::cout << "SM Count        : " << prop.multiProcessorCount << std::endl;
    std::cout << "Warp Size       : " << prop.warpSize << std::endl;
    std::cout << "Max Threads/SM  : " << prop.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "Max Threads/Block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Regs Per SM     : " << prop.regsPerMultiprocessor << std::endl;
    std::cout << "Regs Per Block  : " << prop.regsPerBlock << std::endl;
    std::cout << "Shared Mem Per SM (bytes) : " << prop.sharedMemPerMultiprocessor << std::endl;
    std::cout << "Shared Mem Per Block (bytes) : " << prop.sharedMemPerBlock << std::endl;
    std::cout << "Max Blocks/SM   : " << prop.maxBlocksPerMultiProcessor << std::endl;

    std::cout << "Compute Capability : "
              << prop.major << "." << prop.minor << std::endl;

    std::cout << "Clock Rate (KHz)   : " << prop.clockRate << std::endl;

    return 0;
}

/*
===== GPU Device Info =====
Name            : NVIDIA GeForce RTX 3080 Ti Laptop GPU
SM Count        : 58
Warp Size       : 32
Max Threads/SM  : 1536
Max Threads/Block: 1024
Regs Per SM     : 65536
Regs Per Block  : 65536
Shared Mem Per SM (bytes) : 102400
Shared Mem Per Block (bytes) : 49152
Max Blocks/SM   : 16
Compute Capability : 8.6
Clock Rate (KHz)   : 1125000
*/
