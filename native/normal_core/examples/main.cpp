#include <iostream>
#include <vector>
#include <chrono>
#include "normal/normal.hpp"

int main() {
    const std::size_t N = 1 << 20; // 1M
    const float mean = 2.0f, stdev = 3.0f;
    const std::uint64_t seed = 42;

    std::vector<float> buf(N);

    auto t0 = std::chrono::high_resolution_clock::now();
    normal::generate_cpu(buf.data(), N, mean, stdev, seed);
    auto t1 = std::chrono::high_resolution_clock::now();

    double mu=0, sd=0;
    normal::estimate_stats(buf.data(), N, mu, sd);

    std::chrono::duration<double, std::milli> ms = t1 - t0;

    std::cout << "[CPU] N=" << N << " time=" << ms.count() << " ms  " << "mean =~ " << mu << " std =~ " << sd << "\n";

#ifdef NORMAL_WITH_CUDA
    // 같은 버퍼에 다시 생성해 보자 (GPU -> Host 복사 포함)
    t0 = std::chrono::high_resolution_clock::now();
    normal::generate_cuda(buf.data(), N, mean, stdev, seed);
    t1 = std::chrono::high_resolution_clock::now();
    normal::estimate_stats(buf.data(), N, mu, sd);
    ms = t1 - t0;
    std::cout << "[CUDA] N=" << N << " time=" << ms.count() << " ms  "
              << "mean =~ " << mu << " std =~ " << sd << "\n";
#endif
    return 0;
}
