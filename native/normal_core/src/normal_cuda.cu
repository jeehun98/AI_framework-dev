#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <math_constants.h>   // CUDART_PI_F
#include <cmath>
#include "normal/normal.hpp"

namespace normal {

// ------------------------ RNG utils ------------------------
// xorshift128+ PRNG
struct XorShift128p {
    unsigned long long s0, s1;

    __device__ __forceinline__ unsigned long long next_u64() {
        // Vigna (2014) xorshift128+
        unsigned long long x = s0;
        const unsigned long long y = s1;
        s0 = y;
        x ^= x << 23;
        s1 = x ^ y ^ (x >> 17) ^ (y >> 26);
        return s1 + y;
    }

    // 53-bit uniform in [0,1): use upper 53 bits and scale by 2^-53
    __device__ __forceinline__ float next_uniform01_open_open() {
        // produce a double in (0,1) with 53 bits; then cast to float
        // avoid exact 0 or 1 by adding a tiny offset and clamping
        const unsigned long long u = next_u64();
        // take top 53 bits
        const unsigned long long r53 = u >> 11;
        double d = (double)r53 * (1.0 / 9007199254740992.0); // 2^-53
        // open interval (0,1): push away from edges
        // 2^-54 ~ 1-2^-54
        const double eps = 5.551115123125783e-17; // 2^-54
        d = fmin(fmax(d, eps), 1.0 - eps);
        return (float)d;
    }
};

// SplitMix64: good seeding mixer
__device__ __forceinline__ unsigned long long splitmix64(unsigned long long& x) {
    unsigned long long z = (x += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

// ------------------------ Kernel ------------------------
__global__ void normal_box_muller_kernel(float* out, std::size_t n,
                                         float mean, float stdev,
                                         unsigned long long seed) {
    const std::size_t tid   = blockIdx.x * (std::size_t)blockDim.x + threadIdx.x;
    const std::size_t lanes = gridDim.x  * (std::size_t)blockDim.x;

    // per-thread RNG state seeding via SplitMix64
    unsigned long long x = seed ^ (0xD2B74407B1CE6E93ull + tid);
    XorShift128p rng;
    rng.s0 = splitmix64(x);
    rng.s1 = splitmix64(x);

    // Each iteration writes a pair (z0, z1)
    for (std::size_t i = tid * 2; i < n; i += lanes * 2) {
        float u1 = rng.next_uniform01_open_open();
        float u2 = rng.next_uniform01_open_open();

        // Box–Muller: r * (cos θ, sin θ)
        const float r = sqrtf(-2.f * __logf(u1));      // __logf: fast-math variant
        float s, c;
        __sincosf(2.f * CUDART_PI_F * u2, &s, &c);     // sincos in one shot

        const float z0 = r * c;
        const float z1 = r * s;

        if (i < n)       out[i]     = mean + stdev * z0;
        if (i + 1 < n)   out[i + 1] = mean + stdev * z1;
    }
}

// helper
static inline int round_up_div(int a, int b) { return (a + b - 1) / b; }

// d_out: device pointer
void generate_cuda_device(float* d_out, std::size_t n,
                          float mean, float stdev,
                          std::uint64_t seed, void* stream_v) {
    if (!d_out || n == 0) return;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_v);

    // 한 스레드가 2개의 샘플을 생산하므로, 필요한 thread 수는 ceil(n/2)
    const int block = 256;
    const int pairs = static_cast<int>((n + 1) / 2);
    const int grid  = max(1, round_up_div(pairs, block));

    normal_box_muller_kernel<<<grid, block, 0, stream>>>(
        d_out, n, mean, stdev, (unsigned long long)(seed ? seed : 0x9E3779B97F4A7C15ull)
    );
}

void generate_cuda(float* out, std::size_t n,
                   float mean, float stdev,
                   std::uint64_t seed, void* stream_v) {
    if (!out || n == 0) return;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_v);

    // detect if out is device pointer
    cudaPointerAttributes attr{};
#if CUDART_VERSION >= 10000
    cudaError_t q = cudaPointerGetAttributes(&attr, out);
    bool is_device = (q == cudaSuccess) && (attr.type == cudaMemoryTypeDevice);
#else
    cudaError_t q = cudaPointerGetAttributes(&attr, out);
    bool is_device = (q == cudaSuccess) && (attr.memoryType == cudaMemoryTypeDevice);
#endif

    if (is_device) {
        generate_cuda_device(out, n, mean, stdev, seed, stream);
    } else {
        float* d_out = nullptr;
        cudaMalloc(&d_out, n * sizeof(float));
        generate_cuda_device(d_out, n, mean, stdev, seed, stream);
        cudaMemcpyAsync(out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        cudaFree(d_out);
    }
}

} // namespace normal
#endif // __CUDACC__
