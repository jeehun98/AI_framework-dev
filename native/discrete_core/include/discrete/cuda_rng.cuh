#pragma once
#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <math_constants.h>

struct XorShift128p {
    unsigned long long s0, s1;
    __device__ __forceinline__ unsigned long long next_u64() {
        unsigned long long x = s0;
        const unsigned long long y = s1;
        s0 = y;
        x ^= x << 23;
        s1 = x ^ y ^ (x >> 17) ^ (y >> 26);
        return s1 + y;
    }
    __device__ __forceinline__ float uniform01_open_open() {
        const unsigned long long u = next_u64();
        const unsigned long long r53 = u >> 11;
        double d = (double)r53 * (1.0/9007199254740992.0); // 2^-53
        const double eps = 5.551115123125783e-17;          // 2^-54
        d = fmin(fmax(d, eps), 1.0 - eps);
        return (float)d;
    }
};

__device__ __forceinline__ unsigned long long splitmix64(unsigned long long& x) {
    unsigned long long z = (x += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

struct ThreadRNG {
    XorShift128p rng;
    __device__ ThreadRNG(unsigned long long seed, unsigned long long tid) {
        unsigned long long x = seed ^ (0xD2B74407B1CE6E93ull + tid);
        rng.s0 = splitmix64(x);
        rng.s1 = splitmix64(x);
    }
    __device__ __forceinline__ float u01() { return rng.uniform01_open_open(); }
};
#endif
