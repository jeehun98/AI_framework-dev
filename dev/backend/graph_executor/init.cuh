#pragma once
#include <curand_kernel.h>

__device__ float xavier_init(int fan_in, int fan_out, int seed, int idx) {
    curandState state;
    curand_init(seed, idx, 0, &state);
    float stddev = sqrtf(2.0f / (fan_in + fan_out));
    return curand_normal(&state) * stddev;
}
