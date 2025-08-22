// reduce_stride.cuh
#pragma once
#include <cuda_runtime.h>

static __global__ void reduce_mod_stride_kernel(const float* __restrict__ in, // [rowsB] = [batch*rows_per_sample]
                                                float* __restrict__ out,      // [rows_per_sample]
                                                int rows_per_sample,
                                                int batch_size) {
    int k = blockIdx.x * blockDim.x + threadIdx.x; // channel idx
    if (k >= rows_per_sample) return;
    float acc = 0.f;
    // in layout: [b * rows_per_sample + k]
    for (int b = 0; b < batch_size; ++b) {
        acc += in[b * rows_per_sample + k];
    }
    out[k] = acc;
}

inline void launch_reduce_mod_stride(const float* in, float* out,
                                     int rows_per_sample, int batch_size,
                                     cudaStream_t stream = 0) {
    int block = 256;
    int grid  = (rows_per_sample + block - 1) / block;
    reduce_mod_stride_kernel<<<grid, block, 0, stream>>>(in, out, rows_per_sample, batch_size);
}
