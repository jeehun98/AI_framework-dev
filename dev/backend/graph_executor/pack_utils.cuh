// pack_utils.cuh
#pragma once
#include <cuda_runtime.h>

void launch_pack_rm_to_nchw(
    const float* rm,   // [N, Cout, Hout*Wout] contiguous
    float* nchw,       // [N, Cout, Hout, Wout]
    int N, int Cout, int Hout, int Wout,
    cudaStream_t stream = 0);
