// rnn_kernels.cuh
#pragma once
#include <cuda_runtime.h>

enum RnnActivation { RNN_TANH=0, RNN_RELU=1, RNN_SIGMOID=2 };

void launch_rnn_forward_simple(
    const float* X, const float* Wx, const float* Wh, const float* b,
    const float* h0, float* H_T, float* H_seq,
    int B, int T, int D, int H, RnnActivation act,
    cudaStream_t stream);

void launch_rnn_backward_simple(
    const float* X, const float* Wx, const float* Wh, const float* b,
    const float* h0, const float* H_seq,
    const float* dY_T, const float* dY_seq,
    float* dX, float* dWx, float* dWh, float* db, float* dH0,
    int B, int T, int D, int H, RnnActivation act,
    cudaStream_t stream);
