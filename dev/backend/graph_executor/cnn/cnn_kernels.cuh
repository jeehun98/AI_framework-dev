// cnn/cnn_kernels.cuh
#pragma once
#include <cuda_runtime.h>

// NCHW 레이아웃의 나이브 Conv2D forward (zero-padding)
//  X: [N, Cin, Hin, Win], W: [Cout, Cin, Kh, Kw], Y: [N, Cout, Hout, Wout]
void launch_conv2d_forward_nchw(
    const float* X, const float* W, float* Y,
    int N, int Hin, int Win, int Cin,
    int Hout, int Wout, int Cout,
    int Kh, int Kw, int Sh, int Sw, int Ph, int Pw,
    cudaStream_t stream);

// ---- Backward (naive) ------------------------------------------------------

// dX (input gradient):
//   dX[n, ci, hi, wi] += sum_{co, kh, kw} dY[n, co, ho, wo] * W[co, ci, kh, kw]
//   where ho = (hi + Ph - kh)/Sh, wo = (wi + Pw - kw)/Sw  (정수/경계 체크)
void launch_conv2d_backward_input_nchw(
    const float* dY,  // [N, Cout, Hout, Wout]
    const float* W,   // [Cout, Cin, Kh, Kw]
    float* dX,        // [N, Cin, Hin, Win]
    int N, int Hin, int Win, int Cin,
    int Hout, int Wout, int Cout,
    int Kh, int Kw, int Sh, int Sw, int Ph, int Pw,
    cudaStream_t stream);

// dW (weight gradient):
//   dW[co,ci,kh,kw] += sum_{n,ho,wo} X[n,ci,hi,wi] * dY[n,co,ho,wo]
//   where hi = ho*Sh - Ph + kh, wi = wo*Sw - Pw + kw
void launch_conv2d_backward_weight_nchw(
    const float* dY,  // [N, Cout, Hout, Wout]
    const float* X,   // [N, Cin, Hin, Win]
    float* dW,        // [Cout, Cin, Kh, Kw]
    int N, int Hin, int Win, int Cin,
    int Hout, int Wout, int Cout,
    int Kh, int Kw, int Sh, int Sw, int Ph, int Pw,
    cudaStream_t stream);
