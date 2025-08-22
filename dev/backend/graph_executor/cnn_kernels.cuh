#pragma once
#include <cuda_runtime.h>

// ==================== Conv2D Forward (NCHW) ====================
// X: [N, Cin, Hin, Win]
// W: [Cout, Cin, Kh, Kw]
// Y: [N, Cout, Hout, Wout]
void launch_conv2d_forward_nchw(
    const float* X, const float* W, float* Y,
    int N, int Hin, int Win, int Cin,
    int Hout, int Wout, int Cout,
    int Kh, int Kw, int Sh, int Sw, int Ph, int Pw,
    cudaStream_t stream);

// ==================== Conv2D Backward dX (NCHW) ====================
// dY: [N, Cout, Hout, Wout]
// W : [Cout, Cin, Kh, Kw]
// dX: [N, Cin, Hin, Win]
void launch_conv2d_backward_input_nchw(
    const float* dY, const float* W, float* dX,
    int N, int Hin, int Win, int Cin,
    int Hout, int Wout, int Cout,
    int Kh, int Kw, int Sh, int Sw, int Ph, int Pw,
    cudaStream_t stream);

// ==================== Conv2D Backward dW (NCHW) ====================
// dY: [N, Cout, Hout, Wout]
// X : [N, Cin, Hin, Win]
// dW: [Cout, Cin, Kh, Kw]
void launch_conv2d_backward_weight_nchw(
    const float* dY, const float* X, float* dW,
    int N, int Hin, int Win, int Cin,
    int Hout, int Wout, int Cout,
    int Kh, int Kw, int Sh, int Sw, int Ph, int Pw,
    cudaStream_t stream);
