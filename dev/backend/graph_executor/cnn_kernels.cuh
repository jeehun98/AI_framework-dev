#pragma once
#include <cuda_runtime.h>

// NHWC 입력, W: (OC, IC, KH, KW)
// 출력 Y: NHWC (B, Hout, Wout, OC)
// stride=(Sh,Sw), pad=(Ph,Pw)
void launch_conv2d_forward_nhwc(
    const float* X, const float* W, float* Y,
    int B, int Hin, int Win, int Cin,
    int Hout, int Wout, int Cout,
    int Kh, int Kw, int Sh, int Sw, int Ph, int Pw,
    cudaStream_t stream = 0);

// dX = dY (*) W
void launch_conv2d_backward_input_nhwc(
    const float* dY, const float* W, float* dX,
    int B, int Hin, int Win, int Cin,
    int Hout, int Wout, int Cout,
    int Kh, int Kw, int Sh, int Sw, int Ph, int Pw,
    cudaStream_t stream = 0);

// dW = X (*) dY
void launch_conv2d_backward_weight_nhwc(
    const float* dY, const float* X, float* dW,
    int B, int Hin, int Win, int Cin,
    int Hout, int Wout, int Cout,
    int Kh, int Kw, int Sh, int Sw, int Ph, int Pw,
    cudaStream_t stream = 0);
