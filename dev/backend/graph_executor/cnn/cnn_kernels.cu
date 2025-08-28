// cnn/cnn_kernels.cu
#include "cnn_kernels.cuh"
#include <cuda_runtime.h>
#include "../ge/cuda_check.cuh"

// cnn/cnn_kernels.cu
#include "cnn_kernels.cuh"
#include <cuda_runtime.h>

// --------------------------- Forward (naive, NCHW) --------------------------

__global__ void conv2d_forward_nchw_naive_kernel(
    const float* __restrict__ X,  // [N,Cin,Hin,Win]
    const float* __restrict__ W,  // [Cout,Cin,Kh,Kw]
    float* __restrict__ Y,        // [N,Cout,Hout,Wout]
    int N, int Hin, int Win, int Cin,
    int Hout, int Wout, int Cout,
    int Kh, int Kw, int Sh, int Sw, int Ph, int Pw)
{
    const long long nHW  = (long long)Hout * Wout;
    const long long nCHW = (long long)Cout * nHW;
    const long long idx  = blockIdx.x * blockDim.x + threadIdx.x; // [0 .. N*Cout*Hout*Wout)
    const long long tot  = (long long)N * nCHW;
    if (idx >= tot) return;

    long long t = idx;
    const int n  = (int)(t / nCHW); t -= (long long)n * nCHW;
    const int co = (int)(t / nHW);  t -= (long long)co * nHW;
    const int ho = (int)(t / Wout);
    const int wo = (int)(t - (long long)ho * Wout);

    const int hi0 = ho * Sh - Ph;
    const int wi0 = wo * Sw - Pw;

    float acc = 0.f;

    for (int ci = 0; ci < Cin; ++ci) {
        for (int kh = 0; kh < Kh; ++kh) {
            const int hi = hi0 + kh;
            if (hi < 0 || hi >= Hin) continue;
            for (int kw = 0; kw < Kw; ++kw) {
                const int wi = wi0 + kw;
                if (wi < 0 || wi >= Win) continue;

                const long long x_off = ((long long)n * Cin + ci) * Hin * Win
                                      + (long long)hi * Win + wi;
                const long long w_off = ((long long)co * Cin + ci) * Kh * Kw
                                      + (long long)kh * Kw + kw;
                acc += X[x_off] * W[w_off];
            }
        }
    }

    const long long y_off = ((long long)n * Cout + co) * Hout * Wout
                          + (long long)ho * Wout + wo;
    Y[y_off] = acc;
}

void launch_conv2d_forward_nchw(
    const float* X, const float* W, float* Y,
    int N, int Hin, int Win, int Cin,
    int Hout, int Wout, int Cout,
    int Kh, int Kw, int Sh, int Sw, int Ph, int Pw,
    cudaStream_t stream)
{
    const long long tot = (long long)N * Cout * Hout * Wout;
    const int threads = 256;
    const int blocks  = (int)((tot + threads - 1) / threads);
    conv2d_forward_nchw_naive_kernel<<<blocks, threads, 0, stream>>>(
        X, W, Y, N, Hin, Win, Cin, Hout, Wout, Cout, Kh, Kw, Sh, Sw, Ph, Pw
    );
    CUDA_CHECK(cudaGetLastError());
}

// --------------------------- Backward: dX (naive) ---------------------------
// thread per (n, co, ho, wo) with atomicAdd into dX
__global__ void conv2d_backward_input_nchw_naive_kernel(
    const float* __restrict__ dY, // [N,Cout,Hout,Wout]
    const float* __restrict__ W,  // [Cout,Cin,Kh,Kw]
    float* __restrict__ dX,       // [N,Cin,Hin,Win]
    int N, int Hin, int Win, int Cin,
    int Hout, int Wout, int Cout,
    int Kh, int Kw, int Sh, int Sw, int Ph, int Pw)
{
    const long long nHW  = (long long)Hout * Wout;
    const long long nCHW = (long long)Cout * nHW;
    const long long idx  = blockIdx.x * blockDim.x + threadIdx.x; // [0 .. N*Cout*Hout*Wout)
    const long long tot  = (long long)N * nCHW;
    if (idx >= tot) return;

    long long t = idx;
    const int n  = (int)(t / nCHW); t -= (long long)n * nCHW;
    const int co = (int)(t / nHW);  t -= (long long)co * nHW;
    const int ho = (int)(t / Wout);
    const int wo = (int)(t - (long long)ho * Wout);

    const long long dy_off = ((long long)n * Cout + co) * Hout * Wout
                           + (long long)ho * Wout + wo;
    const float g = dY[dy_off];

    const int hi0 = ho * Sh - Ph;
    const int wi0 = wo * Sw - Pw;

    for (int ci = 0; ci < Cin; ++ci) {
        for (int kh = 0; kh < Kh; ++kh) {
            const int hi = hi0 + kh;
            if (hi < 0 || hi >= Hin) continue;
            for (int kw = 0; kw < Kw; ++kw) {
                const int wi = wi0 + kw;
                if (wi < 0 || wi >= Win) continue;

                const long long w_off = ((long long)co * Cin + ci) * Kh * Kw
                                      + (long long)kh * Kw + kw;
                const long long dx_off = ((long long)n * Cin + ci) * Hin * Win
                                       + (long long)hi * Win + wi;
                atomicAdd(&dX[dx_off], g * W[w_off]);
            }
        }
    }
}

void launch_conv2d_backward_input_nchw(
    const float* dY, const float* W, float* dX,
    int N, int Hin, int Win, int Cin,
    int Hout, int Wout, int Cout,
    int Kh, int Kw, int Sh, int Sw, int Ph, int Pw,
    cudaStream_t stream)
{
    const long long tot = (long long)N * Cout * Hout * Wout;
    const int threads = 256;
    const int blocks  = (int)((tot + threads - 1) / threads);
    conv2d_backward_input_nchw_naive_kernel<<<blocks, threads, 0, stream>>>(
        dY, W, dX, N, Hin, Win, Cin, Hout, Wout, Cout, Kh, Kw, Sh, Sw, Ph, Pw
    );
    CUDA_CHECK(cudaGetLastError());
}

// -------------------------- Backward: dW (naive) ----------------------------
// thread per (n, co, ho, wo) with atomicAdd into dW
__global__ void conv2d_backward_weight_nchw_naive_kernel(
    const float* __restrict__ dY, // [N,Cout,Hout,Wout]
    const float* __restrict__ X,  // [N,Cin,Hin,Win]
    float* __restrict__ dW,       // [Cout,Cin,Kh,Kw]
    int N, int Hin, int Win, int Cin,
    int Hout, int Wout, int Cout,
    int Kh, int Kw, int Sh, int Sw, int Ph, int Pw)
{
    const long long nHW  = (long long)Hout * Wout;
    const long long nCHW = (long long)Cout * nHW;
    const long long idx  = blockIdx.x * blockDim.x + threadIdx.x; // [0 .. N*Cout*Hout*Wout)
    const long long tot  = (long long)N * nCHW;
    if (idx >= tot) return;

    long long t = idx;
    const int n  = (int)(t / nCHW); t -= (long long)n * nCHW;
    const int co = (int)(t / nHW);  t -= (long long)co * nHW;
    const int ho = (int)(t / Wout);
    const int wo = (int)(t - (long long)ho * Wout);

    const long long dy_off = ((long long)n * Cout + co) * Hout * Wout
                           + (long long)ho * Wout + wo;
    const float g = dY[dy_off];

    const int hi0 = ho * Sh - Ph;
    const int wi0 = wo * Sw - Pw;

    for (int ci = 0; ci < Cin; ++ci) {
        for (int kh = 0; kh < Kh; ++kh) {
            const int hi = hi0 + kh;
            if (hi < 0 || hi >= Hin) continue;
            for (int kw = 0; kw < Kw; ++kw) {
                const int wi = wi0 + kw;
                if (wi < 0 || wi >= Win) continue;

                const long long x_off = ((long long)n * Cin + ci) * Hin * Win
                                      + (long long)hi * Win + wi;
                const long long dw_off = ((long long)co * Cin + ci) * Kh * Kw
                                       + (long long)kh * Kw + kw;
                atomicAdd(&dW[dw_off], X[x_off] * g);
            }
        }
    }
}

void launch_conv2d_backward_weight_nchw(
    const float* dY, const float* X, float* dW,
    int N, int Hin, int Win, int Cin,
    int Hout, int Wout, int Cout,
    int Kh, int Kw, int Sh, int Sw, int Ph, int Pw,
    cudaStream_t stream)
{
    const long long tot = (long long)N * Cout * Hout * Wout;
    const int threads = 256;
    const int blocks  = (int)((tot + threads - 1) / threads);
    conv2d_backward_weight_nchw_naive_kernel<<<blocks, threads, 0, stream>>>(
        dY, X, dW, N, Hin, Win, Cin, Hout, Wout, Cout, Kh, Kw, Sh, Sw, Ph, Pw
    );
    CUDA_CHECK(cudaGetLastError());
}
