#include "cnn_kernels.cuh"

#define TW 16

// ==================== Forward (NCHW) ====================
// X: [N, Cin, Hin, Win]
// W: [Cout, Cin, Kh, Kw]
// Y: [N, Cout, Hout, Wout]
__global__ void conv2d_fwd_nchw_kernel(
    const float* __restrict__ X,
    const float* __restrict__ W,
    float* __restrict__ Y,
    int N, int Hin, int Win, int Cin,
    int Hout, int Wout, int Cout,
    int Kh, int Kw, int Sh, int Sw, int Ph, int Pw)
{
    // grid.z = N*Cout
    int bz = blockIdx.z;
    int n  = bz / Cout;
    int oc = bz % Cout;

    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N || oc >= Cout || oh >= Hout || ow >= Wout) return;

    float acc = 0.f;

    // (oh,ow) ↔ (h,w)
    const int h_base = oh * Sh - Ph;
    const int w_base = ow * Sw - Pw;

    // sum over Cin,Kh,Kw
    for (int ic = 0; ic < Cin; ++ic) {
        for (int kh = 0; kh < Kh; ++kh) {
            int h = h_base + kh;
            if ((unsigned)h >= (unsigned)Hin) continue;
            for (int kw = 0; kw < Kw; ++kw) {
                int w = w_base + kw;
                if ((unsigned)w >= (unsigned)Win) continue;

                // X[n,ic,h,w]
                int x_idx = (((n * Cin + ic) * Hin + h) * Win + w);
                // W[oc,ic,kh,kw]
                int w_idx = ((((oc * Cin) + ic) * Kh + kh) * Kw + kw);
                acc += X[x_idx] * W[w_idx];
            }
        }
    }

    // Y[n,oc,oh,ow]
    int y_idx = (((n * Cout + oc) * Hout + oh) * Wout + ow);
    Y[y_idx] = acc;
}

void launch_conv2d_forward_nchw(
    const float* X, const float* W, float* Y,
    int N, int Hin, int Win, int Cin,
    int Hout, int Wout, int Cout,
    int Kh, int Kw, int Sh, int Sw, int Ph, int Pw,
    cudaStream_t stream)
{
    dim3 block(TW, TW, 1);
    dim3 grid((Wout + TW - 1) / TW,
              (Hout + TW - 1) / TW,
              N * Cout);
    conv2d_fwd_nchw_kernel<<<grid, block, 0, stream>>>(
        X, W, Y,
        N, Hin, Win, Cin,
        Hout, Wout, Cout,
        Kh, Kw, Sh, Sw, Ph, Pw);
}



// ==================== Backward dX (NCHW) ====================
// dY: [N, Cout, Hout, Wout]
// W : [Cout, Cin, Kh, Kw]
// dX: [N, Cin, Hin, Win]
__global__ void conv2d_bwd_input_nchw_kernel(
    const float* __restrict__ dY,
    const float* __restrict__ W,
    float* __restrict__ dX,
    int N, int Hin, int Win, int Cin,
    int Hout, int Wout, int Cout,
    int Kh, int Kw, int Sh, int Sw, int Ph, int Pw)
{
    // grid.z = N*Cin
    int bz = blockIdx.z;
    int n  = bz / Cin;
    int ic = bz % Cin;

    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N || ic >= Cin || h >= Hin || w >= Win) return;

    float acc = 0.f;

    // accumulate over oc, kh, kw
    for (int oc = 0; oc < Cout; ++oc) {
        for (int kh = 0; kh < Kh; ++kh) {
            int oh = (h + Ph - kh);
            if (oh % Sh != 0) continue;
            oh /= Sh;
            if ((unsigned)oh >= (unsigned)Hout) continue;

            for (int kw = 0; kw < Kw; ++kw) {
                int ow = (w + Pw - kw);
                if (ow % Sw != 0) continue;
                ow /= Sw;
                if ((unsigned)ow >= (unsigned)Wout) continue;

                // dY[n,oc,oh,ow]
                int dy_idx = (((n * Cout + oc) * Hout + oh) * Wout + ow);
                // W[oc,ic,kh,kw]
                int w_idx  = ((((oc * Cin) + ic) * Kh + kh) * Kw + kw);
                acc += dY[dy_idx] * W[w_idx];
            }
        }
    }

    // dX[n,ic,h,w]
    int dx_idx = (((n * Cin + ic) * Hin + h) * Win + w);
    dX[dx_idx] = acc;
}

void launch_conv2d_backward_input_nchw(
    const float* dY, const float* W, float* dX,
    int N, int Hin, int Win, int Cin,
    int Hout, int Wout, int Cout,
    int Kh, int Kw, int Sh, int Sw, int Ph, int Pw,
    cudaStream_t stream)
{
    dim3 block(TW, TW, 1);
    dim3 grid((Win + TW - 1) / TW,
              (Hin + TW - 1) / TW,
              N * Cin);
    conv2d_bwd_input_nchw_kernel<<<grid, block, 0, stream>>>(
        dY, W, dX,
        N, Hin, Win, Cin,
        Hout, Wout, Cout,
        Kh, Kw, Sh, Sw, Ph, Pw);
}



// ==================== Backward dW (NCHW) ====================
// dY: [N, Cout, Hout, Wout]
// X : [N, Cin, Hin, Win]
// dW: [Cout, Cin, Kh, Kw]
__global__ void conv2d_bwd_weight_nchw_kernel(
    const float* __restrict__ dY,
    const float* __restrict__ X,
    float* __restrict__ dW,
    int N, int Hin, int Win, int Cin,
    int Hout, int Wout, int Cout,
    int Kh, int Kw, int Sh, int Sw, int Ph, int Pw)
{
    // grid.z = Cout, grid.y = Cin, grid.x = Kh
    int oc = blockIdx.z;
    int ic = blockIdx.y;
    int kh = blockIdx.x;
    int kw = threadIdx.x;  // 1D within kernel width
    if (oc >= Cout || ic >= Cin || kh >= Kh || kw >= Kw) return;

    float acc = 0.f;

    // sum over batch and output spatial
    for (int n = 0; n < N; ++n) {
        for (int oh = 0; oh < Hout; ++oh) {
            int h = oh * Sh - Ph + kh;
            if ((unsigned)h >= (unsigned)Hin) continue;

            for (int ow = 0; ow < Wout; ++ow) {
                int w = ow * Sw - Pw + kw;
                if ((unsigned)w >= (unsigned)Win) continue;

                // X[n,ic,h,w]
                int x_idx  = (((n * Cin + ic) * Hin + h) * Win + w);
                // dY[n,oc,oh,ow]
                int dy_idx = (((n * Cout + oc) * Hout + oh) * Wout + ow);
                acc += X[x_idx] * dY[dy_idx];
            }
        }
    }

    int w_idx = ((((oc * Cin) + ic) * Kh + kh) * Kw + kw);
    dW[w_idx] = acc;
}

void launch_conv2d_backward_weight_nchw(
    const float* dY, const float* X, float* dW,
    int N, int Hin, int Win, int Cin,
    int Hout, int Wout, int Cout,
    int Kh, int Kw, int Sh, int Sw, int Ph, int Pw,
    cudaStream_t stream)
{
    dim3 block(Kw, 1, 1);   // thread.x = kw
    dim3 grid(Kh, Cin, Cout);
    conv2d_bwd_weight_nchw_kernel<<<grid, block, 0, stream>>>(
        dY, X, dW,
        N, Hin, Win, Cin,
        Hout, Wout, Cout,
        Kh, Kw, Sh, Sw, Ph, Pw);
}
