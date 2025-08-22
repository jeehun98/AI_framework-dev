#include "cnn_kernels.cuh"

#define TW 16

// -------------------- Forward (NHWC) --------------------
__global__ void conv2d_fwd_nhwc_kernel(
    const float* __restrict__ X,   // [B,Hin,Win,Cin]
    const float* __restrict__ Wt,  // [Cout,Cin,Kh,Kw]
    float* __restrict__ Y,         // [B,Hout,Wout,Cout]
    int B, int Hin, int Win, int Cin,
    int Hout, int Wout, int Cout,
    int Kh, int Kw, int Sh, int Sw, int Ph, int Pw)
{
    // grid.z = B*Cout
    int bz = blockIdx.z;
    int b  = bz / Cout;
    int oc = bz % Cout;

    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B || oc >= Cout || oh >= Hout || ow >= Wout) return;

    float acc = 0.f;

    // (oh,ow) ↔ (h,w) 입력 좌표
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

                // X[b,h,w,ic]
                int x_idx = ((b * Hin + h) * Win + w) * Cin + ic;
                // W[oc,ic,kh,kw]
                int w_idx = (((oc * Cin) + ic) * Kh + kh) * Kw + kw;
                acc += X[x_idx] * Wt[w_idx];
            }
        }
    }

    // Y[b,oh,ow,oc]
    int y_idx = ((b * Hout + oh) * Wout + ow) * Cout + oc;
    Y[y_idx] = acc;
}

void launch_conv2d_forward_nhwc(
    const float* X, const float* W, float* Y,
    int B, int Hin, int Win, int Cin,
    int Hout, int Wout, int Cout,
    int Kh, int Kw, int Sh, int Sw, int Ph, int Pw,
    cudaStream_t stream)
{
    dim3 block(TW, TW, 1);
    dim3 grid((Wout + TW - 1) / TW,
              (Hout + TW - 1) / TW,
              B * Cout);
    conv2d_fwd_nhwc_kernel<<<grid, block, 0, stream>>>(
        X, W, Y,
        B, Hin, Win, Cin,
        Hout, Wout, Cout,
        Kh, Kw, Sh, Sw, Ph, Pw);
}



// -------------------- Backward dX (NHWC) --------------------
__global__ void conv2d_bwd_input_nhwc_kernel(
    const float* __restrict__ dY,  // [B,Hout,Wout,Cout]
    const float* __restrict__ Wt,  // [Cout,Cin,Kh,Kw]
    float* __restrict__ dX,        // [B,Hin,Win,Cin]
    int B, int Hin, int Win, int Cin,
    int Hout, int Wout, int Cout,
    int Kh, int Kw, int Sh, int Sw, int Ph, int Pw)
{
    // grid.z = B*Cin
    int bz = blockIdx.z;
    int b  = bz / Cin;
    int ic = bz % Cin;

    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B || ic >= Cin || h >= Hin || w >= Win) return;

    float acc = 0.f;

    // dY 위치 순회: oh,ow such that h = oh*Sh - Ph + kh, w = ow*Sw - Pw + kw
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

                // dY[b,oh,ow,oc]
                int dy_idx = ((b * Hout + oh) * Wout + ow) * Cout + oc;
                // W[oc,ic,kh,kw]
                int w_idx  = (((oc * Cin) + ic) * Kh + kh) * Kw + kw;
                acc += dY[dy_idx] * Wt[w_idx];
            }
        }
    }

    int dx_idx = ((b * Hin + h) * Win + w) * Cin + ic;
    dX[dx_idx] = acc;
}

void launch_conv2d_backward_input_nhwc(
    const float* dY, const float* W, float* dX,
    int B, int Hin, int Win, int Cin,
    int Hout, int Wout, int Cout,
    int Kh, int Kw, int Sh, int Sw, int Ph, int Pw,
    cudaStream_t stream)
{
    dim3 block(TW, TW, 1);
    dim3 grid((Win + TW - 1) / TW,
              (Hin + TW - 1) / TW,
              B * Cin);
    conv2d_bwd_input_nhwc_kernel<<<grid, block, 0, stream>>>(
        dY, W, dX,
        B, Hin, Win, Cin,
        Hout, Wout, Cout,
        Kh, Kw, Sh, Sw, Ph, Pw);
}



// -------------------- Backward dW (NHWC) --------------------
__global__ void conv2d_bwd_weight_nhwc_kernel(
    const float* __restrict__ dY,  // [B,Hout,Wout,Cout]
    const float* __restrict__ X,   // [B,Hin,Win,Cin]
    float* __restrict__ dW,        // [Cout,Cin,Kh,Kw]
    int B, int Hin, int Win, int Cin,
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
    for (int b = 0; b < B; ++b) {
        for (int oh = 0; oh < Hout; ++oh) {
            int h = oh * Sh - Ph + kh;
            if ((unsigned)h >= (unsigned)Hin) continue;

            for (int ow = 0; ow < Wout; ++ow) {
                int w = ow * Sw - Pw + kw;
                if ((unsigned)w >= (unsigned)Win) continue;

                // X[b,h,w,ic]
                int x_idx = ((b * Hin + h) * Win + w) * Cin + ic;
                // dY[b,oh,ow,oc]
                int dy_idx = ((b * Hout + oh) * Wout + ow) * Cout + oc;
                acc += X[x_idx] * dY[dy_idx];
            }
        }
    }

    int w_idx = (((oc * Cin) + ic) * Kh + kh) * Kw + kw;
    dW[w_idx] = acc;
}

void launch_conv2d_backward_weight_nhwc(
    const float* dY, const float* X, float* dW,
    int B, int Hin, int Win, int Cin,
    int Hout, int Wout, int Cout,
    int Kh, int Kw, int Sh, int Sw, int Ph, int Pw,
    cudaStream_t stream)
{
    dim3 block(Kw, 1, 1);   // thread.x = kw
    dim3 grid(Kh, Cin, Cout);
    conv2d_bwd_weight_nhwc_kernel<<<grid, block, 0, stream>>>(
        dY, X, dW,
        B, Hin, Win, Cin,
        Hout, Wout, Cout,
        Kh, Kw, Sh, Sw, Ph, Pw);
}
