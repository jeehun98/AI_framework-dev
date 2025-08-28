// rnn/rnn_kernels.cu  — cleaned & fixed
#include "rnn_kernels.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>  // std::min

#include "../ge/cuda_check.cuh"

/* ============================ helpers ============================ */

__device__ __forceinline__ float act_forward(float z, RnnActivation a) {
    switch (a) {
        case RNN_TANH:    return tanhf(z);
        case RNN_RELU:    return z > 0.f ? z : 0.f;
        case RNN_SIGMOID: return 1.f / (1.f + expf(-z));
    }
    return z;
}

__device__ __forceinline__ float act_backward_from_y(float y, RnnActivation a) {
    switch (a) {
        case RNN_TANH:    return 1.f - y * y;
        case RNN_RELU:    return y > 0.f ? 1.f : 0.f;
        case RNN_SIGMOID: return y * (1.f - y);
    }
    return 1.f;
}

// row-major offsets
__device__ __forceinline__ size_t off_BTH(int B, int T, int H, int b, int t, int h) {
    return ((size_t)b * T + t) * H + h;
}
__device__ __forceinline__ size_t off_BTD(int B, int T, int D, int b, int t, int d) {
    return ((size_t)b * T + t) * D + d;
}
__device__ __forceinline__ size_t off_BH(int B, int H, int b, int h) {
    return (size_t)b * H + h;
}
__device__ __forceinline__ size_t off_DH(int D, int H, int d, int h) {
    return (size_t)d * H + h;
}
__device__ __forceinline__ size_t off_HH(int H, int i, int j) {
    return (size_t)i * H + j;
}

/* ============================ forward ============================ */

// 한 시점 forward: grid=(ceil(H/blk), B), block=(blk)
__global__ void rnn_forward_timestep_kernel(
    const float* __restrict__ X_base, // [B,T,D]
    int B, int T, int D, int H,
    const float* __restrict__ Wx,     // [D,H]
    const float* __restrict__ Wh,     // [H,H]
    const float* __restrict__ b,      // [H] or nullptr
    const float* __restrict__ Hprev,  // [B,H] (t==0이면 h0 또는 0)
    float* __restrict__ H_t,          // [B,H]
    int t, RnnActivation act)
{
    int h = blockIdx.x * blockDim.x + threadIdx.x;  // 0..H-1
    int bidx = blockIdx.y;                          // 0..B-1
    if (h >= H || bidx >= B) return;

    float z = b ? b[h] : 0.f;

    // x_t · Wx[:,h]
    const float* x_row = X_base + off_BTD(B, T, D, bidx, t, 0);
    for (int d = 0; d < D; ++d)
        z += x_row[d] * Wx[off_DH(D, H, d, h)];

    // h_{t-1} · Wh[:,h]
    const float* hprev_row = Hprev + off_BH(B, H, bidx, 0);
    for (int hp = 0; hp < H; ++hp)
        z += hprev_row[hp] * Wh[off_HH(H, hp, h)];

    H_t[off_BH(B, H, bidx, h)] = act_forward(z, act);
}

// Hseq[b,t,h] = Hcurr[b,h]
__global__ void pack_Hcurr_into_Hseq(
    const float* __restrict__ Hcurr, // [B,H]
    float* __restrict__ Hseq,        // [B,T,H]
    int B, int T, int H, int t)
{
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;
    if (b >= B || h >= H) return;
    const float v = Hcurr[off_BH(B, H, b, h)];
    Hseq[off_BTH(B, T, H, b, t, h)] = v;
}

void launch_rnn_forward_simple(
    const float* X, const float* Wx, const float* Wh, const float* b,
    const float* h0, float* H_T, float* H_seq,
    int B, int T, int D, int H, RnnActivation act,
    cudaStream_t stream)
{
    float* Hprev = nullptr;
    float* Hcurr = nullptr;
    CUDA_CHECK(cudaMalloc(&Hprev, (size_t)B * H * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&Hcurr, (size_t)B * H * sizeof(float)));

    if (h0) CUDA_CHECK(cudaMemcpyAsync(Hprev, h0, (size_t)B * H * sizeof(float),
                                       cudaMemcpyDeviceToDevice, stream));
    else    CUDA_CHECK(cudaMemsetAsync(Hprev, 0, (size_t)B * H * sizeof(float), stream));

    const int blk = std::min(256, H);
    const dim3 gridF((H + blk - 1) / blk, B);

    for (int t = 0; t < T; ++t) {
        rnn_forward_timestep_kernel<<<gridF, blk, 0, stream>>>(
            X, B, T, D, H, Wx, Wh, b, Hprev, Hcurr, t, act
        );
        CUDA_CHECK(cudaGetLastError());

        if (H_seq) {
            const dim3 gridP((H + 255) / 256, B);
            pack_Hcurr_into_Hseq<<<gridP, 256, 0, stream>>>(Hcurr, H_seq, B, T, H, t);
            CUDA_CHECK(cudaGetLastError());
        }

        // swap(Hprev, Hcurr)
        float* tmp = Hprev; Hprev = Hcurr; Hcurr = tmp;
    }

    if (H_T) {
        CUDA_CHECK(cudaMemcpyAsync(H_T, Hprev, (size_t)B * H * sizeof(float),
                                   cudaMemcpyDeviceToDevice, stream));
    }

    CUDA_CHECK(cudaFree(Hprev));
    CUDA_CHECK(cudaFree(Hcurr));
}

/* ============================ backward ============================ */

// 한 시점 backward
__global__ void rnn_backward_timestep_kernel(
    const float* __restrict__ X_base,   // [B,T,D]
    const float* __restrict__ Hseq,     // [B,T,H]
    const float* __restrict__ h0,       // [B,H] (nullable)
    const float* __restrict__ dHin,     // [B,H] (미래에서 온 dH)
    const float* __restrict__ dY_T,     // [B,H] (nullable; t==T-1)
    const float* __restrict__ dY_seq,   // [B,T,H] (nullable)
    const float* __restrict__ Wx,       // [D,H]
    const float* __restrict__ Wh,       // [H,H]
    float* __restrict__ dX_base,        // [B,T,D] (accumulate)
    float* __restrict__ dWx,            // [D,H]   (accumulate)
    float* __restrict__ dWh,            // [H,H]   (accumulate)
    float* __restrict__ db,             // [H]     (accumulate)
    float* __restrict__ dHout_prev,     // [B,H]   (write)
    int B, int T, int D, int H, int t,
    RnnActivation act, bool is_last_t)
{
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;
    if (h >= H || b >= B) return;

    float dht = dHin[off_BH(B, H, b, h)];
    if (dY_seq) dht += dY_seq[off_BTH(B, T, H, b, t, h)];
    if (is_last_t && dY_T) dht += dY_T[off_BH(B, H, b, h)];

    const float h_t = Hseq[off_BTH(B, T, H, b, t, h)];
    const float* hprev_row =
        (t > 0) ? (Hseq + off_BTH(B, T, H, b, t-1, 0))
                : (h0 ? (h0 + off_BH(B, H, b, 0)) : nullptr);

    const float dhraw = dht * act_backward_from_y(h_t, act);

    // db
    atomicAdd(&db[h], dhraw);

    // dWx & dX_t
    const float* x_row = X_base + off_BTD(B, T, D, b, t, 0);
    for (int d = 0; d < D; ++d) {
        atomicAdd(&dWx[off_DH(D, H, d, h)], x_row[d] * dhraw);
        // dX[b,t,d] += Wx[d,h] * dhraw
        atomicAdd(&dX_base[off_BTD(B, T, D, b, t, d)], Wx[off_DH(D, H, d, h)] * dhraw);
    }

    // dWh & dHprev
    for (int hp = 0; hp < H; ++hp) {
        const float hprev_v = (t > 0) ? hprev_row[hp] : (h0 ? hprev_row[hp] : 0.f);
        atomicAdd(&dWh[off_HH(H, hp, h)], hprev_v * dhraw);
        atomicAdd(&dHout_prev[off_BH(B, H, b, hp)], Wh[off_HH(H, hp, h)] * dhraw);
    }
}

void launch_rnn_backward_simple(
    const float* X, const float* Wx, const float* Wh, const float* b,
    const float* h0, const float* H_seq,
    const float* dY_T, const float* dY_seq,
    float* dX, float* dWx, float* dWh, float* db, float* dH0,
    int B, int T, int D, int H, RnnActivation act,
    cudaStream_t stream)
{
    // work buffers
    float* dH_in  = nullptr;
    float* dH_out = nullptr;
    CUDA_CHECK(cudaMalloc(&dH_in,  (size_t)B * H * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dH_out, (size_t)B * H * sizeof(float)));
    CUDA_CHECK(cudaMemsetAsync(dH_in,  0, (size_t)B * H * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(dH_out, 0, (size_t)B * H * sizeof(float), stream));

    // dX accum
    CUDA_CHECK(cudaMemsetAsync(dX,  0, (size_t)B * T * D * sizeof(float), stream));
    // dWx/dWh/db는 호출자가 0 init하는 것을 권장 (원하면 아래 주석 해제)
    // CUDA_CHECK(cudaMemsetAsync(dWx, 0, (size_t)D * H * sizeof(float), stream));
    // CUDA_CHECK(cudaMemsetAsync(dWh, 0, (size_t)H * H * sizeof(float), stream));
    // CUDA_CHECK(cudaMemsetAsync(db,  0, (size_t)H     * sizeof(float), stream));

    const int blk = std::min(256, H);
    const dim3 gridB((H + blk - 1) / blk, B);

    for (int t = T - 1; t >= 0; --t) {
        CUDA_CHECK(cudaMemsetAsync(dH_out, 0, (size_t)B * H * sizeof(float), stream));

        rnn_backward_timestep_kernel<<<gridB, blk, 0, stream>>>(
            X, H_seq, h0, dH_in,
            dY_T, dY_seq,
            Wx, Wh,
            dX, dWx, dWh, db, dH_out,
            B, T, D, H, t, act, /*is_last_t=*/(t == T - 1)
        );
        CUDA_CHECK(cudaGetLastError());

        float* tmp = dH_in; dH_in = dH_out; dH_out = tmp;
    }

    if (dH0) {
        CUDA_CHECK(cudaMemcpyAsync(dH0, dH_in, (size_t)B * H * sizeof(float),
                                   cudaMemcpyDeviceToDevice, stream));
    }

    CUDA_CHECK(cudaFree(dH_in));
    CUDA_CHECK(cudaFree(dH_out));
}
