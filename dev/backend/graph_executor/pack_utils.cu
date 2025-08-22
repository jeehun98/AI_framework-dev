// pack_utils.cu
#include "pack_utils.cuh"

__global__ void pack_rm_to_nchw_kernel(
    const float* __restrict__ rm, float* __restrict__ out,
    int N, int Cout, int Hout, int Wout)
{
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int bz = blockIdx.z;              // N*Cout
    int n  = bz / Cout;
    int oc = bz % Cout;

    if (n >= N || oc >= Cout || oh >= Hout || ow >= Wout) return;

    const int S = Hout * Wout;
    // rm[n, oc, oh*Wout + ow]
    size_t rm_idx  = ((size_t)n * Cout + oc) * S + (oh * Wout + ow);
    // out[n, oc, oh, ow]
    size_t out_idx = ((size_t)n * Cout + oc) * (size_t)(Hout*Wout) + (oh * Wout + ow);
    // 위 out_idx는 [N,Cout,Hout,Wout]를 플랫으로 본 것이므로 동일
    out[out_idx] = rm[rm_idx];
}

void launch_pack_rm_to_nchw(
    const float* rm, float* nchw, int N, int Cout, int Hout, int Wout, cudaStream_t stream)
{
    dim3 block(16, 16, 1);
    dim3 grid((Wout + block.x - 1)/block.x,
              (Hout + block.y - 1)/block.y,
              N * Cout);
    pack_rm_to_nchw_kernel<<<grid, block, 0, stream>>>(rm, nchw, N, Cout, Hout, Wout);
}
