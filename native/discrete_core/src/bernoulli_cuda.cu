#ifdef __CUDACC__
#include <cuda_runtime.h>
#include "discrete/bernoulli.hpp"
#include "discrete/cuda_rng.cuh"

namespace disc {
__global__ void bernoulli_kernel(float* out, std::size_t n, float p, unsigned long long seed){
    const std::size_t tid = blockIdx.x*(std::size_t)blockDim.x + threadIdx.x;
    const std::size_t stride = gridDim.x*(std::size_t)blockDim.x;
    p = fminf(fmaxf(p, 1e-12f), 1.f-1e-12f);
    ThreadRNG tr(seed?seed:0x9E3779B97F4A7C15ull, tid);
    for(std::size_t i=tid;i<n;i+=stride) out[i] = (tr.u01() < p) ? 1.f : 0.f;
}
static inline int rup(int a,int b){return (a+b-1)/b;}

static void bernoulli_device(float* d_out, std::size_t n, float p, std::uint64_t seed, void* stream_v){
    if(!d_out || n==0) return;
    cudaStream_t s = (cudaStream_t)stream_v;
    const int block=256, grid=max(1, rup((int)n, block));
    bernoulli_kernel<<<grid,block,0,s>>>(d_out,n,p,(unsigned long long)seed);
}

void bernoulli_cuda(float* out, std::size_t n, float p, std::uint64_t seed, void* stream_v){
    if(!out || n==0) return;
    cudaStream_t s = (cudaStream_t)stream_v;
    cudaPointerAttributes attr{};
#if CUDART_VERSION >= 10000
    bool is_dev = (cudaPointerGetAttributes(&attr,out)==cudaSuccess) && (attr.type==cudaMemoryTypeDevice);
#else
    bool is_dev = (cudaPointerGetAttributes(&attr,out)==cudaSuccess) && (attr.memoryType==cudaMemoryTypeDevice);
#endif
    if(is_dev){ bernoulli_device(out,n,p,seed,s); }
    else{
        float* d=nullptr; cudaMalloc(&d,n*sizeof(float));
        bernoulli_device(d,n,p,seed,s);
        cudaMemcpyAsync(out,d,n*sizeof(float),cudaMemcpyDeviceToHost,s);
        cudaStreamSynchronize(s); cudaFree(d);
    }
}

void bernoulli_logpmf_cuda(const float* x, std::size_t n, float p, float* out, void* stream_v){
    // 간단 구현: Host에서 계산 (필요 시 커널화)
    if(!x || !out || n==0) return;
    cudaStream_t s = (cudaStream_t)stream_v;
    cudaPointerAttributes ax{}, ao{};
#if CUDART_VERSION >= 10000
    bool xd = (cudaPointerGetAttributes(&ax,x)==cudaSuccess)&&(ax.type==cudaMemoryTypeDevice);
    bool od = (cudaPointerGetAttributes(&ao,out)==cudaSuccess)&&(ao.type==cudaMemoryTypeDevice);
#else
    bool xd = (cudaPointerGetAttributes(&ax,x)==cudaSuccess)&&(ax.memoryType==cudaMemoryTypeDevice);
    bool od = (cudaPointerGetAttributes(&ao,out)==cudaSuccess)&&(ao.memoryType==cudaMemoryTypeDevice);
#endif
    float *hx=nullptr,*ho=nullptr;
    if(xd){ hx=(float*)malloc(n*sizeof(float)); cudaMemcpyAsync(hx,x,n*sizeof(float),cudaMemcpyDeviceToHost,s); cudaStreamSynchronize(s);}
    else  { hx=(float*)x; }
    if(od){ ho=(float*)malloc(n*sizeof(float)); } else { ho=out; }
    bernoulli_logpmf_cpu(hx,n,p,ho);
    if(xd) free(hx);
    if(od){ cudaMemcpyAsync(out,ho,n*sizeof(float),cudaMemcpyHostToDevice,s); cudaStreamSynchronize(s); free(ho); }
}
} // namespace disc
#endif
