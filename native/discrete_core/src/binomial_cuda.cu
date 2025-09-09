#ifdef __CUDACC__
#include <cuda_runtime.h>
#include "discrete/binomial.hpp"
#include "discrete/cuda_rng.cuh"

namespace disc {
__global__ void binom_smalln_kernel(float* out, std::size_t m, int n, float p, unsigned long long seed){
    const std::size_t tid = blockIdx.x*(std::size_t)blockDim.x + threadIdx.x;
    const std::size_t stride = gridDim.x*(std::size_t)blockDim.x;
    p = fminf(fmaxf(p, 1e-12f), 1.f-1e-12f);
    ThreadRNG tr(seed?seed:0x9E3779B97F4A7C15ull, tid);
    for(std::size_t i=tid;i<m;i+=stride){
        int k=0; for(int t=0;t<n;++t) k += (tr.u01() < p);
        out[i] = (float)k;
    }
}
static inline int rup(int a,int b){return (a+b-1)/b;}

static void binom_device(float* d_out, std::size_t m, int n, float p, std::uint64_t seed, void* stream_v){
    if(!d_out || m==0 || n<=0) return;
    cudaStream_t s = (cudaStream_t)stream_v;
    const int block=256, grid=max(1, rup((int)m, block));
    binom_smalln_kernel<<<grid,block,0,s>>>(d_out,m,n,p,(unsigned long long)seed);
}

void binomial_cuda(float* out, std::size_t m, int n, float p, std::uint64_t seed, void* stream_v){
    if(!out || m==0 || n<=0) return;
    cudaStream_t s = (cudaStream_t)stream_v;
    cudaPointerAttributes attr{};
#if CUDART_VERSION >= 10000
    bool is_dev = (cudaPointerGetAttributes(&attr,out)==cudaSuccess) && (attr.type==cudaMemoryTypeDevice);
#else
    bool is_dev = (cudaPointerGetAttributes(&attr,out)==cudaSuccess) && (attr.memoryType==cudaMemoryTypeDevice);
#endif
    if(is_dev){ binom_device(out,m,n,p,seed,s); }
    else{
        float* d=nullptr; cudaMalloc(&d,m*sizeof(float));
        binom_device(d,m,n,p,seed,s);
        cudaMemcpyAsync(out,d,m*sizeof(float),cudaMemcpyDeviceToHost,s);
        cudaStreamSynchronize(s); cudaFree(d);
    }
}

void binomial_logpmf_cuda(const float* k, std::size_t nk, int n, float p, float* out, void* stream_v){
    // 간단 구현: Host 재사용
    if(!k || !out || nk==0 || n<0) return;
    cudaStream_t s = (cudaStream_t)stream_v;
    cudaPointerAttributes ak{}, ao{};
#if CUDART_VERSION >= 10000
    bool kd = (cudaPointerGetAttributes(&ak,k)==cudaSuccess)&&(ak.type==cudaMemoryTypeDevice);
    bool od = (cudaPointerGetAttributes(&ao,out)==cudaSuccess)&&(ao.type==cudaMemoryTypeDevice);
#else
    bool kd = (cudaPointerGetAttributes(&ak,k)==cudaSuccess)&&(ak.memoryType==cudaMemoryTypeDevice);
    bool od = (cudaPointerGetAttributes(&ao,out)==cudaSuccess)&&(ao.memoryType==cudaMemoryTypeDevice);
#endif
    float *hk=nullptr,*ho=nullptr;
    if(kd){ hk=(float*)malloc(nk*sizeof(float)); cudaMemcpyAsync(hk,k,nk*sizeof(float),cudaMemcpyDeviceToHost,s); cudaStreamSynchronize(s);}
    else  { hk=(float*)k; }
    if(od){ ho=(float*)malloc(nk*sizeof(float)); } else { ho=out; }
    binomial_logpmf_cpu(hk,nk,n,p,ho);
    if(kd) free(hk);
    if(od){ cudaMemcpyAsync(out,ho,nk*sizeof(float),cudaMemcpyHostToDevice,s); cudaStreamSynchronize(s); free(ho); }
}
} // namespace disc
#endif
