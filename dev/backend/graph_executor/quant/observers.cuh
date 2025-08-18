#pragma once
#include "quant/quant_types.cuh"

namespace quant {

// x: float[N]의 min/max를 블록별로 구해 host에서 reduce(안전/간단)
__global__ void k_block_minmax(const float* __restrict__ x, int n, float* bmin, float* bmax){
    extern __shared__ float smem[];
    float* smin = smem;
    float* smax = smem + blockDim.x;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float vmin = 1e30f, vmax = -1e30f;
    for (; i < n; i += gridDim.x * blockDim.x) {
        float v = x[i];
        vmin = fminf(vmin, v);
        vmax = fmaxf(vmax, v);
    }
    smin[threadIdx.x] = vmin; smax[threadIdx.x] = vmax;
    __syncthreads();

    for (int s = blockDim.x>>1; s>0; s>>=1){
        if (threadIdx.x < s){
            smin[threadIdx.x] = fminf(smin[threadIdx.x], smin[threadIdx.x+s]);
            smax[threadIdx.x] = fmaxf(smax[threadIdx.x], smax[threadIdx.x+s]);
        }
        __syncthreads();
    }
    if (threadIdx.x==0){
        bmin[blockIdx.x] = smin[0];
        bmax[blockIdx.x] = smax[0];
    }
}

inline void observe_activation(const std::string& tensor_id, const float* dptr, int n){
    using namespace quant;
    if (!runtime().observe_enabled) return;
    const int threads = 256;
    const int blocks = std::min( (n + threads - 1)/threads, 2048 );
    float *d_bmin=nullptr, *d_bmax=nullptr;
    CUDA_CHECK(cudaMalloc(&d_bmin, blocks*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bmax, blocks*sizeof(float)));
    size_t shmem = threads * sizeof(float) * 2;
    k_block_minmax<<<blocks, threads, shmem>>>(dptr, n, d_bmin, d_bmax);
    CUDA_CHECK(cudaPeekAtLastError());

    std::vector<float> hmin(blocks), hmax(blocks);
    CUDA_CHECK(cudaMemcpy(hmin.data(), d_bmin, blocks*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hmax.data(), d_bmax, blocks*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_bmin)); CUDA_CHECK(cudaFree(d_bmax));

    float gmin=1e30f, gmax=-1e30f;
    for (int i=0;i<blocks;++i){ gmin = std::min(gmin, hmin[i]); gmax = std::max(gmax, hmax[i]); }

    auto& mm = cache().act_minmax[tensor_id];
    if (!mm.initialized){
        mm.min = gmin; mm.max = gmax; mm.initialized = true;
    } else {
        mm.min = std::min(mm.min, gmin);
        mm.max = std::max(mm.max, gmax);
    }
}

inline QuantParams calc_qparams_activation(const MinMax& mm){
    // 비대칭 per-tensor (int8)
    const int qmin = -128, qmax = 127;
    float x_min = mm.min, x_max = mm.max;
    // 보호: 상수 입력 등 degenerate
    if (x_max <= x_min){ x_max = x_min + 1e-6f; }
    QuantParams qp;
    qp.scale = (x_max - x_min) / float(qmax - qmin);
    float zf = float(qmin) - x_min / qp.scale;
    qp.zero_point = (int)lrintf(zf);
    // 클램프
    qp.zero_point = std::max(qmin, std::min(qmax, qp.zero_point));
    return qp;
}

inline void freeze_act_qparams(){
    for (auto& kv : cache().act_minmax){
        cache().act_qparams[kv.first] = calc_qparams_activation(kv.second);
    }
}

inline void enable_observers(bool on){ runtime().observe_enabled = on; }

} // namespace quant
