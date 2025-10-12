// backends/cuda/ops/slice/kernels.cu
#include <cuda_runtime.h>
#include <cstdint>

namespace {

__global__ void slice_copy_kernel(const float* __restrict__ x, float* __restrict__ y,
                                  const int64_t* __restrict__ xshape,
                                  const int64_t* __restrict__ yshape,
                                  const int* __restrict__ starts,
                                  int rank, int64_t total_y)
{
  // strides for X and Y (row-major)
  int64_t xs[4]={1,1,1,1}, ys[4]={1,1,1,1}, shx[4]={1,1,1,1}, shy[4]={1,1,1,1};
  for(int d=0; d<rank; ++d){ shx[d]=xshape[d]; shy[d]=yshape[d]; }
  xs[rank-1]=1; ys[rank-1]=1;
  for(int d=rank-2; d>=0; --d){ xs[d]=xs[d+1]*shx[d+1]; ys[d]=ys[d+1]*shy[d+1]; }

  int64_t tid = blockIdx.x*blockDim.x + threadIdx.x;
  if (tid>=total_y) return;

  // y index → coords → x index with starts
  int64_t tmp=tid, coord[4]={0,0,0,0};
  for(int d=0; d<rank; ++d){ coord[d]=tmp/ys[d]; tmp-=coord[d]*ys[d]; }
  int64_t xoff=0;
  for(int d=0; d<rank; ++d){ xoff += (coord[d]+starts[d]) * xs[d]; }
  y[tid] = x[xoff];
}

} // anon

namespace ai {

void slice_copy_launcher(const float* x, float* y,
                         const int64_t* xshape, const int64_t* yshape,
                         const int* starts, int rank, cudaStream_t s)
{
  int64_t total=1; for(int d=0; d<rank; ++d) total *= yshape[d];
  dim3 block(256), grid((unsigned)((total + block.x - 1)/block.x));
  slice_copy_kernel<<<grid, block, 0, s>>>(x,y,xshape,yshape,starts,rank,total);
}

} // namespace ai
