// backends/cuda/ops/concat/kernels.cu
#include <cuda_runtime.h>
#include <cstdint>

namespace {

__global__ void concat_copy_kernel(
  const float* const* __restrict__ in_ptrs,
  const int64_t* __restrict__ sizes_axis,
  int n_inputs,
  float* __restrict__ out,
  const int64_t* __restrict__ shape, int rank,
  int axis)
{
  // RowMajor 연속 인덱스 → (n0,n1,n2,n3) 해석
  // strides
  int64_t stride[4]={0,0,0,0}, sh[4]={1,1,1,1};
  for(int d=0; d<rank; ++d) sh[d]=(int)shape[d];
  stride[rank-1]=1;
  for(int d=rank-2; d>=0; --d) stride[d]=stride[d+1]*sh[d+1];
  const int64_t total = stride[0]*sh[0];

  int64_t axis_stride = stride[axis];
  int64_t inner = (axis==rank-1) ? 1 : stride[axis+1];
  int64_t outer = total / (sh[axis]*inner);

  int64_t tid = blockIdx.x*blockDim.x + threadIdx.x;
  if (tid>=total) return;

  // 좌표 복원
  int64_t tmp = tid;
  int64_t coord[4]={0,0,0,0};
  for(int d=0; d<rank; ++d){
    coord[d] = tmp / stride[d];
    tmp -= coord[d]*stride[d];
  }

  // 입력 선택: coord[axis]가 어느 input의 축에 속하는지
  int64_t aidx = coord[axis];
  int which = 0;
  int64_t off = aidx;
  for(; which<n_inputs; ++which){
    int64_t sz = sizes_axis[which];
    if (off < sz) break;
    off -= sz;
  }
  if (which==n_inputs) return; // shouldn't happen

  // 입력 포인터의 위치 계산:
  // 입력 텐서의 동일 좌표에서 axis만 'off'로 교체
  int64_t in_index = 0;
  for(int d=0; d<rank; ++d){
    int64_t v = (d==axis) ? off : coord[d];
    // 입력 stride 동일(모든 입력은 base와 동일한 shape except axis)
    // stride 재사용 가능
    in_index += v * stride[d];
  }
  out[tid] = in_ptrs[which][in_index];
}

} // anon

namespace ai {

void concat_copy_launcher(const float* const* in_ptrs,
                          const int64_t* sizes_axis,
                          int n_inputs,
                          float* out,
                          const int64_t* shape, int rank,
                          int axis,
                          cudaStream_t s)
{
  int64_t total = 1;
  for(int d=0; d<rank; ++d) total *= shape[d];
  dim3 block(256), grid((unsigned)((total + block.x - 1)/block.x));
  concat_copy_kernel<<<grid, block, 0, s>>>(in_ptrs, sizes_axis, n_inputs, out, shape, rank, axis);
}

} // namespace ai
