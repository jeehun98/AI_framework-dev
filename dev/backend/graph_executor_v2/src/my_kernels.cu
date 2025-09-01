#include <cuda_runtime.h>
#include <cstdio>
extern "C" {

// 아주 단순한 데모 커널
__global__ void dummy_gemm_bias_act_tc_f16_kernel() {}

int ge2_launch_gemm_bias_act_tc_f16(const uintptr_t* buffers, int nbufs, void* stream) {
  // TODO: buffers[]에서 실제 device ptr 가져와 구현
  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  dummy_gemm_bias_act_tc_f16_kernel<<<1,1,0,s>>>();
  return cudaPeekAtLastError();
}

int ge2_launch_gemm_bias_act_f32(const uintptr_t* buffers, int nbufs, void* stream) {
  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  dummy_gemm_bias_act_tc_f16_kernel<<<1,1,0,s>>>();
  return cudaPeekAtLastError();
}

} // extern "C"
