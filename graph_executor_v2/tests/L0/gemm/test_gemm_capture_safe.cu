// tests/l0/gemm/test_gemm_capture_safe.cu
#include "tests/common/cuda_check.hpp"
#include "tests/common/capture_safety.hpp"
#include "tests/common/tensor_builder.hpp"
#include "backends/cuda/ops/gemm/api.hpp"

using namespace tbuild;

int main(){
  const int64_t M=16,N=16,K=16;
  float *A,*B,*Y,*Bias,*Z;
  CUDA_CHECK(cudaMalloc(&A,sizeof(float)*M*K));
  CUDA_CHECK(cudaMalloc(&B,sizeof(float)*K*N));
  CUDA_CHECK(cudaMalloc(&Y,sizeof(float)*M*N));
  CUDA_CHECK(cudaMalloc(&Bias,sizeof(float)*N));
  CUDA_CHECK(cudaMalloc(&Z,sizeof(float)*M*N));

  ai::Tensor tA = make_tensor2d_f32_ld(A,M,K,K);
  ai::Tensor tB = make_tensor2d_f32_ld(B,K,N,N);
  ai::Tensor tY = make_tensor2d_f32_ld(Y,M,N,N);
  ai::Tensor tZ = make_tensor2d_f32_ld(Z,M,N,N);
  ai::Tensor tBias = make_bias_perN(Bias,N);

  ai::GemmAttrs attrs{};
  attrs.act      = ai::ActKind::ReLU;
  attrs.save_z   = true;
  attrs.with_bias= true;

  check_capture_safe([&](cudaStream_t cs){
    ai::StreamHandle s = reinterpret_cast<ai::StreamHandle>(cs);
    ai::Status st = ai::GemmCudaLaunch(tA,tB,&tBias,tY,attrs,s,&tZ,nullptr);
    if(st!=ai::Status::Ok) throw std::runtime_error("launch failed in capture");
  });

  cudaFree(A); cudaFree(B); cudaFree(Y); cudaFree(Bias); cudaFree(Z);
  return 0;
}
