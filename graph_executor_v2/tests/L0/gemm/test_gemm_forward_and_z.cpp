// tests/l0/gemm/test_gemm_forward_and_z.cpp
#include <vector>
#include <random>
#include <cstdio>
#include "tests/common/cuda_check.hpp"
#include "tests/common/device_buffer.hpp"
#include "tests/common/host_utils.hpp"
#include "tests/common/tensor_builder.hpp"
#include "backends/cuda/ops/gemm/api.hpp"

using namespace tbuild;

static void cpu_gemm_bias_relu(const float* A,const float* B,const float* bias,float* Y,
                               int64_t M,int64_t N,int64_t K,int64_t ldA,int64_t ldB,int64_t ldY,
                               std::vector<float>* Zopt){
  for(int64_t i=0;i<M;i++){
    for(int64_t j=0;j<N;j++){
      float acc=0.f;
      for(int64_t k=0;k<K;k++) acc += A[i*ldA+k]*B[k*ldB+j];
      if (bias) acc += bias[j];
      if (Zopt) (*Zopt)[i*ldY+j]=acc;
      Y[i*ldY+j] = acc>0.f?acc:0.f;
    }
  }
}

int main(){
  const int64_t M=32,N=48,K=16;
  const int64_t ldA=K, ldB=N, ldY=N;

  std::mt19937 rng(123);
  std::vector<float> hA(M*K), hB(K*N), hBias(N), hY_ref(M*N), hZ_ref(M*N);
  for(auto& v:hA) v=randf(rng);
  for(auto& v:hB) v=randf(rng);
  for(auto& v:hBias) v=randf(rng);

  cpu_gemm_bias_relu(hA.data(),hB.data(),hBias.data(),hY_ref.data(),
                     M,N,K,ldA,ldB,ldY,&hZ_ref);

  DeviceBuffer<float> dA(M*K), dB(K*N), dBias(N), dY(M*N), dZ(M*N);
  dA.h2d(hA.data()); dB.h2d(hB.data()); dBias.h2d(hBias.data());

  cudaStream_t cs; CUDA_CHECK(cudaStreamCreate(&cs));
  ai::StreamHandle s = reinterpret_cast<ai::StreamHandle>(cs);

  ai::Tensor A = make_tensor2d_f32_ld(dA.data(), M,K,ldA);
  ai::Tensor B = make_tensor2d_f32_ld(dB.data(), K,N,ldB);
  ai::Tensor Y = make_tensor2d_f32_ld(dY.data(), M,N,ldY);
  ai::Tensor Zs= make_tensor2d_f32_ld(dZ.data(), M,N,ldY);
  ai::Tensor Bias = make_bias_perN(dBias.data(), N);

  ai::GemmAttrs attrs{};
  attrs.act      = ai::ActKind::ReLU;
  attrs.save_z   = true;
  attrs.with_bias= true;

  ai::Status st = ai::GemmCudaLaunch(A,B,&Bias,Y,attrs,s,&Zs,nullptr);
  if(st!=ai::Status::Ok){ fprintf(stderr,"FWD failed\n"); return 1; }
  CUDA_CHECK(cudaStreamSynchronize(cs));

  std::vector<float> hY(M*N), hZ(M*N);
  dY.d2h(hY.data()); dZ.d2h(hZ.data());

  if(!allclose(hY,hY_ref,1e-4f,1e-4f)){ fprintf(stderr,"Y mismatch\n"); return 2; }
  if(!allclose(hZ,hZ_ref,1e-4f,1e-4f)){ fprintf(stderr,"Z mismatch\n"); return 3; }

  // alias 케이스: Z_saved == Y
  st = ai::GemmCudaLaunch(A,B,&Bias,Y,attrs,s,/*Z_saved*/&Y,nullptr);
  if(st!=ai::Status::Ok){ fprintf(stderr,"FWD alias failed\n"); return 4; }
  CUDA_CHECK(cudaStreamSynchronize(cs));

  CUDA_CHECK(cudaStreamDestroy(cs));
  return 0;
}
