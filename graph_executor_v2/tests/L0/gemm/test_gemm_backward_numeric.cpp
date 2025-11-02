// tests/l0/gemm/test_gemm_backward_numeric.cpp
#include <vector>
#include <random>
#include <cstdio>
#include <cmath>
#include "tests/common/cuda_check.hpp"
#include "tests/common/device_buffer.hpp"
#include "tests/common/host_utils.hpp"
#include "tests/common/tensor_builder.hpp"
#include "backends/cuda/ops/gemm/api.hpp"

using namespace tbuild;

static inline float d_relu(float z){ return z>0.f?1.f:0.f; }

static void cpu_forward(const float* A,const float* B,const float* bias,
                        float* Z,float* Y,
                        int64_t M,int64_t N,int64_t K,int64_t ldA,int64_t ldB,int64_t ldY){
  for(int64_t i=0;i<M;i++){
    for(int64_t j=0;j<N;j++){
      float acc=0.f;
      for(int64_t k=0;k<K;k++) acc+=A[i*ldA+k]*B[k*ldB+j];
      if(bias) acc+=bias[j];
      Z[i*ldY+j]=acc;
      Y[i*ldY+j]=acc>0.f?acc:0.f;
    }
  }
}
static float cpu_loss(const float* Y,int64_t M,int64_t N,int64_t ldY){
  double s=0.0; for(int64_t i=0;i<M;i++) for(int64_t j=0;j<N;j++){ float y=Y[i*ldY+j]; s+=0.5*y*y; }
  return (float)s;
}
static void cpu_backward_analytic(const float* A,const float* B,const float* Z,const float* Y,
                                  float* gA,float* gB,float* gBias,
                                  int64_t M,int64_t N,int64_t K,int64_t ldA,int64_t ldB,int64_t ldY){
  std::vector<float> gZ(M*N);
  for(int64_t i=0;i<M;i++) for(int64_t j=0;j<N;j++){
    float gy=Y[i*ldY+j]; gZ[i*N+j]= gy * d_relu(Z[i*ldY+j]);
  }
  for(int64_t i=0;i<M;i++) for(int64_t k=0;k<K;k++){
    double acc=0.0; for(int64_t j=0;j<N;j++) acc+= gZ[i*N+j]*B[k*ldB+j];
    gA[i*ldA+k]=(float)acc;
  }
  for(int64_t k=0;k<K;k++) for(int64_t j=0;j<N;j++){
    double acc=0.0; for(int64_t i=0;i<M;i++) acc+= A[i*ldA+k]*gZ[i*N+j];
    gB[k*ldB+j]=(float)acc;
  }
  for(int64_t j=0;j<N;j++){
    double acc=0.0; for(int64_t i=0;i<M;i++) acc+= gZ[i*N+j];
    gBias[j]=(float)acc;
  }
}

int main(){
  const int64_t M=4,N=5,K=3;
  const int64_t ldA=K, ldB=N, ldY=N;

  std::mt19937 rng(7);
  std::vector<float> hA(M*K), hB(K*N), hBias(N);
  for(auto& v:hA)v=randf(rng);
  for(auto& v:hB)v=randf(rng);
  for(auto& v:hBias)v=randf(rng);

  DeviceBuffer<float> dA(M*K), dB(K*N), dBias(N), dZ(M*N), dY(M*N);
  dA.h2d(hA.data()); dB.h2d(hB.data()); dBias.h2d(hBias.data());

  cudaStream_t cs; CUDA_CHECK(cudaStreamCreate(&cs));
  ai::StreamHandle s = reinterpret_cast<ai::StreamHandle>(cs);

  ai::Tensor A    = make_tensor2d_f32_ld(dA.data(), M,K,ldA);
  ai::Tensor B    = make_tensor2d_f32_ld(dB.data(), K,N,ldB);
  ai::Tensor Bias = make_bias_perN(dBias.data(), N);
  ai::Tensor Z    = make_tensor2d_f32_ld(dZ.data(), M,N,ldY);
  ai::Tensor Y    = make_tensor2d_f32_ld(dY.data(), M,N,ldY);

  ai::GemmAttrs attrs{};
  attrs.act       = ai::ActKind::ReLU;
  attrs.save_z    = true;
  attrs.with_bias = true;

  // Forward (save Z)
  if(ai::GemmCudaLaunch(A,B,&Bias,Y,attrs,s,&Z,nullptr)!=ai::Status::Ok){ fprintf(stderr,"FWD fail\n"); return 1; }
  CUDA_CHECK(cudaStreamSynchronize(cs));

  std::vector<float> hZ(M*N), hY(M*N);
  dZ.d2h(hZ.data()); dY.d2h(hY.data());

  // Analytic reference
  std::vector<float> gA_ref(M*K), gB_ref(K*N), gBias_ref(N);
  cpu_backward_analytic(hA.data(),hB.data(),hZ.data(),hY.data(),
                        gA_ref.data(),gB_ref.data(),gBias_ref.data(),
                        M,N,K,ldA,ldB,ldY);

  // API backward
  DeviceBuffer<float> d_gA(M*K), d_gB(K*N), d_gBias(N);
  ai::Tensor gA_t   = make_tensor2d_f32_ld(d_gA.data(), M,K,ldA);
  ai::Tensor gB_t   = make_tensor2d_f32_ld(d_gB.data(), K,N,ldB);
  ai::Tensor gBias_t= make_bias_perN(d_gBias.data(), N);

  ai::GemmWorkspace ws{}; // scratch/lt_ws는 nullptr 허용(내부 체크)
  ai::Status stb = ai::GemmCudaBackward(A,B,/*C*/nullptr, /*gY*/Y, /*Z*/Z,
                                        /*gA*/&gA_t, /*gB*/&gB_t, /*gC*/nullptr, /*gBias*/&gBias_t,
                                        attrs,s,&ws);
  if(stb!=ai::Status::Ok){ fprintf(stderr,"BWD fail\n"); return 2; }
  CUDA_CHECK(cudaStreamSynchronize(cs));

  // ⬇⬇ 호스트로 가져올 버퍼 이름도 충돌 피하기 위해 _h 접미사 사용
  std::vector<float> gA_h(M*K), gB_h(K*N), gBias_h(N);
  d_gA.d2h(gA_h.data()); d_gB.d2h(gB_h.data()); d_gBias.d2h(gBias_h.data());

  if(!allclose(gA_h,gA_ref,5e-4f,5e-4f)){ fprintf(stderr,"gA mismatch\n"); return 3; }
  if(!allclose(gB_h,gB_ref,5e-4f,5e-4f)){ fprintf(stderr,"gB mismatch\n"); return 4; }
  if(!allclose(gBias_h,gBias_ref,5e-4f,5e-4f)){ fprintf(stderr,"gBias mismatch\n"); return 5; }

  // 유한 차분(샘플)
  const float eps=1e-3f;
  std::vector<float> Yp(M*N), Ym(M*N), Ztmp(M*N);
  { auto Awork=hA; Awork[0]+=eps; cpu_forward(Awork.data(),hB.data(),hBias.data(),Ztmp.data(),Yp.data(),M,N,K,ldA,ldB,ldY); }
  { auto Awork=hA; Awork[0]-=eps; cpu_forward(Awork.data(),hB.data(),hBias.data(),Ztmp.data(),Ym.data(),M,N,K,ldA,ldB,ldY); }
  float Lp=cpu_loss(Yp.data(),M,N,ldY), Lm=cpu_loss(Ym.data(),M,N,ldY);
  float dnum=(Lp-Lm)/(2*eps), dan=gA_h[0]; // ← host gradient에서 비교
  if(std::fabs(dnum-dan) > 3e-3f + 1e-2f*std::fabs(dan)){
    fprintf(stderr,"finite-diff mismatch: num=%f an=%f\n",dnum,dan); return 6;
  }

  CUDA_CHECK(cudaStreamDestroy(cs));
  return 0;
}
