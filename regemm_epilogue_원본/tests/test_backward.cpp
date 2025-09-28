// tests/test_backward.cpp
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#include <cassert>
#include <stdio.h>
#define _USE_MATH_DEFINES
#include <math.h>



#include "regemm/api.h"

using namespace regemm;

static float gelu(float x){
  const float k0 = std::sqrt(2.0f/M_PI);
  const float k1 = 0.044715f;
  float t = k0*(x + k1*x*x*x);
  return 0.5f*x*(1.0f + std::tanh(t));
}
static float dgelu(float x){
  const float c = std::sqrt(2.0f/M_PI);
  const float k1 = 0.044715f;
  float x2 = x*x;
  float t = c*(x + k1*x*x2);
  float th = std::tanh(t);
  float sech2 = 1.0f - th*th;
  float dt = c*(1.0f + 3.0f*k1*x2);
  return 0.5f*(1.0f + th) + 0.5f*x*sech2*dt;
}
static float relu(float x){ return x>0?x:0; }
static float drelu(float x){ return x>0?1.0f:0.0f; }
static float leaky(float x, float a){ return x>0?x:a*x; }
static float dleaky(float x, float a){ return x>0?1.0f:a; }
static float sigmoid(float x){ return 1.0f/(1.0f+std::exp(-x)); }
static float dsigmoid(float x){ float s=sigmoid(x); return s*(1.0f-s); }
static float dtanhf(float x){ float t=std::tanh(x); return 1.0f - t*t; }

static float act(float x, ActKind ak, float a){
  switch(ak){
    case ActKind::ReLU: return relu(x);
    case ActKind::LeakyReLU: return leaky(x,a);
    case ActKind::GELU: return gelu(x);
    case ActKind::Sigmoid: return sigmoid(x);
    case ActKind::Tanh: return std::tanh(x);
    case ActKind::None: default: return x;
  }
}
static float dact(float x, ActKind ak, float a){
  switch(ak){
    case ActKind::ReLU: return drelu(x);
    case ActKind::LeakyReLU: return dleaky(x,a);
    case ActKind::GELU: return dgelu(x);
    case ActKind::Sigmoid: return dsigmoid(x);
    case ActKind::Tanh: return dtanhf(x);
    case ActKind::None: default: return 1.0f;
  }
}

static void cpu_matmul_MxK_KxN(const float* A, const float* B, float* C, int M,int N,int K, int lda,int ldb,int ldc) {
  for(int m=0;m<M;++m){
    for(int n=0;n<N;++n){
      float acc=0.f;
      for(int k=0;k<K;++k){
        acc += A[m*lda+k]*B[k*ldb+n];
      }
      C[m*ldc+n]=acc;
    }
  }
}

static float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b){
  assert(a.size()==b.size());
  float m=0; for(size_t i=0;i<a.size();++i){ m=std::max(m, std::fabs(a[i]-b[i])); }
  return m;
}

int main(){
  const int M=64, N=48, K=32;
  const float alpha=1.1f, beta=0.9f, leaky=0.02f;
  const ActKind actk = ActKind::GELU;          // 필요시 바꿔가며 확인
  const BiasKind bkind = BiasKind::PerN;       // Scalar/PerM/PerN 교체해서 확인

  std::mt19937 rng(0);
  std::uniform_real_distribution<float> U(-1,1);

  std::vector<float> hA(M*K), hB(K*N), hC(M*N), hBias, hGY(M*N);
  for(auto& x: hA) x=U(rng);
  for(auto& x: hB) x=U(rng);
  for(auto& x: hC) x=U(rng);
  for(auto& x: hGY) x=U(rng);

  if(bkind==BiasKind::Scalar){ hBias.resize(1); hBias[0]=0.1f; }
  else if(bkind==BiasKind::PerM){ hBias.resize(M); std::fill(hBias.begin(), hBias.end(), 0.1f); }
  else if(bkind==BiasKind::PerN){ hBias.resize(N); std::fill(hBias.begin(), hBias.end(), 0.1f); }

  // ---- CPU ref: Z, Y, gZ, gA, gB, gC, gBias ----
  std::vector<float> hZ(M*N), hAB(M*N);
  cpu_matmul_MxK_KxN(hA.data(), hB.data(), hAB.data(), M,N,K, K,N,N);
  for(int m=0;m<M;++m){
    for(int n=0;n<N;++n){
      float pre = alpha*hAB[m*N+n] + beta*hC[m*N+n];
      if(bkind==BiasKind::Scalar) pre += hBias[0];
      else if(bkind==BiasKind::PerM) pre += hBias[m];
      else if(bkind==BiasKind::PerN) pre += hBias[n];
      hZ[m*N+n]=pre;
    }
  }
  // gZ = gY * act'(Z)
  std::vector<float> hGZ(M*N);
  for(int i=0;i<M*N;++i){ hGZ[i]= hGY[i] * dact(hZ[i], actk, leaky); }

  // gA = gZ @ B^T  (MxK)
  std::vector<float> hGA(M*K,0.f);
  for(int m=0;m<M;++m){
    for(int k=0;k<K;++k){
      float acc=0.f;
      for(int n=0;n<N;++n) acc += hGZ[m*N+n]*hB[k*N+n];
      hGA[m*K+k]=acc;
    }
  }
  // gB = A^T @ gZ  (KxN)
  std::vector<float> hGB(K*N,0.f);
  for(int k=0;k<K;++k){
    for(int n=0;n<N;++n){
      float acc=0.f;
      for(int m=0;m<M;++m) acc += hA[m*K+k]*hGZ[m*N+n];
      hGB[k*N+n]=acc;
    }
  }
  // gC = beta * gZ
  std::vector<float> hGC(M*N);
  for(int i=0;i<M*N;++i) hGC[i]= beta*hGZ[i];

  // gBias
  std::vector<float> hGBias;
  if(bkind==BiasKind::Scalar){
    hGBias.resize(1,0.f);
    for(float v: hGZ) hGBias[0]+=v;
  }else if(bkind==BiasKind::PerM){
    hGBias.resize(M,0.f);
    for(int m=0;m<M;++m){ float s=0; for(int n=0;n<N;++n) s+=hGZ[m*N+n]; hGBias[m]=s; }
  }else if(bkind==BiasKind::PerN){
    hGBias.resize(N,0.f);
    for(int n=0;n<N;++n){ float s=0; for(int m=0;m<M;++m) s+=hGZ[m*N+n]; hGBias[n]=s; }
  }

  // ---- GPU call: forward EX로 Z 저장 후 backward ----
  float *dA,*dB,*dC,*dD,*dZ,*dBias,*dGY,*dGA,*dGB,*dGC,*dGBias=nullptr;
  cudaStream_t stream; cudaStreamCreate(&stream);
  cudaMalloc(&dA, M*K*sizeof(float));
  cudaMalloc(&dB, K*N*sizeof(float));
  cudaMalloc(&dC, M*N*sizeof(float));
  cudaMalloc(&dD, M*N*sizeof(float));
  cudaMalloc(&dZ, M*N*sizeof(float));
  cudaMalloc(&dGY, M*N*sizeof(float));
  if(!hBias.empty()) cudaMalloc(&dBias, hBias.size()*sizeof(float));
  cudaMalloc(&dGA, M*K*sizeof(float));
  cudaMalloc(&dGB, K*N*sizeof(float));
  cudaMalloc(&dGC, M*N*sizeof(float));
  if(!hGBias.empty()) cudaMalloc(&dGBias, hGBias.size()*sizeof(float));

  cudaMemcpyAsync(dA, hA.data(), M*K*sizeof(float), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(dB, hB.data(), K*N*sizeof(float), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(dC, hC.data(), M*N*sizeof(float), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(dGY, hGY.data(), M*N*sizeof(float), cudaMemcpyHostToDevice, stream);
  if(!hBias.empty()) cudaMemcpyAsync(dBias, hBias.data(), hBias.size()*sizeof(float), cudaMemcpyHostToDevice, stream);

  // forward EX (Z stash)
  GemmBiasActParamsEx fwd{};
  fwd.M=M; fwd.N=N; fwd.K=K;
  fwd.A=dA; fwd.lda=K;
  fwd.B=dB; fwd.ldb=N;
  fwd.C=dC; fwd.ldc=N;
  fwd.D=dD; fwd.ldd=N;
  fwd.alpha=alpha; fwd.beta=beta;
  fwd.bias=dBias; fwd.bias_kind=bkind;
  fwd.act=actk; fwd.leaky_slope=leaky;
  fwd.Z=dZ; fwd.ldZ=0; fwd.save_preact=1;

  gemm_bias_act_f32_ex(fwd, stream);

  // backward
  GemmBiasActBwdParams bwd{};
  bwd.M=M; bwd.N=N; bwd.K=K;
  bwd.A=dA; bwd.lda=K;
  bwd.B=dB; bwd.ldb=N;
  bwd.C=dC; bwd.ldc=N;
  bwd.gY=dGY; bwd.ldgY=N;
  bwd.Z=dZ;  bwd.ldZ=N;
  bwd.gA=dGA; bwd.ldgA=K;
  bwd.gB=dGB; bwd.ldgB=N;
  bwd.gC=dGC; bwd.ldgC=N;
  bwd.gBias=dGBias;
  bwd.alpha=alpha; bwd.beta=beta;
  bwd.bias_kind=bkind; bwd.act=actk; bwd.leaky_slope=leaky;

  gemm_bias_act_bwd_f32(bwd, stream);

  // copy back
  std::vector<float> GA(M*K), GB(K*N), GC(M*N), GBias(hGBias.size());
  cudaMemcpyAsync(GA.data(), dGA, M*K*sizeof(float), cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(GB.data(), dGB, K*N*sizeof(float), cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(GC.data(), dGC, M*N*sizeof(float), cudaMemcpyDeviceToHost, stream);
  if(dGBias) cudaMemcpyAsync(GBias.data(), dGBias, GBias.size()*sizeof(float), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  float tol = 3e-4f; // 넉넉히
  float dA_max = max_abs_diff(GA, hGA);
  float dB_max = max_abs_diff(GB, hGB);
  float dC_max = max_abs_diff(GC, hGC);
  float dBias_max = (hGBias.empty()?0.0f:max_abs_diff(GBias, hGBias));

  printf("[bwd] dA max|diff| = %.8f\n", dA_max);
  printf("[bwd] dB max|diff| = %.8f\n", dB_max);
  printf("[bwd] dC max|diff| = %.8f\n", dC_max);
  if(!hGBias.empty()) printf("[bwd] dBias max|diff| = %.8f\n", dBias_max);

  bool ok = (dA_max<tol && dB_max<tol && dC_max<tol && (hGBias.empty() || dBias_max<tol));
  printf("[RESULT] OK=%s (tol=%.1e)\n", ok?"True":"False", tol);

  // cleanup
  if(dGBias) cudaFree(dGBias);
  cudaFree(dGC); cudaFree(dGB); cudaFree(dGA);
  if(dBias) cudaFree(dBias);
  cudaFree(dGY); cudaFree(dZ); cudaFree(dD); cudaFree(dC); cudaFree(dB); cudaFree(dA);
  cudaStreamDestroy(stream);

  return ok?0:1;
}
