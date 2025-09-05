#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#include <cuda_runtime.h>
#include "regemm/api.h"

using namespace regemm;

static void cpu_ref(const GemmBiasActParams& p,
                    const std::vector<float>& A,
                    const std::vector<float>& B,
                    const std::vector<float>& C,
                    const std::vector<float>& bias,
                    std::vector<float>& Dref) {
  auto idx = [&](int r,int c,int ld){ return r*ld + c; };
  for (int m=0;m<p.M;++m){
    for(int n=0;n<p.N;++n){
      float acc = 0.f;
      for (int k=0;k<p.K;++k) acc += A[idx(m,k,p.lda)] * B[idx(k,n,p.ldb)];
      acc *= p.alpha;
      if (p.beta!=0.f && p.C) acc += p.beta * C[idx(m,n,p.ldc)];
      float b = 0.f;
      if (p.bias) {
        if (p.bias_kind==BiasKind::PerN) b = bias[n];
        else if (p.bias_kind==BiasKind::PerM) b = bias[m];
        else if (p.bias_kind==BiasKind::Scalar) b = bias[0];
      }
      acc += b;
      if (p.act==ActKind::ReLU) acc = acc>0.f?acc:0.f;
      Dref[idx(m,n,p.ldd)] = acc;
    }
  }
}

int main(){
  int M=128,N=96,K=64;
  GemmBiasActParams p{};
  p.M=M; p.N=N; p.K=K;
  p.alpha=1.f; p.beta=1.f;
  p.lda=K; p.ldb=N; p.ldc=N; p.ldd=N;
  p.bias_kind=BiasKind::PerN; p.act=ActKind::ReLU; p.dtype=DType::F32;

  std::mt19937 rng(0); std::uniform_real_distribution<float> U(-1,1);
  std::vector<float> hA(M*K), hB(K*N), hC(M*N), hBias(N), hDref(M*N);
  for (auto& x:hA) x=U(rng); for(auto& x:hB) x=U(rng);
  for (auto& x:hC) x=U(rng); for(auto& x:hBias) x=U(rng);

  // CPU ref
  {
    GemmBiasActParams pc=p; pc.A=hA.data(); pc.B=hB.data(); pc.C=hC.data(); pc.bias=hBias.data(); pc.D=hDref.data();
    cpu_ref(pc,hA,hB,hC,hBias,hDref);
  }

  // GPU
  float *dA,*dB,*dC,*dD,*dBias;
  cudaMalloc(&dA, M*K*sizeof(float));
  cudaMalloc(&dB, K*N*sizeof(float));
  cudaMalloc(&dC, M*N*sizeof(float));
  cudaMalloc(&dD, M*N*sizeof(float));
  cudaMalloc(&dBias, N*sizeof(float));
  cudaMemcpy(dA,hA.data(),M*K*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(dB,hB.data(),K*N*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(dC,hC.data(),M*N*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(dBias,hBias.data(),N*sizeof(float),cudaMemcpyHostToDevice);

  GemmBiasActParams pg = p;
  pg.A=dA; pg.B=dB; pg.C=dC; pg.D=dD; pg.bias=dBias;
  int err = gemm_bias_act(pg, nullptr);
  if (err) { std::fprintf(stderr,"launch error %d\n",err); return 1; }
  cudaDeviceSynchronize();

  std::vector<float> hD(M*N);
  cudaMemcpy(hD.data(), dD, M*N*sizeof(float), cudaMemcpyDeviceToHost);

  // Check
  int mism=0; double max_abs=0;
  for (int i=0;i<M*N;++i){
    double diff = std::fabs(hD[i]-hDref[i]);
    max_abs = std::max(max_abs,diff);
    if (diff>1e-4) ++mism;
  }
  std::printf("mismatch=%d, max_abs=%.3g\n", mism, max_abs);
  bool ok = mism==0;
  std::puts(ok ? "PASS" : "FAIL");

  cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dD); cudaFree(dBias);
  return ok?0:1;
}
