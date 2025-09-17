#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <random>
#include <chrono>

#include "regemm/api.h"

using namespace regemm;

static void* dalloc(size_t bytes) {
  void* p=nullptr; cudaMalloc(&p, bytes); return p;
}
static void h2d(void* dst, const void* src, size_t bytes){ cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice); }
static void d2h(void* dst, const void* src, size_t bytes){ cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost); }

int main() {
  const int M=1024, N=1024, K=1024;
  std::vector<float> hA(M*K), hB(K*N), hC(M*N), hD(M*N), hBiasN(N, 0.1f);

  std::mt19937 rng(123);
  std::uniform_real_distribution<float> U(-1,1);
  for (auto& x: hA) x = U(rng);
  for (auto& x: hB) x = U(rng);
  for (auto& x: hC) x = U(rng);

  float *dA=(float*)dalloc(sizeof(float)*M*K);
  float *dB=(float*)dalloc(sizeof(float)*K*N);
  float *dC=(float*)dalloc(sizeof(float)*M*N);
  float *dD=(float*)dalloc(sizeof(float)*M*N);
  float *dBiasN=(float*)dalloc(sizeof(float)*N);

  h2d(dA, hA.data(), sizeof(float)*M*K);
  h2d(dB, hB.data(), sizeof(float)*K*N);
  h2d(dC, hC.data(), sizeof(float)*M*N);
  h2d(dD, hD.data(), sizeof(float)*M*N);
  h2d(dBiasN, hBiasN.data(), sizeof(float)*N);

  GemmBiasActParams p{};
  p.M=M; p.N=N; p.K=K;
  p.A=dA; p.lda=K;
  p.B=dB; p.ldb=N;
  p.C=dC; p.ldc=N;
  p.D=dD; p.ldd=N;
  p.alpha=1.f; p.beta=1.f;
  p.bias=dBiasN; p.bias_kind=BiasKind::PerN;
  p.act=ActKind::ReLU;

  cudaStream_t s; cudaStreamCreate(&s);

  // Warmup
  for(int i=0;i<5;i++) launch_gemm_bias_act_f32_tiled(p, s);
  cudaStreamSynchronize(s);

  // Timing
  auto beg = std::chrono::high_resolution_clock::now();
  for(int it=0; it<50; ++it) launch_gemm_bias_act_f32_tiled(p, s);
  cudaStreamSynchronize(s);
  auto end = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration<double, std::milli>(end-beg).count()/50.0;

  double flops = 2.0 * M * N * K; // FMA = 2 FLOPs
  double tflops = flops / (ms*1e-3) / 1e12;

  printf("[bench] %dx%dx%d: %.3f ms  (%.2f TFLOP/s)\\n", M,N,K, ms, tflops);

  cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dD); cudaFree(dBiasN);
  cudaStreamDestroy(s);
  return 0;
}
