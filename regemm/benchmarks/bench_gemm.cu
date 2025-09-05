// benchmarks/bench_gemm.cu
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include "regemm/api.h"

using namespace regemm;

static double tflops(double ms, long long M, long long N, long long K) {
  // GEMM FLOPs (FMA = 2 FLOPs)
  long double flops = 2.0L * M * N * K;
  return double(flops / (ms * 1e-3) / 1e12);
}

// 아주 러프한 메모리 트래픽 추정(바이트)
// A: M*K, B: K*N, C: (beta!=0) M*N, bias: PerN→N / PerM→M / Scalar→1, D: M*N
static double bytes_moved(long long M, long long N, long long K,
                          bool useC, BiasKind bk, size_t sizeofT = sizeof(float)) {
  long double bytes = 0;
  bytes += (long double)M * K * sizeofT;   // A read
  bytes += (long double)K * N * sizeofT;   // B read
  if (useC) bytes += (long double)M * N * sizeofT; // C read
  // bias read (rough; 실제 캐시 재사용은 더 적게 읽음)
  if (bk == BiasKind::PerN) bytes += (long double)N * sizeofT;
  else if (bk == BiasKind::PerM) bytes += (long double)M * sizeofT;
  else if (bk == BiasKind::Scalar) bytes += (long double)1 * sizeofT;
  // D write
  bytes += (long double)M * N * sizeofT;
  return double(bytes);
}

static void print_usage(const char* prog){
  std::printf(
    "Usage: %s M N K [reps=50] [alpha=1.0] [beta=0.0] [act=0:none|1:ReLU] [bias=0:none|1:perN|2:perM|3:scalar]\n"
    "Example: %s 2048 2048 2048 30 1.0 0.0 0 1\n", prog, prog);
}

int main(int argc, char** argv){
  long long M=1024, N=1024, K=1024;
  int reps=50; float alpha=1.f, beta=0.f;
  int act_i=0, bias_i=0;

  if (argc < 4) {
    print_usage(argv[0]);
    std::puts("Running default: 1024 1024 1024 reps=50 alpha=1 beta=0 act=0 bias=0");
  } else {
    M = std::atoll(argv[1]); N = std::atoll(argv[2]); K = std::atoll(argv[3]);
    if (argc > 4) reps   = std::atoi(argv[4]);
    if (argc > 5) alpha  = std::atof(argv[5]);
    if (argc > 6) beta   = std::atof(argv[6]);
    if (argc > 7) act_i  = std::atoi(argv[7]);
    if (argc > 8) bias_i = std::atoi(argv[8]);
  }

  if (M<=0 || N<=0 || K<=0 || reps<=0){
    std::fprintf(stderr, "Invalid args.\n");
    return 1;
  }

  ActKind  act  = (act_i==1)? ActKind::ReLU : ActKind::None;
  BiasKind bias = BiasKind::None;
  if (bias_i==1) bias = BiasKind::PerN;
  else if (bias_i==2) bias = BiasKind::PerM;
  else if (bias_i==3) bias = BiasKind::Scalar;

  // Host init (간단히 1/0.5로 채움; 성능 측정 목적)
  size_t szA = size_t(M)*K, szB=size_t(K)*N, szC=size_t(M)*N, szD=szC;
  std::vector<float> hA(szA, 1.f), hB(szB, 1.f), hC(szC, 0.5f);

  // Device alloc
  float *dA=nullptr,*dB=nullptr,*dC=nullptr,*dD=nullptr,*dBias=nullptr;
  cudaMalloc(&dA, szA*sizeof(float));
  cudaMalloc(&dB, szB*sizeof(float));
  cudaMalloc(&dC, szC*sizeof(float));
  cudaMalloc(&dD, szD*sizeof(float));

  // bias 크기: 기본 perN(열). perM이면 M, scalar면 1.
  size_t bias_len = 0;
  if (bias == BiasKind::PerN) bias_len = size_t(N);
  else if (bias == BiasKind::PerM) bias_len = size_t(M);
  else if (bias == BiasKind::Scalar) bias_len = 1;
  if (bias_len) cudaMalloc(&dBias, bias_len*sizeof(float));

  // Upload
  cudaMemcpy(dA, hA.data(), szA*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dB, hB.data(), szB*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dC, hC.data(), szC*sizeof(float), cudaMemcpyHostToDevice);
  if (dBias) cudaMemset(dBias, 0, bias_len*sizeof(float)); // 0-bias

  // Params
  GemmBiasActParams p{};
  p.M=int(M); p.N=int(N); p.K=int(K);
  p.alpha=alpha; p.beta=beta;
  p.A=dA; p.B=dB; p.C=(beta!=0.f)? dC : nullptr; // beta==0이면 굳이 C 안 씀
  p.D=dD;
  p.lda=int(K); p.ldb=int(N); p.ldc=int(N); p.ldd=int(N);
  p.bias=dBias; p.bias_kind=bias;
  p.act=act; p.dtype=DType::F32;

  // Warmup
  int warmup = (reps>5)? 5 : 1;
  for (int i=0;i<warmup;i++) gemm_bias_act(p, nullptr);
  cudaDeviceSynchronize();

  // Timing
  cudaEvent_t evs, eve; cudaEventCreate(&evs); cudaEventCreate(&eve);
  cudaEventRecord(evs);
  for (int i=0;i<reps;i++) gemm_bias_act(p, nullptr);
  cudaEventRecord(eve); cudaEventSynchronize(eve);
  float total_ms=0.f; cudaEventElapsedTime(&total_ms, evs, eve);
  float avg_ms = total_ms / reps;

  // Metrics
  double tf = tflops(avg_ms, M,N,K);
  double bytes = bytes_moved(M,N,K, (beta!=0.f), bias);
  double gbs = bytes / (avg_ms * 1e-3) / 1e9;

  // Print
  std::printf("path=auto (launcher selects)\n");
  std::printf("M=%lld N=%lld K=%lld reps=%d alpha=%.3f beta=%.3f act=%d bias=%d\n",
              M,N,K,reps,alpha,beta,act_i,bias_i);
  std::printf("avg=%.3f ms  TFLOP/s=%.3f  approxGB/s=%.1f\n", avg_ms, tf, gbs);

  // Cleanup
  cudaEventDestroy(evs); cudaEventDestroy(eve);
  cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dD); if (dBias) cudaFree(dBias);
  return 0;
}
