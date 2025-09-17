#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <string>
#include <algorithm>

#include "regemm/api.h"  // BiasKind, ActKind, GemmBiasActParams
// 커널 런처 심볼 (필요에 맞게 선택)
// #include "regemm/api.h" 안에서 선언되어 있지 않다면 아래 선언을 추가하세요.

using namespace regemm;

// --- CUDA helpers ------------------------------------------------------------
static void* dalloc(size_t bytes) { void* p=nullptr; cudaMalloc(&p, bytes); return p; }
static void h2d(void* dst, const void* src, size_t bytes){ cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice); }
static void d2h(void* dst, const void* src, size_t bytes){ cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost); }

// --- Host-side activations (커널과 동일한 정의) -------------------------------
static inline float act_host(float x, ActKind k) {
  switch (k) {
    case ActKind::ReLU:       return x > 0.f ? x : 0.f;
    case ActKind::LeakyReLU:  { const float a = 0.01f; return x > 0.f ? x : a*x; }
    case ActKind::GELU:       { // tanh approx: 0.5*x*(1 + tanh(√(2/π)*(x + 0.044715 x^3)))
      const float k0 = 0.7978845608f; // sqrt(2/pi)
      const float k1 = 0.044715f;
      float x3 = x*x*x;
      float t  = k0 * (x + k1 * x3);
      return 0.5f * x * (1.f + std::tanh(t));
    }
    case ActKind::Sigmoid:    return 1.f / (1.f + std::exp(-x));
    case ActKind::Tanh:       return std::tanh(x);
    case ActKind::None:
    default:                  return x;
  }
}

// --- Metrics -----------------------------------------------------------------
struct ErrStats { double l2=0, maxabs=0; };
static ErrStats compare_arrays(const std::vector<float>& a, const std::vector<float>& b) {
  ErrStats e{};
  double acc=0, m=0;
  size_t n = a.size();
  for (size_t i=0;i<n;i++){
    double d = (double)a[i] - (double)b[i];
    acc += d*d;
    m = std::max(m, std::abs(d));
  }
  e.l2 = std::sqrt(acc / std::max<size_t>(1,n));
  e.maxabs = m;
  return e;
}

static void print_samples(const std::vector<float>& v, int M, int N, const char* tag) {
  // 몇 개만 샘플 로그 (좌상단, 중간, 우하단 등)
  int idxs[6] = {0, N/2, N-1, (M/2)*N, (M/2)*N + N/2, M*N - 1};
  printf("  [%s] samples: ", tag);
  for (int k=0;k<6;k++){
    int i = std::clamp(idxs[k], 0, M*N-1);
    printf("%+.5f  ", v[i]);
  }
  printf("\n");
}

static const char* act_name(ActKind k) {
  switch (k) {
    case ActKind::None:      return "None";
    case ActKind::ReLU:      return "ReLU";
    case ActKind::LeakyReLU: return "LeakyReLU";
    case ActKind::GELU:      return "GELU";
    case ActKind::Sigmoid:   return "Sigmoid";
    case ActKind::Tanh:      return "Tanh";
    default:                 return "Unknown";
  }
}

int main() {
  // --- Problem size ----------------------------------------------------------
  const int M=1024, N=1024, K=1024;

  // --- Host buffers ----------------------------------------------------------
  std::vector<float> hA(M*K), hB(K*N), hC(M*N), hD(M*N), hBiasN(N, 0.1f);
  std::vector<float> hD_pre(M*N);   // pre-activation (Act=None 결과 저장)
  std::vector<float> hD_dev(M*N);   // device 결과 복사용
  std::vector<float> hD_host(M*N);  // host에서 activation 적용 결과

  // --- Init data -------------------------------------------------------------
  std::mt19937 rng(123);
  std::uniform_real_distribution<float> U(-1,1);
  for (auto& x: hA) x = U(rng);
  for (auto& x: hB) x = U(rng);
  for (auto& x: hC) x = U(rng);

  // --- Device buffers --------------------------------------------------------
  float *dA=(float*)dalloc(sizeof(float)*M*K);
  float *dB=(float*)dalloc(sizeof(float)*K*N);
  float *dC=(float*)dalloc(sizeof(float)*M*N);
  float *dD=(float*)dalloc(sizeof(float)*M*N);
  float *dBiasN=(float*)dalloc(sizeof(float)*N);

  h2d(dA, hA.data(), sizeof(float)*M*K);
  h2d(dB, hB.data(), sizeof(float)*K*N);
  h2d(dC, hC.data(), sizeof(float)*M*N);
  h2d(dD, hD.data(), sizeof(float)*M*N);         // 초기화 용도(필수는 아님)
  h2d(dBiasN, hBiasN.data(), sizeof(float)*N);

  // --- Params ----------------------------------------------------------------
  GemmBiasActParams p{};
  p.M=M; p.N=N; p.K=K;
  p.A=dA; p.lda=K;
  p.B=dB; p.ldb=N;
  p.C=dC; p.ldc=N;
  p.D=dD; p.ldd=N;
  p.alpha=1.f; p.beta=1.f;
  p.bias=dBiasN; p.bias_kind=BiasKind::PerN;

  cudaStream_t s; cudaStreamCreate(&s);

  // --- 0) Pre-activation 만들기 (Act=None) -----------------------------------
  p.act = ActKind::None;
  for (int i=0;i<3;i++) launch_gemm_bias_act_f32_tiled(p, s); // 가벼운 워밍업
  cudaStreamSynchronize(s);
  d2h(hD_pre.data(), dD, sizeof(float)*M*N);     // D0 = αAB + βC + bias

  // --- 검증 대상 activation 목록 ---------------------------------------------
  const ActKind acts[] = {
    ActKind::None, ActKind::ReLU, ActKind::LeakyReLU,
    ActKind::GELU, ActKind::Sigmoid, ActKind::Tanh
  };

  printf("== Activation correctness & timing check (M=%d N=%d K=%d) ==\n", M,N,K);

  for (ActKind ak: acts) {
    // 1) GPU 실행
    p.act = ak;

    // warmup
    for (int i=0;i<5;i++) launch_gemm_bias_act_f32_tiled(p, s);
    cudaStreamSynchronize(s);

    // timing
    auto t0 = std::chrono::high_resolution_clock::now();
    const int iters = 30;
    for (int it=0; it<iters; ++it) launch_gemm_bias_act_f32_tiled(p, s);
    cudaStreamSynchronize(s);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1-t0).count()/iters;

    // 결과 가져오기
    d2h(hD_dev.data(), dD, sizeof(float)*M*N);

    // 2) Host에서 동일 activation 적용
    if (ak == ActKind::None) {
      hD_host = hD_pre; // 그대로
    } else {
      for (int i=0;i<M*N;i++) hD_host[i] = act_host(hD_pre[i], ak);
    }

    // 3) 비교
    ErrStats e = compare_arrays(hD_dev, hD_host);

    // 4) TFLOPs (GEMM FLOPs 기준)
    double flops = 2.0 * (double)M * (double)N * (double)K; // FMA=2 FLOPs
    double tflops = flops / (ms*1e-3) / 1e12;

    // 5) 로그
    printf("  %-9s : %7.3f ms  (%.2f TFLOP/s) | L2=%.3e  Max=%.3e\n",
           act_name(ak), ms, tflops, e.l2, e.maxabs);

    // 샘플 몇 개 출력 (GPU/Host 둘 다)
    print_samples(hD_dev,  M, N, "GPU");
    print_samples(hD_host, M, N, "CPU");
  }

  // --- cleanup ----------------------------------------------------------------
  cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dD); cudaFree(dBiasN);
  cudaStreamDestroy(s);
  return 0;
}
