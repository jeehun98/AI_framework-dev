#include <cassert>
#include <cmath>
#include <vector>
#include <random>
#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#include "regemm/api.h"

using namespace regemm;

// =======================
// 기본 허용치(완화 버전)
// =======================
#ifndef TEST_ATOL
#define TEST_ATOL 1e-3f        // 절대 허용오차 (작은 값 보호)
#endif
#ifndef TEST_RTOL
#define TEST_RTOL 5e-2f        // 상대 허용오차 (5%)
#endif
#ifndef TEST_MAX_BAD_FRAC
#define TEST_MAX_BAD_FRAC 1e-3 // 허용되는 mismatch 비율(0.1%)
#endif
#ifndef PRINT_MAX_MISMATCH
#define PRINT_MAX_MISMATCH 25  // 상세 로그 최대 개수
#endif

// CUDA error check
#define CUDA_CHECK(x) do { \
  cudaError_t _e = (x); \
  if (_e != cudaSuccess) { \
    std::cerr << "CUDA error: " << cudaGetErrorString(_e) \
              << " @ " << __FILE__ << ":" << __LINE__ << std::endl; \
    std::exit(2); \
  } \
} while(0)

// Device alloc/transfer helpers
static void* dalloc(size_t bytes){ void* p=nullptr; CUDA_CHECK(cudaMalloc(&p, bytes)); return p; }
static void h2d(void* dst,const void* src,size_t bytes){ CUDA_CHECK(cudaMemcpy(dst,src,bytes,cudaMemcpyHostToDevice)); }
static void d2h(void* dst,const void* src,size_t bytes){ CUDA_CHECK(cudaMemcpy(dst,src,bytes,cudaMemcpyDeviceToHost)); }

// ----------------------------
// CPU reference (float FMA)
// ----------------------------
static void gemm_ref(const std::vector<float>& A, const std::vector<float>& B,
                     const std::vector<float>& C, std::vector<float>& D,
                     int M,int N,int K,float alpha,float beta,
                     const std::vector<float>* biasN, ActKind act)
{
  auto actf = [&](float x){ return (act==ActKind::ReLU) ? (x>0.f?x:0.f) : x; };
  for (int m=0; m<M; ++m){
    for (int n=0; n<N; ++n){
      float acc = 0.f;
      for (int k=0; k<K; ++k){
        // float FMA 경로 사용: GPU와 라운딩 경로를 최대한 맞춤
        acc = std::fma((float)A[m*K+k], (float)B[k*N+n], acc);
      }
      float v = alpha*acc + beta*C[m*N+n];
      if (biasN) v += (*biasN)[n];
      D[m*N+n] = actf(v);
    }
  }
}

// 혼합 허용오차 판정
inline bool close_mixed(float a, float b, float atol, float rtol){
  float diff  = std::fabs(a-b);
  float scale = std::max(std::fabs(a), std::fabs(b));
  return diff <= (atol + rtol*scale);
}

// 간단한 파라미터 구조체 (CLI/환경변수로 세팅)
struct TestCfg {
  int   M = 128, N = 160, K = 96;
  float atol = TEST_ATOL;
  float rtol = TEST_RTOL;
  float max_bad_frac = TEST_MAX_BAD_FRAC; // 허용되는 mismatch 비율
  bool  compare_gpu_vs_gpu = false;       // true면 smoke vs tiled 비교
  int   print_max = PRINT_MAX_MISMATCH;
};

static void load_cfg_from_env(TestCfg& cfg){
  if (const char* s = std::getenv("TEST_M")) cfg.M = std::max(1, std::atoi(s));
  if (const char* s = std::getenv("TEST_N")) cfg.N = std::max(1, std::atoi(s));
  if (const char* s = std::getenv("TEST_K")) cfg.K = std::max(1, std::atoi(s));
  if (const char* s = std::getenv("TEST_ATOL")) cfg.atol = std::atof(s);
  if (const char* s = std::getenv("TEST_RTOL")) cfg.rtol = std::atof(s);
  if (const char* s = std::getenv("TEST_MAX_BAD_FRAC")) cfg.max_bad_frac = std::atof(s);
  if (const char* s = std::getenv("TEST_COMPARE_GPU")) cfg.compare_gpu_vs_gpu = (std::atoi(s) != 0);
}

static void load_cfg_from_argv(TestCfg& cfg, int argc, char** argv){
  // 사용법: test_basic [--M 128 --N 160 --K 96 --atol 1e-3 --rtol 5e-2 --badfrac 1e-3 --g2g 0/1]
  for (int i=1; i<argc; ++i){
    auto eq = [&](const char* a,const char* b){ return std::strcmp(a,b)==0; };
    if (eq(argv[i],"--M")   && i+1<argc) cfg.M = std::max(1, std::atoi(argv[++i]));
    else if (eq(argv[i],"--N") && i+1<argc) cfg.N = std::max(1, std::atoi(argv[++i]));
    else if (eq(argv[i],"--K") && i+1<argc) cfg.K = std::max(1, std::atoi(argv[++i]));
    else if (eq(argv[i],"--atol") && i+1<argc) cfg.atol = std::atof(argv[++i]);
    else if (eq(argv[i],"--rtol") && i+1<argc) cfg.rtol = std::atof(argv[++i]);
    else if (eq(argv[i],"--badfrac") && i+1<argc) cfg.max_bad_frac = std::atof(argv[++i]);
    else if (eq(argv[i],"--g2g") && i+1<argc) cfg.compare_gpu_vs_gpu = (std::atoi(argv[++i]) != 0);
  }
}

int main(int argc, char** argv){
  TestCfg cfg;
  load_cfg_from_env(cfg);
  load_cfg_from_argv(cfg, argc, argv);

  // -----------------------------
  // 호스트 데이터 생성
  // -----------------------------
  std::vector<float> A(cfg.M*cfg.K), B(cfg.K*cfg.N), C(cfg.M*cfg.N), BiasN(cfg.N, 0.123f);
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> U(-1.f, 1.f);
  for (auto& x: A) x = U(rng);
  for (auto& x: B) x = U(rng);
  for (auto& x: C) x = U(rng);

  // -----------------------------
  // 디바이스 메모리
  // -----------------------------
  float *dA=(float*)dalloc(sizeof(float)*cfg.M*cfg.K);
  float *dB=(float*)dalloc(sizeof(float)*cfg.K*cfg.N);
  float *dC=(float*)dalloc(sizeof(float)*cfg.M*cfg.N);
  float *dD=(float*)dalloc(sizeof(float)*cfg.M*cfg.N);
  float *dBiasN=(float*)dalloc(sizeof(float)*cfg.N);

  h2d(dA, A.data(), sizeof(float)*cfg.M*cfg.K);
  h2d(dB, B.data(), sizeof(float)*cfg.K*cfg.N);
  h2d(dC, C.data(), sizeof(float)*cfg.M*cfg.N);
  h2d(dBiasN, BiasN.data(), sizeof(float)*cfg.N);

  GemmBiasActParams p{};
  p.M=cfg.M; p.N=cfg.N; p.K=cfg.K;
  p.A=dA; p.lda=cfg.K;
  p.B=dB; p.ldb=cfg.N;
  p.C=dC; p.ldc=cfg.N;
  p.D=dD; p.ldd=cfg.N;
  p.alpha=0.7f; p.beta=0.3f;
  p.bias=dBiasN; p.bias_kind=BiasKind::PerN;
  p.act=ActKind::ReLU;

  cudaStream_t s; CUDA_CHECK(cudaStreamCreate(&s));

  // -----------------------------
  // 기준 결과 준비 (Ref)
  // -----------------------------
  std::vector<float> Ref(cfg.M*cfg.N);

  if (cfg.compare_gpu_vs_gpu) {
    // GPU smoke → Ref
    launch_gemm_bias_act_f32_smoke(p, s);
    CUDA_CHECK(cudaStreamSynchronize(s));
    d2h(Ref.data(), dD, sizeof(float)*cfg.M*cfg.N);
  } else {
    // CPU reference → Ref
    gemm_ref(A, B, C, Ref, cfg.M, cfg.N, cfg.K, p.alpha, p.beta, &BiasN, p.act);
  }

  // -----------------------------
  // 테스트 대상 (D): GPU tiled
  // -----------------------------
  launch_gemm_bias_act_f32_tiled(p, s);
  CUDA_CHECK(cudaStreamSynchronize(s));
  std::vector<float> D(cfg.M*cfg.N);
  d2h(D.data(), dD, sizeof(float)*cfg.M*cfg.N);

  // -----------------------------
  // 비교 + 통계
  // -----------------------------
  int bad = 0;
  double max_abs = 0.0, sum_abs = 0.0;

  for (int i=0; i<cfg.M*cfg.N; ++i){
    float g = D[i], r = Ref[i];
    float ad = std::fabs(g - r);
    max_abs = std::max(max_abs, (double)ad);
    sum_abs += ad;

    if (!close_mixed(g, r, cfg.atol, cfg.rtol)) {
      if (bad < cfg.print_max) {
        double scale = std::max(std::fabs(g), std::fabs(r));
        double rel   = (scale>0.0) ? (ad/scale) : 0.0;
        std::cerr << "Mismatch["<<bad<<"] idx="<<i
                  << " GPU="<<g<<" REF="<<r
                  << " |diff|="<<ad<<" rel="<<rel << "\n";
      }
      ++bad;
    }
  }

  const double mae = sum_abs / double(cfg.M*cfg.N);
  const double bad_frac = double(bad) / double(cfg.M*cfg.N);

  std::cout << "[compare] "
            << (cfg.compare_gpu_vs_gpu? "GPU_vs_GPU(smoke-tiled)" : "CPU_vs_GPU")
            << " bad=" << bad << " (" << bad_frac*100.0 << "%)"
            << "  max|diff|=" << max_abs
            << "  MAE=" << mae
            << "  tol(atol=" << cfg.atol << ", rtol=" << cfg.rtol << ")"
            << "  allow_bad_frac=" << cfg.max_bad_frac
            << "\n";

  bool pass = (bad_frac <= cfg.max_bad_frac);

  if (pass) {
    std::cout << "[test_basic] OK\n";
  } else {
    std::cerr << "[test_basic] FAIL\n";
  }

  CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
  CUDA_CHECK(cudaFree(dD)); CUDA_CHECK(cudaFree(dBiasN)); CUDA_CHECK(cudaStreamDestroy(s));
  return pass ? 0 : 1;
}
