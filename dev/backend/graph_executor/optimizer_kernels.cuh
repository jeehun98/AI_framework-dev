// optimizer_kernels.cuh
#pragma once

#include "optimizer_types.cuh"

// ===================== Build-time toggles (필요 시 -D로 켜기) =====================
#ifndef GRAD_CLIP_ENABLE          // 요소값 클리핑(절댓값 기준 clamp)
#define GRAD_CLIP_ENABLE 0
#endif
#ifndef GRAD_CLIP_THRESH
#define GRAD_CLIP_THRESH 1e4f
#endif

#ifndef GLOBAL_NORM_CLIP_ENABLE   // 글로벌 L2 노름 클리핑(2-pass)
#define GLOBAL_NORM_CLIP_ENABLE 0
#endif

#ifndef WEIGHT_DECAY_ENABLE       // Decoupled WD (SGD-WD/AdamW)
#define WEIGHT_DECAY_ENABLE 0
#endif

#ifndef NESTEROV_ENABLE           // Momentum의 Nesterov 모드
#define NESTEROV_ENABLE 0
#endif

#ifndef AMSGRAD_ENABLE            // Adam의 AMSGrad 변형
#define AMSGRAD_ENABLE 0
#endif

#ifndef DEBUG_KERNEL              // 커널 로그/동기화 토글
#define DEBUG_KERNEL 0
#endif

// ===================== Kernel Prototypes =====================
// NOTE: 모든 커널에서 grad는 const 로 유지. size/n은 요소 개수.

/// SGD (Decoupled WD 선택적)
__global__ void sgd_kernel(
    float* __restrict__ param,
    const float* __restrict__ grad,
#if WEIGHT_DECAY_ENABLE
    float weight_decay,
#endif
    float lr,
    int   n);

/// Momentum (classic/Nesterov, Decoupled WD 선택적)
__global__ void momentum_kernel(
    float* __restrict__ param,
    const float* __restrict__ grad,
    float* __restrict__ velocity,
#if WEIGHT_DECAY_ENABLE
    float weight_decay,
#endif
    float lr,
    float beta,   // momentum 계수
    int   n);

/// Adam / AdamW (+ AMSGrad 선택적)
__global__ void adam_kernel(
    float* __restrict__ param,
    const float* __restrict__ grad,
    float* __restrict__ m,
    float* __restrict__ v,
#if AMSGRAD_ENABLE
    float* __restrict__ vhat_max,    // AMSGrad용 최대 v̂ 유지 버퍼
#endif
#if WEIGHT_DECAY_ENABLE
    float  weight_decay,             // AdamW: decoupled WD
#endif
    float  lr,
    float  beta1,
    float  beta2,
    float  epsilon,
    int    t,    // 1부터 증가 권장(편향보정 안정)
    int    n);

// (옵션) 글로벌 노름 클리핑용 내부 커널을 다른 TU에서 쓰고 싶다면 공개:
// 기본 구현은 .cu 내부 static으로 두는 것을 권장하므로 주석 처리.
// #if GLOBAL_NORM_CLIP_ENABLE
// __global__ void grad_sqsum_kernel(const float* grad, double* partial, int n);
// __global__ void scale_grad_kernel(const float* grad_in, float* grad_out, float scale, int n);
// #endif

// ===================== Host Launcher =====================
// 프레임워크에서 한 API로 호출.
// - SGD: beta1 미사용(0 전달), weight_decay=0이면 off
// - MOMENTUM: beta1 -> beta 로 사용
// - ADAM: t는 1부터 증가
// - AMSGRAD/WD는 빌드 토글 및 인자 유무로 자동 분기
void optimizer_update_cuda(
    float*       param,
    const float* grad,
    float*       velocity,
    float*       m,
    float*       v,
#if AMSGRAD_ENABLE
    float*       vhat_max,
#endif
    float lr, float beta1, float beta2, float epsilon,
#if WEIGHT_DECAY_ENABLE
    float        weight_decay,
#endif
    int          size,
    OptimizerType opt_type,
    int          t
);