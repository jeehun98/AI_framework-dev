// include/regemm/config.h
#pragma once

// --- 타일 크기 ---
#ifndef REGEMM_TILE_M
#define REGEMM_TILE_M 128
#endif
#ifndef REGEMM_TILE_N
#define REGEMM_TILE_N 128
#endif
#ifndef REGEMM_TILE_K
#define REGEMM_TILE_K 16   // BK=32는 성능↑/레지스터↑ 트레이드오프
#endif

// --- 블록 스레드 배치 (blockDim.x * blockDim.y = 256) ---
#ifndef REGEMM_BLOCK_TDX
#define REGEMM_BLOCK_TDX 16
#endif
#ifndef REGEMM_BLOCK_TDY
#define REGEMM_BLOCK_TDY 16
#endif

// --- 마이크로 타일 (thread당 출력 크기) ---
// BM = TDY * THR_M, BN = TDX * THR_N 이어야 함
#ifndef REGEMM_THREAD_TILE_M
#define REGEMM_THREAD_TILE_M 8
#endif
#ifndef REGEMM_THREAD_TILE_N
#define REGEMM_THREAD_TILE_N 8
#endif

// --- 최적화 옵션 ---
#ifndef REGEMM_USE_VECIO
#define REGEMM_USE_VECIO 1      // Global IO float4 벡터화
#endif
#ifndef REGEMM_SMEM_PADK
#define REGEMM_SMEM_PADK 1      // SMEM K차원 padding(BK+1)로 은행충돌 완화
#endif
#ifndef REGEMM_USE_DB
#define REGEMM_USE_DB 1         // K-loop double-buffering (stage 2)
#endif

// --- 정렬 제약(벡터화 안전 조건) ---
#ifndef REGEMM_VEC_ALIGN_ELEMS
#define REGEMM_VEC_ALIGN_ELEMS 4   // float4 = 16B
#endif
