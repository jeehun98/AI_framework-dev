// sm80+ 전용
#if __CUDA_ARCH__ >= 800
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

template<int BM, int BN, int BK> // block tile (e.g., 128x128x32)
__global__ void gemm_bias_act_fp16_tc(
    const half* __restrict__ A,  // [M,K]
    const half* __restrict__ B,  // [K,N]
    const half* __restrict__ bias, // [N] (Per-N)
          half* __restrict__ C,  // [M,N]
    int M, int N, int K, 
    float act_alpha /*e.g., GELU/Leaky slope*/
) {
  // --- 1) shared memory double buffer ---
  extern __shared__ char smem_raw[];
  half* smemA0 = (half*)smem_raw;
  half* smemB0 = smemA0 + BM*BK;
  half* smemA1 = smemB0 + BK*BN;
  half* smemB1 = smemA1 + BM*BK;

  // --- 2) block tile origin (row,col) ---
  const int block_row = blockIdx.y * BM;
  const int block_col = blockIdx.x * BN;

  // --- 3) warp tile 설정 (예: 4 warps → 각 64x64 등) ---
  const int warp_id  = (threadIdx.x >> 5);   // 0..(warps_per_block-1)
  const int lane_id  = (threadIdx.x & 31);   // 0..31

  // warp별 서브타일 오프셋 계산(프로젝트 고정 규칙 표 따르기)
  // int warp_m_off = ...; int warp_n_off = ...;

  // --- 4) FP32 accumulator fragment 초기화 ---
  wmma::fragment<wmma::accumulator, 16,16,16, float> acc[/*num_frag*/];
  #pragma unroll
  for (int i=0; i<sizeof(acc)/sizeof(acc[0]); ++i) wmma::fill_fragment(acc[i], 0.f);

  // --- 5) preload: K단 처음 타일을 cp.async로 올림 ---
  // 각 thread는 자기 gmem 주소/바이트 범위를 계산해 16B 단위로 push
  // (coalesced + 128B aligned 되도록 iterator 설계)
  // cp.async.ca.shared.global [smemA0 + t_off], [A + g_off], 16;
  // cp.async.ca.shared.global [smemB0 + t_off], [B + g_off], 16;
  asm volatile("cp.async.commit_group;");       // 첫 그룹 고정
  asm volatile("cp.async.wait_group 0;");       // preload 완료 보장
  __syncthreads(); // 공유메모리 가시성

  // --- 6) main K-loop: compute(N) + preload(N+1) ---
  int k_tiles = (K + BK - 1) / BK;
  for (int kt = 0; kt < k_tiles; ++kt) {

    // 6-1) 다음 타일 프리패치 (있다면) -> 반대 버퍼로
    if (kt + 1 < k_tiles) {
      // cp.async ... -> (kt&1 ? smemA0/B0 : smemA1/B1) 로 프리패치
      asm volatile("cp.async.commit_group;");
    }

    // 6-2) 현재 타일로부터 warp fragment 로드 (ldmatrix)
    // ldmatrix.x4.trans/shared → a_frag , ldmatrix.x4.shared → b_frag
    wmma::fragment<wmma::matrix_a, 16,16,16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16,16,16, half, wmma::col_major> b_frag;

    // 타일 내 서브패널 반복 (warp tile을 16x16 블록들의 모자이크로 덮음)
    #pragma unroll
    for (int ks = 0; ks < BK; ks += 16) {
      // (공유메모리 주소 → a_frag, b_frag로)
      // wmma::load_matrix_sync(a_frag, smemA_cur + ..., strideA);
      // wmma::load_matrix_sync(b_frag, smemB_cur + ..., strideB);

      // 6-3) MMA: acc += a_frag * b_frag
      // acc[p] = a∗b + acc[p] (FP32 accumulate)
      // wmma::mma_sync(acc[p], a_frag, b_frag, acc[p]);
    }

    // 6-4) 다음 타일 준비 보장 및 경계 동기화
    if (kt + 1 < k_tiles) {
      asm volatile("cp.async.wait_group 0;");
    }
    __syncthreads();  // 다음 iteration에서 공유버퍼 role swap 안전화
  }

  // --- 7) Epilogue: bias + activation (+optional dropout/save-Z) ---
  // acc → 적용 → FP16 cast → gmem C 저장
  // wmma::store_matrix_sync(c_tile, acc[p], ldc, wmma::mem_row_major);
  // Per-N bias: bias[block_col + n_off] 더하고 act 적용

  // 경계 체크(M, N 넘어가는 쓰기 skip), 벡터화 store(float4/half2)
}
#endif
