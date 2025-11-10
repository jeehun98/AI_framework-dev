#include <cuda_runtime.h>
#include <cstdint>
#include <math.h>

namespace ai {

// ===== RNN-local activation & derivative =====
__device__ __forceinline__ float act_eval(int act, float z, float slope){
  enum { None=0, ReLU=1, LeakyReLU=2, GELU=3, Sigmoid=4, Tanh=5 };
  switch (act) {
    case 0:  return z;
    case 1:  return z > 0.f ? z : 0.f;
    case 2:  return z > 0.f ? z : slope * z;
    case 4:  return 1.f / (1.f + __expf(-z));
    case 5:  return tanhf(z);
    case 3: {
      const float c = sqrtf(2.f / 3.1415926535f);
      float z3 = z*z*z;
      float th = tanhf(c*(z + 0.044715f*z3));
      return 0.5f * z * (1.f + th);
    }
    default: return z;
  }
}

__device__ __forceinline__ float dact_eval(int act, float z, float gy, float slope){
  enum { None=0, ReLU=1, LeakyReLU=2, GELU=3, Sigmoid=4, Tanh=5 };
  switch (act) {
    case 0:  return gy;
    case 1:  return (z > 0.f) ? gy : 0.f;
    case 2:  return (z > 0.f) ? gy : slope * gy;
    case 4: {
      float s = 1.f / (1.f + __expf(-z));
      return gy * s * (1.f - s);
    }
    case 5: {
      float t = tanhf(z);
      return gy * (1.f - t*t);
    }
    case 3: {
      const float c = sqrtf(2.f / 3.1415926535f);
      float z2 = z*z;
      float u  = c * (z + 0.044715f * z2 * z);
      float th = tanhf(u);
      float sech2 = 1.f - th*th;
      float du = c * (1.f + 0.134145f * z2);
      float dgelu = 0.5f*(1.f + th) + 0.5f*z*sech2*du;
      return gy * dgelu;
    }
    default: return gy;
  }
}

// [N,H] : Y = act(Z)
__global__ void k_apply_act_rows_local(const float* __restrict__ Z,
                                       float* __restrict__ Y,
                                       int N, int H,
                                       int act, float slope)
{
  int h = blockIdx.x * blockDim.x + threadIdx.x;
  int n = blockIdx.y * blockDim.y + threadIdx.y;
  if (n < N && h < H) {
    size_t idx = (size_t)n * (size_t)H + (size_t)h;
    Y[idx] = act_eval(act, Z[idx], slope);
  }
}

static inline void apply_act_rows_local_launcher(const float* Z, float* Y,
                                                 int N, int H, int act, float slope,
                                                 cudaStream_t s)
{
  dim3 block(128, 1), grid((H + block.x - 1)/block.x, N);
  k_apply_act_rows_local<<<grid, block, 0, s>>>(Z, Y, N, H, act, slope);
}

// [N,H] : gZ = dact(Z, gY_post)
__global__ void k_apply_dact_rows_local(const float* __restrict__ gy_post,
                                        const float* __restrict__ Z,
                                        float* __restrict__ gy,
                                        int N, int H,
                                        int act, float slope)
{
  int h = blockIdx.x * blockDim.x + threadIdx.x;
  int n = blockIdx.y * blockDim.y + threadIdx.y;
  if (n < N && h < H) {
    size_t idx = (size_t)n * (size_t)H + (size_t)h;
    gy[idx] = dact_eval(act, Z[idx], gy_post[idx], slope);
  }
}

static inline void apply_dact_rows_local_launcher(const float* gy_post, const float* Z, float* gy,
                                                  int N, int H, int act, float slope,
                                                  cudaStream_t s)
{
  dim3 block(128, 1), grid((H + block.x - 1)/block.x, N);
  k_apply_dact_rows_local<<<grid, block, 0, s>>>(gy_post, Z, gy, N, H, act, slope);
}

// === 공용 시그니처 alias (api.hpp의 심볼명과 맞춤) ===
void apply_act_rows_launcher(const float* Z_rows, float* H_rows,
                             int M, int N, int act_code, float slope, cudaStream_t s){
  apply_act_rows_local_launcher(Z_rows, H_rows, M, N, act_code, slope, s);
}
void apply_dact_rows_launcher(const float* gy_post, const float* Z_rows, float* gy_rows,
                              int M, int N, int act_code, float slope, cudaStream_t s){
  apply_dact_rows_local_launcher(gy_post, Z_rows, gy_rows, M, N, act_code, slope, s);
}

// G_rows += dXH[:, I:]
__global__ void k_add_dhnext_into_grows(const float* __restrict__ dXH,
                                        float* __restrict__ dG,
                                        int N, int I, int H)
{
  int h = blockIdx.x * blockDim.x + threadIdx.x;
  int n = blockIdx.y * blockDim.y + threadIdx.y;
  if (n < N && h < H) {
    const size_t ldXH = (size_t)(I + H);
    const size_t off  = (size_t)n * ldXH + (size_t)I + (size_t)h;
    const size_t idxG = (size_t)n * (size_t)H + (size_t)h;
    dG[idxG] += dXH[off];
  }
}

void add_dhnext_into_grows_launcher(const float* dXH, float* dG,
                                    int N, int I, int H, cudaStream_t s)
{
  dim3 block(128, 1), grid((H + block.x - 1)/block.x, N);
  k_add_dhnext_into_grows<<<grid, block, 0, s>>>(dXH, dG, N, I, H);
}

// row-major transpose: in[M,N] -> out[N,M]
__global__ void k_transpose_MN(const float* __restrict__ A, float* __restrict__ AT,
                               int M, int N)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (y < M && x < N) {
    AT[(size_t)x * (size_t)M + (size_t)y] = A[(size_t)y * (size_t)N + (size_t)x];
  }
}
void transpose_kernel_launcher(const float* A, float* AT, int M, int N, cudaStream_t s){
  dim3 block(32, 8);
  dim3 grid((N + block.x - 1)/block.x, (M + block.y - 1)/block.y);
  k_transpose_MN<<<grid, block, 0, s>>>(A, AT, M, N);
}

// dWcat[I+H,H] += (Tmp[H,I+H])^T
__global__ void k_add_transpose_into(float* __restrict__ dWcat,
                                     const float* __restrict__ Tmp_H_IpH,
                                     int IpH, int H)
{
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < IpH && j < H) {
    dWcat[(size_t)i*(size_t)H + (size_t)j] += Tmp_H_IpH[(size_t)j*(size_t)IpH + (size_t)i];
  }
}
void add_transpose_into_launcher(float* dWcat, const float* Tmp_H_IpH,
                                 int IpH, int H, cudaStream_t s){
  dim3 block(32, 8);
  dim3 grid((H + block.x - 1)/block.x, (IpH + block.y - 1)/block.y);
  k_add_transpose_into<<<grid, block, 0, s>>>(dWcat, Tmp_H_IpH, IpH, H);
}

// db[h] += sum_{n=0..N-1} G[n,h]
__global__ void k_reduce_db_rows_NH(const float* __restrict__ G, float* __restrict__ db,
                                    int N, int H)
{
  int h = blockIdx.x * blockDim.x + threadIdx.x;
  if (h >= H) return;
  float acc = 0.f;
  for (int n = 0; n < N; ++n) acc += G[(size_t)n * (size_t)H + (size_t)h];
  atomicAdd(&db[h], acc);
}
void reduce_db_rows_NH_launcher(const float* G, float* db, int N, int H, cudaStream_t s){
  constexpr int BS = 256;
  dim3 block(BS), grid((H + BS - 1)/BS);
  k_reduce_db_rows_NH<<<grid, block, 0, s>>>(G, db, N, H);
}

// === 공용 이름으로도 노출(기존 시그니처와 매핑) ===
void reduce_db_rows_kernel_launcher(const float* G_MN, float* db_N,
                                    int M, int N, cudaStream_t s){
  // 여기서는 M=N(행방향 집계) 가정과 동일 — 기존 호출부 호환
  reduce_db_rows_NH_launcher(G_MN, db_N, M, N, s);
}

} // namespace ai
