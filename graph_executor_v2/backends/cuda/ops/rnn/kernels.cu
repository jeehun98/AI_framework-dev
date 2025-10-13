#include <cuda_runtime.h>
#include <cstdint>
#include <float.h>

namespace { // ---- TU-local device/templated kernels ----

  
__device__ __forceinline__ float act_eval(int act, float z, float slope){
  enum { None=0, ReLU=1, LeakyReLU=2, Sigmoid=3, Tanh=4, GELU=5 };
  switch (act) {
    case None:      return z;
    case ReLU:      return z > 0.f ? z : 0.f;
    case LeakyReLU: return z > 0.f ? z : slope * z;
    case Sigmoid:   return 1.f / (1.f + __expf(-z));
    case Tanh:      return tanhf(z);
    case GELU: {
      const float c = sqrtf(2.f / 3.1415926535f);
      float z3 = z*z*z;
      float th = tanhf(c*(z + 0.044715f*z3));
      return 0.5f * z * (1.f + th);
    }
    default: return z;
  }
}

template<int BSX, int BSY>
__global__ void apply_act_rows_kernel(const float* __restrict__ Z_rows,
                                      float* __restrict__ H_rows,
                                      int M, int N, int act, float slope)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  int m = blockIdx.y * blockDim.y + threadIdx.y;
  if (m < M && n < N) {
    size_t idx = (size_t)m * N + n;
    H_rows[idx] = act_eval(act, Z_rows[idx], slope);
  }
}
// --- dact ---
__device__ __forceinline__ float dact(int act, float z, float gy, float slope) {
  // ai::ActKind 값을 정수로 전달 (header 의존 최소화)
  enum { None=0, ReLU=1, LeakyReLU=2, Sigmoid=3, Tanh=4, GELU=5 };
  switch (act) {
    case None:      return gy;
    case ReLU:      return (z > 0.f) ? gy : 0.f;
    case LeakyReLU: return (z > 0.f) ? gy : slope * gy;
    case Sigmoid: {
      float s = 1.f / (1.f + __expf(-z));
      return gy * s * (1.f - s);
    }
    case Tanh: {
      float t = tanhf(z);
      return gy * (1.f - t*t);
    }
    case GELU: {
      const float c = sqrtf(2.f / 3.1415926535f);
      float z3 = z*z*z;
      float th = tanhf(c*(z + 0.044715f*z3));
      float dtanh = (1 - th*th) * c * (1 + 0.134145f*z*z);
      return gy * (0.5f*(1 + th) + 0.5f*z*dtanh);
    }
    default: return gy;
  }
}

template<int BSX, int BSY>
__global__ void apply_dact_rows_kernel(const float* __restrict__ gy_post,
                                       const float* __restrict__ Z_rows,
                                       float* __restrict__ gy_rows,
                                       int M, int N, int act, float slope)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x; // col
  int m = blockIdx.y * blockDim.y + threadIdx.y; // row
  if (m < M && n < N) {
    size_t idx = (size_t)m * N + n;
    gy_rows[idx] = dact(act, Z_rows[idx], gy_post[idx], slope);
  }
}

template<int BSX, int BSY>
__global__ void add_rows_strided_kernel(float* __restrict__ A,        // [M,N]
                                        const float* __restrict__ B,  // [M,strideB]
                                        int M, int N, int strideB, int offsetB)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  int m = blockIdx.y * blockDim.y + threadIdx.y;
  if (m < M && n < N) {
    A[(size_t)m*N + n] += B[(size_t)m*strideB + offsetB + n];
  }
}

template<int BS>
__global__ void reduce_db_rows_kernel(const float* __restrict__ G, // [M,N]
                                      float* __restrict__ db,      // [N]
                                      int M, int N)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x; // over M*N
  int total = M * N;
  if (idx >= total) return;
  int n = idx % N;
  atomicAdd(&db[n], G[idx]);
}

template<int BS>
__global__ void kadd_vec_kernel(float* __restrict__ A, const float* __restrict__ B, int n){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) A[i] += B[i];
}

template<int BS>
__global__ void transpose_kernel(const float* __restrict__ A, float* __restrict__ AT,
                                 int M, int N)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = M * N;
  if (idx >= total) return;
  int n = idx % N;
  int m = idx / N;
  AT[(size_t)n * M + m] = A[(size_t)m * N + n];
}

__global__ void pack_wcat_from_wx_wh_kernel(const float* __restrict__ Wx, // [I,H]
                                            const float* __restrict__ Wh, // [H,H]
                                            float* __restrict__ Wcat,     // [I+H,H]
                                            int I, int H)
{
  int h = blockIdx.x * blockDim.x + threadIdx.x; // 0..H-1
  int r = blockIdx.y * blockDim.y + threadIdx.y; // 0..I+H-1
  if (h >= H) return;
  if (r < I) {
    Wcat[(size_t)r*H + h] = Wx[(size_t)r*H + h];
  } else if (r < I + H) {
    int rr = r - I;
    Wcat[(size_t)r*H + h] = Wh[(size_t)rr*H + h];
  }
}

} // anonymous


namespace ai { // ---- public launchers ----

void apply_act_rows_launcher(const float* Z_rows, float* H_rows,
                             int M, int N, int act_code, float slope, cudaStream_t s)
{
  constexpr int BSX = 128, BSY = 1;
  dim3 block(BSX, BSY);
  dim3 grid((N + BSX - 1)/BSX, (M + BSY - 1)/BSY);
  apply_act_rows_kernel<BSX,BSY><<<grid, block, 0, s>>>(Z_rows, H_rows, M, N, act_code, slope);
}


void apply_dact_rows_launcher(const float* gy_post, const float* Z_rows, float* gy_rows,
                              int M, int N, int act_code, float slope, cudaStream_t s)
{
  constexpr int BSX = 128, BSY = 1;
  dim3 block(BSX, BSY);
  dim3 grid((N + BSX - 1) / BSX, (M + BSY - 1) / BSY);
  apply_dact_rows_kernel<BSX,BSY><<<grid, block, 0, s>>>(gy_post, Z_rows, gy_rows, M, N, act_code, slope);
}

void add_rows_strided_launcher(float* A_MN, const float* B_Mstride,
                               int M, int N, int strideB, int offsetB, cudaStream_t s)
{
  constexpr int BSX = 128, BSY = 1;
  dim3 block(BSX, BSY);
  dim3 grid((N + BSX - 1) / BSX, (M + BSY - 1) / BSY);
  add_rows_strided_kernel<BSX,BSY><<<grid, block, 0, s>>>(A_MN, B_Mstride, M, N, strideB, offsetB);
}

void reduce_db_rows_kernel_launcher(const float* G_MN, float* db_N,
                                    int M, int N, cudaStream_t s)
{
  const int total = M * N;
  constexpr int BS = 256;
  dim3 block(BS), grid((total + BS - 1)/BS);
  reduce_db_rows_kernel<BS><<<grid, block, 0, s>>>(G_MN, db_N, M, N);
}

void kadd_vec_launcher(float* A, const float* B, int n, cudaStream_t s){
  constexpr int BS = 256;
  dim3 block(BS), grid((n + BS - 1)/BS);
  kadd_vec_kernel<BS><<<grid, block, 0, s>>>(A, B, n);
}

void transpose_kernel_launcher(const float* A, float* AT, int M, int N, cudaStream_t s)
{
  const int total = M * N;
  constexpr int BS = 256;
  dim3 block(BS), grid((total + BS - 1) / BS);
  transpose_kernel<BS><<<grid, block, 0, s>>>(A, AT, M, N);
}

void pack_wcat_from_wx_wh_launcher(const float* Wx, const float* Wh, float* Wcat,
                                   int I, int H, cudaStream_t s)
{
  constexpr int BSX = 128, BSY = 4;
  dim3 block(BSX, BSY);
  dim3 grid((H + BSX - 1)/BSX, (I + H + BSY - 1)/BSY);
  pack_wcat_from_wx_wh_kernel<<<grid, block, 0, s>>>(Wx, Wh, Wcat, I, H);
}

} // namespace ai
