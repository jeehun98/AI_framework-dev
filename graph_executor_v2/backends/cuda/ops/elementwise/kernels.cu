#include <cuda_runtime.h>
#include <math_constants.h>
#include <stdint.h>

namespace {

// 최대 차원 수(프로젝트에서 필요 시 늘리기)
constexpr int kMaxND = 8;

// Y의 선형 인덱스를 다차원 인덱스로 풀어 입력 텐서 오프셋을 계산하는 공용 유틸
struct Indexer {
  int nd;
  // 각 텐서 별 stride: 브로드캐스트 차원은 stride=0로 세팅
  int64_t y_shape[kMaxND];
  int64_t a_stride[kMaxND];
  int64_t b_stride[kMaxND]; // unary일 때는 사용하지 않음

  __device__ __forceinline__
int64_t offsetA(int64_t linear) const {
  int64_t off = 0;
  // RowMajor: 마지막 축이 fastest → 뒤에서 앞으로
  for (int i = nd - 1; i >= 0; --i) {
    const int64_t dim = y_shape[i];
    const int64_t idx = linear % dim;
    linear /= dim;
    off += idx * a_stride[i];   // 브로드캐스트 축이면 stride=0
  }
  return off;
}

__device__ __forceinline__
int64_t offsetB(int64_t linear) const {
  int64_t off = 0;
  for (int i = nd - 1; i >= 0; --i) {
    const int64_t dim = y_shape[i];
    const int64_t idx = linear % dim;
    linear /= dim;
    off += idx * b_stride[i];
  }
  return off;
}
};

// ---------- Unary ops ----------
__device__ __forceinline__ float gelu_tanh(float x){
  // Hendrycks & Gimpel approximate GELU
  const float k0 = 0.7978845608f; // sqrt(2/pi)
  const float k1 = 0.044715f;
  float x3 = x * x * x;
  float t = k0 * (x + k1 * x3);
  return 0.5f * x * (1.f + tanhf(t));
}

__global__ void ewise_unary_kernel(const float* __restrict__ X,
                                   float* __restrict__ Y,
                                   Indexer indexer,
                                   int64_t nElem,
                                   int op, float alpha, float cmin, float cmax, float eps)
{
  for (int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
       tid < nElem; tid += blockDim.x * gridDim.x)
  {
    const int64_t xo = indexer.offsetA(tid);
    float x = X[xo];
    float y;
    switch(op){
      case 0: y = x; break;                                // Identity
      case 1: y = fmaxf(x, 0.f); break;                    // ReLU
      case 2: y = (x >= 0.f) ? x : alpha * x; break;       // LeakyReLU
      case 3: y = 1.f / (1.f + __expf(-x)); break;         // Sigmoid
      case 4: y = tanhf(x); break;                         // Tanh
      case 5: y = gelu_tanh(x); break;                     // GELU
      case 6: y = __expf(x); break;                        // Exp
      case 7: y = __logf(fmaxf(fabsf(x), eps)); break;     // Log |x|, eps 보호
      case 8: y = sqrtf(fmaxf(x, 0.f)); break;             // Sqrt
      case 9: y = rsqrtf(fmaxf(x, eps)); break;            // Rsqrt
      case 10: y = fminf(fmaxf(x, cmin), cmax); break;     // Clip
      default: y = x; break;
    }
    Y[tid] = y; // Y는 항상 연속 가정(RowMajor contiguous)
  }
}

// ---------- Binary ops ----------
__device__ __forceinline__ float binary_apply(float a, float b, int op, float eps){
  switch(op){
    case 0: return a + b;                          // Add
    case 1: return a - b;                          // Sub
    case 2: return a * b;                          // Mul
    case 3: return a / fmaxf(b, eps);              // Div
    case 4: return fmaxf(a, b);                    // Max
    case 5: return fminf(a, b);                    // Min
    case 6: return __powf(a, b);                   // Pow
  }
  return a + b;
}

__global__ void ewise_binary_kernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ Y,
                                    Indexer indexer,
                                    int64_t nElem,
                                    int op, float alpha, float beta, float eps)
{
  for (int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
       tid < nElem; tid += blockDim.x * gridDim.x)
  {
    const int64_t ao = indexer.offsetA(tid);
    const int64_t bo = indexer.offsetB(tid);
    float a = alpha * A[ao];
    float b = beta  * B[bo];
    Y[tid] = binary_apply(a,b,op,eps);
  }
}

} // anonymous

namespace ai {

// 런처에서 사용할 시그니처(launcher.cu에서 extern으로 선언/정의 매칭!)
void ewise_unary_kernel_launcher(const float* X, float* Y,
                                 const int64_t* y_shape,
                                 const int64_t* a_stride,
                                 int nd, int64_t nElem,
                                 int op, float alpha, float cmin, float cmax, float eps,
                                 cudaStream_t s)
{
  Indexer idx{};
  idx.nd = nd;
  for (int i=0;i<nd;i++){
    idx.y_shape[i] = y_shape[i];
    idx.a_stride[i] = a_stride[i];
    idx.b_stride[i] = 0;
  }
  constexpr int BS=256;
  int64_t grid = (nElem + BS - 1) / BS;
  grid = (grid>65535)?65535:grid;
  ewise_unary_kernel<<<(int)grid, BS, 0, s>>>(X, Y, idx, nElem, op, alpha, cmin, cmax, eps);
}

void ewise_binary_kernel_launcher(const float* A, const float* B, float* Y,
                                  const int64_t* y_shape,
                                  const int64_t* a_stride,
                                  const int64_t* b_stride,
                                  int nd, int64_t nElem,
                                  int op, float alpha, float beta, float eps,
                                  cudaStream_t s)
{
  Indexer idx{};
  idx.nd = nd;
  for (int i=0;i<nd;i++){
    idx.y_shape[i] = y_shape[i];
    idx.a_stride[i] = a_stride[i];
    idx.b_stride[i] = b_stride[i];
  }
  constexpr int BS=256;
  int64_t grid = (nElem + BS - 1) / BS;
  grid = (grid>65535)?65535:grid;
  ewise_binary_kernel<<<(int)grid, BS, 0, s>>>(A, B, Y, idx, nElem, op, alpha, beta, eps);
}

} // namespace ai
