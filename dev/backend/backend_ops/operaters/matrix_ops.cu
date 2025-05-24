#include <cuda_runtime.h>
#include <cuda_fp16.h>  // ✅ float16 지원
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <pybind11/cast.h>

namespace py = pybind11;

#define TILE_WIDTH 32

// ✅ CuPy array → GPU 포인터 추출
float* get_device_ptr(py::object cupy_array) {
    auto interface = cupy_array.attr("__cuda_array_interface__").cast<py::dict>();
    uintptr_t ptr = interface["data"].cast<std::pair<uintptr_t, bool>>().first;
    return reinterpret_cast<float*>(ptr);
}

// ✅ __half 포인터 추출
__half* get_half_device_ptr(py::object cupy_array) {
    auto interface = cupy_array.attr("__cuda_array_interface__").cast<py::dict>();
    uintptr_t ptr = interface["data"].cast<std::pair<uintptr_t, bool>>().first;
    return reinterpret_cast<__half*>(ptr);
}

// ✅ 행렬 덧셈 (float32)
__global__ void matrix_add_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx < cols && idy < rows) {
        int index = idy * cols + idx;
        C[index] = A[index] + B[index];
    }
}

// ✅ 행렬 곱 (float32, shared memory)
__global__ void matrix_mul_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int M, int N, int K) {
    __shared__ float tile_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tile_B[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    float acc = 0.0f;

    for (int t = 0; t < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
        int tiled_col = t * TILE_WIDTH + threadIdx.x;
        int tiled_row = t * TILE_WIDTH + threadIdx.y;

        tile_A[threadIdx.y][threadIdx.x] = (row < M && tiled_col < K)
            ? A[row * K + tiled_col] : 0.0f;
        tile_B[threadIdx.y][threadIdx.x] = (tiled_row < K && col < N)
            ? B[tiled_row * N + col] : 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; ++i)
            acc += tile_A[threadIdx.y][i] * tile_B[i][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = acc;
}

// ✅ 행렬 곱 (float16, shared memory)
__global__ void matrix_mul_half_kernel(const __half* __restrict__ A,
                                       const __half* __restrict__ B,
                                       __half* __restrict__ C,
                                       int M, int N, int K) {
    __shared__ __half tile_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ __half tile_B[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    __half acc = __float2half(0.0f);

    for (int t = 0; t < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
        int tiled_col = t * TILE_WIDTH + threadIdx.x;
        int tiled_row = t * TILE_WIDTH + threadIdx.y;

        tile_A[threadIdx.y][threadIdx.x] = (row < M && tiled_col < K)
            ? A[row * K + tiled_col] : __float2half(0.0f);
        tile_B[threadIdx.y][threadIdx.x] = (tiled_row < K && col < N)
            ? B[tiled_row * N + col] : __float2half(0.0f);

        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; ++i)
            acc = __hadd(acc, __hmul(tile_A[threadIdx.y][i], tile_B[i][threadIdx.x]));

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = acc;
}

// ✅ 덧셈 (float32)
void matrix_add(py::object a, py::object b, py::object c, int rows, int cols) {
    float* A = get_device_ptr(a);
    float* B = get_device_ptr(b);
    float* C = get_device_ptr(c);

    dim3 threads(32, 32);
    dim3 blocks((cols + 31) / 32, (rows + 31) / 32);

    matrix_add_kernel<<<blocks, threads>>>(A, B, C, rows, cols);
    cudaDeviceSynchronize();
}

// ✅ 곱셈 (float32)
void matrix_mul(py::object a, py::object b, py::object c, int rows, int cols, int K) {
    float* A = get_device_ptr(a);
    float* B = get_device_ptr(b);
    float* C = get_device_ptr(c);

    dim3 threads(TILE_WIDTH, TILE_WIDTH);
    dim3 blocks((cols + TILE_WIDTH - 1) / TILE_WIDTH,
                (rows + TILE_WIDTH - 1) / TILE_WIDTH);

    matrix_mul_kernel<<<blocks, threads>>>(A, B, C, rows, cols, K);
    cudaDeviceSynchronize();
}

// ✅ 곱셈 (float16)
void matrix_mul_half(py::object a, py::object b, py::object c, int rows, int cols, int K) {
    __half* A = get_half_device_ptr(a);
    __half* B = get_half_device_ptr(b);
    __half* C = get_half_device_ptr(c);

    dim3 threads(TILE_WIDTH, TILE_WIDTH);
    dim3 blocks((cols + TILE_WIDTH - 1) / TILE_WIDTH,
                (rows + TILE_WIDTH - 1) / TILE_WIDTH);

    matrix_mul_half_kernel<<<blocks, threads>>>(A, B, C, rows, cols, K);
    cudaDeviceSynchronize();
}

// ✅ Pybind11 모듈 정의
PYBIND11_MODULE(matrix_ops, m) {
    m.def("matrix_add", &matrix_add, "Matrix addition (float32)",
          py::arg("a"), py::arg("b"), py::arg("c"),
          py::arg("rows"), py::arg("cols"));

    m.def("matrix_mul", &matrix_mul, "Matrix multiplication (float32 shared)",
          py::arg("a"), py::arg("b"), py::arg("c"),
          py::arg("rows"), py::arg("cols"), py::arg("K"));

    m.def("matrix_mul_half", &matrix_mul_half, "Matrix multiplication (float16 shared)",
          py::arg("a"), py::arg("b"), py::arg("c"),
          py::arg("rows"), py::arg("cols"), py::arg("K"));
}
