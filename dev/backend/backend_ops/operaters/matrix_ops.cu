#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <pybind11/cast.h>

namespace py = pybind11;

#define TILE_WIDTH 16

// ✅ 행렬 덧셈 커널
__global__ void matrix_add_kernel(float* A, float* B, float* C, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < cols && idy < rows) {
        int index = idy * cols + idx;
        C[index] = A[index] + B[index];
    }
}

// ✅ 기본 Global Memory 행렬 곱 커널
__global__ void matrix_mul_kernel(float* A, float* B, float* C, int rows, int cols, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        float sum = 0.0;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * cols + col];
        }
        C[row * cols + col] = sum;
    }
}

// ✅ Shared Memory Tiling 커널
__global__ void tiled_matrix_mul_kernel(float* A, float* B, float* C, int M, int N, int K) {
    __shared__ float tile_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tile_B[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    float acc = 0.0f;

    for (int t = 0; t < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
        int tiled_col = t * TILE_WIDTH + threadIdx.x;
        int tiled_row = t * TILE_WIDTH + threadIdx.y;

        tile_A[threadIdx.y][threadIdx.x] = (row < M && tiled_col < K) ? A[row * K + tiled_col] : 0.0f;
        tile_B[threadIdx.y][threadIdx.x] = (tiled_row < K && col < N) ? B[tiled_row * N + col] : 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; ++i)
            acc += tile_A[threadIdx.y][i] * tile_B[i][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = acc;
}

// ✅ CuPy array → GPU 포인터 가져오기
float* get_device_ptr(py::object cupy_array) {
    auto interface = cupy_array.attr("__cuda_array_interface__").cast<py::dict>();
    uintptr_t ptr = interface["data"].cast<std::pair<uintptr_t, bool>>().first;
    return reinterpret_cast<float*>(ptr);
}

// ✅ 행렬 덧셈 함수
void matrix_add(py::object a, py::object b, py::object c, int rows, int cols) {
    float* A = get_device_ptr(a);
    float* B = get_device_ptr(b);
    float* C = get_device_ptr(c);

    dim3 threads(16, 16);
    dim3 blocks((cols + 15) / 16, (rows + 15) / 16);

    matrix_add_kernel<<<blocks, threads>>>(A, B, C, rows, cols);
    cudaDeviceSynchronize();
}

// ✅ 기본 행렬 곱 호출
void matrix_mul(py::object a, py::object b, py::object c, int rows, int cols, int K) {
    float* A = get_device_ptr(a);
    float* B = get_device_ptr(b);
    float* C = get_device_ptr(c);

    dim3 threads(16, 16);
    dim3 blocks((cols + 15) / 16, (rows + 15) / 16);

    matrix_mul_kernel<<<blocks, threads>>>(A, B, C, rows, cols, K);
    cudaDeviceSynchronize();
}

// ✅ Shared Memory 버전 호출
void matrix_mul_shared(py::object a, py::object b, py::object c, int rows, int cols, int K) {
    float* A = get_device_ptr(a);
    float* B = get_device_ptr(b);
    float* C = get_device_ptr(c);

    dim3 threads(TILE_WIDTH, TILE_WIDTH);
    dim3 blocks((cols + TILE_WIDTH - 1) / TILE_WIDTH,
                (rows + TILE_WIDTH - 1) / TILE_WIDTH);

    tiled_matrix_mul_kernel<<<blocks, threads>>>(A, B, C, rows, cols, K);
    cudaDeviceSynchronize();
}

// ✅ Pybind11 모듈 정의
PYBIND11_MODULE(matrix_ops, m) {
    m.def("matrix_add", &matrix_add, "Matrix addition (global memory)",
          py::arg("a"), py::arg("b"), py::arg("c"),
          py::arg("rows"), py::arg("cols"));

    m.def("matrix_mul", &matrix_mul, "Matrix multiplication (global memory)",
          py::arg("a"), py::arg("b"), py::arg("c"),
          py::arg("rows"), py::arg("cols"), py::arg("K"));

    m.def("matrix_mul_shared", &matrix_mul_shared, "Matrix multiplication (shared memory tiling)",
          py::arg("a"), py::arg("b"), py::arg("c"),
          py::arg("rows"), py::arg("cols"), py::arg("K"));
}
