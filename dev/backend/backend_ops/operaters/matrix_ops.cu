#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <pybind11/cast.h>

namespace py = pybind11;

#define TILE_WIDTH 32

float* get_device_ptr(py::object cupy_array) {
    auto interface = cupy_array.attr("__cuda_array_interface__").cast<py::dict>();
    uintptr_t ptr = interface["data"].cast<py::tuple>()[0].cast<uintptr_t>();
    return reinterpret_cast<float*>(ptr);
}

__half* get_half_device_ptr(py::object cupy_array) {
    auto interface = cupy_array.attr("__cuda_array_interface__").cast<py::dict>();
    uintptr_t ptr = interface["data"].cast<py::tuple>()[0].cast<uintptr_t>();
    return reinterpret_cast<__half*>(ptr);
}

// -----------------------------
// Matrix Multiplication Kernel
// -----------------------------
__global__ void matrix_mul_kernel_optimized(const float* __restrict__ A,
                                            const float* __restrict__ B,
                                            float* __restrict__ C,
                                            int M, int N, int K) {
    __shared__ float tile_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tile_B_rowwise[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    float acc = 0.0f;

    for (int t = 0; t < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
        int tiled_col = t * TILE_WIDTH + threadIdx.x;
        int tiled_row = t * TILE_WIDTH + threadIdx.y;

        tile_A[threadIdx.y][threadIdx.x] = (row < M && tiled_col < K) ? A[row * K + tiled_col] : 0.0f;
        tile_B_rowwise[threadIdx.x][threadIdx.y] = (col < N && tiled_row < K) ? B[tiled_row * N + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k)
            acc += tile_A[threadIdx.y][k] * tile_B_rowwise[threadIdx.x][k];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = acc;
}

// -----------------------------
// Matrix Addition Kernel
// -----------------------------
__global__ void matrix_add_kernel(const float* A, const float* B, float* C, int size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] + B[idx];
    }
}

// -----------------------------
// Host Wrappers
// -----------------------------
void matrix_mul_optimized(py::object a, py::object b, py::object c, int rows, int cols, int K) {
    float* A = get_device_ptr(a);
    float* B = get_device_ptr(b);
    float* C = get_device_ptr(c);

    dim3 threads(TILE_WIDTH, TILE_WIDTH);
    dim3 blocks((cols + TILE_WIDTH - 1) / TILE_WIDTH,
                (rows + TILE_WIDTH - 1) / TILE_WIDTH);
    matrix_mul_kernel_optimized<<<blocks, threads>>>(A, B, C, rows, cols, K);
    cudaDeviceSynchronize();
}

void matrix_add(py::object a, py::object b, py::object c, int size) {
    float* A = get_device_ptr(a);
    float* B = get_device_ptr(b);
    float* C = get_device_ptr(c);

    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    matrix_add_kernel<<<blocks, threads>>>(A, B, C, size);
    cudaDeviceSynchronize();
}

// -----------------------------
// PYBIND11 Bindings
// -----------------------------
PYBIND11_MODULE(matrix_ops, m) {
    m.def("matrix_mul", &matrix_mul_optimized, "Optimized matrix multiplication (float32 shared)",
          py::arg("a"), py::arg("b"), py::arg("c"),
          py::arg("rows"), py::arg("cols"), py::arg("K"));

    m.def("matrix_add", &matrix_add, "Element-wise matrix addition (float32)",
          py::arg("a"), py::arg("b"), py::arg("c"),
          py::arg("size"));
}
