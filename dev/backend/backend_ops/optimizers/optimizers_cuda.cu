#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <cuda_runtime.h>
#include <cmath>

namespace py = pybind11;

// -------------------------
// 🧠 CuPy 배열에서 GPU 포인터 추출
// -------------------------
float* get_device_ptr(py::object cupy_array) {
    auto interface = cupy_array.attr("__cuda_array_interface__").cast<py::dict>();
    uintptr_t ptr = interface["data"].cast<std::pair<uintptr_t, bool>>().first;
    return reinterpret_cast<float*>(ptr);
}

// -------------------------
// 🚀 CUDA 커널 정의
// -------------------------
__global__ void sgd_update_kernel(float* w, const float* dw, float* b, const float* db, float lr, int w_size, int b_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < w_size) w[idx] -= lr * dw[idx];
    if (idx < b_size) b[idx] -= lr * db[idx];
}

__global__ void momentum_update_kernel(float* w, const float* dw, float* b, const float* db,
                                       float* vw, float* vb, float lr, float momentum,
                                       int w_size, int b_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < w_size) {
        vw[idx] = momentum * vw[idx] - lr * dw[idx];
        w[idx] += vw[idx];
    }
    if (idx < b_size) {
        vb[idx] = momentum * vb[idx] - lr * db[idx];
        b[idx] += vb[idx];
    }
}

// -------------------------
// 🧩 Pybind11 Wrapper
// -------------------------
void sgd_update(py::object w, py::object dw,
                py::object b, py::object db,
                float lr) {
    float* w_ptr = get_device_ptr(w);
    float* dw_ptr = get_device_ptr(dw);
    float* b_ptr = get_device_ptr(b);
    float* db_ptr = get_device_ptr(db);

    auto w_shape = w.attr("shape").cast<py::tuple>();
    auto b_shape = b.attr("shape").cast<py::tuple>();
    int w_size = 1, b_size = 1;
    for (auto s : w_shape) w_size *= s.cast<int>();
    for (auto s : b_shape) b_size *= s.cast<int>();

    int blockSize = 256;
    int gridSize = (std::max(w_size, b_size) + blockSize - 1) / blockSize;

    sgd_update_kernel<<<gridSize, blockSize>>>(w_ptr, dw_ptr, b_ptr, db_ptr, lr, w_size, b_size);
    cudaDeviceSynchronize();
}

void momentum_update(py::object w, py::object dw,
                     py::object b, py::object db,
                     py::object vw, py::object vb,
                     float lr, float momentum) {
    float* w_ptr = get_device_ptr(w);
    float* dw_ptr = get_device_ptr(dw);
    float* b_ptr = get_device_ptr(b);
    float* db_ptr = get_device_ptr(db);
    float* vw_ptr = get_device_ptr(vw);
    float* vb_ptr = get_device_ptr(vb);

    auto w_shape = w.attr("shape").cast<py::tuple>();
    auto b_shape = b.attr("shape").cast<py::tuple>();
    int w_size = 1, b_size = 1;
    for (auto s : w_shape) w_size *= s.cast<int>();
    for (auto s : b_shape) b_size *= s.cast<int>();

    int blockSize = 256;
    int gridSize = (std::max(w_size, b_size) + blockSize - 1) / blockSize;

    momentum_update_kernel<<<gridSize, blockSize>>>(w_ptr, dw_ptr, b_ptr, db_ptr,
                                                    vw_ptr, vb_ptr, lr, momentum,
                                                    w_size, b_size);
    cudaDeviceSynchronize();
}

// -------------------------
// 🔗 Pybind11 모듈 정의
// -------------------------
PYBIND11_MODULE(optimizers_cuda, m) {
    m.def("sgd_update", &sgd_update, "SGD 업데이트");
    m.def("momentum_update", &momentum_update, "Momentum SGD 업데이트");
}
