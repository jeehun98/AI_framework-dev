
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>
#include <cmath>

namespace py = pybind11;

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

void launch_sgd(py::array_t<float> w, py::array_t<float> dw,
                py::array_t<float> b, py::array_t<float> db,
                float lr) {
    auto w_buf = w.request(), dw_buf = dw.request();
    auto b_buf = b.request(), db_buf = db.request();

    float* w_ptr = static_cast<float*>(w_buf.ptr);
    float* dw_ptr = static_cast<float*>(dw_buf.ptr);
    float* b_ptr = static_cast<float*>(b_buf.ptr);
    float* db_ptr = static_cast<float*>(db_buf.ptr);

    int w_size = w_buf.size, b_size = b_buf.size;
    int blockSize = 256, gridSize = (std::max(w_size, b_size) + blockSize - 1) / blockSize;

    sgd_update_kernel<<<gridSize, blockSize>>>(w_ptr, dw_ptr, b_ptr, db_ptr, lr, w_size, b_size);
}

void launch_momentum(py::array_t<float> w, py::array_t<float> dw,
                     py::array_t<float> b, py::array_t<float> db,
                     py::array_t<float> vw, py::array_t<float> vb,
                     float lr, float momentum) {
    auto w_buf = w.request(), dw_buf = dw.request();
    auto b_buf = b.request(), db_buf = db.request();
    auto vw_buf = vw.request(), vb_buf = vb.request();

    float* w_ptr = static_cast<float*>(w_buf.ptr);
    float* dw_ptr = static_cast<float*>(dw_buf.ptr);
    float* b_ptr = static_cast<float*>(b_buf.ptr);
    float* db_ptr = static_cast<float*>(db_buf.ptr);
    float* vw_ptr = static_cast<float*>(vw_buf.ptr);
    float* vb_ptr = static_cast<float*>(vb_buf.ptr);

    int w_size = w_buf.size, b_size = b_buf.size;
    int blockSize = 256, gridSize = (std::max(w_size, b_size) + blockSize - 1) / blockSize;

    momentum_update_kernel<<<gridSize, blockSize>>>(w_ptr, dw_ptr, b_ptr, db_ptr,
                                                    vw_ptr, vb_ptr, lr, momentum,
                                                    w_size, b_size);
}

PYBIND11_MODULE(optimizers_cuda, m) {
    m.def("sgd_update", &launch_sgd, "SGD 업데이트");
    m.def("momentum_update", &launch_momentum, "Momentum SGD 업데이트");
}
