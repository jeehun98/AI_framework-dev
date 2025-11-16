// src/bindings/ops_common_pybind.cpp
//
// Common shim-based types for CUDA ops:
//   - ActKind, GemmAttrs
//   - Device, DType, Layout
//   - TensorDesc, Tensor, make_tensor_2d
//
// 이 모듈에서 노출하는 타입들은 모두
// backends/cuda/ops/_common/shim/* 에 정의된 ai::cuda::shim 기준입니다.
// CUDA ops 바인딩(_ops_gemm, _ops_conv2d, ...)과 동일한 타입을 사용하게 해서
// Python → shim Tensor → 각 ops 바인딩 으로 그대로 전달됩니다.

#include <cstdint>
#include <vector>
#include <stdexcept>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "backends/cuda/ops/_common/shim/ai_shim.hpp"

namespace py   = pybind11;
namespace shim = ::ai::cuda::shim;

PYBIND11_MODULE(_ops_common, m) {
    m.attr("__package__") = "graph_executor_v2.ops";
    m.doc() = R"(Common shim-based types for CUDA ops:
- ActKind, GemmAttrs
- Device, DType, Layout
- TensorDesc, Tensor, make_tensor_2d)";

    // ------------------------------------------------------------------
    // ActKind, GemmAttrs (shim:: 기준)
    // ------------------------------------------------------------------
    py::enum_<shim::ActKind>(m, "ActKind", py::arithmetic(), py::module_local(false))
        .value("None",      shim::ActKind::None)
        .value("ReLU",      shim::ActKind::ReLU)
        .value("LeakyReLU", shim::ActKind::LeakyReLU)
        .value("GELU",      shim::ActKind::GELU)
        .value("Sigmoid",   shim::ActKind::Sigmoid)
        .value("Tanh",      shim::ActKind::Tanh)
        .export_values();

    py::class_<shim::GemmAttrs>(m, "GemmAttrs", py::module_local(false))
        .def(py::init<>())
        .def_readwrite("trans_a",     &shim::GemmAttrs::trans_a)
        .def_readwrite("trans_b",     &shim::GemmAttrs::trans_b)
        .def_readwrite("act",         &shim::GemmAttrs::act)
        .def_readwrite("with_bias",   &shim::GemmAttrs::with_bias)
        .def_readwrite("leaky_slope", &shim::GemmAttrs::leaky_slope)
        .def_readwrite("save_z",      &shim::GemmAttrs::save_z);

    // ------------------------------------------------------------------
    // Device / DType / Layout (shim enums)
    // ------------------------------------------------------------------
    py::enum_<shim::Device>(m, "Device", py::arithmetic(), py::module_local(false))
        .value("CPU",  shim::Device::CPU)
        .value("CUDA", shim::Device::CUDA)
        .export_values();

    py::enum_<shim::DType>(m, "DType", py::arithmetic(), py::module_local(false))
        .value("F32",  shim::DType::F32)
        .value("F16",  shim::DType::F16)
        .value("BF16", shim::DType::BF16)
        .value("I32",  shim::DType::I32)
        .value("I8",   shim::DType::I8)
        .export_values();

    py::enum_<shim::Layout>(m, "Layout", py::arithmetic(), py::module_local(false))
        .value("RowMajor", shim::Layout::RowMajor)
        .value("ColMajor", shim::Layout::ColMajor)
        .export_values();

    // ------------------------------------------------------------------
    // TensorDesc / Tensor (shim::ai_tensor.hpp)
    // ------------------------------------------------------------------
    py::class_<shim::TensorDesc>(m, "TensorDesc", py::module_local(false))
        .def(py::init<>())
        .def_readwrite("dtype",  &shim::TensorDesc::dtype)
        .def_readwrite("layout", &shim::TensorDesc::layout)
        .def_readwrite("shape",  &shim::TensorDesc::shape)
        .def_readwrite("stride", &shim::TensorDesc::stride)
        .def("dim", &shim::TensorDesc::dim);

    py::class_<shim::Tensor>(m, "Tensor", py::module_local(false))
        .def(py::init<>())
        // raw pointer를 u64로 노출 (Python 쪽에서 CuPy ptr와 연결)
        .def_property(
            "data_u64",
            [](const shim::Tensor& t) {
                return static_cast<std::uintptr_t>(reinterpret_cast<std::uintptr_t>(t.data));
            },
            [](shim::Tensor& t, std::uintptr_t p) {
                t.data = reinterpret_cast<void*>(p);
            }
        )
        .def_readwrite("desc",         &shim::Tensor::desc)
        .def_readwrite("device",       &shim::Tensor::device)
        .def_readwrite("device_index", &shim::Tensor::device_index)
        .def("is_cuda", &shim::Tensor::is_cuda)
        .def("is_contiguous_rowmajor_2d", &shim::Tensor::is_contiguous_rowmajor_2d);

    // ------------------------------------------------------------------
    // make_tensor_2d : ptr_u64 + shape → shim::Tensor (2D, RowMajor)
    // ------------------------------------------------------------------
    m.def(
        "make_tensor_2d",
        [](std::uintptr_t ptr_u64,
           const std::vector<std::int64_t>& shape,
           shim::DType dtype = shim::DType::F32,
           shim::Device device = shim::Device::CUDA,
           int device_index = 0) {
            if (shape.size() != 2) {
                throw std::invalid_argument("make_tensor_2d: shape must be 2D");
            }
            const std::int64_t rows = shape[0];
            const std::int64_t cols = shape[1];

            // ai_tensor.hpp 의 shim::make_tensor2d 헬퍼 사용
            shim::Tensor t = shim::make_tensor2d(
                reinterpret_cast<void*>(ptr_u64),
                rows,
                cols,
                dtype
            );
            t.device       = device;
            t.device_index = device_index;
            return t;
        },
        py::arg("ptr_u64"),
        py::arg("shape"),
        py::arg("dtype")        = shim::DType::F32,
        py::arg("device")       = shim::Device::CUDA,
        py::arg("device_index") = 0
    );

    // ------------------------------------------------------------------
    // __all__
    // ------------------------------------------------------------------
    m.attr("__all__") = py::make_tuple(
        "ActKind", "GemmAttrs",
        "Device", "DType", "Layout",
        "TensorDesc", "Tensor", "make_tensor_2d"
    );
}
