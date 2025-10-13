#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include "ai/op_schema.hpp"  // ActKind, GemmAttrs
#include "ai/tensor.hpp"     // Device, DType, Layout, TensorDesc, Tensor

PYBIND11_MODULE(_ops_common, m) {
    m.attr("__package__") = "graph_executor_v2.ops";
    m.doc() = R"(Common types:
- ActKind, GemmAttrs
- Device, DType, Layout, TensorDesc, Tensor, make_tensor_2d)";

    // --- ActKind, GemmAttrs (기존 그대로) ---
    py::enum_<ai::ActKind>(m, "ActKind", py::arithmetic(), py::module_local(false))
        .value("None",      ai::ActKind::None)
        .value("ReLU",      ai::ActKind::ReLU)
        .value("LeakyReLU", ai::ActKind::LeakyReLU)
        .value("GELU",      ai::ActKind::GELU)
        .value("Sigmoid",   ai::ActKind::Sigmoid)
        .value("Tanh",      ai::ActKind::Tanh)
        .export_values();

    py::class_<ai::GemmAttrs>(m, "GemmAttrs", py::module_local(false))
        .def(py::init<>())
        .def_readwrite("trans_a",     &ai::GemmAttrs::trans_a)
        .def_readwrite("trans_b",     &ai::GemmAttrs::trans_b)
        .def_readwrite("act",         &ai::GemmAttrs::act)
        .def_readwrite("with_bias",   &ai::GemmAttrs::with_bias)
        .def_readwrite("leaky_slope", &ai::GemmAttrs::leaky_slope)
        .def_readwrite("save_z",      &ai::GemmAttrs::save_z);

    // --- NEW: Tensor 계열 ---
    py::enum_<ai::Device>(m, "Device", py::arithmetic(), py::module_local(false))
        .value("CPU",  ai::Device::CPU)
        .value("CUDA", ai::Device::CUDA)
        .export_values();

    py::enum_<ai::DType>(m, "DType", py::arithmetic(), py::module_local(false))
        .value("F32",  ai::DType::F32).value("F16", ai::DType::F16)
        .value("BF16", ai::DType::BF16).value("I32", ai::DType::I32)
        .value("I8",   ai::DType::I8)
        .export_values();

    py::enum_<ai::Layout>(m, "Layout", py::arithmetic(), py::module_local(false))
        .value("RowMajor", ai::Layout::RowMajor)
        .value("ColMajor", ai::Layout::ColMajor)
        .export_values();

    py::class_<ai::TensorDesc>(m, "TensorDesc", py::module_local(false))
        .def(py::init<>())
        .def_readwrite("dtype",  &ai::TensorDesc::dtype)
        .def_readwrite("layout", &ai::TensorDesc::layout)
        .def_readwrite("shape",  &ai::TensorDesc::shape)
        .def_readwrite("stride", &ai::TensorDesc::stride)
        .def("dim", &ai::TensorDesc::dim);

    py::class_<ai::Tensor>(m, "Tensor", py::module_local(false))
        .def(py::init<>())
        .def_property("data_u64",
            [](const ai::Tensor& t){ return (uintptr_t)t.data; },
            [](ai::Tensor& t, uintptr_t p){ t.data = reinterpret_cast<void*>(p); })
        .def_readwrite("desc",         &ai::Tensor::desc)
        .def_readwrite("device",       &ai::Tensor::device)
        .def_readwrite("device_index", &ai::Tensor::device_index)
        .def("is_cuda", &ai::Tensor::is_cuda)
        .def("is_contiguous_rowmajor_2d", &ai::Tensor::is_contiguous_rowmajor_2d);

    m.def("make_tensor_2d",
        [](uintptr_t ptr_u64, std::vector<int64_t> shape,
           ai::DType dtype=ai::DType::F32, ai::Device dev=ai::Device::CUDA, int device_index=0) {
            if (shape.size()!=2) throw std::invalid_argument("shape must be 2D");
            ai::Tensor t;
            t.data = reinterpret_cast<void*>(ptr_u64);
            t.device = dev;
            t.device_index = device_index;
            t.desc.dtype = dtype;
            t.desc.layout = ai::Layout::RowMajor;
            t.desc.shape = shape;
            const int64_t ld = shape[1];
            t.desc.stride = { ld, 1 };
            return t;
        },
        py::arg("ptr_u64"), py::arg("shape"),
        py::arg("dtype")=ai::DType::F32,
        py::arg("device")=ai::Device::CUDA,
        py::arg("device_index")=0);

    m.attr("__all__") = py::make_tuple(
        "ActKind","GemmAttrs",
        "Device","DType","Layout","TensorDesc","Tensor","make_tensor_2d"
    );
}
