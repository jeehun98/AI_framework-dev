// src/bindings/memory_pybind.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>

#ifdef BUILD_STANDALONE_OPS
  #include "backends/cuda/ops/_common/shim/ai_shim.hpp"
#else
  #include "ai/tensor.hpp"
  #include "ai/dispatch.hpp"
#endif

#include "backends/cuda/ops/memory/api.hpp"

namespace py = pybind11;
using namespace ai;

static Tensor make_tensor_nd(uintptr_t ptr_u64,
                             const std::vector<int64_t>& shape,
                             DType dtype,
                             Device dev=Device::CUDA)
{
  Tensor t;
  t.data = reinterpret_cast<void*>(ptr_u64);
  t.device = dev;
  t.device_index = 0;
  t.desc.dtype  = dtype;
  t.desc.layout = Layout::RowMajor;
  t.desc.shape  = shape;
  const size_t R=shape.size();
  t.desc.stride.resize(R);
  if (R){
    t.desc.stride[R-1]=1;
    for (int i=(int)R-2;i>=0;--i)
      t.desc.stride[(size_t)i]=t.desc.stride[(size_t)i+1]*shape[(size_t)i+1];
  }
  return t;
}

static void throw_if_bad(Status st, const char* where){
  if (st!=Status::Ok)
    throw std::runtime_error(std::string(where)+" failed with Status="+std::to_string((int)st));
}

PYBIND11_MODULE(_ops_memory, m){
  m.doc() = "Memory utility ops (scalar fill) — CUDA Graph capture safe";

  m.def("fill_f32",
    [](uintptr_t dst_ptr, const std::vector<int64_t>& shape,
       float value, uintptr_t stream_ptr)
    {
      Tensor dst = make_tensor_nd(dst_ptr, shape, DType::F32);
      auto st = FillScalarF32CudaLaunch(dst, value, reinterpret_cast<StreamHandle>(stream_ptr));
      throw_if_bad(st, "FillScalarF32CudaLaunch");
    },
    py::arg("dst_ptr"), py::arg("shape"), py::arg("value"),
    py::arg("stream")=static_cast<uintptr_t>(0)
  );

  m.def("fill_i32",
    [](uintptr_t dst_ptr, const std::vector<int64_t>& shape,
       int32_t value, uintptr_t stream_ptr)
    {
      Tensor dst = make_tensor_nd(dst_ptr, shape, DType::I32);
      auto st = FillScalarI32CudaLaunch(dst, value, reinterpret_cast<StreamHandle>(stream_ptr));
      throw_if_bad(st, "FillScalarI32CudaLaunch");
    },
    py::arg("dst_ptr"), py::arg("shape"), py::arg("value"),
    py::arg("stream")=static_cast<uintptr_t>(0)
  );
}
