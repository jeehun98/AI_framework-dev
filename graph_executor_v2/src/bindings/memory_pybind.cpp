// src/bindings/memory_pybind.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace pybind11::literals;  // "_a" 리터럴 활성화


#ifdef BUILD_STANDALONE_OPS
  #include "backends/cuda/ops/_common/shim/ai_shim.hpp"
#else
  #include "ai/tensor.hpp"
  #include "ai/dispatch.hpp"
#endif

#include "backends/cuda/ops/memory/api.hpp"  // FillF32Cuda/FillI32Cuda, Memory*Cuda API

namespace py = pybind11;
using namespace ai;

// 재사용 유틸: RowMajor 텐서 생성 (dropout_pybind.cpp와 동일 톤)
static Tensor make_tensor_rm(uintptr_t ptr_u64,
                             const std::vector<int64_t>& shape,
                             DType dtype,
                             Device dev = Device::CUDA) {
  Tensor t;
  t.data = reinterpret_cast<void*>(ptr_u64);
  t.device = dev; t.device_index = 0;
  t.desc.dtype = dtype;
  t.desc.layout = Layout::RowMajor;
  t.desc.shape = shape;
  t.desc.stride.resize(shape.size());
  if (!shape.empty()) {
    t.desc.stride.back() = 1;
    for (int i = (int)shape.size() - 2; i >= 0; --i) {
      t.desc.stride[i] = shape[i + 1] * t.desc.stride[i + 1];
    }
  }
  return t;
}

static void throw_if_bad(Status st, const char* where) {
  if (st != Status::Ok) {
    throw std::runtime_error(std::string("[_ops_memory::") + where +
                             "] failed with Status=" +
                             std::to_string(static_cast<int>(st)));
  }
}

PYBIND11_MODULE(_ops_memory, m) {
  m.attr("__package__") = "graph_executor_v2.ops";
  m.doc() = "Capture-safe memory ops (CUDA backend): fills + arena adapter";

  // (선택) 공용 모듈 로드 — 드롭아웃 바인딩과 톤을 맞추기 위해 유지
  py::module_ common = py::module_::import("graph_executor_v2.ops._ops_common");

  // -------- Fill APIs --------
  // fill_f32(dst_ptr, dst_shape, value, stream=0)
  m.def("fill_f32",
    [](uintptr_t dst_ptr, const std::vector<int64_t>& dst_shape,
       float value, uintptr_t stream_ptr) {
      Tensor D = make_tensor_rm(dst_ptr, dst_shape, DType::F32);
      StreamHandle stream = reinterpret_cast<StreamHandle>(stream_ptr);
      auto st = FillF32Cuda(D, value, stream);
      throw_if_bad(st, "fill_f32");
    },
    py::arg("dst_ptr"), py::arg("dst_shape"),
    py::arg("value"),
    py::arg("stream") = (uintptr_t)0
  );

  // fill_i32(dst_ptr, dst_shape, value, stream=0)
  m.def("fill_i32",
    [](uintptr_t dst_ptr, const std::vector<int64_t>& dst_shape,
       int32_t value, uintptr_t stream_ptr) {
      Tensor D = make_tensor_rm(dst_ptr, dst_shape, DType::I32);
      StreamHandle stream = reinterpret_cast<StreamHandle>(stream_ptr);
      auto st = FillI32Cuda(D, value, stream);
      throw_if_bad(st, "fill_i32");
    },
    py::arg("dst_ptr"), py::arg("dst_shape"),
    py::arg("value"),
    py::arg("stream") = (uintptr_t)0
  );

  // -------- Arena controls (capture-safe allocator) --------
  // reserve_bytes(bytes)
  m.def("reserve_bytes",
    [](uint64_t bytes) {
      auto st = MemoryReserveBytesCuda(bytes);
      throw_if_bad(st, "reserve_bytes");
    },
    py::arg("bytes")
  );

  // reset_pool()
  m.def("reset_pool",
    []() {
      auto st = MemoryResetPoolCuda();
      throw_if_bad(st, "reset_pool");
    }
  );

  // stats() -> dict(total_reserved, peak_in_use, curr_in_use, slabs)
  m.def("stats",
    []() {
      MemoryStats s{};
      auto st = MemoryStatsCuda(s);
      throw_if_bad(st, "stats");
      return py::dict(
        "total_reserved"_a = s.total_reserved,
        "peak_in_use"_a    = s.peak_in_use,
        "curr_in_use"_a    = s.curr_in_use,
        "slabs"_a          = s.slabs
      );
    }
  );

  // -------- Temp workspace (token-based) --------
  // alloc_temp(nbytes, align=256, stream=0) -> token(uint64)
  m.def("alloc_temp",
    [](uint64_t nbytes, uint32_t align, uintptr_t stream_ptr) {
      uint64_t token = 0;
      StreamHandle stream = reinterpret_cast<StreamHandle>(stream_ptr);
      auto st = MemoryAllocTempCuda(nbytes, align ? align : 256, token, stream);
      throw_if_bad(st, "alloc_temp");
      return token;
    },
    py::arg("nbytes"),
    py::arg("align") = (uint32_t)256,
    py::arg("stream") = (uintptr_t)0
  );

  // free_temp(token, stream=0)
  // (MVP 구현이 no-op일 수 있음. Arena가 토큰 free 재사용을 지원하면 내부에서 반납됨)
  m.def("free_temp",
    [](uint64_t token, uintptr_t stream_ptr) {
      StreamHandle stream = reinterpret_cast<StreamHandle>(stream_ptr);
      auto st = MemoryFreeTempCuda(token, stream);
      throw_if_bad(st, "free_temp");
    },
    py::arg("token"),
    py::arg("stream") = (uintptr_t)0
  );
}
