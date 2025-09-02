#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include "ge_v2_api.h"

/**
 * @file bindings_min_api.cpp
 * @brief 파이썬에서 호출하는 public API(pybind11)
 *
 * - query_capability(op_type, in_descs, out_descs) -> {kernel_name: score}
 * - launch_kernel(kernel_name, buffers, descs, stream=0) -> None(예외 on error)
 * - query_kernels() -> [kernel_name, ...]  # 디버그용
 */

namespace py = pybind11;

// 네이티브 capability에서 op_type에 해당하는 항목만 파싱
static py::dict query_capability_py(const std::string& op_type,
                                    py::dict /*in_descs*/,
                                    py::dict /*out_descs*/) {
  py::dict scores;
  const auto& cap = ge_v2_capability_table_raw();
  const std::string prefix = op_type + "__";
  for (const auto& kv : cap) {
    if (kv.first.rfind(prefix, 0) == 0) {
      const std::string kernel_name = kv.first.substr(prefix.size());
      scores[py::str(kernel_name)] = kv.second;
    }
  }
  return scores;
}

// 커널 이름으로 런치
static void launch_kernel_py(const std::string& kernel_name,
                             std::vector<ge2_uintptr> buffers,
                             py::dict /*descs*/,
                             ge2_uintptr stream_opaque = 0) {
  const auto& tab = ge_v2_kernel_table_raw();
  auto it = tab.find(kernel_name);
  if (it == tab.end()) {
    throw std::runtime_error("unknown kernel: " + kernel_name);
  }

  ge2_kernel_fn fn = it->second;
  int rc = fn(buffers.data(),
              static_cast<int>(buffers.size()),
              reinterpret_cast<void*>(stream_opaque));
  if (rc != 0) {
    throw std::runtime_error("kernel launch failed rc=" + std::to_string(rc));
  }
}

// 등록된 커널 이름 나열(디버그)
static py::list query_kernels_py() {
  py::list L;
  for (const auto& kv : ge_v2_kernel_table_raw()) {
    L.append(py::str(kv.first));
  }
  return L;
}

PYBIND11_MODULE(graph_executor_v2, m) {
  m.doc() = "V2 bridge (CUDA-ready) for Python compiler";

  m.def("query_capability", &query_capability_py,
        py::arg("op_type"), py::arg("in_descs") = py::dict(),
        py::arg("out_descs") = py::dict(),
        "Return {kernel_name: score} filtered by op_type.");

  m.def("launch_kernel", &launch_kernel_py,
        py::arg("kernel_name"), py::arg("buffers"),
        py::arg("descs") = py::dict(), py::arg("stream") = 0,
        "Launch kernel by name with device pointers and optional stream.");

  m.def("query_kernels", &query_kernels_py, "List registered kernel names.");

  m.attr("GE2_API_VERSION") = GE2_API_VERSION;
}
