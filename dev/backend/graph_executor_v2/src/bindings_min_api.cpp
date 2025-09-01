#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <unordered_map>
#include <string>
#include <cstdint>

namespace py = pybind11;

using KernelFn = int(*)(const uintptr_t*, int, void*);

// launch_table.cpp에서 제공
const std::unordered_map<std::string, KernelFn>& ge_v2_kernel_table_raw();
const std::unordered_map<std::string, int>&      ge_v2_capability_table_raw();

static py::dict query_capability_py(const std::string& op_type,
                                    py::dict /*in_descs*/,
                                    py::dict /*out_descs*/) {
  py::dict scores;
  const auto& cap = ge_v2_capability_table_raw();
  for (auto& kv : cap) {
    const auto& key = kv.first; // "GEMM_BIAS_ACT__gemm_bias_act_tc_f16"
    if (key.rfind(op_type + "__", 0) == 0) { // 접두 일치
      scores[py::str(key.substr(op_type.size() + 2))] = kv.second;
    }
  }
  return scores;
}

static void launch_kernel_py(const std::string& kernel_name,
                             std::vector<uintptr_t> buffers,
                             py::dict /*descs*/,
                             std::uintptr_t stream_opaque) {
  const auto& tab = ge_v2_kernel_table_raw();
  auto it = tab.find(kernel_name);
  if (it == tab.end()) throw std::runtime_error("unknown kernel: " + kernel_name);
  KernelFn fn = it->second;
  int rc = fn(buffers.data(), static_cast<int>(buffers.size()),
              reinterpret_cast<void*>(stream_opaque));
  if (rc != 0) throw std::runtime_error("kernel launch failed rc=" + std::to_string(rc));
}

PYBIND11_MODULE(graph_executor_v2, m) {
  m.doc() = "V2 bridge (CUDA-ready) for Python compiler";
  m.def("query_capability", &query_capability_py);
  m.def("launch_kernel", &launch_kernel_py,
        py::arg("kernel_name"), py::arg("buffers"),
        py::arg("descs"), py::arg("stream") = 0);
}
