/**
 * @file bindings_min_api.cpp
 * @brief pybind11 바인딩: 파이썬에서 호출하는 public API를 노출
 *
 * 노출 함수:
 *  - query_capability(op_type, in_descs, out_descs) -> Dict[kernel_name] = score
 *  - launch_kernel(kernel_name, buffers, descs, stream=0) -> None(예외 on error)
 *  - (디버그) query_kernels() -> List[kernel_name]
 *
 * 주의:
 *  - buffers 는 uintptr_t(Device pointer) 목록입니다. 순서는 "입력..출력.." 고정.
 *  - stream 은 CUDA면 cudaStream_t 를 정수로 캐스팅하여 전달받습니다.
 *  - descs 는 현재 opaque(dict)로 받지만, 필요 시 C++에서 검증/해석하도록 확장 가능합니다.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include "ge_v2_api.h"

namespace py = pybind11;

// --------------------------- capability 조회 ---------------------------

/**
 * @brief 네이티브가 보유한 캡 테이블에서 특정 op_type 의 후보만 뽑아 리턴
 *        key 형식은 "<OPTYPE>__<KERNEL_NAME>"
 */
static py::dict query_capability_py(const std::string& op_type,
                                    py::dict /*in_descs*/,
                                    py::dict /*out_descs*/) {
  py::dict scores;
  const auto& cap = ge_v2_capability_table_raw();
  const std::string prefix = op_type + "__";
  for (const auto& kv : cap) {
    // 접두 일치로 필터링
    if (kv.first.rfind(prefix, 0) == 0) {
      const std::string kernel_name = kv.first.substr(prefix.size());
      scores[py::str(kernel_name)] = kv.second;
    }
  }
  return scores;
}

// ----------------------------- 커널 런치 -------------------------------

/**
 * @brief 이름으로 커널 진입점 찾아 launch
 * @throws std::runtime_error on unknown kernel or non-zero rc
 */
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

  // (옵션) 간단 검증: 너무 적은 버퍼 등
  // 실제 요건은 kernel 별로 다를 수 있으므로, 커널 측에서 재검증하는 게 안전합니다.

  int rc = fn(buffers.data(),
              static_cast<int>(buffers.size()),
              reinterpret_cast<void*>(stream_opaque));
  if (rc != 0) {
    throw std::runtime_error("kernel launch failed rc=" + std::to_string(rc));
  }
}

// ---------------------------- 디버그 유틸 -----------------------------

/**
 * @brief 등록된 커널 이름 나열 (디버깅 편의)
 */
static py::list query_kernels_py() {
  py::list L;
  for (const auto& kv : ge_v2_kernel_table_raw()) {
    L.append(py::str(kv.first));
  }
  return L;
}

// ------------------------------- 모듈 ---------------------------------

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

  // 디버그: 등록된 커널 목록
  m.def("query_kernels", &query_kernels_py, "List registered kernel names.");

  // (옵션) 버전 노출
  m.attr("GE2_API_VERSION") = GE2_API_VERSION;
}
