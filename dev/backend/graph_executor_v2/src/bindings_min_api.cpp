// bindings_min_api.cpp
//
// 노출 API (Python):
//  - query_capability(op_type, in_descs={}, out_descs={}) -> { kernel_name: score }
//    * capability 테이블에서 "OPTYPE__KERNEL" 형태의 엔트리를 찾아서 반환.
//    * in_descs/out_descs 는 향후 확장을 대비한 자리(현재는 미사용).
//
//  - launch_kernel(kernel_name, buffers, descs={}, stream=0) -> None
//    * 커널 테이블에서 kernel_name으로 함수 포인터를 찾아 호출.
//    * buffers: [inputs ...] + [outputs ...] + [옵션: Host 파라미터 블록 주소]
//      - 각 항목은 uintptr_t(정수)로 표현된 device/host 포인터.
//    * descs: 디버깅/로깅용(네이티브 커널은 사용하지 않아도 됨).
//    * stream: CUDA 스트림 주소(uintptr_t). 0이면 default stream.
//
//  - query_kernels() -> [kernel_name, ...]
//    * 현재 등록된 커널 이름 목록을 반환.
//
// 구현 메모:
//  - ge_v2_api.h 에 선언된
//      ge_v2_kernel_table_raw() : {name -> fn}
//      ge_v2_capability_table_raw() : {"OPTYPE__KERNEL" -> score}
//    를 그대로 래핑해 Python 바인딩으로 노출.
//  - launch_kernel 호출 시 CUDA 작업 동안 GIL을 해제하여
//    다른 Python 스레드가 진행될 수 있도록 함.
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include "ge_v2_api.h"

namespace py = pybind11;

// ---------------------------------------------------------------------------
// capability 조회: op_type 에 해당하는 커널 스코어 목록을 dict로 반환
// ---------------------------------------------------------------------------
// in_descs/out_descs 는 현재 미사용(미래: dtype/shape/stride 제약 반영 가능)
static py::dict query_capability_py(const std::string& op_type,
                                    py::dict /*in_descs*/,
                                    py::dict /*out_descs*/) {
  py::dict scores;

  // 전체 capability 테이블에서 "OPTYPE__" 접두사를 가진 엔트리만 선택
  const auto& cap = ge_v2_capability_table_raw();
  const std::string prefix = op_type + "__";

  for (const auto& kv : cap) {
    // rfind(prefix, 0) == 0  <=> 문자열 시작이 prefix 와 일치
    if (kv.first.rfind(prefix, 0) == 0) {
      const std::string kernel_name = kv.first.substr(prefix.size());
      // Python dict 에 { kernel_name: score } 로 입력
      scores[py::str(kernel_name)] = kv.second;
    }
  }
  return scores;
}

// ---------------------------------------------------------------------------
// 커널 실행: 이름으로 함수포인터를 찾아 buffers / stream 으로 호출
// ---------------------------------------------------------------------------
// 주의:
//  - 네이티브 커널은 동기/비동기 모두 가능. 보통은 스트림에 비동기 enqueue.
//  - 여기서는 반환코드(rc != 0) 시 Python 예외를 던진다.
//  - CUDA 작업 중 GIL 해제(py::gil_scoped_release)로 다른 파이썬 스레드 진행 허용.
//
static void launch_kernel_py(const std::string& kernel_name,
                             std::vector<ge2_uintptr> buffers,
                             py::dict /*descs*/,
                             ge2_uintptr stream_opaque = 0) {
  // 1) 커널 테이블 조회
  const auto& tab = ge_v2_kernel_table_raw();
  auto it = tab.find(kernel_name);
  if (it == tab.end()) {
    throw std::runtime_error("unknown kernel: " + kernel_name);
  }

  ge2_kernel_fn fn = it->second;

  // 2) 커널 호출 (CUDA 구간: GIL 해제)
  int rc = 0;
  {
    // GIL 해제 스코프: 블록을 벗어나면 자동으로 GIL 재획득
    py::gil_scoped_release release;

    rc = fn(
      buffers.data(),                           // ge2_uintptr* (연속 메모리)
      static_cast<int>(buffers.size()),         // 버퍼 개수
      reinterpret_cast<void*>(stream_opaque)    // opaque stream 포인터
    );
  }

  // 3) 오류 처리
  if (rc != 0) {
    // rc 매핑 규약(ge_v2_api.h):
    //  - -1: invalid args
    //  - -2: device/cublas error
    //  - -3: not implemented
    throw std::runtime_error(
      "kernel launch failed: name=" + kernel_name + " rc=" + std::to_string(rc)
    );
  }
}

// ---------------------------------------------------------------------------
// 등록된 커널 목록 조회(디버그/점검용)
// ---------------------------------------------------------------------------
static py::list query_kernels_py() {
  py::list L;
  for (const auto& kv : ge_v2_kernel_table_raw()) {
    L.append(py::str(kv.first));
  }
  return L;
}

// ---------------------------------------------------------------------------
// pybind11 모듈 등록부
// ---------------------------------------------------------------------------
PYBIND11_MODULE(graph_executor_v2, m) {
  // 모듈 문서
  m.doc() = "Graph-Executor V2 native bridge (CUDA/cuBLASLt)";

  // 기능 1) capability 조회
  m.def("query_capability", &query_capability_py,
        // 인자 기본값: in/out desc는 미래 확장용
        py::arg("op_type"),
        py::arg("in_descs") = py::dict(),
        py::arg("out_descs") = py::dict(),
        R"pbdoc(
        Query candidate kernels and their preference scores for an op_type.

        Parameters
        ----------
        op_type : str
            Logical operator type (e.g., "GEMM_BIAS_ACT")
        in_descs, out_descs : dict
            Reserved for future constraints (dtype/shape/stride). Currently unused.

        Returns
        -------
        dict
            { kernel_name: score }
        )pbdoc");

  // 기능 2) 커널 실행
  m.def("launch_kernel", &launch_kernel_py,
        py::arg("kernel_name"),
        py::arg("buffers"),
        py::arg("descs") = py::dict(),
        py::arg("stream") = 0,
        R"pbdoc(
        Launch a native kernel by name.

        Parameters
        ----------
        kernel_name : str
            Must exist in ge_v2_kernel_table_raw()
        buffers : List[int]
            ABI: inputs... + outputs... + [optional host-param-block address]
            Every entry is an integer-encoded pointer (uintptr_t).
        descs : dict
            Optional metadata for logging/debug (native kernels may ignore this).
        stream : int
            CUDA stream pointer encoded as uintptr_t. 0 means default stream.

        Raises
        ------
        RuntimeError
            If the kernel name is unknown or the native function returns a non-zero code.
        )pbdoc");

  // 기능 3) 등록된 커널 이름 나열
  m.def("query_kernels", &query_kernels_py,
        R"pbdoc(Return a list of registered kernel names.)pbdoc");

  // 네이티브 ABI 버전 상수 노출(런타임 호환성 체크용)
  m.attr("GE2_API_VERSION") = GE2_API_VERSION;
}
