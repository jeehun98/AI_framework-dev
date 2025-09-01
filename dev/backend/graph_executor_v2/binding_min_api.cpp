#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

// NOTE: 아래 함수 내부는 일단 "연결 고리"만 잡는다.
//  - 실제로는 op_type/kernel_name에 따라 네가 가진 CUDA 함수들을 호출하면 됨.
//  - 시작은 더미/스텁으로 두고, 하나씩 연결해가면 된다.

static py::dict query_capability_py(const std::string& op_type,
                                    py::dict /*in_descs*/,
                                    py::dict /*out_descs*/) {
    py::dict scores;
    // 예시: 어떤 이름의 커널을 지원하는지만 리턴(점수는 0~100)
    if (op_type == "GEMM_BIAS_ACT") {
        scores["gemm_bias_act_tc_f16"] = 80;
        scores["gemm_bias_act_f32"]    = 50;
    }
    return scores;
}

static void launch_kernel_py(const std::string& kernel_name,
                             py::list /*buffers*/,
                             py::dict /*descs*/,
                             size_t /*stream*/) {
    // TODO: 여기서 kernel_name에 따라 네 CUDA 커널 호출
    //  ex) if (kernel_name == "gemm_bias_act_tc_f16") { launch_gemm_bias_relu_tc(...); }
    // 지금은 스텁
}

PYBIND11_MODULE(graph_executor, m) {
    m.doc() = "minimal api for python-compiler -> cuda bridge";
    m.def("query_capability", &query_capability_py, "Return {kernel_name: score, ...}");
    m.def("launch_kernel", &launch_kernel_py, "Launch a specific fused kernel by name");
}
