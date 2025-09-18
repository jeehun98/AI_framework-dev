// ==== 기존 include 들은 그대로 유지 ====
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include "ge_v2_api.h"
#include "ge_v2_api_ex.h"

namespace py = pybind11;

static ge2_uintptr ptr_from_obj(py::handle obj) {
  if (py::isinstance<py::int_>(obj)) {
    return static_cast<ge2_uintptr>(obj.cast<uintptr_t>());
  }
  if (py::isinstance<py::capsule>(obj)) {
    auto cap = obj.cast<py::capsule>();
    return reinterpret_cast<ge2_uintptr>(cap.get_pointer());
  }
  if (obj.is_none()) return 0;
  throw std::runtime_error("Unsupported pointer object; pass int(address) or capsule or None");
}

PYBIND11_MODULE(graph_executor_v2, m) {
  m.doc() = "GE v2 bindings (legacy + extended) routed to regemm_epilogue";

  // ---- 기존 레거시 파라미터/함수 (유지) ----
  py::class_<ge2_gemm_bias_act_params_t>(m, "GemmBiasActParams")
      .def(py::init<>())
      .def_readwrite("M", &ge2_gemm_bias_act_params_t::M)
      .def_readwrite("N", &ge2_gemm_bias_act_params_t::N)
      .def_readwrite("K", &ge2_gemm_bias_act_params_t::K)
      .def_readwrite("has_bias", &ge2_gemm_bias_act_params_t::has_bias)
      .def_readwrite("act", &ge2_gemm_bias_act_params_t::act);

  m.def("gemm_bias_act_f32", [](py::object A, py::object B, py::object D, py::object bias,
                                ge2_gemm_bias_act_params_t p, py::object stream_obj) {
    ge2_uintptr bufs[5]; int n = 0;
    bufs[n++] = ptr_from_obj(A);
    bufs[n++] = ptr_from_obj(B);
    int has_bias = p.has_bias && !bias.is_none();
    if (has_bias) bufs[n++] = ptr_from_obj(bias);
    bufs[n++] = ptr_from_obj(D);
    bufs[n++] = reinterpret_cast<ge2_uintptr>(&p);

    void* stream = stream_obj.is_none() ? nullptr : reinterpret_cast<void*>(ptr_from_obj(stream_obj));
    int st = ge2_launch_gemm_bias_act_f32(bufs, n, stream);
    if (st != 0) throw std::runtime_error("ge2_launch_gemm_bias_act_f32 failed: " + std::to_string(st));
  }, py::arg("A"), py::arg("B"), py::arg("D"), py::arg("bias") = py::none(),
     py::arg("params"), py::arg("stream") = py::none());

  // ---- 확장 enum/struct ----
  py::enum_<ge2_bias_kind_t>(m, "BiasKind")
      .value("Scalar", GE2_BIAS_SCALAR)
      .value("PerM",   GE2_BIAS_PER_M)
      .value("PerN",   GE2_BIAS_PER_N);

  py::enum_<ge2_act_kind_t>(m, "ActKind")
      .value("None",      GE2_ACT_NONE)
      .value("ReLU",      GE2_ACT_RELU)
      .value("LeakyReLU", GE2_ACT_LEAKY_RELU)
      .value("GELU",      GE2_ACT_GELU)
      .value("Sigmoid",   GE2_ACT_SIGMOID)
      .value("Tanh",      GE2_ACT_TANH);

  // ---- EX Forward 파라미터 ----
  py::class_<ge2_gemm_bias_act_params_ex_t>(m, "GemmBiasActParamsEx")
      .def(py::init<>())
      .def_readwrite("M", &ge2_gemm_bias_act_params_ex_t::M)
      .def_readwrite("N", &ge2_gemm_bias_act_params_ex_t::N)
      .def_readwrite("K", &ge2_gemm_bias_act_params_ex_t::K)
      .def_readwrite("lda", &ge2_gemm_bias_act_params_ex_t::lda)
      .def_readwrite("ldb", &ge2_gemm_bias_act_params_ex_t::ldb)
      .def_readwrite("ldc", &ge2_gemm_bias_act_params_ex_t::ldc)
      .def_readwrite("ldd", &ge2_gemm_bias_act_params_ex_t::ldd)
      .def_readwrite("alpha", &ge2_gemm_bias_act_params_ex_t::alpha)
      .def_readwrite("beta",  &ge2_gemm_bias_act_params_ex_t::beta)
      .def_readwrite("use_C", &ge2_gemm_bias_act_params_ex_t::use_C)
      .def_readwrite("has_bias", &ge2_gemm_bias_act_params_ex_t::has_bias)
      .def_readwrite("bias_kind", &ge2_gemm_bias_act_params_ex_t::bias_kind)
      .def_readwrite("act_kind",  &ge2_gemm_bias_act_params_ex_t::act_kind)
      .def_readwrite("leaky_slope", &ge2_gemm_bias_act_params_ex_t::leaky_slope)
      // ✅ NEW: Z stash 옵션
      .def_readwrite("save_preact", &ge2_gemm_bias_act_params_ex_t::save_preact)
      .def_readwrite("ldZ", &ge2_gemm_bias_act_params_ex_t::ldZ);

  // ---- EX Forward: bufs 레이아웃 A,B,(C),D,(bias),(Z),params_ex ----
  m.def("gemm_bias_act_f32_ex",
        [](py::object A, py::object B, py::object C, py::object D, py::object bias, py::object Z,
           ge2_gemm_bias_act_params_ex_t px, py::object stream_obj) {
            ge2_uintptr bufs[10]; int n = 0;
            bufs[n++] = ptr_from_obj(A);
            bufs[n++] = ptr_from_obj(B);
            if (px.use_C) {
              if (C.is_none()) throw std::runtime_error("use_C=1 but C is None");
              bufs[n++] = ptr_from_obj(C);
            }
            bufs[n++] = ptr_from_obj(D);
            if (px.has_bias) {
              if (bias.is_none()) throw std::runtime_error("has_bias=1 but bias is None");
              bufs[n++] = ptr_from_obj(bias);
            }
            if (px.save_preact) {
              if (Z.is_none()) throw std::runtime_error("save_preact=1 but Z is None");
              bufs[n++] = ptr_from_obj(Z);
            }
            bufs[n++] = reinterpret_cast<ge2_uintptr>(&px);

            void* stream = stream_obj.is_none() ? nullptr : reinterpret_cast<void*>(ptr_from_obj(stream_obj));
            int st = ge2_launch_gemm_bias_act_f32_ex(bufs, n, stream);
            if (st != 0) throw std::runtime_error("ge2_launch_gemm_bias_act_f32_ex failed: " + std::to_string(st));
        },
        py::arg("A"), py::arg("B"), py::arg("C") = py::none(),
        py::arg("D"), py::arg("bias") = py::none(), py::arg("Z") = py::none(),
        py::arg("params"), py::arg("stream") = py::none());

  // ---- Backward(EX) 파라미터 ----
  py::class_<ge2_gemm_bias_act_bwd_params_t>(m, "GemmBiasActBwdParams")
      .def(py::init<>())
      .def_readwrite("M", &ge2_gemm_bias_act_bwd_params_t::M)
      .def_readwrite("N", &ge2_gemm_bias_act_bwd_params_t::N)
      .def_readwrite("K", &ge2_gemm_bias_act_bwd_params_t::K)
      .def_readwrite("lda", &ge2_gemm_bias_act_bwd_params_t::lda)
      .def_readwrite("ldb", &ge2_gemm_bias_act_bwd_params_t::ldb)
      .def_readwrite("ldc", &ge2_gemm_bias_act_bwd_params_t::ldc)
      .def_readwrite("ldgY", &ge2_gemm_bias_act_bwd_params_t::ldgY)
      .def_readwrite("ldZ",  &ge2_gemm_bias_act_bwd_params_t::ldZ)
      .def_readwrite("ldgA", &ge2_gemm_bias_act_bwd_params_t::ldgA)
      .def_readwrite("ldgB", &ge2_gemm_bias_act_bwd_params_t::ldgB)
      .def_readwrite("ldgC", &ge2_gemm_bias_act_bwd_params_t::ldgC)
      .def_readwrite("alpha", &ge2_gemm_bias_act_bwd_params_t::alpha)
      .def_readwrite("beta",  &ge2_gemm_bias_act_bwd_params_t::beta)
      .def_readwrite("bias_kind", &ge2_gemm_bias_act_bwd_params_t::bias_kind)
      .def_readwrite("act_kind",  &ge2_gemm_bias_act_bwd_params_t::act_kind)
      .def_readwrite("leaky_slope", &ge2_gemm_bias_act_bwd_params_t::leaky_slope);

  // ---- Backward(EX): bufs 레이아웃 A,B,(C), gY, Z, gA, gB, (gC), (gBias), params_bwd ----
  // ---- Backward(EX): bufs 레이아웃을 항상 고정 인덱스로 ----
  m.def("gemm_bias_act_bwd_f32_ex",
    [](py::object A, py::object B, py::object C,
      py::object gY, py::object Z,
      py::object gA, py::object gB,
      py::object gC, py::object gBias,
      ge2_gemm_bias_act_bwd_params_t pb, py::object stream_obj) {

      ge2_uintptr bufs[11]; int n = 0;
      bufs[n++] = ptr_from_obj(A);         // 0: A
      bufs[n++] = ptr_from_obj(B);         // 1: B
      // 2: C (없으면 0을 넣는다)
      bufs[n++] = C.is_none() ? 0 : ptr_from_obj(C);
      bufs[n++] = ptr_from_obj(gY);        // 3: gY
      bufs[n++] = ptr_from_obj(Z);         // 4: Z
      bufs[n++] = ptr_from_obj(gA);        // 5: gA
      bufs[n++] = ptr_from_obj(gB);        // 6: gB
      // 7: gC (없으면 0)
      bufs[n++] = gC.is_none() ? 0 : ptr_from_obj(gC);
      // 8: gBias (없으면 0)
      bufs[n++] = gBias.is_none() ? 0 : ptr_from_obj(gBias);
      bufs[n++] = reinterpret_cast<ge2_uintptr>(&pb); // 9: &pb

      void* stream = stream_obj.is_none() ? nullptr
                      : reinterpret_cast<void*>(ptr_from_obj(stream_obj));
      int st = ge2_launch_gemm_bias_act_bwd_f32_ex(bufs, n, stream);
      if (st != 0)
        throw std::runtime_error("ge2_launch_gemm_bias_act_bwd_f32_ex failed: " + std::to_string(st));
  });

}
