// src/bindings/batchnorm_pybind.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#ifdef BUILD_STANDALONE_OPS
  #include "backends/cuda/ops/_common/shim/ai_shim.hpp"
#else
  #include "ai/tensor.hpp"
  #include "ai/dispatch.hpp"
#endif

#include "backends/cuda/ops/batchnorm/api.hpp"

namespace py = pybind11;
using namespace ai;

// ---------- helpers: contiguous row-major Tensor makers ----------
static Tensor make_tensor_4d(uintptr_t ptr_u64,
                             const std::vector<int64_t>& shape, // [N,C,H,W] or [N,H,W,C]
                             DType dtype = DType::F32,
                             Device dev  = Device::CUDA) {
  if (shape.size() != 4) throw std::invalid_argument("shape must be 4D");
  Tensor t;
  t.data = reinterpret_cast<void*>(ptr_u64);
  t.device = dev;
  t.device_index = 0;
  t.desc.dtype = dtype;
  t.desc.layout = Layout::RowMajor;
  t.desc.shape = shape;
  // contiguous row-major stride
  t.desc.stride.resize(4);
  t.desc.stride[3] = 1;
  t.desc.stride[2] = shape[3] * t.desc.stride[3];
  t.desc.stride[1] = shape[2] * t.desc.stride[2];
  t.desc.stride[0] = shape[1] * t.desc.stride[1];
  return t;
}

static Tensor make_tensor_1d(uintptr_t ptr_u64,
                             int64_t len,
                             DType dtype = DType::F32,
                             Device dev  = Device::CUDA) {
  Tensor t;
  t.data = reinterpret_cast<void*>(ptr_u64);
  t.device = dev;
  t.device_index = 0;
  t.desc.dtype = dtype;
  t.desc.layout = Layout::RowMajor;
  t.desc.shape  = { len };
  t.desc.stride = { 1 };
  return t;
}

// ---------- error helper ----------
static void throw_if_bad(Status st, const char* where) {
  if (st != Status::Ok) {
    throw std::runtime_error(
      std::string("[_ops_batchnorm::") + where + "] failed with Status=" +
      std::to_string(static_cast<int>(st))
    );
  }
}

PYBIND11_MODULE(_ops_batchnorm, m) {
  m.attr("__package__") = "graph_executor_v2.ops";
  m.doc() = R"(Independent BatchNorm CUDA ops binding (standalone shim compatible)
- Training: reduce mean/var, update running stats, output save_mean/save_invstd.
- Inference: use running stats; invstd buffer required from caller (capture-safe).
- Layout: row-major contiguous tensors. X/Y are 4D; gamma/beta/running/save are 1D [C].
)";

  // (선택) 공통 enum/타입 재노출 예시:
  // py::module_ common = py::module_::import("graph_executor_v2.ops._ops_common");
  // m.attr("DType") = common.attr("DType"); ...

  // ---------- Attrs ----------
  py::class_<BatchNormAttrs>(m, "BatchNormAttrs")
    .def(py::init<>())
    .def_readwrite("channels_last", &BatchNormAttrs::channels_last)
    .def_readwrite("eps",           &BatchNormAttrs::eps)
    .def_readwrite("momentum",      &BatchNormAttrs::momentum)
    .def_readwrite("training",      &BatchNormAttrs::training)
    .def_readwrite("with_affine",   &BatchNormAttrs::with_affine)
    .def_readwrite("use_welford",   &BatchNormAttrs::use_welford)
    .def_readwrite("num_groups",    &BatchNormAttrs::num_groups);

  // ========================= forward =========================
  // forward(x_ptr,x_shape, y_ptr,y_shape, gamma_ptr|None, beta_ptr|None,
  //         running_mean_ptr, running_var_ptr, attrs, stream,
  //         save_mean_ptr|None, save_invstd_ptr|None)
  m.def("forward",
    [](uintptr_t x_ptr, const std::vector<int64_t>& x_shape,   // [N,*,*,*]
       uintptr_t y_ptr, const std::vector<int64_t>& y_shape,   // same as X
       py::object gamma_ptr_obj,                                // int or None
       py::object beta_ptr_obj,                                 // int or None
       uintptr_t running_mean_ptr,                              // [C]
       uintptr_t running_var_ptr,                               // [C]
       BatchNormAttrs attrs,
       uintptr_t stream_ptr,
       py::object save_mean_ptr_obj = py::none(),               // training 필수
       py::object save_invstd_ptr_obj = py::none()              // training/infer 모두 권장
    ){
      Tensor X = make_tensor_4d(x_ptr, x_shape);
      Tensor Y = make_tensor_4d(y_ptr, y_shape);
      if (X.desc.shape != Y.desc.shape) {
        throw std::invalid_argument("[_ops_batchnorm.forward] X and Y shape mismatch");
      }

      // 채널 수 추론
      int64_t C = attrs.channels_last ? X.desc.shape[3] : X.desc.shape[1];

      const Tensor* gamma = nullptr;
      const Tensor* beta  = nullptr;
      Tensor gammaT, betaT;

      if (!gamma_ptr_obj.is_none()) {
        auto gptr = gamma_ptr_obj.cast<uintptr_t>();
        gammaT = make_tensor_1d(gptr, C);
        gamma = &gammaT;
      }
      if (!beta_ptr_obj.is_none()) {
        auto bptr = beta_ptr_obj.cast<uintptr_t>();
        betaT = make_tensor_1d(bptr, C);
        beta = &betaT;
      }
      if (attrs.with_affine && (!gamma || !beta)) {
        throw std::invalid_argument("[_ops_batchnorm.forward] with_affine=True requires gamma_ptr and beta_ptr");
      }

      Tensor running_mean = make_tensor_1d(running_mean_ptr, C);
      Tensor running_var  = make_tensor_1d(running_var_ptr,  C);

      Tensor *save_mean=nullptr, *save_invstd=nullptr;
      Tensor save_meanT, save_invstdT;

      if (!save_mean_ptr_obj.is_none()) {
        auto sp = save_mean_ptr_obj.cast<uintptr_t>();
        save_meanT = make_tensor_1d(sp, C);
        save_mean = &save_meanT;
      }
      if (!save_invstd_ptr_obj.is_none()) {
        auto sp = save_invstd_ptr_obj.cast<uintptr_t>();
        save_invstdT = make_tensor_1d(sp, C);
        save_invstd = &save_invstdT;
      }

      if (attrs.training) {
        if (!save_mean || !save_invstd) {
          throw std::invalid_argument("[_ops_batchnorm.forward] training=True requires save_mean_ptr and save_invstd_ptr");
        }
      } else {
        // inference에서도 캡처-세이프를 위해 invstd 출력 버퍼를 제공하도록 강제
        if (!save_invstd) {
          throw std::invalid_argument("[_ops_batchnorm.forward] inference requires save_invstd_ptr (capture-safe)");
        }
      }

      StreamHandle stream = reinterpret_cast<StreamHandle>(stream_ptr);
      auto st = BatchNormCudaLaunch(
        X, gamma, beta, &running_mean, &running_var, Y,
        attrs, stream, save_mean, save_invstd, /*ws_fwd*/ nullptr
      );
      throw_if_bad(st, "forward");
    },
    py::arg("x_ptr"), py::arg("x_shape"),
    py::arg("y_ptr"), py::arg("y_shape"),
    py::arg("gamma_ptr") = py::none(),
    py::arg("beta_ptr")  = py::none(),
    py::arg("running_mean_ptr"),
    py::arg("running_var_ptr"),
    py::arg("attrs")  = BatchNormAttrs{},
    py::arg("stream") = static_cast<uintptr_t>(0),
    py::arg("save_mean_ptr")   = py::none(),
    py::arg("save_invstd_ptr") = py::none()
  );

  // ========================= backward =========================
  // backward(dy_ptr,dy_shape, x_ptr,x_shape, gamma_ptr|None,
  //          save_mean_ptr, save_invstd_ptr,
  //          dx_ptr|None, dgamma_ptr|None, dbeta_ptr|None,
  //          attrs, stream)
  m.def("backward",
    [](uintptr_t dy_ptr, const std::vector<int64_t>& dy_shape,
       uintptr_t x_ptr,  const std::vector<int64_t>& x_shape,
       py::object gamma_ptr_obj,                     // int or None
       uintptr_t save_mean_ptr,                      // [C]
       uintptr_t save_invstd_ptr,                    // [C]
       py::object dx_ptr_obj,                        // int or None
       py::object dgamma_ptr_obj,                    // int or None
       py::object dbeta_ptr_obj,                     // int or None
       BatchNormAttrs attrs,
       uintptr_t stream_ptr)
    {
      Tensor dY = make_tensor_4d(dy_ptr, dy_shape);
      Tensor X  = make_tensor_4d(x_ptr,  x_shape);
      if (dY.desc.shape != X.desc.shape) {
        throw std::invalid_argument("[_ops_batchnorm.backward] dY and X shape mismatch");
      }

      int64_t C = attrs.channels_last ? X.desc.shape[3] : X.desc.shape[1];

      const Tensor* gamma = nullptr;
      Tensor gammaT;
      if (!gamma_ptr_obj.is_none()) {
        auto gptr = gamma_ptr_obj.cast<uintptr_t>();
        gammaT = make_tensor_1d(gptr, C);
        gamma = &gammaT;
      } else if (attrs.with_affine) {
        // with_affine=True인데 gamma를 안 주면 dX 계산에서 γ=1로 처리해도 되지만,
        // 일반적으로는 사용자가 전달하도록 강제하는 편이 안전.
        // 필요 시 완화 가능.
      }

      Tensor save_mean   = make_tensor_1d(save_mean_ptr,   C);
      Tensor save_invstd = make_tensor_1d(save_invstd_ptr, C);

      Tensor *dX=nullptr, *dgamma=nullptr, *dbeta=nullptr;
      Tensor dX_T, dgamma_T, dbeta_T;

      if (!dx_ptr_obj.is_none()) {
        auto p = dx_ptr_obj.cast<uintptr_t>();
        dX_T = make_tensor_4d(p, x_shape);
        dX = &dX_T;
      }
      if (!dgamma_ptr_obj.is_none()) {
        auto p = dgamma_ptr_obj.cast<uintptr_t>();
        dgamma_T = make_tensor_1d(p, C);
        dgamma = &dgamma_T;
      }
      if (!dbeta_ptr_obj.is_none()) {
        auto p = dbeta_ptr_obj.cast<uintptr_t>();
        dbeta_T = make_tensor_1d(p, C);
        dbeta = &dbeta_T;
      }

      StreamHandle stream = reinterpret_cast<StreamHandle>(stream_ptr);
      auto st = BatchNormCudaBackwardLaunch(
        dY, X, gamma, save_mean, save_invstd,
        dX, dgamma, dbeta, attrs, stream, /*ws_bwd*/ nullptr
      );
      throw_if_bad(st, "backward");
    },
    py::arg("dy_ptr"), py::arg("dy_shape"),
    py::arg("x_ptr"),  py::arg("x_shape"),
    py::arg("gamma_ptr") = py::none(),
    py::arg("save_mean_ptr"),
    py::arg("save_invstd_ptr"),
    py::arg("dx_ptr")     = py::none(),
    py::arg("dgamma_ptr") = py::none(),
    py::arg("dbeta_ptr")  = py::none(),
    py::arg("attrs")  = BatchNormAttrs{},
    py::arg("stream") = static_cast<uintptr_t>(0)
  );
}
