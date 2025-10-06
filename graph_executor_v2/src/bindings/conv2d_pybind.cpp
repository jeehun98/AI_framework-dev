// src/bindings/conv2d_pybind.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#ifdef BUILD_STANDALONE_OPS
  #include "backends/cuda/ops/_common/shim/ai_shim.hpp"
#else
  #include "ai/tensor.hpp"
  #include "ai/dispatch.hpp"
#endif

#include "backends/cuda/ops/conv2d/api.hpp"

namespace py = pybind11;
using namespace ai;

// ----------------- helpers: contiguous row-major Tensor makers -----------------
static Tensor make_tensor_4d(uintptr_t ptr_u64,
                             const std::vector<int64_t>& shape, // [N,C,H,W]
                             DType dtype = DType::F32,
                             Device dev  = Device::CUDA) {
  if (shape.size() != 4) throw std::invalid_argument("shape must be 4D [N,C,H,W]");
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

// 예외 변환: Status != Ok -> Python RuntimeError
static void throw_if_bad(Status st, const char* where) {
  if (st != Status::Ok) {
    throw std::runtime_error(
      std::string("[_ops_conv2d::") + where + "] failed with Status=" +
      std::to_string(static_cast<int>(st))
    );
  }
}

PYBIND11_MODULE(_ops_conv2d, m) {
  m.attr("__package__") = "graph_executor_v2.ops";
  m.doc() = R"(Independent conv2d CUDA ops binding (standalone shim compatible)
- Forward/Backward support fused epilogue (act, bias) and Z(pre) save/consume.
- All tensors are float32, row-major contiguous.
- Shapes:
  * X: [N,Cin,H,W], W: [Cout,Cin,Kh,Kw], B: [Cout]
  * Y/Z: [N,Cout,Hout,Wout]
- Workspaces:
  * Forward: dCol[HWo,K], W_KC[K,Cout], Y_tmp[HWo,Cout], (optional) Z_rows[HWo,Cout]
  * Backward: dCol[HWo,K], dTmp[max(Cout*K, HWo*K)], (optional) W_CK[Cout,K], dY_HT[HWo,Cout], dWpack[Cout,K], gy_rows[Cout,HWo], Z_rows[Cout,HWo]
)";

  // ----- ActKind 노출 -----
  py::enum_<ActKind>(m, "ActKind")
    .value("None",      ActKind::None)
    .value("ReLU",      ActKind::ReLU)
    .value("LeakyReLU", ActKind::LeakyReLU)
    .value("GELU",      ActKind::GELU)
    .value("Sigmoid",   ActKind::Sigmoid)
    .value("Tanh",      ActKind::Tanh)
    .export_values();

  // ----- Conv2DAttrs -----
  py::class_<Conv2DAttrs>(m, "Conv2DAttrs")
    .def(py::init<>())
    .def_readwrite("stride_h",    &Conv2DAttrs::stride_h)
    .def_readwrite("stride_w",    &Conv2DAttrs::stride_w)
    .def_readwrite("pad_h",       &Conv2DAttrs::pad_h)
    .def_readwrite("pad_w",       &Conv2DAttrs::pad_w)
    .def_readwrite("dil_h",       &Conv2DAttrs::dil_h)
    .def_readwrite("dil_w",       &Conv2DAttrs::dil_w)
    .def_readwrite("groups",      &Conv2DAttrs::groups)
    .def_readwrite("with_bias",   &Conv2DAttrs::with_bias)
    .def_readwrite("act",         &Conv2DAttrs::act)
    .def_readwrite("leaky_slope", &Conv2DAttrs::leaky_slope)
    .def_readwrite("save_z",      &Conv2DAttrs::save_z);

  // ========================= forward =========================
  // forward(..., z_ptr=None, attrs, stream, *, dCol_ptr=0, W_KC_ptr=0, Y_tmp_ptr=0, Z_rows_ptr=0)
  // attrs.save_z=True면 z_ptr과 Z_rows_ptr 필수
  m.def("forward",
    [](uintptr_t x_ptr, const std::vector<int64_t>& x_shape,        // [N,Cin,H,W]
       uintptr_t w_ptr, const std::vector<int64_t>& w_shape,        // [Cout,Cin,Kh,Kw]
       uintptr_t y_ptr, const std::vector<int64_t>& y_shape,        // [N,Cout,Hout,Wout]
       py::object b_ptr_obj,                                        // int or None (bias=[Cout])
       py::object z_ptr_obj,                                        // int or None (Z_saved=[N,Cout,Ho,Wo])
       Conv2DAttrs attrs,
       uintptr_t stream_ptr,
       // --- workspace pointers (optional, default 0=none) ---
       uintptr_t dCol_ptr,
       uintptr_t W_KC_ptr,
       uintptr_t Y_tmp_ptr,
       uintptr_t Z_rows_ptr) {

        Tensor X = make_tensor_4d(x_ptr, x_shape);
        Tensor W = make_tensor_4d(w_ptr, w_shape);
        Tensor Y = make_tensor_4d(y_ptr, y_shape);

        const Tensor* Bptr = nullptr;
        Tensor B;
        if (!b_ptr_obj.is_none()) {
          auto b_ptr = b_ptr_obj.cast<uintptr_t>();
          B = make_tensor_1d(b_ptr, w_shape.at(0)); // Cout
          Bptr = &B;
        }

        Tensor* Zptr = nullptr;
        Tensor Ztmp;
        if (!z_ptr_obj.is_none()) {
          auto z_ptr = z_ptr_obj.cast<uintptr_t>();
          Ztmp = make_tensor_4d(z_ptr, y_shape); // Z_saved shape == Y shape
          Zptr = &Ztmp;
        }

        // Workspace struct
        Conv2DWorkspaceFwd ws{};
        ws.dCol   = reinterpret_cast<float*>(dCol_ptr);
        ws.W_KC   = reinterpret_cast<float*>(W_KC_ptr);
        ws.Y_tmp  = reinterpret_cast<float*>(Y_tmp_ptr);
        ws.Z_rows = reinterpret_cast<float*>(Z_rows_ptr);

        // 필수성 체크
        if (!ws.dCol || !ws.W_KC || !ws.Y_tmp) {
          throw std::invalid_argument("[_ops_conv2d.forward] workspace pointers (dCol, W_KC, Y_tmp) are required");
        }
        if (attrs.save_z && (Zptr == nullptr)) {
          throw std::invalid_argument("[_ops_conv2d.forward] attrs.save_z=True requires z_ptr (Z_saved)");
        }
        if (attrs.save_z && (!ws.Z_rows)) {
          throw std::invalid_argument("[_ops_conv2d.forward] attrs.save_z=True requires Z_rows workspace");
        }

        StreamHandle stream = reinterpret_cast<StreamHandle>(stream_ptr);
        auto st = Conv2DCudaLaunch(X, W, Bptr, Y, attrs, stream, Zptr, &ws);
        throw_if_bad(st, "forward");
      },
    py::arg("x_ptr"), py::arg("x_shape"),
    py::arg("w_ptr"), py::arg("w_shape"),
    py::arg("y_ptr"), py::arg("y_shape"),
    py::arg("bias_ptr") = py::none(),
    py::arg("z_ptr")    = py::none(),
    py::arg("attrs")    = Conv2DAttrs{},
    py::arg("stream")   = static_cast<uintptr_t>(0),
    // workspace args (keyword-only 권장)
    py::arg("dCol_ptr")   = static_cast<uintptr_t>(0),
    py::arg("W_KC_ptr")   = static_cast<uintptr_t>(0),
    py::arg("Y_tmp_ptr")  = static_cast<uintptr_t>(0),
    py::arg("Z_rows_ptr") = static_cast<uintptr_t>(0)
  );

  // ========================= backward =========================
  // backward(..., z_ptr,z_shape, ..., *, dCol_ptr, dTmp_ptr, W_CK_ptr, dWpack_ptr, dY_HT_ptr, gy_rows_ptr, Z_rows_ptr)
  m.def("backward",
    [](uintptr_t x_ptr, const std::vector<int64_t>& x_shape,        // [N,Cin,H,W]
       uintptr_t w_ptr, const std::vector<int64_t>& w_shape,        // [Cout,Cin,Kh,Kw]
       uintptr_t dy_ptr, const std::vector<int64_t>& dy_shape,      // [N,Cout,Hout,Wout]
       uintptr_t z_ptr,  const std::vector<int64_t>& z_shape,       // [N,Cout,Hout,Wout]
       py::object dw_ptr_obj,                                       // int or None
       py::object db_ptr_obj,                                       // int or None
       py::object dx_ptr_obj,                                       // int or None
       Conv2DAttrs attrs,
       uintptr_t stream_ptr,
       // --- workspace pointers (all required to avoid mallocs) ---
       uintptr_t dCol_ptr,
       uintptr_t dTmp_ptr,
       uintptr_t W_CK_ptr,
       uintptr_t dWpack_ptr,
       uintptr_t dY_HT_ptr,
       uintptr_t gy_rows_ptr,
       uintptr_t Z_rows_ptr) {

        Tensor X  = make_tensor_4d(x_ptr,  x_shape);
        Tensor W  = make_tensor_4d(w_ptr,  w_shape);
        Tensor dY = make_tensor_4d(dy_ptr, dy_shape);
        Tensor Z  = make_tensor_4d(z_ptr,  z_shape);

        Tensor *dW=nullptr, *dB=nullptr, *dX=nullptr;
        Tensor dW_t, dB_t, dX_t;

        if (!dw_ptr_obj.is_none()) {
          auto dw_ptr = dw_ptr_obj.cast<uintptr_t>();
          dW_t = make_tensor_4d(dw_ptr, w_shape);
          dW = &dW_t;
        }
        if (!db_ptr_obj.is_none()) {
          auto db_ptr = db_ptr_obj.cast<uintptr_t>();
          dB_t = make_tensor_1d(db_ptr, w_shape.at(0)); // Cout
          dB = &dB_t;
        }
        if (!dx_ptr_obj.is_none()) {
          auto dx_ptr = dx_ptr_obj.cast<uintptr_t>();
          dX_t = make_tensor_4d(dx_ptr, x_shape);
          dX = &dX_t;
        }

        // Workspace struct
        Conv2DWorkspaceBwd ws{};
        ws.dCol    = reinterpret_cast<float*>(dCol_ptr);
        ws.dTmp    = reinterpret_cast<float*>(dTmp_ptr);
        ws.W_CK    = reinterpret_cast<float*>(W_CK_ptr);
        ws.dWpack  = reinterpret_cast<float*>(dWpack_ptr);
        ws.dY_HT   = reinterpret_cast<float*>(dY_HT_ptr);
        ws.gy_rows = reinterpret_cast<float*>(gy_rows_ptr);
        ws.Z_rows  = reinterpret_cast<float*>(Z_rows_ptr);

        // 필수성 체크
        if (!ws.dCol || !ws.dTmp) {
          throw std::invalid_argument("[_ops_conv2d.backward] workspace dCol_ptr and dTmp_ptr are required");
        }
        if (!ws.gy_rows || !ws.Z_rows) {
          throw std::invalid_argument("[_ops_conv2d.backward] workspace gy_rows_ptr and Z_rows_ptr are required");
        }
        // dX 경로면 W_CK, dY_HT 필요
        // dW 경로면 dWpack 필요
        if (dX && (!ws.W_CK || !ws.dY_HT)) {
          throw std::invalid_argument("[_ops_conv2d.backward] gX path requires W_CK_ptr and dY_HT_ptr workspaces");
        }
        if (dW && (!ws.dWpack)) {
          throw std::invalid_argument("[_ops_conv2d.backward] gW path requires dWpack_ptr workspace");
        }

        StreamHandle stream = reinterpret_cast<StreamHandle>(stream_ptr);
        auto st = Conv2DCudaBackwardLaunch(X, W, dY, Z, dW, dB, dX, attrs, stream, &ws);
        throw_if_bad(st, "backward");
      },
    py::arg("x_ptr"),  py::arg("x_shape"),
    py::arg("w_ptr"),  py::arg("w_shape"),
    py::arg("dy_ptr"), py::arg("dy_shape"),
    py::arg("z_ptr"),  py::arg("z_shape"),
    py::arg("dw_ptr") = py::none(),
    py::arg("db_ptr") = py::none(),
    py::arg("dx_ptr") = py::none(),
    py::arg("attrs")  = Conv2DAttrs{},
    py::arg("stream") = static_cast<uintptr_t>(0),
    // workspace args (keyword-only 권장)
    py::arg("dCol_ptr")    = static_cast<uintptr_t>(0),
    py::arg("dTmp_ptr")    = static_cast<uintptr_t>(0),
    py::arg("W_CK_ptr")    = static_cast<uintptr_t>(0),
    py::arg("dWpack_ptr")  = static_cast<uintptr_t>(0),
    py::arg("dY_HT_ptr")   = static_cast<uintptr_t>(0),
    py::arg("gy_rows_ptr") = static_cast<uintptr_t>(0),
    py::arg("Z_rows_ptr")  = static_cast<uintptr_t>(0)
  );
}
