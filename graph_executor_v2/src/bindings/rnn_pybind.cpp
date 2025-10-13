// src/bindings/rnn_pybind.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#ifdef BUILD_STANDALONE_OPS
  #include "backends/cuda/ops/_common/shim/ai_shim.hpp"
#else
  #include "ai/tensor.hpp"
  #include "ai/dispatch.hpp"
#endif

#include "backends/cuda/ops/rnn/api.hpp"

namespace py = pybind11;
using namespace ai;

// ----------------- helpers: contiguous row-major Tensor makers -----------------
static Tensor make_tensor_3d(uintptr_t ptr_u64,
                             const std::vector<int64_t>& shape, // [N,T,I] or [N,T,H]
                             DType dtype = DType::F32,
                             Device dev  = Device::CUDA) {
  if (shape.size() != 3) throw std::invalid_argument("shape must be 3D");
  Tensor t;
  t.data = reinterpret_cast<void*>(ptr_u64);
  t.device = dev;
  t.device_index = 0;
  t.desc.dtype = dtype;
  t.desc.layout = Layout::RowMajor;
  t.desc.shape = shape;
  // contiguous row-major stride
  t.desc.stride.resize(3);
  t.desc.stride[2] = 1;
  t.desc.stride[1] = shape[2] * t.desc.stride[2];
  t.desc.stride[0] = shape[1] * t.desc.stride[1];
  return t;
}

static Tensor make_tensor_2d(uintptr_t ptr_u64,
                             const std::vector<int64_t>& shape, // [M,N]
                             DType dtype = DType::F32,
                             Device dev  = Device::CUDA) {
  if (shape.size() != 2) throw std::invalid_argument("shape must be 2D");
  Tensor t;
  t.data = reinterpret_cast<void*>(ptr_u64);
  t.device = dev;
  t.device_index = 0;
  t.desc.dtype = dtype;
  t.desc.layout = Layout::RowMajor;
  t.desc.shape = shape;
  t.desc.stride.resize(2);
  t.desc.stride[1] = 1;
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
      std::string("[_ops_rnn::") + where + "] failed with Status=" +
      std::to_string(static_cast<int>(st))
    );
  }
}

PYBIND11_MODULE(_ops_rnn, m) {
  m.attr("__package__") = "graph_executor_v2.ops";
  m.doc() = R"(Independent RNN (Elman) CUDA ops binding (standalone shim compatible)
- Forward/Backward support fused epilogue (act, bias) and Z(pre) save/consume.
- All tensors are float32, row-major contiguous.
- Shapes:
  * X : [N,T,I], Wx: [I,H], Wh: [H,H], h0: [N,H], B(optional): [H]
  * Y/Z: [N,T,H]
- Workspaces:
  * Forward:  XH_cat[N,I+H], Y_rows[N,H], W_cat[I+H,H], (optional) Z_rows[N,H]
  * Backward: XH_cat[N,I+H], G_rows[N,H], Z_rows[N,H], W_cat[I+H,H],
              dXH_cat[N,I+H], dWcat[I+H,H], TmpW[I+H,H]
)";
  // conv2d처럼 동일하게 설정 (중복 줄도 동일 톤로 유지해도 무방)
  m.attr("__package__") = "graph_executor_v2.ops";
  m.doc() = R"(Independent rnn CUDA ops binding ... )";

  // ===== re-export common types to avoid duplicate registration =====
  py::module_ common = py::module_::import("graph_executor_v2.ops._ops_common");
  m.attr("ActKind") = common.attr("ActKind");
  // (필요하면 Device/DType/Layout/TensorDesc/Tensor 등도 같은 방식으로 재노출 가능)

  // ----- RnnAttrs -----
  py::class_<RnnAttrs>(m, "RnnAttrs")
    .def(py::init<>())
    .def_readwrite("act",         &RnnAttrs::act)
    .def_readwrite("leaky_slope", &RnnAttrs::leaky_slope)
    .def_readwrite("with_bias",   &RnnAttrs::with_bias)
    .def_readwrite("save_z",      &RnnAttrs::save_z);

  // ========================= forward =========================
  // forward(..., z_ptr=None, attrs, stream, *, XH_cat_ptr=0, Y_rows_ptr=0, W_cat_ptr=0, Z_rows_ptr=0)
  // attrs.save_z=True면 z_ptr과 Z_rows_ptr 필수
  m.def("forward",
    [](uintptr_t x_ptr,  const std::vector<int64_t>& x_shape,   // [N,T,I]
       uintptr_t wx_ptr, const std::vector<int64_t>& wx_shape,  // [I,H]
       uintptr_t wh_ptr, const std::vector<int64_t>& wh_shape,  // [H,H]
       uintptr_t h0_ptr, const std::vector<int64_t>& h0_shape,  // [N,H]
       uintptr_t y_ptr,  const std::vector<int64_t>& y_shape,   // [N,T,H]
       py::object b_ptr_obj,                                     // int or None (B=[H])
       py::object z_ptr_obj,                                     // int or None (Z_saved=[N,T,H])
       RnnAttrs attrs,
       uintptr_t stream_ptr,
       // --- workspace pointers (optional, default 0=none) ---
       uintptr_t XH_cat_ptr,
       uintptr_t Y_rows_ptr,
       uintptr_t W_cat_ptr,
       uintptr_t Z_rows_ptr) {

        Tensor X  = make_tensor_3d(x_ptr,  x_shape);
        Tensor Wx = make_tensor_2d(wx_ptr, wx_shape);
        Tensor Wh = make_tensor_2d(wh_ptr, wh_shape);
        Tensor h0 = make_tensor_2d(h0_ptr, h0_shape);
        Tensor Y  = make_tensor_3d(y_ptr,  y_shape);

        const Tensor* Bptr = nullptr;
        Tensor B;
        if (!b_ptr_obj.is_none()) {
          auto b_ptr = b_ptr_obj.cast<uintptr_t>();
          B = make_tensor_1d(b_ptr, wx_shape.at(1)); // H
          Bptr = &B;
        }

        Tensor* Zptr = nullptr;
        Tensor Ztmp;
        if (!z_ptr_obj.is_none()) {
          auto z_ptr = z_ptr_obj.cast<uintptr_t>();
          Ztmp = make_tensor_3d(z_ptr, y_shape); // Z_saved shape == Y shape
          Zptr = &Ztmp;
        }

        // Workspace struct
        RnnWorkspaceFwd ws{};
        ws.XH_cat = reinterpret_cast<float*>(XH_cat_ptr);
        ws.Y_rows = reinterpret_cast<float*>(Y_rows_ptr);
        ws.W_cat  = reinterpret_cast<float*>(W_cat_ptr);
        ws.Z_rows = reinterpret_cast<float*>(Z_rows_ptr);

        // 필수성 체크
        if (!ws.XH_cat || !ws.Y_rows || !ws.W_cat) {
          throw std::invalid_argument("[_ops_rnn.forward] workspace pointers (XH_cat, Y_rows, W_cat) are required");
        }
        if (attrs.save_z && (Zptr == nullptr)) {
          throw std::invalid_argument("[_ops_rnn.forward] attrs.save_z=True requires z_ptr (Z_saved)");
        }
        if (attrs.save_z && (!ws.Z_rows)) {
          throw std::invalid_argument("[_ops_rnn.forward] attrs.save_z=True requires Z_rows workspace");
        }

        StreamHandle stream = reinterpret_cast<StreamHandle>(stream_ptr);
        auto st = RnnCudaLaunch(X, Wx, Wh, Bptr, h0, Y, attrs, stream, Zptr, &ws);
        throw_if_bad(st, "forward");
      },
    py::arg("x_ptr"),  py::arg("x_shape"),
    py::arg("wx_ptr"), py::arg("wx_shape"),
    py::arg("wh_ptr"), py::arg("wh_shape"),
    py::arg("h0_ptr"), py::arg("h0_shape"),
    py::arg("y_ptr"),  py::arg("y_shape"),
    py::arg("bias_ptr") = py::none(),
    py::arg("z_ptr")    = py::none(),
    py::arg("attrs")    = RnnAttrs{},
    py::arg("stream")   = static_cast<uintptr_t>(0),
    // workspace args (keyword-only 권장)
    py::arg("XH_cat_ptr") = static_cast<uintptr_t>(0),
    py::arg("Y_rows_ptr") = static_cast<uintptr_t>(0),
    py::arg("W_cat_ptr")  = static_cast<uintptr_t>(0),
    py::arg("Z_rows_ptr") = static_cast<uintptr_t>(0)
  );

  // ========================= backward =========================
  // backward(..., z_ptr,z_shape, ..., *, XH_cat_ptr, G_rows_ptr, Z_rows_ptr, W_cat_ptr, dXH_cat_ptr, dWcat_ptr, TmpW_ptr)
  m.def("backward",
    [](uintptr_t x_ptr,   const std::vector<int64_t>& x_shape,    // [N,T,I]
       uintptr_t wx_ptr,  const std::vector<int64_t>& wx_shape,   // [I,H]
       uintptr_t wh_ptr,  const std::vector<int64_t>& wh_shape,   // [H,H]
       uintptr_t h0_ptr,  const std::vector<int64_t>& h0_shape,   // [N,H]
       uintptr_t dy_ptr,  const std::vector<int64_t>& dy_shape,   // [N,T,H]
       uintptr_t z_ptr,   const std::vector<int64_t>& z_shape,    // [N,T,H]
       py::object dwx_ptr_obj,                                    // int or None
       py::object dwh_ptr_obj,                                    // int or None
       py::object db_ptr_obj,                                     // int or None
       py::object dh0_ptr_obj,                                    // int or None
       py::object dx_ptr_obj,                                     // int or None
       RnnAttrs attrs,
       uintptr_t stream_ptr,
       // --- workspace pointers (all required to avoid mallocs) ---
       uintptr_t XH_cat_ptr,
       uintptr_t G_rows_ptr,
       uintptr_t Z_rows_ptr,
       uintptr_t W_cat_ptr,
       uintptr_t dXH_cat_ptr,
       uintptr_t dWcat_ptr,
       uintptr_t TmpW_ptr) {

        // Inputs
        Tensor X   = make_tensor_3d(x_ptr,  x_shape);
        Tensor Wx  = make_tensor_2d(wx_ptr, wx_shape);
        Tensor Wh  = make_tensor_2d(wh_ptr, wh_shape);
        Tensor h0  = make_tensor_2d(h0_ptr, h0_shape);
        Tensor dYp = make_tensor_3d(dy_ptr, dy_shape);
        Tensor Z   = make_tensor_3d(z_ptr,  z_shape);

        // Optional outputs
        Tensor *dWx=nullptr, *dWh=nullptr, *dB=nullptr, *dh0=nullptr, *dX=nullptr;
        Tensor dWx_t, dWh_t, dB_t, dh0_t, dX_t;

        if (!dwx_ptr_obj.is_none()) {
          auto p = dwx_ptr_obj.cast<uintptr_t>();
          dWx_t = make_tensor_2d(p, wx_shape); // [I,H]
          dWx = &dWx_t;
        }
        if (!dwh_ptr_obj.is_none()) {
          auto p = dwh_ptr_obj.cast<uintptr_t>();
          dWh_t = make_tensor_2d(p, wh_shape); // [H,H]
          dWh = &dWh_t;
        }
        if (!db_ptr_obj.is_none()) {
          auto p = db_ptr_obj.cast<uintptr_t>();
          dB_t = make_tensor_1d(p, wx_shape.at(1)); // [H]
          dB = &dB_t;
        }
        if (!dh0_ptr_obj.is_none()) {
          auto p = dh0_ptr_obj.cast<uintptr_t>();
          dh0_t = make_tensor_2d(p, h0_shape); // [N,H]
          dh0 = &dh0_t;
        }
        if (!dx_ptr_obj.is_none()) {
          auto p = dx_ptr_obj.cast<uintptr_t>();
          dX_t = make_tensor_3d(p, x_shape); // [N,T,I]
          dX = &dX_t;
        }

        // Workspaces
        RnnWorkspaceBwd ws{};
        ws.XH_cat  = reinterpret_cast<float*>(XH_cat_ptr);
        ws.G_rows  = reinterpret_cast<float*>(G_rows_ptr);
        ws.Z_rows  = reinterpret_cast<float*>(Z_rows_ptr);
        ws.W_cat   = reinterpret_cast<float*>(W_cat_ptr);
        ws.dXH_cat = reinterpret_cast<float*>(dXH_cat_ptr);
        ws.dWcat   = reinterpret_cast<float*>(dWcat_ptr);
        ws.TmpW    = reinterpret_cast<float*>(TmpW_ptr);

        // 필수성 체크
        if (!ws.XH_cat || !ws.G_rows || !ws.Z_rows || !ws.W_cat || !ws.dXH_cat || !ws.dWcat || !ws.TmpW) {
          throw std::invalid_argument("[_ops_rnn.backward] all workspace pointers (XH_cat,G_rows,Z_rows,W_cat,dXH_cat,dWcat,TmpW) are required");
        }

        StreamHandle stream = reinterpret_cast<StreamHandle>(stream_ptr);
        auto st = RnnCudaBackwardLaunch(X, Wx, Wh, /*B=*/nullptr, h0, dYp, Z,
                                        dWx, dWh, dB, dh0, dX,
                                        attrs, stream, &ws);
        throw_if_bad(st, "backward");
      },
    py::arg("x_ptr"),   py::arg("x_shape"),
    py::arg("wx_ptr"),  py::arg("wx_shape"),
    py::arg("wh_ptr"),  py::arg("wh_shape"),
    py::arg("h0_ptr"),  py::arg("h0_shape"),
    py::arg("dy_ptr"),  py::arg("dy_shape"),
    py::arg("z_ptr"),   py::arg("z_shape"),
    py::arg("dwx_ptr")  = py::none(),
    py::arg("dwh_ptr")  = py::none(),
    py::arg("db_ptr")   = py::none(),
    py::arg("dh0_ptr")  = py::none(),
    py::arg("dx_ptr")   = py::none(),
    py::arg("attrs")    = RnnAttrs{},
    py::arg("stream")   = static_cast<uintptr_t>(0),
    // workspace args (keyword-only 권장)
    py::arg("XH_cat_ptr")  = static_cast<uintptr_t>(0),
    py::arg("G_rows_ptr")  = static_cast<uintptr_t>(0),
    py::arg("Z_rows_ptr")  = static_cast<uintptr_t>(0),
    py::arg("W_cat_ptr")   = static_cast<uintptr_t>(0),
    py::arg("dXH_cat_ptr") = static_cast<uintptr_t>(0),
    py::arg("dWcat_ptr")   = static_cast<uintptr_t>(0),
    py::arg("TmpW_ptr")    = static_cast<uintptr_t>(0)
  );
}
