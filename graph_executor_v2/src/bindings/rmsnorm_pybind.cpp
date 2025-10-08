// src/bindings/rmsnorm_pybind.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#ifdef BUILD_STANDALONE_OPS
  #include "backends/cuda/ops/_common/shim/ai_shim.hpp"
#else
  #include "ai/tensor.hpp"
  #include "ai/dispatch.hpp"
#endif

#include "backends/cuda/ops/rmsnorm/api.hpp"

namespace py = pybind11;
using namespace ai;

static Tensor make_tensor_nd(uintptr_t ptr_u64,
                             const std::vector<int64_t>& shape,
                             DType dtype=DType::F32,
                             Device dev=Device::CUDA)
{
  Tensor t;
  t.data = reinterpret_cast<void*>(ptr_u64);
  t.device = dev;
  t.device_index = 0;
  t.desc.dtype  = dtype;
  t.desc.layout = Layout::RowMajor;
  t.desc.shape  = shape;
  const size_t R=shape.size();
  t.desc.stride.resize(R);
  if (R){
    t.desc.stride[R-1]=1;
    for (int i=(int)R-2;i>=0;--i)
      t.desc.stride[(size_t)i]=t.desc.stride[(size_t)i+1]*shape[(size_t)i+1];
  }
  return t;
}

static void throw_if_bad(Status st, const char* where){
  if (st!=Status::Ok)
    throw std::runtime_error(std::string(where)+" failed with Status="+std::to_string((int)st));
}

PYBIND11_MODULE(_ops_rmsnorm, m){
  m.doc() = "Independent CUDA RMSNorm (forward/backward)";

  py::class_<RMSNormAttrs>(m, "RMSNormAttrs")
    .def(py::init<>())
    .def_readwrite("eps", &RMSNormAttrs::eps);

  // Forward
  m.def("forward",
    [](uintptr_t x_ptr, const std::vector<int64_t>& x_shape,
       py::object gamma_ptr_or_none, const std::vector<int64_t>& gamma_shape,
       py::object beta_ptr_or_none,  const std::vector<int64_t>& beta_shape,
       uintptr_t y_ptr, const std::vector<int64_t>& y_shape,
       const RMSNormAttrs& attrs, uintptr_t stream_ptr){

        if (x_shape.size()!=2 || y_shape.size()!=2)
          throw std::invalid_argument("X,Y must be [M,N]");
        if (x_shape!=y_shape)
          throw std::invalid_argument("X and Y shape must match");

        const int64_t N = x_shape[1];

        Tensor X = make_tensor_nd(x_ptr, x_shape, DType::F32);
        Tensor Y = make_tensor_nd(y_ptr, y_shape, DType::F32);

        Tensor gT{}, bT{}; const Tensor* gamma=nullptr; const Tensor* beta=nullptr;
        if (!gamma_ptr_or_none.is_none()){
          if (gamma_shape.size()!=1 || gamma_shape[0]!=N)
            throw std::invalid_argument("gamma must be [N]");
          auto gptr = gamma_ptr_or_none.cast<uintptr_t>();
          gT = make_tensor_nd(gptr, {N}, DType::F32); gamma=&gT;
        }
        if (!beta_ptr_or_none.is_none()){
          if (beta_shape.size()!=1 || beta_shape[0]!=N)
            throw std::invalid_argument("beta must be [N]");
          auto bptr = beta_ptr_or_none.cast<uintptr_t>();
          bT = make_tensor_nd(bptr, {N}, DType::F32); beta=&bT;
        }

        auto st = RMSNormCudaLaunch(X, gamma, beta, Y, attrs, reinterpret_cast<StreamHandle>(stream_ptr));
        throw_if_bad(st, "RMSNormCudaLaunch");
      },
    py::arg("x_ptr"), py::arg("x_shape"),
    py::arg("gamma_ptr")=py::none(), py::arg("gamma_shape")=std::vector<int64_t>{},
    py::arg("beta_ptr") =py::none(), py::arg("beta_shape") =std::vector<int64_t>{},
    py::arg("y_ptr"),   py::arg("y_shape"),
    py::arg("attrs"),
    py::arg("stream")=static_cast<uintptr_t>(0)
  );

  // Backward
  m.def("backward",
    [](uintptr_t x_ptr, const std::vector<int64_t>& x_shape,
       py::object gamma_ptr_or_none, const std::vector<int64_t>& gamma_shape,
       uintptr_t dy_ptr, const std::vector<int64_t>& dy_shape,
       uintptr_t dx_ptr, const std::vector<int64_t>& dx_shape,
       py::object dgamma_ptr_or_none, const std::vector<int64_t>& dgamma_shape,
       py::object dbeta_ptr_or_none,  const std::vector<int64_t>& dbeta_shape,
       const RMSNormAttrs& attrs, uintptr_t stream_ptr){

        if (x_shape.size()!=2 || dy_shape.size()!=2 || dx_shape.size()!=2)
          throw std::invalid_argument("X,dY,dX must be [M,N]");
        if (x_shape!=dy_shape || x_shape!=dx_shape)
          throw std::invalid_argument("Shapes of X,dY,dX must match");

        const int64_t N = x_shape[1];

        Tensor X  = make_tensor_nd(x_ptr,  x_shape,  DType::F32);
        Tensor dY = make_tensor_nd(dy_ptr, dy_shape, DType::F32);
        Tensor dX = make_tensor_nd(dx_ptr, dx_shape, DType::F32);

        Tensor gT{}; const Tensor* gamma=nullptr;
        if (!gamma_ptr_or_none.is_none()){
          if (gamma_shape.size()!=1 || gamma_shape[0]!=N)
            throw std::invalid_argument("gamma must be [N]");
          auto gptr = gamma_ptr_or_none.cast<uintptr_t>();
          gT = make_tensor_nd(gptr, {N}, DType::F32); gamma=&gT;
        }

        Tensor dgammaT{}, dbetaT{}; Tensor* dgamma=nullptr; Tensor* dbeta=nullptr;
        if (!dgamma_ptr_or_none.is_none()){
          if (dgamma_shape.size()!=1 || dgamma_shape[0]!=N)
            throw std::invalid_argument("dgamma must be [N]");
          auto gop = dgamma_ptr_or_none.cast<uintptr_t>();
          dgammaT = make_tensor_nd(gop, {N}, DType::F32); dgamma=&dgammaT;
        }
        if (!dbeta_ptr_or_none.is_none()){
          if (dbeta_shape.size()!=1 || dbeta_shape[0]!=N)
            throw std::invalid_argument("dbeta must be [N]");
          auto bop = dbeta_ptr_or_none.cast<uintptr_t>();
          dbetaT = make_tensor_nd(bop, {N}, DType::F32); dbeta=&dbetaT;
        }

        auto st = RMSNormCudaBackwardLaunch(X, gamma, dY, dX, dgamma, dbeta, attrs,
                                            reinterpret_cast<StreamHandle>(stream_ptr));
        throw_if_bad(st, "RMSNormCudaBackwardLaunch");
      },
    py::arg("x_ptr"), py::arg("x_shape"),
    py::arg("gamma_ptr")=py::none(), py::arg("gamma_shape")=std::vector<int64_t>{},
    py::arg("dy_ptr"), py::arg("dy_shape"),
    py::arg("dx_ptr"), py::arg("dx_shape"),
    py::arg("dgamma_ptr")=py::none(), py::arg("dgamma_shape")=std::vector<int64_t>{},
    py::arg("dbeta_ptr") =py::none(), py::arg("dbeta_shape") =std::vector<int64_t>{},
    py::arg("attrs"),
    py::arg("stream")=static_cast<uintptr_t>(0)
  );
}
