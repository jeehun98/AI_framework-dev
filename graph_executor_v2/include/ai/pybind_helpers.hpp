#pragma once
#include <pybind11/pybind11.h>
#include "ai/dispatch.hpp"

namespace ai { namespace py {

template<typename RegisterFn>
void bind_op_module(pybind11::module_& m, const char* opname, RegisterFn reg) {
    // 1) 파이썬 함수 등록(선택)
    reg(m);
    // 2) import 시 디스패처에 커널 등록
    ai::dispatch::RegisterKernelsFor(opname); // 내부에서 backends/cuda/register_ops.cpp 의 심볼 실행
}

}} // namespace
