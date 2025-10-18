#include <cuda_runtime.h>
#include "../api/epilogue.h"

namespace epi {
Status run_fp32(const Plan& plan, const Tensors& ts, DType bdt, void* stream);
Status run_fp16(const Plan& plan, const Tensors& ts, DType bdt, void* stream);

Status run(const Plan& plan, const Tensors& ts,
           DType xdt, DType ydt, DType bdt, void* stream){
  if (xdt==DType::F32 && ydt==DType::F32) return run_fp32(plan, ts, bdt, stream);
  if (xdt==DType::F16 && ydt==DType::F16) return run_fp16(plan, ts, bdt, stream);
  return {false, "Unsupported dtype combo (MVP supports F32->F32 or F16->F16)"};
}

} // namespace epi
