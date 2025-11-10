// _common/shim/layout.hpp
#pragma once
#include <cstddef>
namespace ai::cuda::shim {
inline bool valid_ld_rowmajor(int rows,int cols,int ld){
  if (rows<=0||cols<=0) return false;
  if (ld==0) return true;
  return ld>=cols;
}
inline int resolve_ld(int ld,int fallback_cols){ return (ld==0)? fallback_cols: ld; }
} // ns
