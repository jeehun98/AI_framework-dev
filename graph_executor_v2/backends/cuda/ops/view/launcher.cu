#include "backends/cuda/ops/view/api.hpp"
#include <numeric>
#include <algorithm>

namespace ai {

Status PermuteMakeView(const TensorDesc& in, const std::vector<int>& perm, TensorDesc& out) {
  const int r = (int)in.shape.size();
  if ((int)perm.size() != r) return Status::Invalid;

  std::vector<int> seen(r,0);
  for (int i=0;i<r;++i){
    if (perm[i] < 0 || perm[i] >= r) return Status::Invalid;
    if (seen[perm[i]]) return Status::Invalid;
    seen[perm[i]] = 1;
  }

  out = in;
  out.shape.resize(r);
  out.stride.resize(r);
  for (int i=0;i<r;++i){
    out.shape[i]  = in.shape[perm[i]];
    out.stride[i] = in.stride[perm[i]];
  }
  return Status::Ok;
}

Status Transpose2DMakeView(const TensorDesc& in, int d0, int d1, TensorDesc& out) {
  const int r = (int)in.shape.size();
  if (r < 2) return Status::Invalid;
  if (d0<0) d0 += r; if (d1<0) d1 += r;
  if (d0<0||d0>=r||d1<0||d1>=r||d0==d1) return Status::Invalid;

  std::vector<int> perm(r);
  std::iota(perm.begin(), perm.end(), 0);
  std::swap(perm[d0], perm[d1]);
  return PermuteMakeView(in, perm, out);
}

Status ExpandMakeView(const TensorDesc& in, const std::vector<int64_t>& out_shape, TensorDesc& out) {
  const int rin  = (int)in.shape.size();
  const int rout = (int)out_shape.size();
  const int r = std::max(rin, rout);

  // 뒤축 정렬
  std::vector<int64_t> ish(r,1), istr(r,0), osh(r,1);
  for (int i=0;i<rin;++i){  ish[r-rin+i]  = in.shape[i]; istr[r-rin+i] = in.stride[i]; }
  for (int i=0;i<rout;++i){ osh[r-rout+i] = out_shape[i]; }

  std::vector<int64_t> oshape(r), ostride(r);
  for (int i=0;i<r;++i){
    if (osh[i] == ish[i]) {           // 동일 크기 → stride 유지
      oshape[i]  = osh[i];
      ostride[i] = istr[i];
    } else if (ish[i] == 1 && osh[i] > 1) { // broadcast → stride=0 뷰
      oshape[i]  = osh[i];
      ostride[i] = 0;
    } else {
      return Status::ShapeMismatch;
    }
  }

  out = in;
  out.shape = std::move(oshape);
  out.stride= std::move(ostride);
  return Status::Ok;
}

} // namespace ai
