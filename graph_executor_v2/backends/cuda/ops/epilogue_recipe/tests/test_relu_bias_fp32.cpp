#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
#include "../api/epilogue.h"

using namespace epi;

int main(){
  const int M=8, N=16, size=M*N;
  std::vector<float> hx(size), hy(size), hb(N);
  for(int i=0;i<size;++i) hx[i] = (i%11)-5.0f;
  for(int i=0;i<N;++i) hb[i]=0.5f;

  float *dx,*dy,*db;
  cudaMalloc(&dx, size*sizeof(float));
  cudaMalloc(&dy, size*sizeof(float));
  cudaMalloc(&db, N*sizeof(float));
  cudaMemcpy(dx, hx.data(), size*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemset(dy, 0, size*sizeof(float));
  cudaMemcpy(db, hb.data(), N*sizeof(float), cudaMemcpyHostToDevice);

  Plan plan;
  plan.attrs.act=ActKind::ReLU;
  plan.attrs.bias=BiasKind::PerN;
  plan.attrs.dropout=false; plan.attrs.p_drop=0.0f;

  Tensors ts{};
  ts.x=dx; ts.y=dy; ts.bias=db; ts.M=M; ts.N=N; ts.ld_x=N; ts.ld_y=N;
  ts.rng_seed=0x12345678ABCDEF00ull; ts.rng_offset=0ull;

  auto st = run(plan, ts, DType::F32, DType::F32, DType::F32, nullptr);
  if(!st.ok){ printf("run failed: %s\n", st.msg); return 1; }

  hy.resize(size);
  cudaMemcpy(hy.data(), dy, size*sizeof(float), cudaMemcpyDeviceToHost);

  int errors=0;
  for(int m=0;m<M;++m){
    for(int n=0;n<N;++n){
      float v = hx[m*N+n] + hb[n];
      v = v>0.f? v:0.f;
      float got = hy[m*N+n];
      if (fabsf(v-got) > 1e-6f) ++errors;
    }
  }
  printf("OK. errors=%d\n", errors);

  cudaFree(dx); cudaFree(dy); cudaFree(db);
  return errors?1:0;
}
