#pragma once
#include <vector>
#include <random>
#include <cmath>
inline float randf(std::mt19937& g){ std::uniform_real_distribution<float> d(-1.f,1.f); return d(g); }
inline bool allclose(const std::vector<float>& a,const std::vector<float>& b, float atol=1e-5f, float rtol=1e-4f){
  if(a.size()!=b.size()) return false;
  for(size_t i=0;i<a.size();++i){
    float diff = std::fabs(a[i]-b[i]);
    float tol = atol + rtol*std::fabs(b[i]);
    if(diff>tol) return false;
  }
  return true;
}
