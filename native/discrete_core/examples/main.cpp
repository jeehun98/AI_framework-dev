#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include "discrete/bernoulli.hpp"
#include "discrete/binomial.hpp"

static void stats(const float* x, size_t n, double& mu, double& sd){
    long double m=0.0L, m2=0.0L;
    for(size_t i=0;i<n;++i){ long double d=x[i]-m; m+=d/(i+1); m2+=d*(x[i]-m); }
    mu=(double)m; sd=(n>1)? std::sqrt((double)(m2/(n-1))) : 0.0;
}

int main(){
    const size_t N = 1<<20; // 1M
    std::vector<float> buf(N);
    double mu, sd;

    // Bernoulli p=0.3
    auto t0=std::chrono::high_resolution_clock::now();
    disc::bernoulli_cpu(buf.data(), N, 0.3f, 42);
    auto t1=std::chrono::high_resolution_clock::now();
    stats(buf.data(), N, mu, sd);
    std::chrono::duration<double,std::milli> ms=t1-t0;
    std::cout << "[Bernoulli CPU] time="<<ms.count()<<" ms mean~"<<mu<<" var~"<<sd*sd<<"\n";

#ifdef DISCRETE_WITH_CUDA
    t0=std::chrono::high_resolution_clock::now();
    disc::bernoulli_cuda(buf.data(), N, 0.3f, 42);
    t1=std::chrono::high_resolution_clock::now();
    stats(buf.data(), N, mu, sd);
    ms=t1-t0;
    std::cout << "[Bernoulli CUDA] time="<<ms.count()<<" ms mean~"<<mu<<" var~"<<sd*sd<<"\n";
#endif

    // Binomial n=10, p=0.2
    t0=std::chrono::high_resolution_clock::now();
    disc::binomial_cpu(buf.data(), N, 10, 0.2f, 123);
    t1=std::chrono::high_resolution_clock::now();
    stats(buf.data(), N, mu, sd);
    ms=t1-t0;
    std::cout << "[Binomial  CPU] time="<<ms.count()<<" ms mean~"<<mu<<" var~"<<sd*sd<<"\n";

#ifdef DISCRETE_WITH_CUDA
    t0=std::chrono::high_resolution_clock::now();
    disc::binomial_cuda(buf.data(), N, 10, 0.2f, 123);
    t1=std::chrono::high_resolution_clock::now();
    stats(buf.data(), N, mu, sd);
    ms=t1-t0;
    std::cout << "[Binomial  CUDA] time="<<ms.count()<<" ms mean~"<<mu<<" var~"<<sd*sd<<"\n";
#endif

    return 0;
}
