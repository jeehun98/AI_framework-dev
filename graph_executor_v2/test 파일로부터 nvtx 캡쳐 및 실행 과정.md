
nsys profile --trace=cuda,nvtx -o run_trace --force-overwrite=true python smoke_static_dynamic_optim_and_pool.py

ncu --target-processes all python smoke_gemm_binding.py
