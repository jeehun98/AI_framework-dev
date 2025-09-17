#!/usr/bin/env bash
set -euo pipefail
app=${1:-bench_gemm}
metrics="sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,lts__t_sectors_srcunit_tex_op_read.sum,smsp__inst_executed_pipe_fma.sum"
build_dir=build
[ -x "${build_dir}/${app}" ] || { echo "Run build first."; exit 1; }
ncu --set full --metrics ${metrics} ${build_dir}/${app}
