// run_graph.cuh
#pragma once

extern "C" void run_graph_cuda(
    int* E, int E_len,
    int* shapes, int shapes_len,
    float* W, float* b,
    int W_rows, int W_cols,
    int activation_type,
    float* out_host);  // ✅ out_host는 host memory
