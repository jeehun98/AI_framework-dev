#ifndef RUN_GRAPH_CUH
#define RUN_GRAPH_CUH

extern "C" void run_graph_cuda(
    int* E, int E_len,
    int* shapes, int shapes_len,
    float* W, float* b,
    int W_rows, int W_cols,
    float* x, float* out
);

#endif
