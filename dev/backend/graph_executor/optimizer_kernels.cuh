#pragma once

#include "op_structs.cuh"

void optimizer_update_cuda(
    float* param, float* grad,
    float* velocity, float* m, float* v,  // momentum/adam
    float learning_rate, float beta1, float beta2, float epsilon,
    int size,
    OptimizerType opt_type,
    int timestep = 1  // Adam용
);
