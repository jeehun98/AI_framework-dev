// sum_columns_kernel.cuh
#pragma once

__global__ void sum_columns_kernel(const float* input, float* output, int rows, int cols);
