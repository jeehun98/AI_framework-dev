#pragma once
#include <cuda_runtime.h>

/** Row-wise bias (len = cols): out[r,c] = in[r,c] + bias[c] */
void launch_add_bias_rowwise(const float* in, const float* bias, float* out,
                             int rows, int cols, cudaStream_t stream = 0);

/** Col-wise / Channel bias (len = rows_per_sample):
 *  out[r,c] = in[r,c] + bias[r % rows_per_sample]
 *  rows = batch_size * rows_per_sample
 */
void launch_add_bias_colwise(const float* in, const float* bias, float* out,
                             int rows, int cols, int rows_per_sample,
                             cudaStream_t stream = 0);