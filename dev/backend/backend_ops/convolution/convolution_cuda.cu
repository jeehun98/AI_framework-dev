#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv2d_forward_kernel(
    const float* input, const float* weights, const float* bias, float* output,
    int batch, int in_c, int in_h, int in_w,
    int out_c, int kernel, int stride, int padding,
    int out_h, int out_w
) {
    int b = blockIdx.x;
    int oc = blockIdx.y;
    int oh = threadIdx.y;
    int ow = threadIdx.x;

    if (oh >= out_h || ow >= out_w) return;

    float value = bias[oc];

    for (int ic = 0; ic < in_c; ++ic) {
        for (int kh = 0; kh < kernel; ++kh) {
            for (int kw = 0; kw < kernel; ++kw) {
                int ih = oh * stride - padding + kh;
                int iw = ow * stride - padding + kw;

                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                    int input_idx = ((b * in_c + ic) * in_h + ih) * in_w + iw;
                    int weight_idx = ((oc * in_c + ic) * kernel + kh) * kernel + kw;
                    value += input[input_idx] * weights[weight_idx];
                }
            }
        }
    }

    int output_idx = ((b * out_c + oc) * out_h + oh) * out_w + ow;
    output[output_idx] = value;
}

torch::Tensor conv2d_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias,
    int stride,
    int padding
) {
    const int batch = input.size(0);
    const int in_c = input.size(1);
    const int in_h = input.size(2);
    const int in_w = input.size(3);
    const int out_c = weights.size(0);
    const int kernel = weights.size(2);
    const int out_h = (in_h + 2 * padding - kernel) / stride + 1;
    const int out_w = (in_w + 2 * padding - kernel) / stride + 1;

    auto output = torch::zeros({batch, out_c, out_h, out_w}, input.options());

    dim3 grid(batch, out_c);
    dim3 block(out_w, out_h);

    conv2d_forward_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weights.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch, in_c, in_h, in_w, out_c, kernel, stride, padding, out_h, out_w
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv2d_forward, "Conv2D forward (CUDA)");
}
