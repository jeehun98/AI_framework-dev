// run_graph_backward.cu
#include <iostream>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <unordered_map>

#include "run_graph.cuh"
#include "backward_kernels_optimized.cuh"
#include "activation_backward.cuh"
#include "cnn_kernels.cuh"
#include "op_structs.cuh"
#include "loss_kernels.cuh"
#include "transpose.cuh"

#ifndef TILE_WIDTH
#define TILE_WIDTH 16
#endif

void run_graph_backward(
    const std::vector<OpStruct>& E,
    std::unordered_map<std::string, float*>& tensors,
    std::unordered_map<std::string, Shape>& shapes,
    std::unordered_map<std::string, float*>& gradients,
    const std::string& final_output_id,
    int batch_size)
{
    std::string grad_start_id = final_output_id;

    // 1. 먼저 LOSS 연산 처리 및 초기 gradient 생성
    if (!E.empty() && E.back().op_type == LOSS) {
        const OpStruct& loss_op = E.back();

        std::string loss_type = loss_op.extra_params.loss_type;
        std::string label_id = loss_op.extra_params.label_id;
        float* y_true = tensors[label_id];
        float* y_pred = tensors[loss_op.input_id];  // e.g., activation_out

        Shape shape = shapes[loss_op.input_id];
        int sz = shape.rows * shape.cols;
        float* dL_dy = nullptr;
        cudaMalloc(&dL_dy, sz * sizeof(float));

        if (loss_type == "bce") {
            bce_loss_backward<<<(sz + 255)/256, 256>>>(y_true, y_pred, dL_dy, sz);
            cudaDeviceSynchronize();

            float* host_loss_grad = new float[sz];
            cudaMemcpy(host_loss_grad, dL_dy, sizeof(float) * sz, cudaMemcpyDeviceToHost);

            // ✅ 여기서 디버깅 코드 추가
            printf("[DEBUG][LOSS] grad_out from BCE backward (first 10): ");
            for (int i = 0; i < std::min(sz, 10); ++i)
                printf("%.5f ", host_loss_grad[i]);
            printf("\n");
        }

        // ✅ gradient 시작점은 loss의 입력 activation 출력
        grad_start_id = loss_op.input_id;
        gradients[loss_op.input_id] = dL_dy;
        gradients[loss_op.output_id] = dL_dy;
    }

    // 2. 나머지 역전파 루프
    for (auto it = E.rbegin(); it != E.rend(); ++it) {
        const OpStruct& op = *it;
        float* input = tensors[op.input_id];
        float* param = (!op.param_id.empty() && tensors.count(op.param_id)) ? tensors[op.param_id] : nullptr;
        float* grad_out = gradients[op.output_id];

        Shape in_shape = shapes[op.input_id];
        Shape out_shape = shapes[op.output_id];
        int in_rows = in_shape.rows, in_cols = in_shape.cols;
        int out_rows = out_shape.rows, out_cols = out_shape.cols;

        float* grad_input = nullptr;
        if (op.op_type != FLATTEN && op.op_type != LOSS)
            cudaMalloc(&grad_input, in_rows * in_cols * sizeof(float));

        printf("[INFO] op_type=%d, output_id=%s, input_id=%s\n", op.op_type, op.output_id.c_str(), op.input_id.c_str());
        printf("[INFO] grad_out ptr = %p, grad_input ptr = %p\n", grad_out, grad_input);

        // ✅ grad_out 값 디버그 (앞 10개)
        if (grad_out) {
            float* debug_gradout = new float[10];
            cudaMemcpy(debug_gradout, grad_out, sizeof(float) * 10, cudaMemcpyDeviceToHost);
            printf("[DEBUG] grad_out values (first 10): ");
            for (int i = 0; i < 10; ++i) printf("%.5f ", debug_gradout[i]);
            printf("\n");
            delete[] debug_gradout;
        }

        // ✅ grad_input 값 디버그 (초기 상태, 앞 10개)
        if (grad_input) {
            float* debug_gradinput = new float[10];
            cudaMemcpy(debug_gradinput, grad_input, sizeof(float) * 10, cudaMemcpyDeviceToHost);
            printf("[DEBUG] grad_input initial values (first 10): ");
            for (int i = 0; i < 10; ++i) printf("%.5f ", debug_gradinput[i]);
            printf("\n");
            delete[] debug_gradinput;
        }


        switch (op.op_type) {
            case MATMUL: {
                if (!param) break;

                float* W_T = nullptr;
                cudaMalloc(&W_T, sizeof(float) * in_cols * out_cols);
                launch_transpose(param, W_T, in_cols, out_cols);
                cudaMemset(grad_input, 0, sizeof(float) * out_rows * in_cols);

                int total_threads = out_rows * in_cols;
                if (total_threads <= 1024) {
                    printf("[run_graph_backward] launching matmul_backward_input_simple | M=%d N=%d K=%d\n", out_rows, out_cols, in_cols);

                    int blockSize = std::min(32, total_threads);
                    int gridSize = (total_threads + blockSize - 1) / blockSize;
                    matmul_backward_input_simple<<<gridSize, blockSize>>>(grad_out, W_T, grad_input, out_rows, out_cols, in_cols);
                    cudaError_t err = cudaGetLastError();
                    if (err != cudaSuccess) {
                        printf("CUDA kernel launch failed input_simple: %s\n", cudaGetErrorString(err));
                    }
                } else {
                    printf("[run_graph_backward] launching matmul_backward_input_shared | M=%d N=%d K=%d\n", out_rows, out_cols, in_cols);

                    dim3 blockDim(16, 16);
                    dim3 gridDim((in_cols + 15) / 16, (out_rows + 15) / 16);
                    matmul_backward_input_shared<<<gridDim, blockDim>>>(grad_out, W_T, grad_input, out_rows, out_cols, in_cols);
                    cudaError_t err = cudaGetLastError();
                    if (err != cudaSuccess) {
                        printf("CUDA kernel launch failed input_shared: %s\n", cudaGetErrorString(err));
                    }
                }

                cudaFree(W_T);

                float* input_T = nullptr;
                cudaMalloc(&input_T, sizeof(float) * in_rows * in_cols);
                launch_transpose(input, input_T, in_rows, in_cols);

                float* grad_weight = nullptr;
                cudaMalloc(&grad_weight, in_cols * out_cols * sizeof(float));

                dim3 blockDimW(16, 16);
                dim3 gridDimW((out_cols + 15) / 16, (in_cols + 15) / 16);
                matmul_backward_weight_shared<<<gridDimW, blockDimW>>>(
                    input_T, grad_out, grad_weight, in_cols, out_cols, in_rows);

                cudaDeviceSynchronize();

                gradients[op.param_id] = grad_weight;
                cudaFree(input_T);
                break;
            }

            case ADD: {
                add_backward_input<<<(out_rows * out_cols + 255) / 256, 256>>>(grad_out, grad_input, out_rows * out_cols);
                float* grad_bias = nullptr;
                cudaMalloc(&grad_bias, out_cols * sizeof(float));
                add_backward_bias<<<(out_cols + 255)/256, 256>>>(grad_out, grad_bias, out_rows, out_cols);
                gradients[op.param_id] = grad_bias;
                break;
            }

            case SIGMOID:
            case RELU:
            case TANH: 
            // 🔍 [1] grad_out & grad_input 디버깅 (커널 실행 전)
                {
                    float* debug_gradout_pre = new float[10];
                    float* debug_gradinput_pre = new float[10];

                    cudaMemcpy(debug_gradout_pre, grad_out, sizeof(float) * 10, cudaMemcpyDeviceToHost);
                    cudaMemcpy(debug_gradinput_pre, grad_input, sizeof(float) * 10, cudaMemcpyDeviceToHost);

                    printf("[DEBUG][PRE] grad_out (first 10): ");
                    for (int i = 0; i < 10; ++i) printf("%.5f ", debug_gradout_pre[i]);
                    printf("\n");

                    printf("[DEBUG][PRE] grad_input (first 10): ");
                    for (int i = 0; i < 10; ++i) printf("%.5f ", debug_gradinput_pre[i]);
                    printf("\n");

                    delete[] debug_gradout_pre;
                    delete[] debug_gradinput_pre;
                }

                // ✅ [2] 커널 실행
                activation_backward<<<(out_rows * out_cols + 255) / 256, 256>>>(
                    grad_out, tensors[op.output_id], grad_input, out_rows, out_cols, op.op_type);

                cudaDeviceSynchronize();  // 필수

                // 🔍 [3] grad_out & grad_input 디버깅 (커널 실행 후)
                {
                    float* debug_output = new float[10];
                    float* debug_gradout = new float[10];
                    float* debug_gradinput = new float[10];

                    cudaMemcpy(debug_output, tensors[op.output_id], sizeof(float) * 10, cudaMemcpyDeviceToHost);
                    cudaMemcpy(debug_gradout, grad_out, sizeof(float) * 10, cudaMemcpyDeviceToHost);
                    cudaMemcpy(debug_gradinput, grad_input, sizeof(float) * 10, cudaMemcpyDeviceToHost);

                    printf("[DEBUG][POST] activation output (first 10): ");
                    for (int i = 0; i < 10; ++i) printf("%.5f ", debug_output[i]);
                    printf("\n");

                    printf("[DEBUG][POST] grad_out (first 10): ");
                    for (int i = 0; i < 10; ++i) printf("%.5f ", debug_gradout[i]);
                    printf("\n");

                    printf("[DEBUG][POST] grad_input (first 10): ");
                    for (int i = 0; i < 10; ++i) printf("%.5f ", debug_gradinput[i]);
                    printf("\n");

                    delete[] debug_output;
                    delete[] debug_gradout;
                    delete[] debug_gradinput;
                }


            case FLATTEN:
                gradients[op.input_id] = grad_out;
                break;

            case LOSS:
                // 이미 처리 완료
                break;
        }

        if (grad_input == nullptr && op.op_type != FLATTEN && op.op_type != LOSS) {
            printf("[ERROR] grad_input is NULL for op_type=%d, input_id=%s\n", op.op_type, op.input_id.c_str());
        }

        if (grad_input && op.op_type != FLATTEN && op.op_type != LOSS)
            gradients[op.input_id] = grad_input;

        if (!op.param_id.empty() && gradients.count(op.param_id)) {
            float* grad = gradients[op.param_id];
            Shape shape = shapes[op.param_id];
            int size = shape.rows * shape.cols;
            float* host_grad = new float[size];
            cudaMemcpy(host_grad, grad, sizeof(float) * size, cudaMemcpyDeviceToHost);
            float min_val = host_grad[0], max_val = host_grad[0], sum = 0.0f;
            for (int i = 0; i < size; ++i) {
                min_val = fminf(min_val, host_grad[i]);
                max_val = fmaxf(max_val, host_grad[i]);
                sum += host_grad[i];
            }
            std::cout << "[GRADIENT] " << op.param_id << " grad → min=" << min_val
                      << ", max=" << max_val << ", mean=" << sum / size << std::endl;

            for (int i = 0; i < std::min(size, 10); ++i)
                std::cout << "  [" << i << "] = " << host_grad[i] << std::endl;
            delete[] host_grad;
        }
    }
}
