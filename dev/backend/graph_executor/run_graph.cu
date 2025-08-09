#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cuda_runtime.h>
#include <unordered_map>

#include "run_graph.cuh"
#include "matmul_shared_optimized_kernel.cuh"
#include "activation.cuh"
#include "add_kernel.cuh"
#include "cnn_kernels.cuh"
#include "op_structs.cuh"

#ifndef TILE_WIDTH
#define TILE_WIDTH 16
#endif

#ifndef CUDA_CHECK
#define CUDA_CHECK(x) do { cudaError_t _e=(x); if(_e!=cudaSuccess){ \
  fprintf(stderr,"[CUDA] %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); } } while(0)
#endif

void run_graph_cuda(
    const std::vector<OpStruct>& E,
    std::unordered_map<std::string, float*>& tensors,
    std::unordered_map<std::string, Shape>& shapes,
    float* out_host,
    const std::string& final_output_id,
    int batch_size)
{
    for (const auto& op : E) {
        float* input = tensors[op.input_id];
        float* param = (!op.param_id.empty() && tensors.find(op.param_id) != tensors.end())
                         ? tensors[op.param_id] : nullptr;

        Shape in_shape = shapes[op.input_id];
        Shape out_shape;

        // per-sample output shape 결정
        if (op.op_type == MATMUL && param != nullptr) {
            Shape w_shape = shapes[op.param_id];      // [K x N]
            out_shape = {in_shape.rows, w_shape.cols};
        } else {
            out_shape = in_shape;
        }

        // output 버퍼 준비 (배치 포함)
        if (tensors.find(op.output_id) == tensors.end()) {
            float* out_ptr = nullptr;
            CUDA_CHECK(cudaMalloc(&out_ptr, (size_t)batch_size * out_shape.rows * out_shape.cols * sizeof(float)));
            tensors[op.output_id] = out_ptr;
            shapes[op.output_id] = out_shape; // per-sample shape 저장
        }

        float* output = tensors[op.output_id];

        const int rows = out_shape.rows;
        const int cols = out_shape.cols;
        const int total = rows * cols;

        // 공통 런치 설정 (per-sample)
        dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
        dim3 dimGrid((cols + TILE_WIDTH - 1) / TILE_WIDTH,
                     (rows + TILE_WIDTH - 1) / TILE_WIDTH);

        int threads = 256;
        int blocks  = (total + threads - 1) / threads;

        // === 배치 루프: per-sample 포인터로 실행 ===
        for (int b = 0; b < batch_size; ++b) {
            const size_t in_stride  = (size_t)in_shape.rows * in_shape.cols;
            const size_t out_stride = (size_t)out_shape.rows * out_shape.cols;

            float* input_b  = input  + b * in_stride;
            float* output_b = output + b * out_stride;

            switch (op.op_type) {
                case MATMUL: {
                    if (!param) { fprintf(stderr, "[MATMUL] missing param\n"); break; }
                    // A[MxK] * W[KxN] = C[MxN]
                    matmul_shared_kernel_coalesced<<<dimGrid, dimBlock>>>(
                        input_b, param, output_b,
                        rows, in_shape.cols, cols);
                    CUDA_CHECK(cudaGetLastError());
                    break;
                }

                case ADD: {
                    if (!param) { fprintf(stderr, "[ADD] missing bias/param\n"); break; }
                    add_kernel<<<blocks, threads>>>(input_b, param, output_b, rows, cols);
                    CUDA_CHECK(cudaGetLastError());
                    break;
                }

                case SIGMOID: {
                    activation_sigmoid<<<blocks, threads>>>(input_b, param, output_b, rows, cols);
                    CUDA_CHECK(cudaGetLastError());
                    break;
                }

                case RELU: {
                    activation_relu<<<blocks, threads>>>(input_b, param, output_b, rows, cols);
                    CUDA_CHECK(cudaGetLastError());
                    break;
                }

                case TANH: {
                    activation_tanh<<<blocks, threads>>>(input_b, param, output_b, rows, cols);
                    CUDA_CHECK(cudaGetLastError());
                    break;
                }

                case FLATTEN: {
                    CUDA_CHECK(cudaMemcpy(output_b, input_b, out_stride * sizeof(float),
                                          cudaMemcpyDeviceToDevice));
                    break;
                }

                case CONV2D: {
                    // 방법 A: 배치 루프 유지 → z축엔 채널만.
                    int KH = op.extra_params.kernel_h;
                    int KW = op.extra_params.kernel_w;
                    int SH = op.extra_params.stride_h;
                    int SW = op.extra_params.stride_w;
                    int PH = op.extra_params.padding_h;
                    int PW = op.extra_params.padding_w;
                    int IH = op.extra_params.input_h;
                    int IW = op.extra_params.input_w;
                    int IC = op.extra_params.input_c;
                    int OC = op.extra_params.output_c;

                    // OH, OW는 per-sample 출력 공간 크기
                    int OW = shapes[op.output_id].cols / OC;
                    int OH = shapes[op.output_id].rows; // 보관 방식에 따라 다를 수 있음 (필요시 계산)

                    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
                    dim3 gridDim((OW + TILE_WIDTH - 1) / TILE_WIDTH,
                                 (OH + TILE_WIDTH - 1) / TILE_WIDTH,
                                 OC); // ★ 배치 루프를 도니 z=OC만

                    // 커널은 input_b/output_b 기준 (한 샘플)
                    conv2d_forward_kernel<<<gridDim, blockDim>>>(
                        input_b, param, output_b,
                        /*batch_size=*/1, IH, IW,
                        IC, OC,
                        KH, KW,
                        OH, OW
                    );
                    CUDA_CHECK(cudaGetLastError());
                    break;
                }

                case LOSS:
                    // forward에서는 패스 (run_graph_with_loss_cuda()에서 처리)
                    continue;
                
                // 그 외 노드만 출력 텐서 할당
                if (tensors.find(op.output_id) == tensors.end()) {
                    float* out_ptr;
                    cudaMalloc(&out_ptr, batch_size * out_shape.rows * out_shape.cols * sizeof(float));
                    tensors[op.output_id] = out_ptr;
                    shapes[op.output_id] = out_shape;
                }

                
                default:
                    fprintf(stderr, "[ERROR] Unsupported op_type: %d\n", op.op_type);
                    break;
            }
        } // end for b

        // 디버그/안전용: op마다 1회 동기화(필요 시만)
        // CUDA_CHECK(cudaDeviceSynchronize());
    } // end for ops

    // 최종 출력 호스트로 복사 (배치 포함 크기)
    Shape out_shape = shapes[final_output_id];
    const size_t out_bytes = (size_t)batch_size * out_shape.rows * out_shape.cols * sizeof(float);
    CUDA_CHECK(cudaMemcpy(out_host, tensors[final_output_id], out_bytes, cudaMemcpyDeviceToHost));
}
