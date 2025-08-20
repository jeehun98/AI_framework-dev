// run_graph.cu (updated: cuBLAS Strided-Batched + bias fuse + single-launch activations)

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <unordered_map>
#include <cuda_runtime.h>

#include <cublas_v2.h>

#include "quant/quant_types.cuh"
#include "quant/observers.cuh"
#include "quant/quant_kernels.cuh"
#include "quant/int8_gemm_dp4a.cuh"
#include "quant/epilogue_kernels.cuh"

#include "run_graph.cuh"
#include "activation_ops.cuh"
#include "softmax_kernels.cuh"
#include "add_bias_rowwise.cuh"
#include "cnn_kernels.cuh"
#include "op_structs.cuh"

#include "ge/cuda_check.cuh"
#include "ge/cublas_utils.cuh"
#include "ge/gemm_rm.cuh"
#include "ge/act_map.cuh"
#include "ge/alloc_utils.cuh"

#ifndef TILE_WIDTH
#define TILE_WIDTH 16
#endif

// 입력 텐서 조회
inline const float* get_tensor_ro(const std::unordered_map<std::string, float*>& tensors,
                                  const std::string& id)
{
    auto it = tensors.find(id);
    if (it == tensors.end() || !it->second) {
        std::fprintf(stderr, "[ERR] Tensor '%s' not found (RO)\n", id.c_str());
        return nullptr;
    }
    return it->second;
}

inline float* get_tensor_rw(std::unordered_map<std::string, float*>& tensors,
                            const std::string& id)
{
    auto it = tensors.find(id);
    if (it == tensors.end() || !it->second) {
        std::fprintf(stderr, "[ERR] Tensor '%s' not found (RW)\n", id.c_str());
        return nullptr;
    }
    return it->second;
}

void run_graph_cuda(
    const std::vector<OpStruct>& E,
    std::unordered_map<std::string, float*>& tensors,
    std::unordered_map<std::string, Shape>& shapes,
    float* out_host,
    const std::string& final_output_id,
    int batch_size)
{
    auto h = ge_cublas();

    for (size_t i = 0; i < E.size(); ++i) {
        const auto& op = E[i];
        if (op.op_type == LOSS) continue;

        // 입력/shape 확보
        auto it_in = tensors.find(op.input_id);
        if (it_in == tensors.end() || it_in->second == nullptr) {
            std::fprintf(stderr, "[ERR] missing input tensor: %s\n", op.input_id.c_str());
            break;
        }
        float* input = it_in->second;

        const auto it_inshape = shapes.find(op.input_id);
        if (it_inshape == shapes.end()) {
            std::fprintf(stderr, "[ERR] missing input shape: %s\n", op.input_id.c_str());
            break;
        }
        const Shape in_shape = it_inshape->second;

        float* param = nullptr;
        if (!op.param_id.empty()) {
            auto it_p = tensors.find(op.param_id);
            if (it_p != tensors.end()) param = it_p->second;
        }

        Shape out_shape = in_shape; // 기본은 동일

        switch (op.op_type) {
        case MATMUL: {
            if (!param) {
                std::fprintf(stderr, "[MATMUL] missing param for %s\n", op.output_id.c_str());
                break;
            }
            auto it_wshape = shapes.find(op.param_id);
            if (it_wshape == shapes.end()) {
                std::fprintf(stderr, "[MATMUL] missing weight shape: %s\n", op.param_id.c_str());
                break;
            }
            const Shape w_shape = it_wshape->second; // [K, N]
            const int M = in_shape.rows;
            const int K = in_shape.cols;
            const int N = w_shape.cols;
            if (w_shape.rows != K) {
                std::fprintf(stderr, "[MATMUL] dim mismatch: in(K=%d) vs W(rows=%d)\n", K, w_shape.rows);
                break;
            }
            out_shape = { M, N };

            // 다음 op가 row-wise ADD면 bias fuse
            bool fuse_bias = false;
            float* bias_ptr = nullptr;
            std::string out_id = op.output_id;

            if ((i + 1) < E.size()) {
                const auto& nx = E[i + 1];
                if (nx.op_type == ADD && nx.input_id == op.output_id &&
                    !nx.param_id.empty() && tensors.count(nx.param_id))
                {
                    const Shape bshape = shapes[nx.param_id];
                    const bool row_bias = (bshape.rows == 1 && bshape.cols == N) ||
                                          (bshape.rows == N && bshape.cols == 1);
                    if (row_bias) {
                        fuse_bias = true;
                        bias_ptr = tensors[nx.param_id];
                        out_id = nx.output_id; // ADD 출력으로 바로 기록
                    }
                }
            }

            float* Y = ge_ensure_output(tensors, shapes, out_id, out_shape, batch_size);

            const long long strideA = (long long)M * K;
            const long long strideC = (long long)M * N;

            gemm_rm_strided_batched_tf32(
                h,
                /*transA=*/false, /*transB=*/false,
                /*M=*/M, /*N=*/N, /*K=*/K,
                /*A*/ input,  /*lda=*/K, /*strideA=*/strideA,
                /*B*/ param,  /*ldb=*/N, /*strideB=*/0LL,   // shared weight
                /*C*/ Y,      /*ldc=*/N, /*strideC=*/strideC,
                /*batch=*/batch_size,
                /*alpha=*/1.f, /*beta=*/0.f
            );

            if (fuse_bias) {
                const int rowsB = batch_size * M;
                const int cols  = N;
                launch_add_bias_rowwise(Y, bias_ptr, Y, rowsB, cols);
                CUDA_CHECK(cudaGetLastError());
                ++i; // 다음 ADD 스킵
            }
            break;
        }

        case ADD: {
            if (!param) {
                std::fprintf(stderr, "[ADD] missing param for %s\n", op.output_id.c_str());
                break;
            }
            out_shape = in_shape;
            float* output = ge_ensure_output(tensors, shapes, op.output_id, out_shape, batch_size);

            const Shape bshape = shapes[op.param_id];
            const bool row_bias = (bshape.rows == 1 && bshape.cols == out_shape.cols) ||
                                  (bshape.rows == out_shape.cols && bshape.cols == 1);
            if (row_bias) {
                const int rowsB = batch_size * out_shape.rows;
                const int cols  = out_shape.cols;
                launch_add_bias_rowwise(input, param, output, rowsB, cols);
                CUDA_CHECK(cudaGetLastError());
            } else {
                std::fprintf(stderr,
                        "[ADD] unsupported shape: input(%d,%d) + param(%d,%d). Expect row-wise bias.\n",
                        in_shape.rows, in_shape.cols, bshape.rows, bshape.cols);
                const size_t bytes = (size_t)batch_size * out_shape.rows * out_shape.cols * sizeof(float);
                CUDA_CHECK(cudaMemcpy(output, input, bytes, cudaMemcpyDeviceToDevice));
            }
            break;
        }

        // ---------- 활성화 ----------
        case SIGMOID:
        case RELU:
        case TANH:
        case LEAKY_RELU:
        case ELU:
        case GELU:
        case SILU: {
            out_shape = (shapes.count(op.output_id) ? shapes[op.output_id] : in_shape);
            float* output = ge_ensure_output(tensors, shapes, op.output_id, out_shape, batch_size);

            const int rowsB = batch_size * out_shape.rows;
            const int cols  = out_shape.cols;

            const float* bias_ptr = nullptr; // 옵션
            if (!op.param_id.empty()) {
                auto itb = tensors.find(op.param_id);
                if (itb != tensors.end()) bias_ptr = itb->second;
            }
            const int act = ge_map_act_type(op.op_type);
            const float alpha = op.extra_params.alpha;
            const int gelu_tanh_flag = op.extra_params.gelu_tanh ? 1 : 0;

            launch_activation_forward(input, bias_ptr, output,
                                      rowsB, cols, act, alpha, gelu_tanh_flag, /*stream*/0);
            CUDA_CHECK(cudaGetLastError());
            break;
        }

        // ---------- Softmax ----------
        case SOFTMAX: {
            out_shape = (shapes.count(op.output_id) ? shapes[op.output_id] : in_shape);
            float* output = ge_ensure_output(tensors, shapes, op.output_id, out_shape, batch_size);

            const int rowsB = batch_size * out_shape.rows;
            const int cols  = out_shape.cols;
            const float temperature = (op.extra_params.temperature > 0.f)
                                        ? op.extra_params.temperature : 1.f;

            launch_softmax_forward(input, output, rowsB, cols, temperature, /*stream*/0);
            CUDA_CHECK(cudaGetLastError());
            break;
        }

        case FLATTEN: {
            auto it_outshape = shapes.find(op.output_id);
            if (it_outshape == shapes.end()) {
                std::fprintf(stderr, "[FLATTEN][ERR] missing out shape for %s\n", op.output_id.c_str());
                break;
            }
            const Shape out_shape_f = it_outshape->second;
            float* output = ge_ensure_output(tensors, shapes, op.output_id, out_shape_f, batch_size);

            const size_t elems_in  = (size_t)batch_size * in_shape.rows      * in_shape.cols;
            const size_t elems_out = (size_t)batch_size * out_shape_f.rows    * out_shape_f.cols;
            if (elems_in != elems_out) {
                std::fprintf(stderr,
                        "[FLATTEN][ERR] elem mismatch: in=%zu out=%zu (B=%d in=(%d,%d) out=(%d,%d))\n",
                        elems_in, elems_out, batch_size,
                        in_shape.rows, in_shape.cols, out_shape_f.rows, out_shape_f.cols);
                break;
            }
            CUDA_CHECK(cudaMemcpy(output, input, elems_in * sizeof(float), cudaMemcpyDeviceToDevice));
            break;
        }

        case CONV2D: {
            auto it_outshape = shapes.find(op.output_id);
            if (it_outshape == shapes.end()) {
                std::fprintf(stderr, "[CONV2D][ERR] missing out shape for %s\n", op.output_id.c_str());
                break;
            }
            const Shape out_shape_c = it_outshape->second;
            float* output = ge_ensure_output(tensors, shapes, op.output_id, out_shape_c, batch_size);

            const int KH = op.extra_params.kernel_h;
            const int KW = op.extra_params.kernel_w;
            const int IH = op.extra_params.input_h;
            const int IW = op.extra_params.input_w;
            const int IC = op.extra_params.input_c;
            const int OC = op.extra_params.output_c;

            const int OW = out_shape_c.cols / OC;
            const int OH = out_shape_c.rows;

            dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
            dim3 gridDim((OW + TILE_WIDTH - 1) / TILE_WIDTH,
                         (OH + TILE_WIDTH - 1) / TILE_WIDTH,
                         OC);

            for (int b = 0; b < batch_size; ++b) {
                float* in_b  = input  + (size_t)b * IH * IW * IC;
                float* out_b = output + (size_t)b * OH * OW * OC;
                conv2d_forward_kernel<<<gridDim, blockDim>>>(
                    in_b, param, out_b,
                    /*batch_size=*/1, IH, IW,
                    IC, OC,
                    KH, KW,
                    OH, OW
                );
                CUDA_CHECK(cudaGetLastError());
            }
            break;
        }

        default:
            std::fprintf(stderr, "[ERR] Unsupported op_type: %d\n", op.op_type);
            break;
        } // switch
    } // for

    // 최종 출력 호스트 복사
    auto it_final  = tensors.find(final_output_id);
    auto it_fshape = shapes.find(final_output_id);
    if (out_host && it_final != tensors.end() && it_fshape != shapes.end()) {
        const Shape out_shape = it_fshape->second;
        const size_t bytes = (size_t)batch_size * out_shape.rows * out_shape.cols * sizeof(float);
        CUDA_CHECK(cudaMemcpy(out_host, it_final->second, bytes, cudaMemcpyDeviceToHost));
    } else if (out_host) {
        std::fprintf(stderr, "[ERR] final output missing: id=%s\n", final_output_id.c_str());
    }
}
