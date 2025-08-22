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

#include "run_graph_utils.cuh"

#include "pooling/pooling_ops.cuh"
#include "pooling/pooling_kernels.cuh"
#include "executor/run_graph.cuh"
#include "activation/activation_ops.cuh"
#include "softmax/softmax_kernels.cuh"
#include "bias/add_bias_rowwise.cuh"
#include "cnn/cnn_kernels.cuh"
#include "op_structs.cuh"
#include "pooling/pooling_kernels.cuh"

#include "ge/cuda_check.cuh"
#include "ge/cublas_utils.cuh"
#include "ge/gemm_rm.cuh"
#include "ge/act_map.cuh"
#include "ge/alloc_utils.cuh"

#ifndef TILE_WIDTH
#define TILE_WIDTH 16
#endif


static void debug_l2(const char* name, const float* dptr, size_t n_elems) {
    std::vector<float> h(n_elems);
    cudaMemcpy(h.data(), dptr, n_elems * sizeof(float), cudaMemcpyDeviceToHost);
    double s = 0.0;
    for (size_t i = 0; i < n_elems; ++i) { double v = h[i]; s += v * v; }
    std::fprintf(stderr, "[GRAD] %s L2=%.6e\n", name, std::sqrt(s));
}


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

            const int rows_per_sample = out_shape.rows; // 예: filters(채널)
            const int cols            = out_shape.cols; // 예: H*W
            const int rowsB           = batch_size * rows_per_sample;

            const bool bias_rowwise = (bshape.rows == 1 && bshape.cols == cols)   // (1, cols)
                                || (bshape.rows == cols && bshape.cols == 1);  // (cols, 1)

            const bool bias_colwise = (bshape.rows == 1 && bshape.cols == rows_per_sample)   // (1, rows)
                                || (bshape.rows == rows_per_sample && bshape.cols == 1);  // (rows, 1)

            if (bias_rowwise) {
                launch_add_bias_rowwise(input, param, output, rowsB, cols);  // stream 없음 오버로드
                CUDA_CHECK(cudaGetLastError());
            } else if (bias_colwise) {
                launch_add_bias_colwise(input, param, output, rowsB, cols, rows_per_sample);
                CUDA_CHECK(cudaGetLastError());
            } else {
                std::fprintf(stderr,
                    "[ADD] unsupported shape: input(%d,%d) + param(%d,%d). "
                    "Expect row-wise (len=cols) or channel-wise (len=rows) bias.\n",
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

        // run_graph.cu — switch(op.op_type) 내부 교체
        case POOL_MAX:
        case POOL_AVG: {
            const OpExtraParams& ex = op.extra_params;

            // 입력 NCHW 크기
            const int N = batch_size;
            const int C = ex.input_c;
            const int H = ex.input_h;
            const int W = ex.input_w;

            const int kH = ex.kernel_h;
            const int kW = ex.kernel_w;
            const int sH = (ex.stride_h > 0 ? ex.stride_h : 1);
            const int sW = (ex.stride_w > 0 ? ex.stride_w : 1);
            const int pH = ex.padding_h;
            const int pW = ex.padding_w;
            const int dH = (ex.dilation_h > 0 ? ex.dilation_h : 1);
            const int dW = (ex.dilation_w > 0 ? ex.dilation_w : 1);

            // 출력 공간 크기 (ceil_mode=false)
            const int Hout = (H + 2*pH - dH*(kH - 1) - 1) / sH + 1;
            const int Wout = (W + 2*pW - dW*(kW - 1) - 1) / sW + 1;

            if (Hout <= 0 || Wout <= 0) {
                std::fprintf(stderr, "[POOL][ERR] invalid output size Hout=%d Wout=%d\n", Hout, Wout);
                break;
            }

            // sample view: (rows=C, cols=Hout*Wout) — 물리 버퍼는 [N,C,Hout,Wout]
            Shape out_shape_p{ C, Hout * Wout };
            shapes[op.output_id] = out_shape_p;

            // 버퍼 (NCHW contiguous)
            const float* X = get_tensor_ro(tensors, op.input_id);        // [N,C,H,W]
            float* Y = ge_ensure_output(tensors, shapes, op.output_id, out_shape_p, N); // [N,C,Hout,Wout]

            if (!X || !Y) {
                std::fprintf(stderr, "[POOL][ERR] missing tensors X(%p) or Y(%p)\n", (void*)X, (void*)Y);
                break;
            }

            // Pool 파라미터 (커널은 NCHW 가정)
            Pool2DParams p{};
            p.N = N; p.C = C; p.H = H; p.W = W;
            p.kernel_h = kH; p.kernel_w = kW;
            p.stride_h = sH; p.stride_w = sW;
            p.pad_h = pH;   p.pad_w = pW;
            p.dilation_h = dH; p.dilation_w = dW;
            p.H_out = Hout; p.W_out = Wout;
            p.avg_inclusive = ex.count_include_pad; // 필요 시 OpExtraParams에 bool 필드 존재해야 함

            if (op.op_type == POOL_MAX) {
                const size_t n_out = (size_t)N * C * Hout * Wout;
                int32_t* argmax = ensure_argmax_ws(tensors, n_out, "__pool_argmax::" + op.output_id);

                maxpool2d_forward(/*x=*/X, /*y=*/Y, /*argmax=*/argmax, /*p=*/p, /*stream=*/0);
                CUDA_CHECK(cudaGetLastError());
            } else { // POOL_AVG
                avgpool2d_forward(/*x=*/X, /*y=*/Y, /*p=*/p, /*stream=*/0);
                CUDA_CHECK(cudaGetLastError());
            }
            break;
        }


        case CONV2D: {
            const OpExtraParams& ex = op.extra_params;

            const int N    = batch_size;
            const int Cin  = ex.input_c;   // NCHW
            const int Hin  = ex.input_h;
            const int Win  = ex.input_w;
            const int Cout = ex.output_c;
            const int Kh   = ex.kernel_h;
            const int Kw   = ex.kernel_w;
            const int Sh   = (ex.stride_h > 0 ? ex.stride_h : 1);
            const int Sw   = (ex.stride_w > 0 ? ex.stride_w : 1);
            const int Ph   = ex.padding_h;
            const int Pw   = ex.padding_w;

            const int Hout = (Hin + 2*Ph - Kh) / Sh + 1;
            const int Wout = (Win + 2*Pw - Kw) / Sw + 1;

            if (Hout <= 0 || Wout <= 0) {
                std::fprintf(stderr, "[CONV2D][ERR] invalid output size Hout=%d Wout=%d\n", Hout, Wout);
                break;
            }

            // sample view: (rows=Cout, cols=Hout*Wout) — 물리 버퍼는 [N,Cout,Hout,Wout]
            Shape out_shape_c{ Cout, Hout * Wout };
            shapes[op.output_id] = out_shape_c;

            // 버퍼는 NCHW로 관리되어야 함
            float* X = tensors[op.input_id];  // [N, Cin, Hin, Win]
            float* W = tensors[op.param_id];  // [Cout, Cin, Kh, Kw] contiguous
            float* Y = ge_ensure_output(tensors, shapes, op.output_id, out_shape_c, N); // [N, Cout, Hout, Wout] 여야 함

            launch_conv2d_forward_nchw(
                /*X=*/X, /*W=*/W, /*Y=*/Y,
                /*N=*/N, /*Hin=*/Hin, /*Win=*/Win, /*Cin=*/Cin,
                /*Hout=*/Hout, /*Wout=*/Wout, /*Cout=*/Cout,
                /*Kh=*/Kh, /*Kw=*/Kw, /*Sh=*/Sh, /*Sw=*/Sw, /*Ph=*/Ph, /*Pw=*/Pw,
                /*stream=*/0);
            CUDA_CHECK(cudaGetLastError());
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
