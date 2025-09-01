// run_graph_backward.cu — TF32 + strided-batched + safe dW accumulate + grad-key dual register
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>          // std::max
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../executor/run_graph.cuh"
#include "../activation/activation_ops.cuh"
#include "../softmax/softmax_kernels.cuh"
#include "../cnn/cnn_kernels.cuh"
#include "../op_structs.cuh"
#include "../loss/loss_kernels.cuh"
#include "../reduce/reduce_stride.cuh"
#include "../reduce/reduce_ops.cuh"
#include "../pooling/pooling_ops.cuh"
#include "../pooling/pooling_kernels.cuh"
#include "../rnn/rnn_kernels.cuh"

#include "run_graph_utils.cuh"
#include "../ge/pack_utils.cuh"
#include "../ge/cuda_check.cuh"
#include "../ge/cublas_utils.cuh"
#include "../ge/gemm_rm.cuh"
#include "../ge/act_map.cuh"
#include "../ge/fill.cuh"

#ifndef TILE_WIDTH
#define TILE_WIDTH 16
#endif

/* ============================ Legacy→Vector normalize ============================ */

static inline void normalize_legacy_local(OpStruct& n) {
    if (!n.input_id.empty() && n.inputs.empty())  n.inputs.push_back(n.input_id);
    if (!n.param_id.empty() && n.params.empty())  n.params.push_back(n.param_id);
}
static inline std::vector<OpStruct> normalize_graph_local(const std::vector<OpStruct>& E) {
    std::vector<OpStruct> out; out.reserve(E.size());
    for (const auto& n_ref : E) {           // 복사 1회만 수행
        OpStruct n = n_ref;
        normalize_legacy_local(n);
        out.emplace_back(std::move(n));
    }
    return out;
}
static inline std::string A_of(const OpStruct& op) {
    return (!op.inputs.empty() ? op.inputs[0] : op.input_id);
}
static inline std::string B_of(const OpStruct& op) {
    if (!op.params.empty()) return op.params[0];
    if (op.inputs.size() >= 2) return op.inputs[1];
    return op.param_id;
}

/* ========================== small helpers / utilities ========================== */

// safer accumulate: sum rows of (B x KN) -> (KN)
__global__ void sum_rows_kernel(const float* X, float* y, int B, int KN) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= KN) return;
    float s = 0.f;
    for (int b = 0; b < B; ++b) s += X[b * KN + i];
    y[i] = s;
}
static inline void sum_rows(const float* X, float* y, int B, int KN) {
    int th = 256, bl = (KN + th - 1) / th;
    sum_rows_kernel<<<bl, th>>>(X, y, B, KN);
    CUDA_CHECK(cudaGetLastError());
}

static inline void reg_grad(std::unordered_map<std::string, float*>& G,
                            const std::string& key, float* ptr) {
    if (!key.empty() && ptr) G[key] = ptr;
}

/* ============================== time utils (BW) ============================== */

// SLICE_TIME backward: dY shape (B,1,D) -> dX shape (B,T,D) (t 위치로 scatter)
__global__ void slice_time_bw_kernel(const float* __restrict__ dY,
                                     float* __restrict__ dX,
                                     int B, int T, int D, int t) {
    int b = blockIdx.z;                                        // 배치는 z축
    int i = blockIdx.x * blockDim.x + threadIdx.x;             // 0..D-1
    if (b >= B || i >= D) return;
    size_t off_y = ((size_t)b * 1 + 0) * D + i;
    size_t off_x = ((size_t)b * T + t) * D + i;
    dX[off_x] = dY[off_y];
}

// CONCAT_TIME backward: dY(B,t1+t2,D) -> dX1(B,t1,D), dX2(B,t2,D)
__global__ void concat_time_bw_kernel(const float* __restrict__ dY,
                                      float* __restrict__ dX1, int t1,
                                      float* __restrict__ dX2, int t2,
                                      int B, int D) {
    int b = blockIdx.z;
    int r = blockIdx.y;                                        // 0..t1+t2-1
    int c = blockIdx.x * blockDim.x + threadIdx.x;             // 0..D-1
    if (b >= B || c >= D) return;
    int T = t1 + t2;
    size_t y_off = ((size_t)b * T + r) * D + c;
    if (r < t1) {
        size_t x1_off = ((size_t)b * t1 + r) * D + c;
        dX1[x1_off] = dY[y_off];
    } else {
        int r2 = r - t1;
        size_t x2_off = ((size_t)b * t2 + r2) * D + c;
        dX2[x2_off] = dY[y_off];
    }
}

/* ================================== main ================================== */

void run_graph_backward(
    const std::vector<OpStruct>& E_in,
    std::unordered_map<std::string, float*>& tensors,
    std::unordered_map<std::string, Shape>&  shapes,
    std::unordered_map<std::string, float*>& gradients,
    const std::string& /*final_output_id_unused*/,
    int batch_size)
{
    auto h = ge_cublas();
    const auto E = normalize_graph_local(E_in);

    bool fused_softmax = false;
    std::string fused_softmax_in_id, fused_softmax_out_id;

    /* ------------------------------- 1) LOSS BW ------------------------------- */
    if (!E.empty() && E.back().op_type == LOSS) {
        const OpStruct& loss_op = E.back();
        const std::string loss_type = loss_op.extra_params.loss_type;
        const std::string label_id  = loss_op.extra_params.label_id;

        const std::string y_pred_id = A_of(loss_op);
        if (!tensors.count(label_id) || !tensors.count(y_pred_id)) {
            std::fprintf(stderr, "[LOSS][BW] missing y_true(%s) or y_pred(%s)\n",
                         label_id.c_str(), y_pred_id.c_str());
            return;
        }
        const float* y_true = tensors[label_id];
        const float* y_pred = tensors[y_pred_id];

        if (!shapes.count(y_pred_id)) {
            std::fprintf(stderr, "[LOSS][BW] missing shape(y_pred=%s)\n", y_pred_id.c_str());
            return;
        }
        const Shape shp = shapes[y_pred_id];
        const int rows_per_sample = shp.rows;
        const int C = shp.cols;
        const int B = batch_size * rows_per_sample;
        const int N = B * C;
        cudaStream_t stream = 0;

        // softmax 바로 뒤 cce면 fused 경로 사용
        const OpStruct* prev = nullptr;
        if (E.size() >= 2) {
            const OpStruct& cand = E[E.size()-2];
            if (cand.op_type == SOFTMAX && cand.output_id == y_pred_id) prev = &cand;
        }

        if (loss_type == "cce" && prev) {
            float* dL_dz = nullptr;
            CUDA_CHECK(cudaMalloc(&dL_dz, (size_t)N * sizeof(float)));
            launch_softmax_xent_fused_backward(y_pred, y_true, dL_dz, B, C, stream);
            fused_softmax = true;
            fused_softmax_in_id  = A_of(*prev);
            fused_softmax_out_id = prev->output_id;
            gradients[fused_softmax_in_id] = dL_dz;
        } else {
            float* dL_dy = nullptr;
            CUDA_CHECK(cudaMalloc(&dL_dy, (size_t)N * sizeof(float)));
            if (loss_type == "bce")      launch_bce_loss_backward(y_true, y_pred, dL_dy, N, B, stream);
            else if (loss_type == "mse") launch_mse_loss_backward(y_true, y_pred, dL_dy, N, stream);
            else if (loss_type == "cce") launch_cce_loss_backward(y_true, y_pred, dL_dy, B, C, stream);
            else std::fprintf(stderr, "[LOSS][BW] unsupported: %s\n", loss_type.c_str());
            gradients[y_pred_id] = dL_dy;
        }
    }

    /* ------------------------------ 2) others BW ----------------------------- */
    for (auto it = E.rbegin(); it != E.rend(); ++it) {
        const OpStruct& op = *it;
        if (op.op_type == LOSS) continue;
        if (fused_softmax && op.op_type == SOFTMAX && op.output_id == fused_softmax_out_id) continue;

        const std::string A_id = A_of(op);
        const std::string B_id = B_of(op);

        if (!tensors.count(A_id)) continue;
        float* input = tensors[A_id];
        float* param = (!B_id.empty() && tensors.count(B_id)) ? tensors[B_id] : nullptr;

        float* grad_out_full = gradients.count(op.output_id) ? gradients[op.output_id] : nullptr;
        if (!grad_out_full && op.op_type != FLATTEN) continue;

        const Shape in_shape  = shapes.count(A_id)         ? shapes[A_id]         : Shape{0,0};
        const Shape out_shape = shapes.count(op.output_id) ? shapes[op.output_id] : in_shape;

        const int M = out_shape.rows;
        const int N = out_shape.cols;
        const int K = in_shape.cols;
        const int in_size  = in_shape.rows  * in_shape.cols;

        float* grad_input_full = nullptr;
        if (op.op_type != FLATTEN) {
            CUDA_CHECK(cudaMalloc(&grad_input_full, (size_t)batch_size * in_size * sizeof(float)));
        }

        switch (op.op_type) {
        /* --------------------------------- GEMM --------------------------------- */
        case MATMUL: {
            if (!param) break;

            // dX = dY · W^T
            gemm_rm_strided_batched_tf32(
                h, /*transA=*/false, /*transB=*/true,
                /*M=*/M, /*N=*/K, /*K=*/N,
                /*A=*/grad_out_full,   /*lda=*/N, /*strideA=*/(long long)M * N,
                /*B=*/param,           /*ldb=*/N, /*strideB=*/0LL,
                /*C=*/grad_input_full, /*ldc=*/K, /*strideC=*/(long long)M * K,
                /*batch=*/batch_size,
                /*alpha=*/1.f, /*beta=*/0.f
            );

            // dW = sum_b (X_b^T · dY_b)  --> 임시버퍼 없이 직접 누적
            float* dW = nullptr; // (K, N)
            CUDA_CHECK(cudaMalloc(&dW, (size_t)K * N * sizeof(float)));

            for (int b = 0; b < batch_size; ++b) {
                const float* Xb  = input + (long long)b * M * K;       // X_b:  (M,K)
                const float* dYb = grad_out_full + (long long)b * M * N; // dY_b: (M,N)

                // dW += X_b^T · dY_b
                gemm_rm_tf32(
                    h,
                    /*transA=*/true,  /* A: X_b^T (K,M) */
                    /*transB=*/false, /* B: dY_b  (M,N) */
                    /*M=*/K, /*N=*/N, /*K=*/M,
                    /*A=*/Xb,  /*lda=*/K,
                    /*B=*/dYb, /*ldb=*/N,
                    /*C=*/dW,  /*ldc=*/N,
                    /*alpha=*/1.f,
                    /*beta=*/(b == 0 ? 0.f : 1.f)  // 첫 배치는 덮어쓰기, 이후는 누적
                );
            }

            // grad key를 param_id와 B_id 모두로 등록(옵티마이저 호환)
            reg_grad(gradients, B_id,        dW);
            reg_grad(gradients, op.param_id, dW);
            break;
        }


        /* ---------------------------------- ADD --------------------------------- */
        case ADD: {
            if (!tensors.count(B_id) || !shapes.count(B_id)) {
                std::fprintf(stderr, "[ADD/BWD] missing bias '%s'\n", B_id.c_str());
                break;
            }
            const Shape outS = out_shape;
            const Shape bS   = shapes[B_id];
            const int rows_per_sample = outS.rows;
            const int cols            = outS.cols;
            const int rowsB           = batch_size * rows_per_sample;

            const size_t bytes = (size_t)rowsB * cols * sizeof(float);
            CUDA_CHECK(cudaMemcpy(grad_input_full, grad_out_full, bytes, cudaMemcpyDeviceToDevice));

            const bool bias_rowwise = (bS.rows == 1 && bS.cols == cols) ||
                                      (bS.rows == cols && bS.cols == 1);
            const bool bias_colwise = (bS.rows == 1 && bS.cols == rows_per_sample) ||
                                      (bS.rows == rows_per_sample && bS.cols == 1);

            if (bias_rowwise) {
                float* grad_bias = nullptr; CUDA_CHECK(cudaMalloc(&grad_bias, (size_t)cols * sizeof(float)));
                launch_reduce_over_rows(grad_out_full, grad_bias, rowsB, cols);
                reg_grad(gradients, B_id,        grad_bias);
                reg_grad(gradients, op.param_id, grad_bias);
            } else if (bias_colwise) {
                float* temp_rows = nullptr; CUDA_CHECK(cudaMalloc(&temp_rows, (size_t)rowsB * sizeof(float)));
                launch_reduce_over_cols(grad_out_full, temp_rows, rowsB, cols);

                float* grad_bias = nullptr; CUDA_CHECK(cudaMalloc(&grad_bias, (size_t)rows_per_sample * sizeof(float)));
                launch_reduce_batch_stride(temp_rows, grad_bias, rows_per_sample, batch_size);
                CUDA_CHECK(cudaFree(temp_rows));

                reg_grad(gradients, B_id,        grad_bias);
                reg_grad(gradients, op.param_id, grad_bias);
            } else {
                const int len = std::max(bS.rows * bS.cols, 1);
                float* grad_bias = nullptr; CUDA_CHECK(cudaMalloc(&grad_bias, (size_t)len * sizeof(float)));
                CUDA_CHECK(cudaMemset(grad_bias, 0, (size_t)len * sizeof(float)));
                reg_grad(gradients, B_id,        grad_bias);
                reg_grad(gradients, op.param_id, grad_bias);
            }
            break;
        }

        /* ------------------------------- ADD_BIAS ------------------------------- */
        case ADD_BIAS: {
            if (!param || !shapes.count(B_id)) {
                std::fprintf(stderr, "[ADD_BIAS/BWD] missing bias '%s'\n", B_id.c_str());
                break;
            }
            const Shape outS = out_shape;
            const int rowsB = batch_size * outS.rows;
            const int C     = outS.cols;

            // dX = dY
            const size_t bytes = (size_t)rowsB * C * sizeof(float);
            CUDA_CHECK(cudaMemcpy(grad_input_full, grad_out_full, bytes, cudaMemcpyDeviceToDevice));

            // db = sum over rows
            float* grad_bias = nullptr; CUDA_CHECK(cudaMalloc(&grad_bias, (size_t)C * sizeof(float)));
            launch_reduce_over_rows(grad_out_full, grad_bias, rowsB, C);
            reg_grad(gradients, B_id,        grad_bias);
            reg_grad(gradients, op.param_id, grad_bias);
            break;
        }

        /* ------------------------------- Activations ---------------------------- */
        case SIGMOID:
        case RELU:
        case TANH:
        case LEAKY_RELU:
        case ELU:
        case GELU:
        case SILU: {
            const int rowsB = batch_size * out_shape.rows;
            const int cols  = out_shape.cols;
            const float* gout = grad_out_full;
            const float* out  = tensors[op.output_id];
            const float* in   = tensors[A_id];
            float* gin        = grad_input_full;

            const int act = ge_map_act_type(op.op_type);
            const float alpha = op.extra_params.alpha;
            const int gelu_tanh_flag = op.extra_params.gelu_tanh ? 1 : 0;

            launch_activation_backward(gout, in, out, gin, rowsB, cols, act, alpha, gelu_tanh_flag, 0);
            CUDA_CHECK(cudaGetLastError());
            break;
        }

        /* --------------------------------- Softmax ------------------------------ */
        case SOFTMAX: {
            const int rowsB = batch_size * out_shape.rows;
            const int cols  = out_shape.cols;
            const float* gout = grad_out_full;
            const float* y    = tensors[op.output_id];
            float* gin        = grad_input_full;
            float temperature = (op.extra_params.temperature > 0.f) ? op.extra_params.temperature : 1.f;
            launch_softmax_backward(gout, y, gin, rowsB, cols, temperature, 0);
            CUDA_CHECK(cudaGetLastError());
            break;
        }

        /* --------------------------------- Flatten ------------------------------ */
        case FLATTEN: {
            // Pass-through
            gradients[A_id] = grad_out_full;
            continue;  // grad_input_full 미할당이므로 공통 등록 스킵
        }

        /* ---------------------------------- Conv -------------------------------- */
        case CONV2D: {
            const OpExtraParams& ex = op.extra_params;
            const int N=batch_size, Cin=ex.input_c, Hin=ex.input_h, Win=ex.input_w;
            const int Cout=ex.output_c, Kh=ex.kernel_h, Kw=ex.kernel_w;
            const int Sh=ex.stride_h>0?ex.stride_h:1, Sw=ex.stride_w>0?ex.stride_w:1;
            const int Ph=ex.padding_h, Pw=ex.padding_w;
            const int Hout = (Hin + 2*Ph - Kh) / Sh + 1;
            const int Wout = (Win + 2*Pw - Kw) / Sw + 1;

            const float* X   = tensors[A_id];
            const float* W   = param;
            const float* dYrm= gradients[op.output_id];
            if (!X || !W || !dYrm) { std::fprintf(stderr,"[BW][CONV2D] missing X/W/dY\n"); break; }

            float* dY = nullptr; CUDA_CHECK(cudaMalloc(&dY, (size_t)N*Cout*Hout*Wout*sizeof(float)));
            launch_pack_rm_to_nchw(dYrm, dY, N, Cout, Hout, Wout);

            float* dX = nullptr; float* dW = nullptr;
            CUDA_CHECK(cudaMalloc(&dX, (size_t)N*Cin*Hin*Win*sizeof(float)));
            CUDA_CHECK(cudaMalloc(&dW, (size_t)Cout*Cin*Kh*Kw*sizeof(float)));
            CUDA_CHECK(cudaMemset(dX, 0, (size_t)N*Cin*Hin*Win*sizeof(float)));
            CUDA_CHECK(cudaMemset(dW, 0, (size_t)Cout*Cin*Kh*Kw*sizeof(float)));

            launch_conv2d_backward_input_nchw(dY, W, dX, N, Hin, Win, Cin, Hout, Wout, Cout, Kh, Kw, Sh, Sw, Ph, Pw, 0);
            CUDA_CHECK(cudaGetLastError());
            launch_conv2d_backward_weight_nchw(dY, X, dW, N, Hin, Win, Cin, Hout, Wout, Cout, Kh, Kw, Sh, Sw, Ph, Pw, 0);
            CUDA_CHECK(cudaGetLastError());

            gradients[A_id] = dX;
            reg_grad(gradients, B_id,        dW);
            reg_grad(gradients, op.param_id, dW);

            CUDA_CHECK(cudaFree(dY));
            break;
        }

        /* ---------------------------------- Pool -------------------------------- */
        case POOL_MAX:
        case POOL_AVG: {
            const OpExtraParams& ex = op.extra_params;
            const int N=batch_size, C=ex.input_c, H=ex.input_h, W=ex.input_w;
            const int kH=ex.kernel_h, kW=ex.kernel_w;
            const int sH=ex.stride_h>0?ex.stride_h:1, sW=ex.stride_w>0?ex.stride_w:1;
            const int pH=ex.padding_h, pW=ex.padding_w;
            const int dH=ex.dilation_h>0?ex.dilation_h:1, dW=ex.dilation_w>0?ex.dilation_w:1;
            const int Hout=(H+2*pH-dH*(kH-1)-1)/sH+1;
            const int Wout=(W+2*pW-dW*(kW-1)-1)/sW+1;

            const float* dYrm = gradients.count(op.output_id)?gradients[op.output_id]:nullptr;
            if (!dYrm) { std::fprintf(stderr,"[BW][POOL] missing dY\n"); break; }

            float* dY=nullptr; CUDA_CHECK(cudaMalloc(&dY,(size_t)N*C*Hout*Wout*sizeof(float)));
            launch_pack_rm_to_nchw(dYrm, dY, N, C, Hout, Wout);

            float* dX=nullptr; CUDA_CHECK(cudaMalloc(&dX,(size_t)N*C*H*W*sizeof(float)));
            CUDA_CHECK(cudaMemset(dX,0,(size_t)N*C*H*W*sizeof(float)));

            Pool2DParams p{};
            p.N=N; p.C=C; p.H=H; p.W=W; p.H_out=Hout; p.W_out=Wout;
            p.kernel_h=kH; p.kernel_w=kW; p.stride_h=sH; p.stride_w=sW;
            p.pad_h=pH; p.pad_w=pW; p.dilation_h=dH; p.dilation_w=dW;
            p.avg_inclusive = ex.count_include_pad;

            if (op.op_type == POOL_MAX) {
                auto itArg = tensors.find("__pool_argmax::" + op.output_id);
                if (itArg == tensors.end() || !itArg->second) {
                    std::fprintf(stderr, "[BW][POOL_MAX] missing argmax\n");
                    CUDA_CHECK(cudaFree(dY)); CUDA_CHECK(cudaFree(dX)); break;
                }
                const int32_t* argmax = reinterpret_cast<const int32_t*>(itArg->second);
                maxpool2d_backward(dY, dX, argmax, p, 0);
            } else {
                avgpool2d_backward(dY, dX, p, 0);
            }
            CUDA_CHECK(cudaGetLastError());
            gradients[A_id] = dX;
            CUDA_CHECK(cudaFree(dY));
            break;
        }

        /* -------------------------------- SLICE_TIME ----------------------------- */
        case SLICE_TIME: {
            const Shape aS = shapes.count(A_id) ? shapes[A_id] : Shape{0,0}; // (T,D)
            const int T = aS.rows;
            const int D = aS.cols;
            int t = op.extra_params.time_index;
            if (t < 0) t = 0; if (t >= T) t = T - 1;

            // dX = 0, 선택된 t 위치로만 scatter
            const size_t bytesX = (size_t)batch_size * T * D * sizeof(float);
            CUDA_CHECK(cudaMemset(grad_input_full, 0, bytesX));

            const dim3 blk(256,1,1);
            const dim3 grd((D + 255) / 256, 1, batch_size);
            slice_time_bw_kernel<<<grd, blk>>>(grad_out_full, grad_input_full, batch_size, T, D, t);
            CUDA_CHECK(cudaGetLastError());
            break;
        }

        /* ------------------------------- CONCAT_TIME ----------------------------- */
        case CONCAT_TIME: {
            const Shape s1 = shapes.count(A_id) ? shapes[A_id] : Shape{0,0}; // (t1,D)
            const Shape s2 = shapes.count(B_id) ? shapes[B_id] : Shape{0,0}; // (t2,D)
            if (s1.cols != s2.cols) { std::fprintf(stderr,"[CONCAT_TIME/BWD] D mismatch\n"); break; }
            const int t1 = s1.rows, t2 = s2.rows, D = s1.cols;

            // dX1 = grad_input_full (이미 malloc됨: size=B*t1*D)
            float* dX2 = nullptr; CUDA_CHECK(cudaMalloc(&dX2, (size_t)batch_size * t2 * D * sizeof(float)));

            // zero-init
            CUDA_CHECK(cudaMemset(grad_input_full, 0, (size_t)batch_size * t1 * D * sizeof(float)));
            CUDA_CHECK(cudaMemset(dX2,            0, (size_t)batch_size * t2 * D * sizeof(float)));

            const dim3 blk(256,1,1);
            const dim3 grd((D + 255) / 256, t1 + t2, batch_size);
            concat_time_bw_kernel<<<grd, blk>>>(grad_out_full, grad_input_full, t1, dX2, t2, batch_size, D);
            CUDA_CHECK(cudaGetLastError());

            // 등록 후, 공통 경로의 중복 등록을 피하려면 continue
            gradients[A_id] = grad_input_full;
            reg_grad(gradients, B_id, dX2);
            continue;
        }

        /* -------------------------------- FILL_ZERO ------------------------------ */
        case FILL_ZERO: {
            // 생성 오퍼: 입력으로 전파 없음
            if (grad_input_full) { CUDA_CHECK(cudaFree(grad_input_full)); grad_input_full = nullptr; }
            continue;  // 공통 등록 스킵
        }

        /* ----------------------------------- RNN --------------------------------- */
        case RNN: {
            const OpExtraParams& ex = op.extra_params;
            const int B=batch_size;
            const int T=(ex.time_steps>0?ex.time_steps:(shapes.count(A_id)?shapes[A_id].rows:0));
            const int D=(ex.input_w   >0?ex.input_w   :(shapes.count(A_id)?shapes[A_id].cols:0));
            const int H=(ex.hidden_size>0?ex.hidden_size:ex.output_c);
            if (H<=0||T<=0||D<=0){ std::fprintf(stderr,"[BW][RNN] bad meta\n"); break; }

            const std::string Wx_id = (op.params.size()>=1)?op.params[0]:B_of(op);
            const std::string Wh_id = (op.params.size()>=2)?op.params[1]:"";
            const std::string b_id  = (op.params.size()>=3)?op.params[2]:"";
            const std::string h0_id = (op.params.size()>=4)?op.params[3]:"";

            float* X  = tensors.count(A_id)?tensors[A_id]:nullptr;
            float* Wx = tensors.count(Wx_id)?tensors[Wx_id]:nullptr;
            float* Wh = tensors.count(Wh_id)?tensors[Wh_id]:nullptr;
            float* b  = (!b_id.empty() && tensors.count(b_id))?tensors[b_id]:nullptr;
            float* h0 = (!h0_id.empty()&& tensors.count(h0_id))?tensors[h0_id]:nullptr;
            if (!X||!Wx||!Wh){ std::fprintf(stderr,"[BW][RNN] missing X/Wx/Wh\n"); break; }

            const std::string HSEQ_id = op.output_id + "::__rnn_hseq";
            float* H_seq = tensors.count(HSEQ_id)?tensors[HSEQ_id]:nullptr;
            if (!H_seq){ std::fprintf(stderr,"[BW][RNN] missing H_seq\n"); break; }

            float* gOut = gradients.count(op.output_id)?gradients[op.output_id]:nullptr;
            if (!gOut){ std::fprintf(stderr,"[BW][RNN] missing grad_out\n"); break; }

            const Shape outS = shapes.count(op.output_id)?shapes[op.output_id]:Shape{1,H};
            const bool ret_seq = (outS.rows == T);
            const float* dH_T=nullptr; const float* dH_seq=nullptr;
            if (ret_seq) dH_seq=gOut; else dH_T=gOut;

            if (!b) {
                const std::string zb="__rnn_zero_bias::"+op.output_id;
                if (!tensors.count(zb)){
                    float* zb_ptr=nullptr; CUDA_CHECK(cudaMalloc(&zb_ptr,(size_t)H*sizeof(float)));
                    CUDA_CHECK(cudaMemset(zb_ptr,0,(size_t)H*sizeof(float)));
                    tensors[zb]=zb_ptr; shapes[zb]={1,H};
                }
                b=tensors[zb];
            }
            if (!h0) {
                const std::string zh0="__rnn_zero_h0::"+op.output_id;
                if (!tensors.count(zh0)){
                    float* zh0_ptr=nullptr; CUDA_CHECK(cudaMalloc(&zh0_ptr,(size_t)B*H*sizeof(float)));
                    CUDA_CHECK(cudaMemset(zh0_ptr,0,(size_t)B*H*sizeof(float)));
                    tensors[zh0]=zh0_ptr; shapes[zh0]={1,H};
                }
                h0=tensors[zh0];
            }

            float* dX=nullptr; float* dWx=nullptr; float* dWh=nullptr; float* db=nullptr; float* dh0=nullptr;
            CUDA_CHECK(cudaMalloc(&dX, (size_t)B*T*D*sizeof(float)));
            CUDA_CHECK(cudaMalloc(&dWx,(size_t)D*H*sizeof(float)));
            CUDA_CHECK(cudaMalloc(&dWh,(size_t)H*H*sizeof(float)));
            CUDA_CHECK(cudaMalloc(&db, (size_t)H*sizeof(float)));
            CUDA_CHECK(cudaMalloc(&dh0,(size_t)B*H*sizeof(float)));
            CUDA_CHECK(cudaMemset(dX,  0,(size_t)B*T*D*sizeof(float)));
            CUDA_CHECK(cudaMemset(dWx, 0,(size_t)D*H*sizeof(float)));
            CUDA_CHECK(cudaMemset(dWh, 0,(size_t)H*H*sizeof(float)));
            CUDA_CHECK(cudaMemset(db,  0,(size_t)H*sizeof(float)));
            CUDA_CHECK(cudaMemset(dh0, 0,(size_t)B*H*sizeof(float)));

            RnnActivation rnn_act = RNN_TANH;
            launch_rnn_backward_simple(X, Wx, Wh, b, h0, H_seq, dH_T, dH_seq,
                                       dX, dWx, dWh, db, dh0, B, T, D, H, rnn_act, 0);
            CUDA_CHECK(cudaGetLastError());

            gradients[A_id] = dX;
            reg_grad(gradients, Wx_id,      dWx);
            reg_grad(gradients, op.param_id, dWx); // dual-key safety
            if (!Wh_id.empty()) reg_grad(gradients, Wh_id, dWh); else CUDA_CHECK(cudaFree(dWh));
            if (!b_id.empty())  reg_grad(gradients, b_id,  db);  else CUDA_CHECK(cudaFree(db));
            if (!h0_id.empty()) reg_grad(gradients, h0_id, dh0); else CUDA_CHECK(cudaFree(dh0));
            break;
        }

        default: break;
        } // switch

        // 공통: grad_input_full 등록 (FLATTEN/특수 continue 케이스 제외)
        if (op.op_type != FLATTEN) {
            if (!grad_input_full) {
                std::fprintf(stderr, "[BW] grad_input_full null: op=%d\n", op.op_type);
            } else {
                gradients[A_id] = grad_input_full;
            }
        }
    } // for
}
