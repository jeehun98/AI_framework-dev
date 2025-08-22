// run_graph_backward.cu (TF32 + strided-batched + fused softmax-xent, with batch-mean grads)

#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "executor/run_graph.cuh"
#include "activation/activation_ops.cuh"
#include "softmax/softmax_kernels.cuh"
#include "cnn/cnn_kernels.cuh"
#include "op_structs.cuh"
#include "loss/loss_kernels.cuh"
#include "reduce/reduce_stride.cuh"
#include "reduce/reduce_ops.cuh" 
#include "pooling/pooling_ops.cuh"
#include "pooling/pooling_kernels.cuh"

#include "ge/pack_utils.cuh"
#include "ge/cuda_check.cuh"
#include "ge/cublas_utils.cuh"
#include "ge/gemm_rm.cuh"
#include "ge/act_map.cuh"
#include "ge/fill.cuh"

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

static inline void checkCudaLast(const char* where) {
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "[CUDA][ERR] %s: %s\n", where, cudaGetErrorString(e));
    }
}
static inline void checkCudaSync(const char* where) {
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "[CUDA][SYNC] %s: %s\n", where, cudaGetErrorString(e));
    }
}

void run_graph_backward(
    const std::vector<OpStruct>& E,
    std::unordered_map<std::string, float*>& tensors,
    std::unordered_map<std::string, Shape>& shapes,
    std::unordered_map<std::string, float*>& gradients,
    const std::string& final_output_id,
    int batch_size)
{
    auto h = ge_cublas();

    std::string grad_start_id = final_output_id;
    bool fused_softmax = false;
    std::string fused_softmax_in_id, fused_softmax_out_id;

    // 1) LOSS backward: dL/dy_pred (혹은 fused면 dL/dz) 생성
    if (!E.empty() && E.back().op_type == LOSS) {
        const OpStruct& loss_op = E.back();
        const std::string loss_type = loss_op.extra_params.loss_type;
        const std::string label_id  = loss_op.extra_params.label_id;

        const float* y_true = tensors[label_id];
        const float* y_pred = tensors[loss_op.input_id];

        Shape shp = shapes[loss_op.input_id];
        const int C = shp.cols;
        const int rows_per_sample = shp.rows;          // 보통 1
        const int B = batch_size * rows_per_sample;
        const int N = B * C;

        cudaStream_t stream = 0;

        // 직전 op이 SOFTMAX인지 확인 (fused 조건)
        const OpStruct* prev = nullptr;
        if (E.size() >= 2) {
            const OpStruct& cand = E[E.size()-2];
            if (cand.op_type == SOFTMAX && cand.output_id == loss_op.input_id) {
                prev = &cand;
            }
        }

        if (loss_type == "cce" && prev) {
            // ∂L/∂z = (p - y) / B
            float* dL_dz = nullptr;
            CUDA_CHECK(cudaMalloc(&dL_dz, (size_t)N * sizeof(float)));
            launch_softmax_xent_fused_backward(y_pred, y_true, dL_dz, B, C, stream);
            checkCudaLast("launch_softmax_xent_fused_backward");
            checkCudaSync("softmax_xent_fused_backward sync");

            fused_softmax = true;
            fused_softmax_in_id  = prev->input_id;   // z
            fused_softmax_out_id = prev->output_id;  // p
            grad_start_id = prev->input_id;
            gradients[prev->input_id] = dL_dz;
        } else {
            float* dL_dy = nullptr;
            CUDA_CHECK(cudaMalloc(&dL_dy, (size_t)N * sizeof(float)));

            if (loss_type == "bce") {
                launch_bce_loss_backward(y_true, y_pred, dL_dy, N, B, stream);
                checkCudaLast("launch_bce_loss_backward");
                checkCudaSync("bce_backward sync");
            } else if (loss_type == "mse") {
                launch_mse_loss_backward(y_true, y_pred, dL_dy, N, stream);
                checkCudaLast("launch_mse_loss_backward");
                checkCudaSync("mse_backward sync");
            } else if (loss_type == "cce") {
                launch_cce_loss_backward(y_true, y_pred, dL_dy, B, C, stream);
                checkCudaLast("launch_cce_loss_backward");
                checkCudaSync("cce_backward sync");
            } else {
                std::fprintf(stderr, "[LOSS][BW] unsupported: %s\n", loss_type.c_str());
            }

            grad_start_id = loss_op.input_id;
            gradients[loss_op.input_id] = dL_dy;
        }
    }

    // 2) 나머지 역전파
    for (auto it = E.rbegin(); it != E.rend(); ++it) {
        const OpStruct& op = *it;
        if (op.op_type == LOSS) continue;

        if (fused_softmax && op.op_type == SOFTMAX && op.output_id == fused_softmax_out_id) {
            continue;
        }

        float* input = tensors[op.input_id];
        float* param = (!op.param_id.empty() && tensors.count(op.param_id))
                         ? tensors[op.param_id] : nullptr;
        float* grad_out_full = gradients[op.output_id];
        if (!grad_out_full && op.op_type != FLATTEN) continue;

        Shape in_shape  = shapes[op.input_id];
        Shape out_shape = shapes[op.output_id];
        if (out_shape.rows == 0 || out_shape.cols == 0) out_shape = in_shape;

        const int M = out_shape.rows;   // == in_shape.rows
        const int N = out_shape.cols;
        const int K = in_shape.cols;
        const int in_size  = in_shape.rows * in_shape.cols;
        const int out_size = out_shape.rows * out_shape.cols;

        float* grad_input_full = nullptr;
        if (op.op_type != FLATTEN) {
            CUDA_CHECK(cudaMalloc(&grad_input_full, (size_t)batch_size * in_size * sizeof(float)));
        }

        if (!gradients.count(op.output_id) && op.op_type != FLATTEN) {
            std::fprintf(stderr,
                "[BW][SKIP] op=%d id=%s: grad_out missing\n",
                op.op_type, op.output_id.c_str());
        }

        switch (op.op_type) {
        case MATMUL: {
            if (!param) break;

            // dX = dY · W^T  (B, M, K)
            gemm_rm_strided_batched_tf32(
                h,
                /*transA=*/false, /*transB=*/true,
                /*M=*/M, /*N=*/K, /*K=*/N,
                /*A =*/ grad_out_full,   /*lda =*/ N, /*strideA =*/ (long long)M * N,
                /*B =*/ param,           /*ldb =*/ N, /*strideB =*/ 0LL,
                /*C =*/ grad_input_full, /*ldc =*/ K, /*strideC =*/ (long long)M * K,
                /*batch=*/batch_size,
                /*alpha=*/1.f, /*beta=*/0.f
            );

            // dW = sum_b (X_b^T · dY_b)
            float* dW_tmp = nullptr; // (B, K, N)
            CUDA_CHECK(cudaMalloc(&dW_tmp, (size_t)batch_size * K * N * sizeof(float)));

            gemm_rm_strided_batched_tf32(
                h,
                /*transA=*/true, /*transB=*/false,
                /*M=*/K, /*N=*/N, /*K=*/M,
                /*A =*/ input,          /*lda =*/ K, /*strideA =*/ (long long)M * K,
                /*B =*/ grad_out_full,  /*ldb =*/ N, /*strideB =*/ (long long)M * N,
                /*C =*/ dW_tmp,         /*ldc =*/ N, /*strideC =*/ (long long)K * N,
                /*batch=*/batch_size,
                /*alpha=*/1.f, /*beta=*/0.f
            );

            // 배치축 합산
            float* dW_accum = nullptr;                 // (K, N)
            CUDA_CHECK(cudaMalloc(&dW_accum, (size_t)K * N * sizeof(float)));

            float* onesB = nullptr;
            CUDA_CHECK(cudaMalloc(&onesB, (size_t)batch_size * sizeof(float)));
            {
                int thr = 256, blk = (batch_size + thr - 1) / thr;
                ge_fill_kernel<<<blk, thr>>>(onesB, 1.0f, batch_size);
            }

            // C(1, K*N) = A(1, B) · B(B, K*N)
            gemm_rm_tf32(
                h, false, false,
                /*M=*/1, /*N=*/(K * N), /*K=*/batch_size,
                /*A=*/onesB,     /*lda=*/batch_size,
                /*B=*/dW_tmp,    /*ldb=*/(K * N),
                /*C=*/dW_accum,  /*ldc=*/(K * N),
                1.f, 0.f
            );

            gradients[op.param_id] = dW_accum;

            CUDA_CHECK(cudaFree(dW_tmp));
            CUDA_CHECK(cudaFree(onesB));
            break;
        }

        case ADD: {
            // ===== 방어적 로깅/가드 =====
            const Shape out_shape = shapes[op.output_id];  // (= in_shape)
            const Shape bshape    = shapes[op.param_id];
            const int rows_per_sample = out_shape.rows;
            const int cols            = out_shape.cols;
            const int rowsB           = batch_size * rows_per_sample;

            // 기존 코드 어딘가에서 쓰는 out_size가 "샘플당 총 요소 수"라면:
            const int out_size_expected = rows_per_sample * cols;
            if (out_size != out_size_expected) {
                std::fprintf(stderr,
                    "[ADD/BWD] size mismatch: out_size=%d but rows*cols=%d*%d=%d (op=%s, param=%s)\n",
                    out_size, rows_per_sample, cols, out_size_expected,
                    op.output_id.c_str(), op.param_id.c_str());
                // 안전하게 리턴하거나 에러 처리
            }

            // 포인터 유효성(기본 체크)
            if (!grad_out_full || !grad_input_full) {
                std::fprintf(stderr, "[ADD/BWD] null grad buffer(s) (op=%s)\n", op.output_id.c_str());
            }

            // dX = dY
            {
                const size_t bytes = (size_t)rowsB * cols * sizeof(float);
                CUDA_CHECK(cudaMemcpy(grad_input_full, grad_out_full, bytes, cudaMemcpyDeviceToDevice));
            }

            // bias shape 판정
            const bool bias_rowwise = (bshape.rows == 1 && bshape.cols == cols)
                                || (bshape.rows == cols && bshape.cols == 1);

            const bool bias_colwise = (bshape.rows == 1 && bshape.cols == rows_per_sample)
                                || (bshape.rows == rows_per_sample && bshape.cols == 1);

            if (bias_rowwise) {
                // dB (len=cols) = sum over rows (rowsB)
                float* grad_bias = nullptr;
                CUDA_CHECK(cudaMalloc(&grad_bias, (size_t)cols * sizeof(float)));
                launch_reduce_over_rows(/*in=*/grad_out_full, /*out=*/grad_bias, rowsB, cols);
                CUDA_CHECK(cudaDeviceSynchronize()); // 🔎 디버깅용
                gradients[op.param_id] = grad_bias;
            }
            else if (bias_colwise) {
                // 1) 각 행 합(열 방향 축소): temp_rows [rowsB]
                float* temp_rows = nullptr;
                CUDA_CHECK(cudaMalloc(&temp_rows, (size_t)rowsB * sizeof(float)));
                launch_reduce_over_cols(/*in=*/grad_out_full, /*out=*/temp_rows, rowsB, cols);
                CUDA_CHECK(cudaDeviceSynchronize()); // 🔎

                // 2) 배치 방향 합: dB [rows_per_sample]
                float* grad_bias = nullptr;
                CUDA_CHECK(cudaMalloc(&grad_bias, (size_t)rows_per_sample * sizeof(float)));
                launch_reduce_batch_stride(/*in=*/temp_rows, /*out=*/grad_bias, rows_per_sample, batch_size);
                CUDA_CHECK(cudaDeviceSynchronize()); // 🔎

                gradients[op.param_id] = grad_bias;
                CUDA_CHECK(cudaFree(temp_rows));
            }
            else {
                std::fprintf(stderr,
                    "[ADD/BWD] unsupported bias shape (%d,%d) for out(%d,%d). "
                    "Expect row-wise(len=cols) or channel-wise(len=rows).\n",
                    bshape.rows, bshape.cols, out_shape.rows, out_shape.cols);

                // 최소한 0으로 된 grad_bias라도 남김
                const int len = std::max(bshape.rows * bshape.cols, 1);
                float* grad_bias = nullptr;
                CUDA_CHECK(cudaMalloc(&grad_bias, (size_t)len * sizeof(float)));
                CUDA_CHECK(cudaMemset(grad_bias, 0, (size_t)len * sizeof(float)));
                gradients[op.param_id] = grad_bias;
            }
            break;
        }

        // -------- 활성화 계열 --------
        case SIGMOID:
        case RELU:
        case TANH:
        case LEAKY_RELU:
        case ELU:
        case GELU:
        case SILU:
        {
            const int rowsB = batch_size * out_shape.rows;
            const int cols  = out_shape.cols;

            const float* gout = grad_out_full;           // dL/dout
            const float* out  = tensors[op.output_id];   // f(z)
            const float* in   = tensors[op.input_id];    // z
            float* gin        = grad_input_full;         // dL/din

            const int act = ge_map_act_type(op.op_type);
            const float alpha = op.extra_params.alpha;
            const int gelu_tanh_flag = op.extra_params.gelu_tanh ? 1 : 0;

            cudaStream_t stream = 0;

            launch_activation_backward(
                gout, in, out, gin,
                rowsB, cols, act, alpha, gelu_tanh_flag, stream
            );
            CUDA_CHECK(cudaGetLastError());
            break;
        }

        case SOFTMAX:
        {
            const int rowsB = batch_size * out_shape.rows;
            const int cols  = out_shape.cols;

            const float* gout = grad_out_full;           // dL/dY
            const float* y    = tensors[op.output_id];   // Y = softmax(X)
            float* gin        = grad_input_full;         // dL/dX

            float temperature = (op.extra_params.temperature > 0.f)
                              ? op.extra_params.temperature : 1.f;
            cudaStream_t stream = 0;

            launch_softmax_backward(
                gout, y, gin, rowsB, cols, temperature, stream
            );
            CUDA_CHECK(cudaGetLastError());
            break;
        }

        case FLATTEN: {
            gradients[op.input_id] = grad_out_full;
            continue; // grad_input_full 할당 안 함
        }

        // run_graph_backward.cu — switch(op.op_type) 내부
        case CONV2D: {
            const OpExtraParams& ex = op.extra_params;

            const int N    = batch_size;
            const int Cin  = ex.input_c;
            const int Hin  = ex.input_h;
            const int Win  = ex.input_w;
            const int Cout = ex.output_c;
            const int Kh   = ex.kernel_h;
            const int Kw   = ex.kernel_w;
            const int Sh   = (ex.stride_h > 0 ? ex.stride_h : 1);
            const int Sw   = (ex.stride_w > 0 ? ex.stride_w : 1);
            const int Ph   = ex.padding_h;
            const int Pw   = ex.padding_w;

            // 엔진과 동일한 식
            const int Hout = (Hin + 2*Ph - Kh) / Sh + 1;
            const int Wout = (Win + 2*Pw - Kw) / Sw + 1;

            // 텐서 포인터 (NCHW)
            const float* X    = tensors[op.input_id];     // [N, Cin, Hin, Win]
            const float* W    = tensors[op.param_id];     // [Cout, Cin, Kh, Kw] (contiguous)
            const float* dYrm = gradients[op.output_id];  // per-sample row-major: [Cout, Hout*Wout]

            if (!X || !W) {
                std::fprintf(stderr, "[BW][CONV2D] missing X or W (X=%p, W=%p) for op=%s param=%s\n",
                            (void*)X, (void*)W, op.output_id.c_str(), op.param_id.c_str());
                break;
            }
            if (!dYrm) {
                std::fprintf(stderr, "[BW][CONV2D] missing grad_out for %s (expect key='%s')\n",
                            op.output_id.c_str(), op.output_id.c_str());
                break;
            }

            // rm(B, Cout, Hout*Wout) → NCHW 패킹
            float* dY = nullptr; // [N, Cout, Hout, Wout]
            CUDA_CHECK(cudaMalloc(&dY, (size_t)N * Cout * Hout * Wout * sizeof(float)));
            launch_pack_rm_to_nchw(/*rm=*/dYrm, /*nchw=*/dY, N, Cout, Hout, Wout);
            CUDA_CHECK(cudaGetLastError());

            // 출력 그라드 버퍼 준비
            float* dX = nullptr; // [N, Cin, Hin, Win]
            float* dW = nullptr; // [Cout, Cin, Kh, Kw]
            CUDA_CHECK(cudaMalloc(&dX, (size_t)N * Cin * Hin * Win * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&dW, (size_t)Cout * Cin * Kh * Kw * sizeof(float)));
            CUDA_CHECK(cudaMemset(dX, 0, (size_t)N * Cin * Hin * Win * sizeof(float)));
            CUDA_CHECK(cudaMemset(dW, 0, (size_t)Cout * Cin * Kh * Kw * sizeof(float)));

            // dX
            launch_conv2d_backward_input_nchw(
                /*dY=*/dY, /*W=*/W, /*dX=*/dX,
                /*N=*/N, /*Hin=*/Hin, /*Win=*/Win, /*Cin=*/Cin,
                /*Hout=*/Hout, /*Wout=*/Wout, /*Cout=*/Cout,
                /*Kh=*/Kh, /*Kw=*/Kw, /*Sh=*/Sh, /*Sw=*/Sw, /*Ph=*/Ph, /*Pw=*/Pw,
                /*stream=*/0);
            CUDA_CHECK(cudaGetLastError());

            // dW
            launch_conv2d_backward_weight_nchw(
                /*dY=*/dY, /*X=*/X, /*dW=*/dW,
                /*N=*/N, /*Hin=*/Hin, /*Win=*/Win, /*Cin=*/Cin,
                /*Hout=*/Hout, /*Wout=*/Wout, /*Cout=*/Cout,
                /*Kh=*/Kh, /*Kw=*/Kw, /*Sh=*/Sh, /*Sw=*/Sw, /*Ph=*/Ph, /*Pw=*/Pw,
                /*stream=*/0);
            CUDA_CHECK(cudaGetLastError());

            // gradients에 등록
            if (op.param_id.empty()) {
                std::fprintf(stderr,
                    "[BW][CONV2D] op.param_id EMPTY for output=%s (Cout=%d Cin=%d KhKw=%dx%d)\n",
                    op.output_id.c_str(), Cout, Cin, Kh, Kw);
            } else {
                gradients[op.input_id] = dX;  // dX는 이전 op로 전달
                gradients[op.param_id] = dW;  // conv weight grad
                std::fprintf(stderr,
                    "[BW][CONV2D] set grad for '%s' (dW elems=%zu) and '%s' (dX elems=%zu)\n",
                    op.param_id.c_str(), (size_t)Cout*Cin*Kh*Kw,
                    op.input_id.c_str(), (size_t)N*Cin*Hin*Win);
            }

            CUDA_CHECK(cudaFree(dY));
            break;
        }

        case POOL_MAX:
        case POOL_AVG: {
            const OpExtraParams& ex = op.extra_params;

            // 입력 NCHW 메타
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

            const int Hout = (H + 2*pH - dH*(kH - 1) - 1) / sH + 1;
            const int Wout = (W + 2*pW - dW*(kW - 1) - 1) / sW + 1;

            if (Hout <= 0 || Wout <= 0) {
                std::fprintf(stderr, "[BW][POOL] invalid out size Hout=%d Wout=%d\n", Hout, Wout);
                break;
            }

            // grad_out (엔진 2D 뷰) -> NCHW 패킹
            const float* dYrm = gradients[op.output_id]; // per-sample: [C, Hout*Wout]
            if (!dYrm) {
                std::fprintf(stderr, "[BW][POOL] missing grad_out for %s\n", op.output_id.c_str());
                break;
            }

            float* dY = nullptr; // [N,C,Hout,Wout]
            CUDA_CHECK(cudaMalloc(&dY, (size_t)N * C * Hout * Wout * sizeof(float)));
            launch_pack_rm_to_nchw(/*rm=*/dYrm, /*nchw=*/dY, N, C, Hout, Wout);
            CUDA_CHECK(cudaGetLastError());

            // dX 준비
            float* dX = nullptr; // [N,C,H,W]
            CUDA_CHECK(cudaMalloc(&dX, (size_t)N * C * H * W * sizeof(float)));
            CUDA_CHECK(cudaMemset(dX, 0, (size_t)N * C * H * W * sizeof(float)));

            // 공통 Pool 파라미터
            Pool2DParams p{};
            p.N=N; p.C=C; p.H=H; p.W=W;
            p.H_out=Hout; p.W_out=Wout;
            p.kernel_h=kH; p.kernel_w=kW;
            p.stride_h=sH; p.stride_w=sW;
            p.pad_h=pH; p.pad_w=pW;
            p.dilation_h=dH; p.dilation_w=dW;
            p.avg_inclusive = ex.count_include_pad;

            if (op.op_type == POOL_MAX) {
                // forward 때 저장해둔 argmax 필요
                auto it = tensors.find("__pool_argmax::" + op.output_id);
                if (it == tensors.end() || !it->second) {
                    std::fprintf(stderr, "[BW][POOL_MAX] missing argmax for %s\n", op.output_id.c_str());
                    CUDA_CHECK(cudaFree(dY));
                    CUDA_CHECK(cudaFree(dX));
                    break;
                }
                const int32_t* argmax = reinterpret_cast<const int32_t*>(it->second);

                maxpool2d_backward(/*grad_y=*/dY, /*grad_x=*/dX, /*argmax=*/argmax, /*p=*/p, /*stream=*/0);
                CUDA_CHECK(cudaGetLastError());
            } else { // POOL_AVG
                avgpool2d_backward(/*grad_y=*/dY, /*grad_x=*/dX, /*p=*/p, /*stream=*/0);
                CUDA_CHECK(cudaGetLastError());
            }

            // 등록: 입력 텐서의 그래디언트
            gradients[op.input_id] = dX;

            // 정리
            CUDA_CHECK(cudaFree(dY));
            break;
        }



        default:
            break;
        }

        if (op.op_type != FLATTEN) {
            if (!grad_input_full) {
                std::fprintf(stderr, "[BW] grad_input_full is null: op=%d\n", op.op_type);
            } else {
                gradients[op.input_id] = grad_input_full;
            }
        }
    }
}
