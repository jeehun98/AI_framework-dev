// run_graph.cu (vector/legacy resolve + time-kernels defined here + CNN/Pooling/RNN)

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <unordered_map>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../quant/quant_types.cuh"
#include "../quant/observers.cuh"
#include "../quant/quant_kernels.cuh"
#include "../quant/int8_gemm_dp4a.cuh"
#include "../quant/epilogue_kernels.cuh"

#include "run_graph_utils.cuh"    // resolve_A_id / resolve_B_id / ge_ensure_output 등

#include "../rnn/rnn_kernels.cuh"
#include "../pooling/pooling_ops.cuh"
#include "../pooling/pooling_kernels.cuh"
#include "../executor/run_graph.cuh"
#include "../activation/activation_ops.cuh"
#include "../softmax/softmax_kernels.cuh"
#include "../bias/add_bias_rowwise.cuh"
#include "../op_structs.cuh"

#include "../ge/cuda_check.cuh"
#include "../ge/cublas_utils.cuh"
#include "../ge/gemm_rm.cuh"
#include "../ge/act_map.cuh"
#include "../ge/alloc_utils.cuh"
#include "../ge/pack_utils.cuh"

// CNN 전용
#include "../cnn/cnn_kernels.cuh"

#ifndef TILE_WIDTH
#define TILE_WIDTH 16
#endif

/* ============================== Debug utils =============================== */

static void debug_l2(const char* name, const float* dptr, size_t n_elems) {
    std::vector<float> h(n_elems);
    cudaMemcpy(h.data(), dptr, n_elems * sizeof(float), cudaMemcpyDeviceToHost);
    double s = 0.0;
    for (size_t i = 0; i < n_elems; ++i) { double v = h[i]; s += v * v; }
    std::fprintf(stderr, "[GRAD] %s L2=%.6e\n", name, std::sqrt(s));
}

/* ======================= Tensor lookup (RO/RW helpers) ==================== */

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

/* ======================= Time utilities (simple kernels) ================== */
// per-sample view: rows=T, cols=D  (물리 메모리: [B, rows, cols])
__global__ void slice_time_kernel(const float* __restrict__ X,
                                  float* __restrict__ Y,
                                  int B, int T, int D, int t) {
    const int b = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x; // 0..D-1
    if (b >= B || i >= D) return;
    const size_t off_in  = ((size_t)b * T + t) * D;
    const size_t off_out = ((size_t)b * 1 + 0) * D;      // (1,D)
    Y[off_out + i] = X[off_in + i];
}

__global__ void concat_time_kernel(const float* __restrict__ X1, int t1,
                                   const float* __restrict__ X2, int t2,
                                   float* __restrict__ Y, int B, int D) {
    // Y rows = t1 + t2, cols = D
    const int b = blockIdx.z;
    const int r = blockIdx.y;  // 0..t1+t2-1
    const int c = blockIdx.x * blockDim.x + threadIdx.x; // 0..D-1
    if (b >= B || c >= D) return;
    const int T = t1 + t2;
    const size_t y_off = ((size_t)b * T + r) * D + c;
    if (r < t1) {
        const size_t x1_off = ((size_t)b * t1 + r) * D + c;
        Y[y_off] = X1[x1_off];
    } else {
        const int r2 = r - t1;
        const size_t x2_off = ((size_t)b * t2 + r2) * D + c;
        Y[y_off] = X2[x2_off];
    }
}

/* ============== Legacy→Vector normalize (defensive for safety) ============ */

static inline void normalize_legacy_local(OpStruct& n) {
    if (!n.input_id.empty() && n.inputs.empty())  n.inputs.push_back(n.input_id);
    if (!n.param_id.empty() && n.params.empty())  n.params.push_back(n.param_id);
}
static inline std::vector<OpStruct> normalize_graph_local(const std::vector<OpStruct>& E) {
    std::vector<OpStruct> out; out.reserve(E.size());
    for (auto n : E) { normalize_legacy_local(n); out.push_back(std::move(n)); }
    return out;
}

/* ================================ Main exec =============================== */

void run_graph_cuda(
    const std::vector<OpStruct>& E_in,
    std::unordered_map<std::string, float*>& tensors,
    std::unordered_map<std::string, Shape>& shapes,
    float* out_host,
    const std::string& final_output_id,
    int batch_size)
{
    auto h = ge_cublas();

    // ✅ 실행 전, 전 노드를 legacy→vector로 정규화
    const auto E = normalize_graph_local(E_in);

    for (size_t i = 0; i < E.size(); ++i) {
        const auto& op = E[i];
        if (op.op_type == LOSS) continue;

        // ---- 공통 입력/shape 조회 (legacy + vector 혼용 대비) ----
        const std::string in_id = resolve_A_id(op);
        auto it_in = tensors.find(in_id);
        if (it_in == tensors.end() || it_in->second == nullptr) {
            std::fprintf(stderr, "[ERR] missing input tensor: %s\n", in_id.c_str());
            break;
        }
        float* input = it_in->second;

        const auto it_inshape = shapes.find(in_id);
        if (it_inshape == shapes.end()) {
            std::fprintf(stderr, "[ERR] missing input shape: %s\n", in_id.c_str());
            break;
        }
        const Shape in_shape = it_inshape->second;

        // (필요 시) 두 번째 피연산자: 가중치/편향 or 다른 입력
        std::string param_id = resolve_B_id(op);
        float* param = nullptr;
        if (!param_id.empty()) {
            auto it_p = tensors.find(param_id);
            if (it_p != tensors.end()) param = it_p->second;
        }

        Shape out_shape = in_shape; // 기본은 동일

        switch (op.op_type) {

        /* ============================== Linear ops ============================== */

        case MATMUL: {
            // A x W
            const std::string A_id = in_id;
            const std::string W_id = param_id;
            auto itA = tensors.find(A_id);
            auto itW = tensors.find(W_id);
            if (itA == tensors.end() || itW == tensors.end()) {
                std::fprintf(stderr, "[MATMUL][ERR] missing A=%s or W=%s\n", A_id.c_str(), W_id.c_str());
                break;
            }
            const Shape aS = shapes[A_id]; // [M,K]
            const Shape wS = shapes[W_id]; // [K,N]
            const int M = aS.rows;
            const int K = aS.cols;
            const int N = wS.cols;
            if (wS.rows != K) {
                std::fprintf(stderr, "[MATMUL][ERR] dim mismatch: A(K=%d) vs W(rows=%d)\n", K, wS.rows);
                break;
            }
            out_shape = { M, N };

            // 🔒 Bias fuse는 디버깅 안정성을 위해 기본 비활성화 (필요 시 아래 주석 해제)
            // bool fuse_bias = false;
            // float* bias_ptr = nullptr;
            // std::string out_id = op.output_id;
            // if ((i + 1) < E.size()) {
            //     const auto& nx = E[i + 1];
            //     const std::string nxA = resolve_A_id(nx);
            //     if ((nx.op_type == ADD || nx.op_type == ADD_BIAS) && nxA == op.output_id) {
            //         const std::string b_id = resolve_B_id(nx);
            //         if (tensors.count(b_id)) {
            //             const Shape bS = shapes[b_id];
            //             const bool row_bias = (bS.rows == 1 && bS.cols == N) || (bS.rows == N && bS.cols == 1);
            //             if (row_bias) {
            //                 fuse_bias = true;
            //                 bias_ptr = tensors[b_id];
            //                 out_id = nx.output_id;
            //             }
            //         }
            //     }
            // }

            float* Y = ge_ensure_output(tensors, shapes, op.output_id /*out_id*/, out_shape, batch_size);

            const long long strideA = (long long)M * K;
            const long long strideC = (long long)M * N;

            gemm_rm_strided_batched_tf32(
                h,
                /*transA=*/false, /*transB=*/false,
                /*M=*/M, /*N=*/N, /*K=*/K,
                /*A*/ itA->second,  /*lda=*/K, /*strideA=*/strideA,
                /*B*/ itW->second,  /*ldb=*/N, /*strideB=*/0LL,   // shared weight
                /*C*/ Y,            /*ldc=*/N, /*strideC=*/strideC,
                /*batch=*/batch_size,
                /*alpha=*/1.f, /*beta=*/0.f
            );

            // if (fuse_bias) {
            //     const int rowsB = batch_size * M;
            //     const int cols  = N;
            //     launch_add_bias_rowwise(Y, bias_ptr, Y, rowsB, cols);
            //     CUDA_CHECK(cudaGetLastError());
            //     ++i; // fused op skip
            // }
            break;
        }

        case ADD_BIAS: {
            // 입력 + (1,C) or (C,1)
            const std::string A_id = in_id;
            const std::string b_id = param_id;
            auto itA = tensors.find(A_id);
            auto itB = tensors.find(b_id);
            if (itA == tensors.end() || itB == tensors.end()) {
                std::fprintf(stderr, "[ADD_BIAS][ERR] missing tensors A=%s or b=%s\n",
                             A_id.c_str(), b_id.c_str());
                break;
            }
            const Shape aS = shapes[A_id];
            const Shape bS = shapes[b_id];

            Shape outS = aS;
            float* Y = ge_ensure_output(tensors, shapes, op.output_id, outS, batch_size);

            const int rowsB = batch_size * outS.rows;
            const int C     = outS.cols;
            if (!((bS.rows == 1 && bS.cols == C) || (bS.rows == C && bS.cols == 1))) {
                std::fprintf(stderr, "[ADD_BIAS][ERR] bias shape must be (1,%d) or (%d,1), got (%d,%d)\n",
                             C, C, bS.rows, bS.cols);
                const size_t bytes = (size_t)rowsB * C * sizeof(float);
                CUDA_CHECK(cudaMemcpy(Y, itA->second, bytes, cudaMemcpyDeviceToDevice));
                break;
            }
            launch_add_bias_rowwise(itA->second, itB->second, Y, rowsB, C);
            CUDA_CHECK(cudaGetLastError());
            break;
        }

        case ADD: {
            const std::string A_id = in_id;
            const std::string B_id = param_id; // inputs[1] or params[0] or legacy
            auto itA = tensors.find(A_id);
            auto itB = tensors.find(B_id);
            if (itA == tensors.end() || itB == tensors.end()) {
                std::fprintf(stderr, "[ADD][ERR] missing inputs %s or %s\n", A_id.c_str(), B_id.c_str());
                break;
            }
            const Shape aS = shapes[A_id];
            const Shape bS = shapes[B_id];

            Shape outS = aS;
            float* Y = ge_ensure_output(tensors, shapes, op.output_id, outS, batch_size);

            const int rows_per_sample = outS.rows;
            const int cols            = outS.cols;
            const int rowsB           = batch_size * rows_per_sample;

            const bool bias_rowwise = (bS.rows == 1 && bS.cols == cols) ||
                                      (bS.rows == cols && bS.cols == 1);
            const bool bias_colwise = (bS.rows == 1 && bS.cols == rows_per_sample) ||
                                      (bS.rows == rows_per_sample && bS.cols == 1);

            if (bias_rowwise) {
                launch_add_bias_rowwise(itA->second, itB->second, Y, rowsB, cols);
                CUDA_CHECK(cudaGetLastError());
            } else if (bias_colwise) {
                launch_add_bias_colwise(itA->second, itB->second, Y, rowsB, cols, rows_per_sample);
                CUDA_CHECK(cudaGetLastError());
            } else {
                // fallback: 동형 add 필요 시 구현. 여기서는 rowwise로 가정.
                const size_t bytes = (size_t)rowsB * cols * sizeof(float);
                CUDA_CHECK(cudaMemcpy(Y, itA->second, bytes, cudaMemcpyDeviceToDevice));
                launch_add_bias_rowwise(Y, itB->second, Y, rowsB, cols);
                CUDA_CHECK(cudaGetLastError());
            }
            break;
        }

        /* ============================== Activations ============================= */

        case SIGMOID:
        case RELU:
        case TANH:
        case LEAKY_RELU:
        case ELU:
        case GELU:
        case SILU: {
            // out_shape 지정이 오면 그걸 우선, 아니면 in_shape
            out_shape = (shapes.count(op.output_id) ? shapes[op.output_id] : in_shape);
            float* output = ge_ensure_output(tensors, shapes, op.output_id, out_shape, batch_size);

            const int rowsB = batch_size * out_shape.rows;
            const int cols  = out_shape.cols;

            const float* bias_ptr = nullptr; // 옵션
            if (!param_id.empty()) {
                auto itb = tensors.find(param_id);
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

        /* ================================ Softmax =============================== */

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

        /* ================================= Flatten ============================== */

        case FLATTEN: {
            auto it_outshape = shapes.find(op.output_id);
            if (it_outshape == shapes.end()) {
                std::fprintf(stderr, "[FLATTEN][ERR] missing out shape for %s\n", op.output_id.c_str());
                break;
            }
            const Shape out_shape_f = it_outshape->second;
            float* output = ge_ensure_output(tensors, shapes, op.output_id, out_shape_f, batch_size);

            const size_t elems_in  = (size_t)batch_size * in_shape.rows    * in_shape.cols;
            const size_t elems_out = (size_t)batch_size * out_shape_f.rows  * out_shape_f.cols;
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

        /* ================================== Pool ================================= */

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
            const float* X = get_tensor_ro(tensors, in_id);        // [N,C,H,W]
            float* Y = ge_ensure_output(tensors, shapes, op.output_id, out_shape_p, N); // [N,C,Hout,Wout]

            if (!X || !Y) {
                std::fprintf(stderr, "[POOL][ERR] missing tensors X(%p) or Y(%p)\n", (void*)X, (void*)Y);
                break;
            }

            Pool2DParams p{};
            p.N = N; p.C = C; p.H = H; p.W = W;
            p.kernel_h = kH; p.kernel_w = kW;
            p.stride_h = sH; p.stride_w = sW;
            p.pad_h = pH;   p.pad_w = pW;
            p.dilation_h = dH; p.dilation_w = dW;
            p.H_out = Hout; p.W_out = Wout;
            p.avg_inclusive = ex.count_include_pad;

            if (op.op_type == POOL_MAX) {
                const size_t n_out = (size_t)N * C * Hout * Wout;
                int32_t* argmax = ensure_argmax_ws(tensors, n_out, "__pool_argmax::" + op.output_id);

                maxpool2d_forward(/*x=*/X, /*y=*/Y, /*argmax=*/argmax, /*p=*/p, /*stream=*/0);
                CUDA_CHECK(cudaGetLastError());
            } else {
                avgpool2d_forward(/*x=*/X, /*y=*/Y, /*p=*/p, /*stream=*/0);
                CUDA_CHECK(cudaGetLastError());
            }
            break;
        }

        /* ================================== Conv ================================= */

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

            float* X = tensors[in_id];    // [N, Cin, Hin, Win]
            float* W = tensors[param_id]; // [Cout, Cin, Kh, Kw] contiguous
            float* Y = ge_ensure_output(tensors, shapes, op.output_id, out_shape_c, N); // [N, Cout, Hout, Wout]

            launch_conv2d_forward_nchw(
                /*X=*/X, /*W=*/W, /*Y=*/Y,
                /*N=*/N, /*Hin=*/Hin, /*Win=*/Win, /*Cin=*/Cin,
                /*Hout=*/Hout, /*Wout=*/Wout, /*Cout=*/Cout,
                /*Kh=*/Kh, /*Kw=*/Kw, /*Sh=*/Sh, /*Sw=*/Sw, /*Ph=*/Ph, /*Pw=*/Pw,
                /*stream=*/0);
            CUDA_CHECK(cudaGetLastError());
            break;
        }

        /* =============================== Time utils ============================= */

        case SLICE_TIME: {
            // input (T,D) -> (1,D) at t
            const Shape aS = shapes[in_id]; // rows=T, cols=D
            const int T = aS.rows;
            const int D = aS.cols;
            int t = op.extra_params.time_index;
            if (t < 0) t = 0;
            if (t >= T) t = T - 1;

            Shape outS{ 1, D };
            float* Y = ge_ensure_output(tensors, shapes, op.output_id, outS, batch_size);

            const dim3 blk(256, 1, 1);
            const dim3 grd((D + 255) / 256, 1, batch_size);
            slice_time_kernel<<<grd, blk>>>(input, Y, batch_size, T, D, t);
            CUDA_CHECK(cudaGetLastError());
            break;
        }

        case CONCAT_TIME: {
            // inputs[0]=(t1,D), inputs[1]=(t2,D) -> (t1+t2, D)
            const std::string X1_id = in_id;
            const std::string X2_id = param_id; // B-operand
            auto it1 = tensors.find(X1_id);
            auto it2 = tensors.find(X2_id);
            if (it1 == tensors.end() || it2 == tensors.end()) {
                std::fprintf(stderr, "[CONCAT_TIME][ERR] missing tensors %s or %s\n",
                             X1_id.c_str(), X2_id.c_str());
                break;
            }
            const Shape s1 = shapes[X1_id]; // (t1,D)
            const Shape s2 = shapes[X2_id]; // (t2,D)
            if (s1.cols != s2.cols) {
                std::fprintf(stderr, "[CONCAT_TIME][ERR] feature mismatch: D1=%d D2=%d\n", s1.cols, s2.cols);
                break;
            }
            const int t1 = s1.rows, t2 = s2.rows, D = s1.cols;
            Shape outS{ t1 + t2, D };

            float* Y = ge_ensure_output(tensors, shapes, op.output_id, outS, batch_size);

            const dim3 blk(256, 1, 1);
            const dim3 grd((D + 255) / 256, t1 + t2, batch_size);
            concat_time_kernel<<<grd, blk>>>(it1->second, t1, it2->second, t2, Y, batch_size, D);
            CUDA_CHECK(cudaGetLastError());
            break;
        }

        case FILL_ZERO: {
            // output_id의 shape에 맞게 0으로 채움
            auto itS = shapes.find(op.output_id);
            if (itS == shapes.end()) {
                std::fprintf(stderr, "[FILL_ZERO][ERR] missing shape for %s\n", op.output_id.c_str());
                break;
            }
            float* Y = ge_ensure_output(tensors, shapes, op.output_id, itS->second, batch_size);
            const size_t bytes = (size_t)batch_size * itS->second.rows * itS->second.cols * sizeof(float);
            CUDA_CHECK(cudaMemsetAsync(Y, 0, bytes));
            CUDA_CHECK(cudaGetLastError());
            break;
        }

        /* ================================== RNN =================================== */
        case RNN: {
            // 메타
            const OpExtraParams& ex = op.extra_params;
            const int B = batch_size;
            const int T = (ex.time_steps  > 0 ? ex.time_steps  : in_shape.rows); // per-sample rows=T
            const int D = (ex.input_w     > 0 ? ex.input_w     : in_shape.cols); // per-sample cols=D
            const int H = (ex.hidden_size > 0 ? ex.hidden_size : ex.output_c);

            if (H <= 0 || T <= 0 || D <= 0) {
                std::fprintf(stderr, "[RNN][ERR] invalid meta: B=%d T=%d D=%d H=%d\n", B,T,D,H);
                break;
            }

            // 텐서 id: inputs / params (vector 우선, legacy 백업)
            const std::string X_id  = resolve_A_id(op);                    // inputs[0] or input_id
            const std::string Wx_id = (op.params.size() >= 1) ? op.params[0] : resolve_B_id(op);
            const std::string Wh_id = (op.params.size() >= 2) ? op.params[1] : "";
            const std::string b_id  = (op.params.size() >= 3) ? op.params[2] : "";
            const std::string h0_id = (op.params.size() >= 4) ? op.params[3] : "";

            // 포인터 조회
            float* X  = get_tensor_rw(tensors, X_id);                              // [B,T,D]
            float* Wx = (!Wx_id.empty() && tensors.count(Wx_id)) ? tensors[Wx_id] : nullptr; // [D,H]
            float* Wh = (!Wh_id.empty() && tensors.count(Wh_id)) ? tensors[Wh_id] : nullptr; // [H,H]
            float* b  = (!b_id.empty()  && tensors.count(b_id))  ? tensors[b_id]  : nullptr; // [H] or [1,H]
            float* h0 = (!h0_id.empty() && tensors.count(h0_id)) ? tensors[h0_id] : nullptr; // [B,H] opt

            if (!X || !Wx || !Wh) {
                std::fprintf(stderr, "[RNN][ERR] missing X/Wx/Wh: X=%p Wx=%p Wh=%p\n",(void*)X,(void*)Wx,(void*)Wh);
                break;
            }

            // bias / h0 없으면 내부 0 버퍼 준비(캐시)
            if (!b) {
                const std::string zb = "__rnn_zero_bias::" + op.output_id;
                if (!tensors.count(zb)) {
                    float* zb_ptr = nullptr;
                    CUDA_CHECK(cudaMalloc(&zb_ptr, (size_t)H * sizeof(float)));
                    CUDA_CHECK(cudaMemset(zb_ptr, 0, (size_t)H * sizeof(float)));
                    tensors[zb] = zb_ptr; shapes[zb] = {1, H};
                }
                b = tensors[zb];
            }
            if (!h0) {
                const std::string zh0 = "__rnn_zero_h0::" + op.output_id;
                if (!tensors.count(zh0)) {
                    float* zh0_ptr = nullptr;
                    CUDA_CHECK(cudaMalloc(&zh0_ptr, (size_t)B * H * sizeof(float)));
                    CUDA_CHECK(cudaMemset(zh0_ptr, 0, (size_t)B * H * sizeof(float)));
                    tensors[zh0] = zh0_ptr; shapes[zh0] = {1, H}; // per-sample view
                }
                h0 = tensors[zh0];
            }

            // 출력/시퀀스 버퍼
            // 출력 shape는 shape_map에 사전 지정되었으면 그것을 우선 사용
            Shape outS = (shapes.count(op.output_id) ? shapes[op.output_id] : Shape{1, H});
            float* H_T = ge_ensure_output(tensors, shapes, op.output_id, outS, B);

            // 항상 H_seq 저장(역전파 대비)
            const std::string HSEQ_id = op.output_id + "::__rnn_hseq";
            Shape seqS{ T, H };
            float* H_seq = ge_ensure_output(tensors, shapes, HSEQ_id, seqS, B);

            // 활성화(기본 tanh)
            RnnActivation rnn_act = RNN_TANH;

            // 실행
            launch_rnn_forward_simple(
                /*X=*/X, /*Wx=*/Wx, /*Wh=*/Wh, /*b=*/b,
                /*h0=*/h0, /*H_T=*/H_T, /*H_seq=*/H_seq,
                /*B,T,D,H*/ B, T, D, H, rnn_act,
                /*stream=*/0
            );
            CUDA_CHECK(cudaGetLastError());

            // shape 등록 명시
            shapes[op.output_id] = outS; // (1,H) 또는 (T,H)
            shapes[HSEQ_id]      = seqS; // (T,H)
            break;
        }

        /* ================================ default =============================== */

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
