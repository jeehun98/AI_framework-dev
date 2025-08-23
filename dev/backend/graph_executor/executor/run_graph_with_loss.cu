// run_graph_with_loss.cu (revised)
#include "run_graph_with_loss.cuh"
#include "run_graph.cuh"
#include "../loss/loss_kernels.cuh"
#include "../op_structs.cuh"

#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <cstdio>
#include <cmath>

#ifndef DEBUG_LOSS
#define DEBUG_LOSS 0
#endif
#ifndef DEBUG_SYNC
#define DEBUG_SYNC 0
#endif

#define CUDA_OK(stmt) do {                                      \
    cudaError_t __e = (stmt);                                   \
    if (__e != cudaSuccess) {                                   \
        std::fprintf(stderr, "[CUDA][ERR] %s:%d %s\n",          \
                     __FILE__, __LINE__, cudaGetErrorString(__e)); \
        return NAN;                                             \
    }                                                           \
} while(0)

static inline bool cuda_check_last(const char* where) {
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "[CUDA][ERR] %s: %s\n", where, cudaGetErrorString(e));
        return false;
    }
    return true;
}

#if DEBUG_LOSS
static void dump_first_k(const char* tag, const float* dev, int n, int k=8) {
    const int m = (n < k ? n : k);
    std::vector<float> h(m);
    cudaMemcpy(h.data(), dev, sizeof(float)*m, cudaMemcpyDeviceToHost);
    std::printf("[%s] ", tag);
    for (int i=0;i<m;++i) std::printf("%.6f ", h[i]);
    std::printf("\n");
}
#endif

float run_graph_with_loss_cuda(
    const std::vector<OpStruct>& E,
    std::unordered_map<std::string, float*>& tensors,
    std::unordered_map<std::string, Shape>& shapes,
    const std::string& final_output_id,
    const std::string& label_tensor_id,
    const std::string& loss_type,
    int batch_size)
{
    if (batch_size <= 0) {
        std::fprintf(stderr, "[LOSS] invalid batch_size=%d\n", batch_size);
        return NAN;
    }

    // 0) 손실 입력 선택
    std::string pred_id = final_output_id;
    if (!E.empty() && E.back().op_type == LOSS)
        pred_id = E.back().input_id;

    // 0-1) 사용자 제공 텐서 스냅샷
    std::unordered_set<std::string> user_keys;
    user_keys.reserve(tensors.size());
    for (const auto& kv : tensors) user_keys.insert(kv.first);

    // 1) Forward
    run_graph_cuda(E, tensors, shapes, /*out_host=*/nullptr, pred_id, batch_size);
    if (!cuda_check_last("run_graph_cuda")) return NAN;
#if DEBUG_SYNC
    CUDA_OK(cudaDeviceSynchronize());
#endif

    // 2) y_pred / y_true 확인
    auto it_pred = tensors.find(pred_id);
    if (it_pred == tensors.end() || it_pred->second == nullptr) {
        std::fprintf(stderr, "[LOSS] missing y_pred tensor: %s\n", pred_id.c_str());
        return NAN;
    }
    auto it_true = tensors.find(label_tensor_id);
    if (it_true == tensors.end() || it_true->second == nullptr) {
        std::fprintf(stderr, "[LOSS] missing y_true tensor: %s\n", label_tensor_id.c_str());
        return NAN;
    }
    auto it_pshape = shapes.find(pred_id);
    auto it_tshape = shapes.find(label_tensor_id);
    if (it_pshape == shapes.end() || it_tshape == shapes.end()) {
        std::fprintf(stderr, "[LOSS] missing shapes for y_pred/y_true\n");
        return NAN;
    }

    float* y_pred = it_pred->second;
    float* y_true = it_true->second;

    const Shape sp = it_pshape->second;
    const Shape st = it_tshape->second;

    const int rows_per_sample = sp.rows;
    const int num_classes     = sp.cols;
    const int B               = batch_size * rows_per_sample;
    const long long n_pred    = 1LL * B * num_classes;
    const long long n_true    = 1LL * (batch_size * st.rows) * st.cols;

    if (n_pred <= 0 || n_true <= 0) {
        std::fprintf(stderr, "[LOSS] invalid sizes: n_pred=%lld n_true=%lld\n", n_pred, n_true);
        return NAN;
    }
    // y_true 크기 검증: MSE/BCE는 동일 요소수, CCE는 (B,C) 매칭
    if (loss_type == "mse" || loss_type == "binary_crossentropy" || loss_type == "bce") {
        if (n_true != n_pred) {
            std::fprintf(stderr, "[LOSS] size mismatch: pred=%lld true=%lld (MSE/BCE require same)\n",
                         n_pred, n_true);
            return NAN;
        }
    } else if (loss_type == "cce") {
    // Expect y_true shape to match per-sample y_pred shape
    if (st.rows != rows_per_sample || st.cols != num_classes) {
        std::fprintf(stderr,
            "[LOSS] size mismatch for CCE: pred per-sample=(%d,%d), "
            "y_true=(%d,%d), batch=%d\n",
            rows_per_sample, num_classes, st.rows, st.cols, batch_size);
        return NAN;
        }
    }

#if DEBUG_LOSS
    {
        std::vector<float> hp(std::min<long long>(n_pred, 8));
        cudaMemcpy(hp.data(), y_pred, sizeof(float)*hp.size(), cudaMemcpyDeviceToHost);
        bool bad = false;
        for (float v : hp) if (!(v >= 0.0f && v <= 1.0f)) { bad = true; break; }
        if (bad) {
            std::printf("[LOSS][WARN] y_pred not in [0,1]. Check activation or use logits variant.\n");
            dump_first_k("y_pred", y_pred, (int)n_pred);
        }
    }
#endif

    // 3) 손실 계산
    float loss_value = NAN;
    if (loss_type == "mse") {
        loss_value = compute_mse_loss_cuda(y_true, y_pred, (int)n_pred);
    } else if (loss_type == "binary_crossentropy" || loss_type == "bce") {
        loss_value = compute_bce_loss_cuda(y_true, y_pred, (int)n_pred);
    } else if (loss_type == "cce") {
        loss_value = compute_cce_loss_cuda(y_true, y_pred, B, num_classes);
    } else {
        std::fprintf(stderr, "[LOSS] Unsupported loss type: %s\n", loss_type.c_str());
        return NAN;
    }
    if (!cuda_check_last("loss kernels")) return NAN;
#if DEBUG_SYNC
    CUDA_OK(cudaDeviceSynchronize());
#endif

#if DEBUG_LOSS
    {
        const double eps = 1e-7;
        std::vector<float> hp((size_t)n_pred), ht((size_t)n_pred);
        CUDA_OK(cudaMemcpy(hp.data(), y_pred, sizeof(float)*n_pred, cudaMemcpyDeviceToHost));
        CUDA_OK(cudaMemcpy(ht.data(), y_true, sizeof(float)*n_pred, cudaMemcpyDeviceToHost));
        double acc = 0.0;
        if (loss_type == "mse") {
            for (long long i=0;i<n_pred;++i){ double d=hp[i]-ht[i]; acc+=d*d; }
            acc /= (double)n_pred;
        } else if (loss_type == "binary_crossentropy" || loss_type == "bce") {
            for (long long i=0;i<n_pred;++i){
                const double yp = std::min(std::max((double)hp[i], eps), 1.0-eps);
                const double yt = (double)ht[i];
                acc += -(yt*std::log(yp) + (1.0-yt)*std::log(1.0-yp));
            }
            acc /= (double)n_pred;
        } else if (loss_type == "cce") {
            for (int b=0; b<B; ++b) {
                for (int c=0; c<num_classes; ++c) {
                    const long long i = 1LL*b*num_classes + c;
                    const double yt = (double)ht[i];
                    if (yt > 0.0) {
                        const double yp = std::min(std::max((double)hp[i], eps), 1.0-eps);
                        acc += -(yt * std::log(yp));
                    }
                }
            }
            acc /= (double)B;
        }
        std::printf("[LOSS][CHECK] GPU=%.6f CPU=%.6f (n=%lld, B=%d, C=%d)\n",
                    loss_value, (float)acc, n_pred, B, num_classes);
    }
#endif

    // 4) 임시 버퍼 정리
    for (const auto& op : E) {
        if (op.op_type == LOSS) continue;
        const std::string& out_id = op.output_id;
        if (out_id == pred_id) continue;
        auto it = tensors.find(out_id);
        if (it == tensors.end()) continue;
        if (user_keys.count(out_id)) continue;   // 사용자 제공 버퍼는 보존

        cudaFree(it->second);
        tensors.erase(it);
        shapes.erase(out_id);                    // shape도 동기 삭제
    }

    return loss_value;
}
