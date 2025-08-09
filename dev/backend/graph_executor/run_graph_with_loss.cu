// run_graph_with_loss.cu
#include "run_graph_with_loss.cuh"
#include "run_graph.cuh"          // forward 실행
#include "loss_kernels.cuh"       // compute_*_loss_cuda
#include "op_structs.cuh"

#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <cstdio>
#include <cmath>

// ===== Debug toggles =====
#ifndef DEBUG_LOSS
#define DEBUG_LOSS 0
#endif

// ===== CUDA helpers =====
#define CUDA_OK(stmt) do {                                      \
    cudaError_t __e = (stmt);                                   \
    if (__e != cudaSuccess) {                                   \
        std::fprintf(stderr, "[CUDA][ERR] %s:%d %s\n",          \
                     __FILE__, __LINE__, cudaGetErrorString(__e)); \
        return NAN;                                             \
    }                                                           \
} while(0)

static inline void cuda_check_last_and_sync(const char* where) {
    cudaError_t e1 = cudaGetLastError();
    if (e1 != cudaSuccess) {
        std::fprintf(stderr, "[CUDA][ERR] %s (last): %s\n", where, cudaGetErrorString(e1));
    }
    cudaError_t e2 = cudaDeviceSynchronize();
    if (e2 != cudaSuccess) {
        std::fprintf(stderr, "[CUDA][ERR] %s (sync): %s\n", where, cudaGetErrorString(e2));
    }
}

#if DEBUG_LOSS
static void dump_first_k(const char* tag, const float* dev, int n, int k=8) {
    int m = std::min(n, k);
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
    // 0) 손실 입력 보정: 그래프 마지막이 LOSS면 그 입력(= activation 출력)을 y_pred로 사용
    std::string pred_id = final_output_id;
    if (!E.empty() && E.back().op_type == LOSS) {
        pred_id = E.back().input_id;
    }

    // 0-1) 호출 시점에 이미 존재하던(=사용자/파이썬이 넘긴) 텐서 키 스냅샷
    std::unordered_set<std::string> user_keys;
    user_keys.reserve(tensors.size());
    for (const auto& kv : tensors) user_keys.insert(kv.first);

    // 1) Forward (pred_id를 최종 출력으로 사용)
    {
        float dummy = 0.0f; // run_graph_cuda가 out_host에 결과를 복사하지만 여기선 사용 안 함
        run_graph_cuda(E, tensors, shapes, &dummy, pred_id, batch_size);
        cuda_check_last_and_sync("run_graph_cuda");
    }

    // 2) y_pred / y_true 확보 및 크기 계산
    if (!tensors.count(pred_id)) {
        std::fprintf(stderr, "[LOSS] missing y_pred tensor: %s\n", pred_id.c_str());
        return NAN;
    }
    if (!tensors.count(label_tensor_id)) {
        std::fprintf(stderr, "[LOSS] missing y_true tensor: %s\n", label_tensor_id.c_str());
        return NAN;
    }

    float* y_pred = tensors[pred_id];
    float* y_true = tensors[label_tensor_id];

    // per-sample shape
    Shape out_shape = shapes[pred_id];
    const int per_sample = out_shape.rows * out_shape.cols;
    const int n = batch_size * per_sample;
    if (n <= 0) {
        std::fprintf(stderr, "[LOSS] invalid size: n=%d (rows=%d, cols=%d, batch=%d)\n",
                     n, out_shape.rows, out_shape.cols, batch_size);
        return NAN;
    }

#if DEBUG_LOSS
    // 포인터 경로/범위 검사 (필요시만)
    const float* act_out = nullptr;
    const float* dense_out = nullptr;
    for (auto it = E.rbegin(); it != E.rend(); ++it) {
        const auto& op = *it;
        if (op.op_type == SIGMOID || op.op_type == RELU || op.op_type == TANH) {
            auto f = tensors.find(op.output_id);
            if (f != tensors.end()) { act_out = f->second; break; }
        }
    }
    for (auto it = E.rbegin(); it != E.rend(); ++it) {
        const auto& op = *it;
        if (op.op_type == MATMUL) {
            auto f = tensors.find(op.output_id);
            if (f != tensors.end()) { dense_out = f->second; break; }
        }
    }

    std::printf("[PTR] y_pred=%p", (void*)y_pred);
    if (act_out)   std::printf(", act_out=%p",   (void*)act_out);
    if (dense_out) std::printf(", dense_out=%p", (void*)dense_out);
    std::printf("\n");

    // y_pred 범위 체크
    {
        std::vector<float> hp(std::min(n, 8));
        cudaMemcpy(hp.data(), y_pred, sizeof(float)*hp.size(), cudaMemcpyDeviceToHost);
        bool bad = false;
        for (float v : hp) if (!(v >= 0.0f && v <= 1.0f)) { bad = true; break; }
        if (bad) {
            std::printf("[LOSS][WARN] y_pred out of [0,1]. Check activation/loss wiring.\n");
            dump_first_k("CHK: y_pred", y_pred, n);
            if (act_out)   dump_first_k("CHK: activation_out", act_out, n);
            if (dense_out) dump_first_k("CHK: dense_linear_out", dense_out, n);
        }
    }
#endif

    // 3) 손실 계산 (평균)
    float loss_value = 0.0f;
    if (loss_type == "mse") {
        loss_value = compute_mse_loss_cuda(y_true, y_pred, n);
    } else if (loss_type == "binary_crossentropy" || loss_type == "bce") {
        loss_value = compute_bce_loss_cuda(y_true, y_pred, n);
    } else {
        std::fprintf(stderr, "[LOSS] Unsupported loss type: %s\n", loss_type.c_str());
        return NAN;
    }
    cuda_check_last_and_sync("loss kernels");

#if DEBUG_LOSS
    // GPU/CPU 대조 (디버그 시에만)
    {
        std::vector<float> hp(n), ht(n);
        CUDA_OK(cudaMemcpy(hp.data(), y_pred, sizeof(float)*n, cudaMemcpyDeviceToHost));
        CUDA_OK(cudaMemcpy(ht.data(), y_true, sizeof(float)*n, cudaMemcpyDeviceToHost));
        const double eps = 1e-7;
        double acc = 0.0;
        if (loss_type == "mse") {
            for (int i=0;i<n;++i) { double d = (double)hp[i] - (double)ht[i]; acc += d*d; }
            acc /= (double)n;
        } else {
            for (int i=0;i<n;++i) {
                double yp = std::min(std::max((double)hp[i], eps), 1.0 - eps);
                double yt = (double)ht[i];
                acc += -(yt*std::log(yp) + (1.0-yt)*std::log(1.0-yp));
            }
            acc /= (double)n;
        }
        std::printf("[LOSS][CHECK] GPU=%.6f  CPU=%.6f  (n=%d, batch=%d)\n",
                    loss_value, (float)acc, n, batch_size);
    }
#endif

    // 4) 임시 출력 버퍼 정리
    //    - 사용자(파이썬)가 넘긴 포인터(user_keys)는 절대 해제 금지
    //    - pred_id(결과 출력 버퍼)는 호출자가 이후 사용 가능하니 해제 금지
    //    - run_graph_cuda 내에서 새로 cudaMalloc 된 출력들만 해제
    for (const auto& op : E) {
        if (op.op_type == LOSS) continue;           // forward에서 LOSS는 생성하지 않음
        const std::string& out_id = op.output_id;
        if (out_id == pred_id) continue;            // 결과 출력 유지
        auto it = tensors.find(out_id);
        if (it == tensors.end()) continue;          // 없는 키
        if (user_keys.count(out_id)) continue;      // 사용자 제공 포인터

        // 임시 버퍼로 판단 → 해제
        cudaFree(it->second);
        // tensors.erase(it);        // 필요시 정리
        // shapes.erase(out_id);     // 필요시 정리
    }

    return loss_value;
}
