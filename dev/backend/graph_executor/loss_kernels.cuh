#pragma once
#include <cuda_runtime.h>

// ======= 손실 값 계산 (Forward) =======
float compute_mse_loss_cuda(const float* y_true, const float* y_pred, int size);
float compute_bce_loss_cuda(const float* y_true, const float* y_pred, int size);

// ✅ CCE는 (배치 평균) 시그니처 유지
float compute_cce_loss_cuda(const float* y_true, const float* y_pred,
                            int batch_size, int num_classes);

// ======= Backward 래퍼(호스트 함수)만 노출 =======
//  - 커널(__global__)은 .cu 내부에만 두고, 여기서는 호스트 래퍼만 선언합니다.
void launch_bce_loss_backward(const float* y_true, const float* y_pred,
                              float* grad_out, int size, int batch_size,
                              cudaStream_t stream = 0);

void launch_mse_loss_backward(const float* y_true, const float* y_pred,
                              float* grad_out, int size,
                              cudaStream_t stream = 0);

void launch_cce_loss_backward(const float* y_true, const float* y_pred,
                              float* grad_out, int batch_size, int num_classes,
                              cudaStream_t stream = 0);

                              // ✅ NEW: softmax ⊗ CCE fused (∂L/∂z 직생성)
void launch_softmax_xent_fused_backward(const float* y_prob,   // softmax 출력 p
                                        const float* y_true,   // one-hot
                                        float* grad_z,         // ∂L/∂z
                                        int batch_size, int num_classes,
                                        cudaStream_t stream = 0);