#pragma once

float compute_mse_loss_cuda(float* y_true, float* y_pred, int size);
float compute_bce_loss_cuda(float* y_true, float* y_pred, int size);
float compute_cce_loss_cuda(float* y_true, float* y_pred, int batch_size, int num_classes);
