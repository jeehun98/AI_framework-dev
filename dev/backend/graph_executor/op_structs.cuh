// op_structs.h  (또는 동일 헤더)
#pragma once
#include <string>
#include "quant/quant_types.cuh"

enum OpType {
    MATMUL = 0,
    ADD = 1,
    RELU = 2,
    SIGMOID = 3,
    TANH = 4,
    FLATTEN = 5,
    CONV2D = 6,
    LOSS = 7,
    LEAKY_RELU = 8,
    ELU = 9,
    GELU = 10,
    SILU = 11,
    SOFTMAX = 12,
    POOL_MAX = 13,
    POOL_AVG = 14
};

struct Shape {
    int rows;
    int cols;
};

struct OpExtraParams {
    // ---- Convolution / Pooling 공통 파라미터 ----
    int kernel_h   = 0;
    int kernel_w   = 0;
    int stride_h   = 1;
    int stride_w   = 1;
    int padding_h  = 0;
    int padding_w  = 0;

    // ✅ 새로 추가: dilation, pooling 평균 계산 방식
    int dilation_h = 1;          // 기본 1 (미사용 시 1로 해석)
    int dilation_w = 1;          // 기본 1
    bool count_include_pad = false; // AvgPool에서 패딩을 분모에 포함할지

    // 입력/출력 텐서 메타 (NCHW 가정)
    int input_h  = 0;
    int input_w  = 0;
    int input_c  = 1;
    int output_c = 1;
    int batch_size = 1;

    // RNN류 (미사용 시 0/1 유지)
    int time_steps  = 0;
    int hidden_size = 0;
    int num_layers  = 1;

    bool use_bias = true;

    // ---- Loss ----
    std::string label_id = "";
    std::string loss_type = "";  // "mse","bce","cce" 등

    // ---- Activations / Softmax ----
    float alpha = 0.01f;    // LeakyReLU/ELU
    int   gelu_tanh = 1;    // 1=tanh 근사
    float temperature = 1.f; // Softmax 온도
    int   axis = 1;          // Softmax 축(행렬 기준)
};

struct OpStruct {
    int op_type;
    std::string input_id;
    std::string param_id;
    std::string output_id;
    OpExtraParams extra_params;

    OpStruct() = default;
    OpStruct(int type, std::string in, std::string param, std::string out, OpExtraParams extra = {})
        : op_type(type), input_id(std::move(in)), param_id(std::move(param)), output_id(std::move(out)), extra_params(std::move(extra)) {}
};
