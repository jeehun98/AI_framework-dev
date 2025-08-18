// op_struct.h (또는 동일 헤더)
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
    // ✅ 신규 활성화
    LEAKY_RELU = 8,
    ELU = 9,
    GELU = 10,
    SILU = 11,
    SOFTMAX = 12
};

struct Shape {
    int rows;
    int cols;
};

struct OpExtraParams {
    // 기존 CNN/RNN 관련 필드들...
    int kernel_h = 0;
    int kernel_w = 0;
    int stride_h = 1;
    int stride_w = 1;
    int padding_h = 0;
    int padding_w = 0;
    int input_h = 0;
    int input_w = 0;
    int input_c = 1;
    int output_c = 1;
    int batch_size = 1;

    int time_steps = 0;
    int hidden_size = 0;
    int num_layers = 1;

    bool use_bias = true;

    // ✅ 손실
    std::string label_id = "";
    std::string loss_type = "";  // "mse","bce","cce" 등

    // ✅ 활성화 파라미터
    float alpha = 0.01f;     // LeakyReLU/ELU 계수
    int   gelu_tanh = 1;     // 1=tanh 근사, 0=정규 CDF(선택사항, 여기선 tanh)
    float temperature = 1.f; // Softmax 온도(τ)
    int   axis = 1;          // Softmax 축(행렬 기준 1=열방향 class)
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
