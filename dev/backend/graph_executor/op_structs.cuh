#pragma once
#include <string>

enum OpType {
    MATMUL = 0,
    ADD = 1,
    RELU = 2,
    SIGMOID = 3,
    TANH = 4,
    FLATTEN = 5,
    CONV2D = 6,
    LOSS = 7
};

enum OptimizerType {
    SGD = 0,
    MOMENTUM = 1,
    ADAM = 2
};

struct Shape {
    int rows;
    int cols;
};

// ✅ CNN, RNN, Generic 연산에 공통적으로 사용할 확장 가능한 구조체
struct OpExtraParams {
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
};

// ✅ 수정: extra_params 멤버 포함
struct OpStruct {
    int op_type;
    std::string input_id;
    std::string param_id;
    std::string output_id;
    OpExtraParams extra_params;

    // 기본 생성자
    OpStruct() = default;

    // 확장 생성자 (extra_params 포함)
    OpStruct(int type, std::string in, std::string param, std::string out, OpExtraParams extra = {})
        : op_type(type), input_id(std::move(in)), param_id(std::move(param)), output_id(std::move(out)), extra_params(std::move(extra)) {}
};
