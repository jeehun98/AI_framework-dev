
#pragma once
#include <string>

enum OpType {
    MATMUL = 0,
    ADD = 1,
    RELU = 2,
    SIGMOID = 3,
    TANH = 4
};

struct OpStruct {
    int op_type;
    std::string input_id;
    std::string param_id;
    std::string output_id;
};

struct Shape {
    int rows;
    int cols;
};
