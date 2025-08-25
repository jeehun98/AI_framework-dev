// op_structs.h
#pragma once
#include <string>
#include <vector>
#include "quant/quant_types.cuh"

// op_structs.h (또는 동일 헤더)
enum OpType {
    MATMUL    = 0,
    ADD       = 1,
    RELU      = 2,
    SIGMOID   = 3,
    TANH      = 4,
    FLATTEN   = 5,
    CONV2D    = 6,
    LOSS      = 7,
    LEAKY_RELU= 8,
    ELU       = 9,
    GELU      = 10,
    SILU      = 11,
    SOFTMAX   = 12,
    POOL_MAX  = 13,
    POOL_AVG  = 14,

    // ---- 새 primitive (기존 값 보존; 뒤에 추가) ----
    ADD_BIAS   = 15, // y = x + b
    SLICE_TIME = 16, // X[:, t, :] → (B, D)
    CONCAT_TIME= 17, // 시간 축 concat
    FILL_ZERO  = 18, // 버퍼 0 초기화

    // ---- 단일 RNN 오퍼 (추가) ----
    RNN        = 19  // X[B,T,D], Wx[D,H], Wh[H,H], b[H], (opt)h0[B,H] → H_T[B,H] 또는 H_seq[B,T,H]
};


struct Shape {
    int rows;
    int cols;
};

struct OpExtraParams {
    // ---- Convolution / Pooling 공통 ----
    int  kernel_h   = 0;
    int  kernel_w   = 0;
    int  stride_h   = 1;
    int  stride_w   = 1;
    int  padding_h  = 0;
    int  padding_w  = 0;
    int  dilation_h = 1;
    int  dilation_w = 1;
    bool count_include_pad = false;

    // NCHW 메타 (필요 시 사용)
    int input_h  = 0;
    int input_w  = 0;
    int input_c  = 1;
    int output_c = 1;
    int batch_size = 1;

    // RNN 힌트(선택)
    int time_steps  = 0;
    int hidden_size = 0;
    int num_layers  = 1;

    bool use_bias = true;

    // Loss
    std::string label_id = "";
    std::string loss_type = "";  // "mse","bce","cce"

    // Activations / Softmax
    float alpha = 0.01f;  // Leaky/ELU
    int   gelu_tanh = 1;  // 1=tanh 근사
    float temperature = 1.f;
    int   axis = 1;

    // ---- 시간축 유틸 ----
    int time_index   = -1; // SLICE_TIME에서 t 지정
    int concat_count = 0;  // CONCAT_TIME 체이닝 시 선택적으로 사용
};

// ---- 하위호환 + 확장 가능한 OpStruct ----
struct OpStruct {
    int op_type;

    // ✅ 새 경로: 다중 입력/파라미터
    std::vector<std::string> inputs; // 데이터 입력들 (예: ["A","B"])
    std::vector<std::string> params; // 가중치/편향 등 (예: ["W","b"])

    std::string output_id;
    OpExtraParams extra_params;

    // ---- 레거시 필드(하위호환 유지) ----
    // 기존 코드가 사용하는 단일 입력/파라미터
    std::string input_id;  // DEPRECATED
    std::string param_id;  // DEPRECATED

    // ----- 생성자들 -----
    OpStruct() = default;

    // 새 벡터 기반 생성자
    OpStruct(int type,
             std::vector<std::string> in,
             std::vector<std::string> par,
             std::string out,
             OpExtraParams extra = {})
        : op_type(type),
          inputs(std::move(in)),
          params(std::move(par)),
          output_id(std::move(out)),
          extra_params(std::move(extra)) {}

    // 레거시 단일 입력/파라미터 생성자 (기존 파이썬 바인딩과 100% 동일)
    OpStruct(int type,
             std::string in,
             std::string param,
             std::string out,
             OpExtraParams extra = {})
        : op_type(type),
          output_id(std::move(out)),
          extra_params(std::move(extra)),
          input_id(std::move(in)),
          param_id(std::move(param)) {}

    // 실행 전 정규화: 레거시 → 벡터로 이관
    inline void normalize_legacy() {
        if (!input_id.empty() && inputs.empty()) {
            inputs.push_back(input_id);
        }
        if (!param_id.empty() && params.empty()) {
            params.push_back(param_id);
        }
    }
};
