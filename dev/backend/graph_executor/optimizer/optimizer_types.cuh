#pragma once
#include <string>
#include <type_traits>

// 강타입 enum: 암묵적 int 변환 방지, ABI 명확화
enum class OptimizerType : int {
    SGD      = 0,
    MOMENTUM = 1,
    ADAM     = 2
};

// ----- 변환 유틸 -----
static inline const char* to_string(OptimizerType t) {
    switch (t) {
        case OptimizerType::SGD:      return "SGD";
        case OptimizerType::MOMENTUM: return "MOMENTUM";
        case OptimizerType::ADAM:     return "ADAM";
        default:                      return "UNKNOWN";
    }
}

static inline std::string to_std_string(OptimizerType t) {
    return std::string(to_string(t));
}

static inline int to_int(OptimizerType t) {
    return static_cast<int>(t);
}

static inline OptimizerType optimizer_type_from_int(int v) {
    switch (v) {
        case 0: return OptimizerType::SGD;
        case 1: return OptimizerType::MOMENTUM;
        case 2: return OptimizerType::ADAM;
        default: return OptimizerType::ADAM; // 안전 기본값
    }
}

static inline OptimizerType optimizer_type_from_string(const std::string& s) {
    if (s == "SGD" || s == "sgd") return OptimizerType::SGD;
    if (s == "MOMENTUM" || s == "momentum" || s == "mom") return OptimizerType::MOMENTUM;
    if (s == "ADAM" || s == "adam") return OptimizerType::ADAM;
    return OptimizerType::ADAM; // 안전 기본값
}

// ----- 스트림 출력(디버깅용) -----
#include <ostream>
static inline std::ostream& operator<<(std::ostream& os, OptimizerType t) {
    return (os << to_string(t));
}

// ----- unordered_map 키로 쓰고 싶을 때(선택) -----
// <unordered_map> 사용 시 해시 지원이 필요하면 아래 주석 해제
/*
#include <functional>
namespace std {
template<> struct hash<OptimizerType> {
    size_t operator()(const OptimizerType& t) const noexcept {
        return std::hash<int>()(static_cast<int>(t));
    }
};
}
*/
