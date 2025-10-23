#pragma once
#include <cuda_runtime.h>
#include <functional>
#include <string>
#include <vector>
#include <unordered_map>

namespace ai { namespace executor {

// 런타임 컨텍스트 인터페이스(필요시 실제 구현 클래스를 바꿔 끼우세요)
struct RuntimeContext {
    explicit RuntimeContext(int num_streams = 1);
    ~RuntimeContext();
    int num_streams() const;
    cudaStream_t stream(int i) const;
    void synchronize_all();
private:
    struct Impl;
    Impl* p_;
};

// 그래프 노드
struct Node {
    int id = -1;
    std::string name;
    std::vector<int> deps;
    int stream_hint = -1;
    std::function<void(RuntimeContext&, cudaStream_t)> run;
};

// 스케줄 결과 아이템 (무명 struct 대신 "이름 있는 타입")
struct ScheduleItem {
    int node_id;
    int stream_idx;
};

// 실행 플랜
struct Plan {
    std::vector<ScheduleItem> items;
    std::unordered_map<int, cudaEvent_t> done_events;
};

// API
Plan make_plan(const std::vector<Node>& nodes, RuntimeContext& ctx);
void run_plan(const std::vector<Node>& nodes, Plan& plan, RuntimeContext& ctx);

}} // namespace ai::executor
