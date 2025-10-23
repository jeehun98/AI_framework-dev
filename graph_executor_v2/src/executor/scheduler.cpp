// src/executor/scheduler.cpp
#include "src/executor/scheduler.hpp"
#include <queue>
#include <cassert>
#include <algorithm>
#include "backends/cuda/ops/_common/shim/nvtx.hpp"

namespace ai { namespace executor {

// RuntimeContext 구현
struct RuntimeContext::Impl {
    explicit Impl(int num_streams) {
        if (num_streams < 1) num_streams = 1;
        streams.resize(num_streams);
        for (auto& s : streams) cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    }
    ~Impl() {
        for (auto s : streams) cudaStreamDestroy(s);
    }
    std::vector<cudaStream_t> streams;
};

RuntimeContext::RuntimeContext(int n): p_(new Impl(n)) {}
RuntimeContext::~RuntimeContext(){ delete p_; }
int RuntimeContext::num_streams() const { return static_cast<int>(p_->streams.size()); }
cudaStream_t RuntimeContext::stream(int i) const { return p_->streams.at(i); }
void RuntimeContext::synchronize_all(){ for (auto s : p_->streams) cudaStreamSynchronize(s); }

// 내부: 토폴로지 정렬
static std::vector<int> topo_sort(const std::vector<Node>& nodes){
    std::unordered_map<int,int> indeg;
    std::unordered_map<int,std::vector<int>> outs;
    indeg.reserve(nodes.size());
    outs.reserve(nodes.size());
    for (auto& n: nodes){
        indeg[n.id] += static_cast<int>(n.deps.size());
        for (int d: n.deps) outs[d].push_back(n.id);
    }
    std::queue<int> q;
    for (auto& n: nodes) if (indeg[n.id]==0) q.push(n.id);
    std::vector<int> order; order.reserve(nodes.size());
    while(!q.empty()){
        int u=q.front(); q.pop();
        order.push_back(u);
        for(int v: outs[u]) if(--indeg[v]==0) q.push(v);
    }
    return order;
}

Plan make_plan(const std::vector<Node>& nodes, RuntimeContext& ctx){
    NVTX_RANGE("scheduler.make_plan", NVTX_COLOR::Blue);

    auto order = topo_sort(nodes);
    Plan plan; plan.items.reserve(order.size());

    std::unordered_map<int,const Node*> by_id; by_id.reserve(nodes.size());
    for (auto& n: nodes) by_id[n.id] = &n;

    const int S = std::max(1, ctx.num_streams());
    int rr = 1;
    for (int nid : order){
        auto* n = by_id.at(nid);
        int s_idx = (n->stream_hint>=0 && n->stream_hint<S)
                  ? n->stream_hint
                  : (S==1 ? 0 : ((rr++ % std::max(1,S-1)) + 1));
        plan.items.push_back({nid, s_idx});

        cudaEvent_t ev; cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
        plan.done_events[nid] = ev;
    }
    return plan;
}

void run_plan(const std::vector<Node>& nodes, Plan& plan, RuntimeContext& ctx){
    NVTX_RANGE("scheduler.run_plan", NVTX_COLOR::Green);

    std::unordered_map<int,const Node*> by_id; by_id.reserve(nodes.size());
    for (auto& n: nodes) by_id[n.id] = &n;

    std::unordered_map<int,int> node_stream; node_stream.reserve(plan.items.size());
    for (auto& it : plan.items) node_stream[it.node_id] = it.stream_idx;

    for (auto& it : plan.items){
        const Node* n = by_id.at(it.node_id);
        cudaStream_t s = ctx.stream(it.stream_idx);

        for (int d : n->deps){
            if (node_stream.at(d) != it.stream_idx)
                cudaStreamWaitEvent(s, plan.done_events.at(d), 0);
        }

        const std::string label = n->name.empty()
            ? ("node#" + std::to_string(n->id))
            : (n->name + "(#" + std::to_string(n->id) + ")");
        {
            NVTX_RANGE_F(label.c_str(), NVTX_COLOR::Teal);
            n->run(ctx, s);
        }

        cudaEventRecord(plan.done_events.at(n->id), s);
    }

    ctx.synchronize_all();
    for (auto& kv : plan.done_events) cudaEventDestroy(kv.second);
    plan.done_events.clear();
}

}} // namespace ai::executor
