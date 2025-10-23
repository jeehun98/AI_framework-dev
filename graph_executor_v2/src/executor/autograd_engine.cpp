// src/executor/autograd_engine.cpp
#include "src/executor/scheduler.hpp"
#include "backends/cuda/ops/_common/shim/nvtx.hpp"
#include <vector>
#include <string>

namespace ai { namespace executor {

struct Tensor { /* TODO: 실제 텐서 타입 */ };
struct Batch  { std::vector<Tensor> inputs, targets; };

class AutogradEngine {
public:
    void train_step(const Batch& batch) {
        NVTX_RANGE("autograd.step", NVTX_COLOR::Blue);

        { NVTX_RANGE("fwd",   NVTX_COLOR::Cyan);    /* TODO: fwd 준비 */ }
        { NVTX_RANGE("bwd",   NVTX_COLOR::Magenta); /* TODO: bwd 준비 */ }
        { NVTX_RANGE("optim", NVTX_COLOR::Orange);  /* TODO: optim 준비 */ }

        RuntimeContext ctx(/*num_streams=*/4);
        std::vector<Node> nodes;

        nodes.push_back(Node{
            0, "input.encode", {}, -1,
            [](RuntimeContext& c, cudaStream_t s){
                NVTX_RANGE("encode.launch", NVTX_COLOR::Teal);
                // TODO: encode_launcher(..., s);
            }
        });
        nodes.push_back(Node{
            1, "dense.gemm", {0}, -1,
            [](RuntimeContext& c, cudaStream_t s){
                NVTX_RANGE("gemm.launch", NVTX_COLOR::Orange);
                // TODO: gemm_launch(..., s);
            }
        });
        nodes.push_back(Node{
            2, "act.relu", {1}, -1,
            [](RuntimeContext& c, cudaStream_t s){
                NVTX_RANGE("relu.launch", NVTX_COLOR::Yellow);
                // TODO: relu_launch(..., s);
            }
        });
        nodes.push_back(Node{
            3, "loss.softmax_ce", {2}, -1,
            [](RuntimeContext& c, cudaStream_t s){
                NVTX_RANGE("softmax_ce.launch", NVTX_COLOR::Red);
                // TODO: softmax_ce_launch(..., s);
            }
        });

        NVTX_RANGE("build_and_run_plan", NVTX_COLOR::Green);
        Plan plan = make_plan(nodes, ctx);
        run_plan(nodes, plan, ctx);
    }
};

}} // namespace ai::executor

extern "C" void ai_autograd_train_step_stub() {
    using namespace ai::executor;
    AutogradEngine eng;
    Batch dummy;
    eng.train_step(dummy);
}
