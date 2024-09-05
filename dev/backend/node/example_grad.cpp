// gpt 에 물어볼 내용;;
class Node {
public:
    std::string operation;
    double input;
    double output;
    double grad;  // 그래디언트 값
    std::vector<std::shared_ptr<Node>> parents;
    std::vector<std::shared_ptr<Node>> children;

    Node(const std::string& op, double in_val, double out_val)
        : operation(op), input(in_val), output(out_val), grad(0.0) {}

    void add_parent(std::shared_ptr<Node> parent) {
        parents.push_back(parent);
    }

    void add_child(std::shared_ptr<Node> child) {
        children.push_back(child);
    }

    // 각 노드의 연산에 따른 미분값을 계산
    void compute_grad() {
        if (operation == "negate") {
            grad = -1.0 * grad;  // Negate 연산의 미분: -1
        } else if (operation == "exp") {
            grad = output * grad;  // Exp 연산의 미분: exp(x) (출력값 그대로)
        } else if (operation == "add") {
            grad = grad;  // Add 연산의 미분: 1 (입력에 대한 미분이 동일하게 전달됨)
        } else if (operation == "reciprocal") {
            grad = -1.0 * grad / (output * output);  // Reciprocal 연산의 미분: -1/x^2
        }
    }

    // 부모 노드로 그래디언트를 전파
    void backpropagate() {
        compute_grad();
        for (auto& parent : parents) {
            parent->grad += grad;  // 부모 노드에 그래디언트를 누적
            parent->backpropagate();  // 재귀적으로 부모 노드에 전파
        }
    }
};


// 각 operations 이 grad 를 갱신하는 식이 있는 것 
// output 을 통해 이거 나중에 봐도 어떤 느낌인지 알아야 한다 지훈아