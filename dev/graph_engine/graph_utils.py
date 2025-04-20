from collections import OrderedDict

def connect_graphs(parents, children):
    """
    두 계산 그래프의 노드를 연결합니다.
    - parents: 이전 레이어의 root_node_list
    - children: 현재 레이어의 leaf_node_list

    각 children[i] 노드에 대해 parents[i]를 자식으로 연결합니다.
    """
    if not parents or not children:
        raise ValueError("두 node_list 중 하나 이상이 비어 있습니다.")

    if len(parents) != len(children):
        raise ValueError(f"노드 개수 불일치: parents={len(parents)}, children={len(children)}")

    for prev_root, curr_leaf in zip(parents, children):
        curr_leaf.add_child(prev_root)

    return parents


def print_graph(node_list):
    """
    터미널에서 계산 그래프를 트리 형태로 출력합니다.
    - 순환 구조가 있어도 재방문 노드를 감지하여 무한 루프 방지
    """
    def print_node(node, prefix="", is_last=True, visited=None):
        if visited is None:
            visited = OrderedDict()
        if node in visited:
            print(prefix + ("└── " if is_last else "├── ") + f"[{node.operation}] out={node.output} (↺ visited)")
            return
        visited[node] = True

        connector = "└── " if is_last else "├── "
        print(prefix + connector + f"[{node.operation}] out={node.output} weight={node.weight_value}")

        child_count = len(node.children)
        for idx, child in enumerate(node.children):
            is_last_child = (idx == child_count - 1)
            next_prefix = prefix + ("    " if is_last else "│   ")
            print_node(child, next_prefix, is_last_child, visited)

    if not node_list:
        print("그래프 비어있음")
        return

    print("\n[ 계산 그래프 구조 ]")
    visited = OrderedDict()
    for idx, node in enumerate(node_list):
        print_node(node, is_last=(idx == len(node_list) - 1), visited=visited)

