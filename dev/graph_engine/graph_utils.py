from collections import OrderedDict


def get_leaf_nodes(node_list):
    """
    자식이 없는 노드들을 리프 노드로 간주하고 반환합니다.
    """
    if not node_list:
        raise ValueError("node_list가 비어 있습니다.")

    visited = OrderedDict()
    leaf_nodes = []

    def dfs(node):
        if node in visited:
            return
        visited[node] = True
        if not node.children:
            leaf_nodes.append(node)
        else:
            for child in node.children:
                dfs(child)

    for node in node_list:
        dfs(node)

    return leaf_nodes


def get_leaf_nodes_consistent_or_all(node_list):
    """
    node_list에 있는 모든 노드가 동일한 리프 노드 집합을 공유하면 해당 집합을 반환하고,
    그렇지 않으면 전체 노드 리스트를 반환합니다.
    """
    if not node_list:
        raise ValueError("node_list가 비어 있습니다.")

    all_leaf_sets = [set(get_leaf_nodes([node])) for node in node_list]
    first_set = all_leaf_sets[0]

    if all(leaf_set == first_set for leaf_set in all_leaf_sets[1:]):
        return list(first_set)  # ✅ 공유되는 리프 노드만 반환

    return node_list  # ❌ 공유되지 않으면 전체 반환


def get_root_nodes(node_list):
    """
    부모가 없는 노드들을 루트 노드로 간주하고 반환합니다.
    """
    return [node for node in node_list if not node.parents]


def connect_graphs(parents, children):
    """
    children는 루트, parents는 리프로 보고 연결합니다.
    즉, children의 각 노드에 대해 parents의 leaf를 연결합니다.
    """
    if not parents or not children:
        raise ValueError("두 node_list 중 하나 이상이 비어 있습니다.")

    leaf_nodes = get_leaf_nodes_consistent_or_all(parents)
    root_nodes = children

    print(len(leaf_nodes), len(root_nodes), "[DEBUG] leaf vs root 연결 확인")

    if len(leaf_nodes) != len(root_nodes):
        raise ValueError(f"리프 노드({len(leaf_nodes)})와 루트 노드({len(root_nodes)})의 개수가 일치하지 않습니다.")

    for leaf, root in zip(leaf_nodes, root_nodes):
        leaf.add_child(root)
        root.add_parent(leaf)

    return children


def print_graph(node_list):
    """
    터미널에서 보기 좋은 트리 구조로 계산 그래프를 출력합니다.
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

    print("\n[ 그래프 구조 ]")
    visited = OrderedDict()
    for idx, node in enumerate(node_list):
        print_node(node, is_last=(idx == len(node_list) - 1), visited=visited)
