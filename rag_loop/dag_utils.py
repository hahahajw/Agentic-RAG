# ═══════════════════════════════════════════════════════════════════
# DAG 拓扑查询工具函数
# ═══════════════════════════════════════════════════════════════════
# 消费者: Solver (solver.py)、Critic (critic.py)、Planner (planner.py)
# 这些函数是纯数据查询——不修改 DAG，不调用 LLM，不执行搜索。
# 从 solver.py 和 critic.py 提取至此，避免跨模块循环依赖。

from .models import DAG, DAGNode, ChunkInfo, EdgeType, NodeStatus


def get_dependency_answers(dag: DAG, node_id: str) -> list[str]:
    """收集 node_id 的所有 dependency 源节点已求解的非空 answer。

    返回: 非空 answer 字符串列表（已滤除 SOLVED 但 answer="" 的情况）。
    用于 solve_dag 的查询构造和 critic 的推理节点展示。
    """
    answers: list[str] = []
    for edge in dag.edges:
        if edge.to_id == node_id and edge.edge_type == EdgeType.DEPENDENCY:
            src = dag.nodes.get(edge.from_id)
            if src and src.status == NodeStatus.SOLVED and src.answer:
                answers.append(src.answer)
    return answers


def get_child_answers(dag: DAG, node_id: str) -> list[str]:
    """收集 node_id 的所有 decomposition 子节点的已求解非空 answer。

    返回: 非空 answer 字符串列表（SOLVED + answer 非空）。
    用于 solve_dag 的查询构造和 critic 的推理节点展示。
    """
    answers: list[str] = []
    for edge in dag.edges:
        if edge.from_id == node_id and edge.edge_type == EdgeType.DECOMPOSITION:
            child = dag.nodes.get(edge.to_id)
            if child and child.status == NodeStatus.SOLVED and child.answer:
                answers.append(child.answer)
    return answers


def get_direct_children(dag: DAG, node_id: str) -> list[str]:
    """获取 node_id 的所有直接子节点（通过 decomposition 边）。

    仅返回 decomposition 子节点——dependency 目标不视为"子节点"，
    它们是独立的依赖目标。
    """
    children: list[str] = []
    for edge in dag.edges:
        if edge.from_id == node_id and edge.edge_type == EdgeType.DECOMPOSITION:
            if edge.to_id in dag.nodes:
                children.append(edge.to_id)
    return sorted(children)


def get_dependency_sources(dag: DAG, node_id: str) -> list[str]:
    """获取 node_id 的所有 dependency 源节点。

    目标节点依赖这些源节点的 answer 来消解 question 中的指代。
    """
    sources: list[str] = []
    for edge in dag.edges:
        if edge.to_id == node_id and edge.edge_type == EdgeType.DEPENDENCY:
            if edge.from_id in dag.nodes:
                sources.append(edge.from_id)
    return sorted(sources)


def topological_layers(dag: DAG) -> list[list[str]]:
    """BFS 从根出发按层分组节点——符合人类从根到叶子的因果阅读顺序。

    Level 0 = [root], Level 1 = root 的直接子节点, Level 2 = 孙节点, ...

    时间复杂度 O(V + E)，空间 O(V)。
    """
    root = dag.root
    if root is None:
        return [sorted(dag.nodes.keys())]

    children: dict[str, list[str]] = {nid: [] for nid in dag.nodes}
    for edge in dag.edges:
        children[edge.from_id].append(edge.to_id)

    layers: list[list[str]] = []
    visited: set[str] = set()
    current = [root.id]

    while current:
        layers.append(sorted(current))
        visited.update(current)
        next_level: list[str] = []
        for nid in current:
            for child in children.get(nid, []):
                if child not in visited:
                    next_level.append(child)
        current = list(dict.fromkeys(next_level))

    remaining = [nid for nid in dag.nodes if nid not in visited]
    if remaining:
        layers.append(sorted(remaining))

    return layers


def find_leaves(dag: DAG) -> list[str]:
    """找出 DAG 中所有叶子节点（出度为 0 的节点）。

    叶子 = 没有任何出边的节点。在 DAG 语义中，叶子是最底层的原子事实——
    它们不再被进一步分解。

    消费者: critic.py（路径追踪——计算根→叶子的推理链）
    Phase 4 的 Planner 也可能使用此函数判断 DAG 的分解深度。
    """
    outdegree: dict[str, int] = {nid: 0 for nid in dag.nodes}
    for edge in dag.edges:
        outdegree[edge.from_id] += 1
    return sorted(nid for nid, deg in outdegree.items() if deg == 0)


def find_all_paths(
    dag: DAG,
    from_id: str,
    to_id: str,
) -> list[list[str]]:
    """找从 from_id 到 to_id 的所有简单路径（不重复访问节点）。

    使用 DFS + 回溯。对于小型 DAG（< 20 节点），路径数量有限，
    全部枚举的开销可忽略。

    消费者: critic.py（_compute_reasoning_chains——预计算根→各叶子的路径，
    注入 Critic USER 消息供条件②审查）

    Args:
        dag: DAG 图
        from_id: 起点节点 ID
        to_id: 终点节点 ID

    Returns:
        路径列表，每条路径为节点 ID 序列（from_id → ... → to_id）。
        无路径时返回空列表。
    """
    adj: dict[str, list[str]] = {nid: [] for nid in dag.nodes}
    for edge in dag.edges:
        adj[edge.from_id].append(edge.to_id)

    all_paths: list[list[str]] = []
    current_path: list[str] = [from_id]
    visited: set[str] = {from_id}

    def dfs(node: str) -> None:
        if node == to_id:
            all_paths.append(list(current_path))
            return
        for neighbor in adj.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                current_path.append(neighbor)
                dfs(neighbor)
                current_path.pop()
                visited.discard(neighbor)

    if from_id in dag.nodes and to_id in dag.nodes:
        dfs(from_id)

    return all_paths