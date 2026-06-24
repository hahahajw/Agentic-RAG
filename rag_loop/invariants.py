# ═══════════════════════════════════════════════════════════════════
# 不变式检查器
# ═══════════════════════════════════════════════════════════════════
# 纯结构检查，不调用 LLM，不读取 chunk 文本内容。
# 输入 DAG，输出违规列表。
# 收集全部违规——Planner 同轮修订仅 3 次，一次看到所有违规效率更高。

from dataclasses import dataclass

from .models import DAG, DAGEdge, EdgeType


@dataclass
class InvariantViolation:
    """一次不变式违规——可直接注入 Planner 消息的结构化描述"""
    invariant: str
    description: str
    involved_nodes: list[str]
    involved_edges: list[tuple[str, str, str]]


def check_invariants(dag: DAG) -> list[InvariantViolation]:
    """检查 DAG 的 7 条结构不变式 I1-I7。

    检查顺序: I5/I7 → I2 → I4 → I3/I6 → I1
    I3 和 I6 依赖 I4 的结果（需要知道根节点是谁），
    因此必须在 I4 之后执行。I1（环检测）最昂贵，放在最后。

    核心约束（FRAMEWORK.md §5）：纯结构检查，不调用 LLM，不读取 chunk 内容。
    """
    violations: list[InvariantViolation] = []

    # ── I5: node.id 与 dict key 一致 ──
    for nid, node in dag.nodes.items():
        if node.id != nid:
            violations.append(InvariantViolation(
                invariant="I5",
                description=(
                    f"节点以 key='{nid}' 存储，但其 id 字段为 '{node.id}'。"
                    f"两者必须一致——key 是节点在 DAG 中的身份标识"
                ),
                involved_nodes=[nid],
                involved_edges=[]
            ))

    # ── I7: 边类型合法 ──
    valid_edge_types = {EdgeType.DECOMPOSITION.value, EdgeType.DEPENDENCY.value}
    for edge in dag.edges:
        if edge.edge_type.value not in valid_edge_types:
            violations.append(InvariantViolation(
                invariant="I7",
                description=(
                    f"边 {edge.from_id}→{edge.to_id} 的类型为 "
                    f"'{edge.edge_type.value}'，必须为 decomposition 或 dependency"
                ),
                involved_nodes=[edge.from_id, edge.to_id],
                involved_edges=[(edge.from_id, edge.to_id, edge.edge_type.value)]
            ))

    # ── I2: 边端点存在 ──
    for edge in dag.edges:
        missing = []
        if edge.from_id not in dag.nodes:
            missing.append(f"from_id={edge.from_id}")
        if edge.to_id not in dag.nodes:
            missing.append(f"to_id={edge.to_id}")
        if missing:
            violations.append(InvariantViolation(
                invariant="I2",
                description=(
                    f"边 {edge.from_id}→{edge.to_id} 引用了不存在的节点: "
                    f"{', '.join(missing)}"
                ),
                involved_nodes=[
                    nid for nid in (edge.from_id, edge.to_id)
                    if nid not in dag.nodes
                ],
                involved_edges=[(edge.from_id, edge.to_id, edge.edge_type.value)]
            ))

    # ── 计算入度（I4、I3、I6 共用）──
    indegrees: dict[str, int] = {nid: 0 for nid in dag.nodes}
    for edge in dag.edges:
        if edge.to_id in indegrees:
            indegrees[edge.to_id] += 1

    roots = [nid for nid, deg in indegrees.items() if deg == 0]

    # ── I4: 唯一根 ──
    root_id: str | None = None
    if len(roots) == 0:
        violations.append(InvariantViolation(
            invariant="I4",
            description=(
                "没有入度为 0 的根节点——所有节点都有入边，"
                "可能存在覆盖全部节点的有向环"
            ),
            involved_nodes=list(dag.nodes.keys()),
            involved_edges=[]
        ))
    elif len(roots) > 1:
        violations.append(InvariantViolation(
            invariant="I4",
            description=(
                f"多个根节点（入度为 0）: {roots}。"
                f"DAG 必须恰好有一个根，不能是森林"
            ),
            involved_nodes=roots,
            involved_edges=[]
        ))
    else:
        root_id = roots[0]

    # ── I3: 非根节点至少一条入边 ──
    for nid, deg in indegrees.items():
        if deg == 0:
            if root_id is None or nid != root_id:
                violations.append(InvariantViolation(
                    invariant="I3",
                    description=(
                        f"节点 '{nid}' 没有入边——它是孤立节点。"
                        f"除根节点外，每个节点必须至少有一条入边，"
                        f"表明其对回答 Q 有贡献"
                    ),
                    involved_nodes=[nid],
                    involved_edges=[]
                ))

    # ── I6: 根 question == Q ──
    if root_id is not None:
        root_node = dag.nodes[root_id]
        if root_node.question != dag.q:
            violations.append(InvariantViolation(
                invariant="I6",
                description=(
                    f"根节点 '{root_id}' 的 question = '{root_node.question}'，"
                    f"与原始问题 Q = '{dag.q}' 不一致。"
                    f"根是内部 DAG 与理想 DAG 的唯一锚点——两者必须一致"
                ),
                involved_nodes=[root_id],
                involved_edges=[]
            ))

    # ── I1: 无环（Kahn 拓扑排序 + DFS 回溯定位环路径）──
    adj: dict[str, list[str]] = {nid: [] for nid in dag.nodes}
    in_deg: dict[str, int] = {nid: 0 for nid in dag.nodes}
    for edge in dag.edges:
        if edge.from_id in adj and edge.to_id in adj:
            adj[edge.from_id].append(edge.to_id)
            in_deg[edge.to_id] += 1

    queue = [nid for nid, d in in_deg.items() if d == 0]
    sorted_count = 0
    while queue:
        node = queue.pop(0)
        sorted_count += 1
        for neighbor in adj[node]:
            in_deg[neighbor] -= 1
            if in_deg[neighbor] == 0:
                queue.append(neighbor)

    if sorted_count != len(dag.nodes):
        remaining = [nid for nid, d in in_deg.items() if d > 0]
        cycle = _find_cycle(adj, remaining)
        cycle_str = " → ".join(cycle) if cycle else "(未能定位具体环路径)"
        violations.append(InvariantViolation(
            invariant="I1",
            description=f"检测到有向环: {cycle_str}",
            involved_nodes=cycle if cycle else remaining,
            involved_edges=[]
        ))

    return violations


def _find_cycle(
    adj: dict[str, list[str]],
    candidates: list[str]
) -> list[str]:
    """从候选节点中找一条有向环路径。

    使用 DFS + 递归栈回溯。visited 防止重复遍历整个搜索空间，
    rec_set（当前递归路径上的节点）检测 back edge。
    """
    visited: set[str] = set()
    rec_stack: list[str] = []
    rec_set: set[str] = set()
    result: list[str] = []

    def dfs(node: str) -> bool:
        nonlocal result
        visited.add(node)
        rec_stack.append(node)
        rec_set.add(node)

        for neighbor in adj.get(node, []):
            if neighbor not in visited:
                if dfs(neighbor):
                    return True
            elif neighbor in rec_set:
                start_idx = rec_stack.index(neighbor)
                result = rec_stack[start_idx:] + [neighbor]
                return True

        rec_stack.pop()
        rec_set.discard(node)
        return False

    for start in candidates:
        if start not in visited:
            if dfs(start):
                return result
    return []