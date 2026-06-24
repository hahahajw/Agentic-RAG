# ═══════════════════════════════════════════════════════════════════
# DAG 操作 API
# ═══════════════════════════════════════════════════════════════════
# 原语类型 + apply_primitives 三步流水线。
# 将 Planner 的原语序列转换为新 DAG，验证前置条件和不变式。

from dataclasses import dataclass

from .models import DAG, DAGEdge, DAGNode, EdgeType, NodeStatus
from .invariants import InvariantViolation, check_invariants


# ═══════════════════════════════════════════════════════════════════
# 原语类型
# ═══════════════════════════════════════════════════════════════════


@dataclass
class InheritPrimitive:
    """INHERIT: 保留探针——该事实已被正确确立，原封不动带入新 DAG。

    Planner 意图: "上一轮此节点的搜索结果有效，我不需要重新搜索它。"
    效果: 拷贝 node_id 的全部字段（含 search_query）到新 DAG 同名节点，
    仅更新 planner_rationale 和 round_last_updated。
    """
    node_id: str
    new_rationale: str


@dataclass
class InheritAndRelabelPrimitive:
    """INHERIT_AND_RELABEL: 同一事实，换探测问题重新搜索。

    Planner 意图: "这个事实需要被确立，但之前的探测方式（question）不对，
    换个问法重新搜。旧观测保留作为参考。"
    效果: 从旧 DAG 拷贝历史观测字段，question 换为新值，status 重置为 UNSOLVED。
    search_query 不保留——旧搜索词对应旧 question，新搜索词由下一轮 Solver 生成。
    """
    node_id: str
    new_question: str
    new_rationale: str


@dataclass
class InitializePrimitive:
    """INITIALIZE: 新增探针——发现一个之前未被 DAG 覆盖的语义需求。

    Planner 意图: "理想 DAG 中应该有一个对应的事实，但当前 DAG 中没有——
    我新引入一个探针来探测它。"
    效果: 创建全新节点，ID 由 planner_output_to_primitives 预分配
    （占位符 $N → N{round}_{seq}），所有 Solver/Critic 字段为空，status = UNSOLVED。
    """
    question: str
    rationale: str
    node_id: str


@dataclass
class LinkPrimitive:
    """LINK: 声明两个事实之间的关系。

    Planner 意图: "我认为这两个事实之间有 decomposition 或 dependency 关系。"
    效果: 在新 DAG 中添加有向边，前置条件包括端点存在且不引入环。
    """
    from_id: str
    to_id: str
    edge_type: EdgeType


Primitive = InheritPrimitive | InheritAndRelabelPrimitive | InitializePrimitive | LinkPrimitive


# ═══════════════════════════════════════════════════════════════════
# 错误与结果类型
# ═══════════════════════════════════════════════════════════════════


@dataclass
class PrimitiveError:
    """原语前置条件失败或系统一致性检查失败。

    与 InvariantViolation 的区别:
    - PrimitiveError: 原语层问题——"你引用了不存在的节点" / "旧节点未被处理"
    - InvariantViolation: DAG 层问题——"你的 DAG 有环" / "根 question 不对"
    """
    primitive_index: int
    primitive_type: str | None
    description: str


@dataclass
class ApplyResult:
    """apply_primitives 的返回结果——成功或失败的统一载体。

    dag = None 表示应用失败。errors 为空列表表示成功。
    """
    dag: DAG | None
    errors: list[InvariantViolation | PrimitiveError]


# ═══════════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════════


def apply_primitives(
    old_dag: DAG,
    primitives: list[Primitive],
    deleted_nodes: set[str],
    deleted_edges: set[tuple[str, str, str]],
) -> ApplyResult:
    """应用 Planner 的原语序列，将旧 DAG 转换为新 DAG。

    三步流水线（任一步骤失败则立即返回错误，不做部分 DAG 组装）:
    1. 系统一致性检查——旧 DAG 的每个节点/边是否被原语或删除列表完整覆盖
    2. 逐条应用原语——检查各原语的前置条件，构建新 DAG
    3. I1-I7 不变式检查——调用 check_invariants 验证最终 DAG 的合法性

    InitializePrimitive.node_id 由 planner_output_to_primitives 预分配，
    此函数直接使用——不再自行生成 ID。
    """
    new_round = old_dag.round_number + 1

    # ── Step 1: 系统一致性检查 ──
    consistency_errors = _check_consistency(
        old_dag, primitives, deleted_nodes, deleted_edges
    )
    if consistency_errors:
        return ApplyResult(
            dag=None, errors=consistency_errors
        )

    # ── Step 2: 逐条应用原语 ──
    new_dag = DAG(q=old_dag.q, round_number=new_round)

    for i, p in enumerate(primitives):
        match p:
            case InheritPrimitive(node_id=nid, new_rationale=rat):
                error = _apply_inherit(
                    new_dag, old_dag, nid, rat, new_round, i
                )
                if error:
                    return ApplyResult(
                        dag=None, errors=[error]
                    )

            case InheritAndRelabelPrimitive(
                node_id=nid, new_question=new_q, new_rationale=rat
            ):
                error = _apply_inherit_and_relabel(
                    new_dag, old_dag, nid, new_q, rat, new_round, i
                )
                if error:
                    return ApplyResult(
                        dag=None, errors=[error]
                    )

            case InitializePrimitive(question=q, rationale=rat, node_id=nid):
                new_dag.nodes[nid] = DAGNode(
                    id=nid,
                    question=q,
                    planner_rationale=rat,
                    round_created=new_round,
                    round_last_updated=new_round,
                )

            case LinkPrimitive(from_id=fid, to_id=tid, edge_type=et):
                error = _apply_link(new_dag, fid, tid, et, new_round, i)
                if error:
                    return ApplyResult(
                        dag=None, errors=[error]
                    )

    # ── Step 3: I1-I7 不变式检查 ──
    violations = check_invariants(new_dag)
    if violations:
        return ApplyResult(
            dag=None, errors=list(violations)
        )

    return ApplyResult(dag=new_dag, errors=[])


# ═══════════════════════════════════════════════════════════════════
# 一致性检查
# ═══════════════════════════════════════════════════════════════════


def _check_consistency(
    old_dag: DAG,
    primitives: list[Primitive],
    deleted_nodes: set[str],
    deleted_edges: set[tuple[str, str, str]],
) -> list[PrimitiveError]:
    """检查旧 DAG 的每个元素是否被 Planner 的输出完整覆盖。

    旧节点:
      - 出现在 INHERIT / INHERIT_AND_RELABEL 的 node_id → 被保留
      - 出现在 deleted_nodes → 被显式删除
      - 均未出现 → 意外遗漏

    旧边:
      - 出现在 LINK(from_id, to_id, edge_type) → 被重建
      - 出现在 deleted_edges → 被显式撤销
      - 均未出现 → 意外遗漏
    """
    errors: list[PrimitiveError] = []

    # 收集原语中引用的旧节点
    referenced_nodes: set[str] = set()
    for p in primitives:
        if isinstance(p, (InheritPrimitive, InheritAndRelabelPrimitive)):
            referenced_nodes.add(p.node_id)

    # 检测矛盾——同一节点同时出现在原语和删除列表中
    contradiction = referenced_nodes & deleted_nodes
    if contradiction:
        for nid in sorted(contradiction):
            errors.append(PrimitiveError(
                primitive_index=-1,
                primitive_type=None,
                description=(
                    f"节点 '{nid}' 同时出现在原语序列（INHERIT/INHERIT_AND_RELABEL）"
                    f"和删除列表中。Planner 输出矛盾——请决定保留或删除此节点"
                )
            ))

    # 检查每个旧节点是否被覆盖
    for nid in old_dag.nodes:
        if nid not in referenced_nodes and nid not in deleted_nodes:
            errors.append(PrimitiveError(
                primitive_index=-1,
                primitive_type=None,
                description=(
                    f"节点 '{nid}' 在旧 DAG 中存在，但未出现在原语序列"
                    f"或删除列表中。如确认删除，请将其加入 deleted_nodes"
                )
            ))

    # 收集原语中 LINK 创建的边
    linked_edges: set[tuple[str, str, str]] = set()
    for p in primitives:
        if isinstance(p, LinkPrimitive):
            linked_edges.add((p.from_id, p.to_id, p.edge_type.value))

    for edge in old_dag.edges:
        edge_key = (edge.from_id, edge.to_id, edge.edge_type.value)
        if edge_key not in linked_edges and edge_key not in deleted_edges:
            errors.append(PrimitiveError(
                primitive_index=-1,
                primitive_type=None,
                description=(
                    f"边 {edge.from_id}→{edge.to_id} ({edge.edge_type.value}) "
                    f"在旧 DAG 中存在，但未出现在 LINK 原语或删除列表中。"
                    f"如确认撤销此边，请将其加入 deleted_edges"
                )
            ))

    return errors


# ═══════════════════════════════════════════════════════════════════
# 原语应用函数
# ═══════════════════════════════════════════════════════════════════


def _apply_inherit(
    new_dag: DAG,
    old_dag: DAG,
    node_id: str,
    new_rationale: str,
    new_round: int,
    primitive_index: int,
) -> PrimitiveError | None:
    """应用 INHERIT 原语——拷贝旧节点全部字段到新 DAG，仅更新 rationale。

    列表字段做浅拷贝——新 DAG 和旧 DAG 的列表独立，
    但 chunk 对象本身共享（chunk 只读，共享安全）。
    search_query 在复制时保留——INHERIT 带入的节点
    已在上轮由 Solver 搜索过，保留搜索记录供 Critic 正确区分节点形态。
    """
    if node_id not in old_dag.nodes:
        return PrimitiveError(
            primitive_index=primitive_index,
            primitive_type="INHERIT",
            description=(
                f"INHERIT({node_id}) 失败: "
                f"节点 '{node_id}' 在旧 DAG 中不存在"
            )
        )

    old = old_dag.nodes[node_id]
    new_dag.nodes[node_id] = DAGNode(
        id=node_id,
        question=old.question,
        planner_rationale=new_rationale,
        status=old.status,
        retrieved_chunks=list(old.retrieved_chunks),
        supporting_chunks=list(old.supporting_chunks),
        answer=old.answer,
        solver_judgment=old.solver_judgment,
        search_query=old.search_query,
        critic_health=old.critic_health,
        critic_factual_notes=old.critic_factual_notes,
        critic_normative_advice=old.critic_normative_advice,
        round_created=old.round_created,
        round_last_updated=new_round,
    )
    return None


def _apply_inherit_and_relabel(
    new_dag: DAG,
    old_dag: DAG,
    node_id: str,
    new_question: str,
    new_rationale: str,
    new_round: int,
    primitive_index: int,
) -> PrimitiveError | None:
    """应用 INHERIT_AND_RELABEL 原语——换探测问题，从零开始。

    INHERIT_AND_RELABEL 的语义是旧搜索已被证明完全无效
    （否则 Planner 会用 INHERIT）。因此除了槽位身份（id + round_created）
    之外，不保留任何旧数据。

    仅保留:
    - id: 同一事实槽位
    - round_created: 该事实首次被识别的轮次
    替换:
    - question → new_question（新探测方式）
    - planner_rationale → new_rationale（新探测的理由）
    重置:
    - status → UNSOLVED
    - round_last_updated → new_round
    清空（设为默认值）:
    - 所有 Solver 字段: retrieved_chunks, supporting_chunks, answer,
      solver_judgment, search_query
    - 所有 Critic 字段: critic_health, critic_factual_notes,
      critic_normative_advice
    """
    if node_id not in old_dag.nodes:
        return PrimitiveError(
            primitive_index=primitive_index,
            primitive_type="INHERIT_AND_RELABEL",
            description=(
                f"INHERIT_AND_RELABEL({node_id}) 失败: "
                f"节点 '{node_id}' 在旧 DAG 中不存在"
            )
        )

    old = old_dag.nodes[node_id]
    new_dag.nodes[node_id] = DAGNode(
        id=node_id,
        question=new_question,
        planner_rationale=new_rationale,
        status=NodeStatus.UNSOLVED,
        # 所有 Solver 字段清空（使用默认值）
        # 所有 Critic 字段清空（使用默认值）
        # search_query 清空——旧搜索词对应旧 question
        round_created=old.round_created,
        round_last_updated=new_round,
    )
    return None


def _apply_link(
    new_dag: DAG,
    from_id: str,
    to_id: str,
    edge_type: EdgeType,
    new_round: int,
    primitive_index: int,
) -> PrimitiveError | None:
    """应用 LINK 原语——添加有向边，含端点检查和环检测前置条件。

    前置条件:
    1. from_id 和 to_id 在新 DAG 节点集合中存在
    2. 添加 from→to 后不会创建有向环
    """
    if from_id not in new_dag.nodes:
        return PrimitiveError(
            primitive_index=primitive_index,
            primitive_type="LINK",
            description=(
                f"LINK({from_id}→{to_id}) 失败: "
                f"from_id='{from_id}' 在新 DAG 中不存在。"
                f"请确保 INITIALIZE 该节点后再 LINK"
            )
        )
    if to_id not in new_dag.nodes:
        return PrimitiveError(
            primitive_index=primitive_index,
            primitive_type="LINK",
            description=(
                f"LINK({from_id}→{to_id}) 失败: "
                f"to_id='{to_id}' 在新 DAG 中不存在。"
                f"请确保 INITIALIZE 该节点后再 LINK"
            )
        )

    if _would_create_cycle(new_dag, from_id, to_id):
        cycle_path = _find_path(new_dag, to_id, from_id)
        path_str = " → ".join(cycle_path) if cycle_path else "(无路径详情)"
        return PrimitiveError(
            primitive_index=primitive_index,
            primitive_type="LINK",
            description=(
                f"LINK({from_id}→{to_id}) 失败: 添加此边会创建有向环。"
                f"当前图中已有路径使 {to_id} 可达 {from_id}: {path_str}"
            )
        )

    new_dag.edges.append(DAGEdge(
        from_id=from_id,
        to_id=to_id,
        edge_type=edge_type,
        round_created=new_round,
    ))
    return None


# ═══════════════════════════════════════════════════════════════════
# 图算法辅助函数
# ═══════════════════════════════════════════════════════════════════


def _would_create_cycle(dag: DAG, from_id: str, to_id: str) -> bool:
    """检查添加边 from_id→to_id 是否会创建有向环。

    等价判断: to_id 在当前 DAG 中是否已存在到 from_id 的路径。
    使用迭代 DFS（栈）而非递归——避免深层 DAG 上的递归深度风险。
    """
    if from_id == to_id:
        return True

    adj: dict[str, list[str]] = {nid: [] for nid in dag.nodes}
    for edge in dag.edges:
        adj[edge.from_id].append(edge.to_id)

    visited: set[str] = set()
    stack = [to_id]
    while stack:
        node = stack.pop()
        if node == from_id:
            return True
        if node not in visited:
            visited.add(node)
            for neighbor in adj.get(node, []):
                if neighbor not in visited:
                    stack.append(neighbor)
    return False


def _find_path(
    dag: DAG,
    from_id: str,
    to_id: str,
) -> list[str]:
    """在 DAG 中找从 from_id 到 to_id 的一条路径。

    使用迭代 DFS + parent 字典回溯路径。
    仅在 _would_create_cycle 确认存在路径后调用——
    用于生成 "已有路径: N1→N2→N3→N1" 形式的错误消息。
    """
    adj: dict[str, list[str]] = {nid: [] for nid in dag.nodes}
    for edge in dag.edges:
        adj[edge.from_id].append(edge.to_id)

    visited: set[str] = set()
    parent: dict[str, str] = {}
    stack = [from_id]
    visited.add(from_id)

    while stack:
        node = stack.pop()
        if node == to_id:
            path = [to_id]
            current = to_id
            while current != from_id:
                current = parent[current]
                path.append(current)
            path.reverse()
            return path

        for neighbor in adj.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = node
                stack.append(neighbor)

    return []