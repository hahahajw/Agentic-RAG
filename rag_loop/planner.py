# ═══════════════════════════════════════════════════════════════════
# Planner 组件
# ═══════════════════════════════════════════════════════════════════
# 4.1: Pydantic 输出模型 + 类型转换函数
# 4.2: USER 消息组装 (build_planner_user_message)
# 4.3: SYSTEM prompt + 世界观内容
# 4.4: 调用入口 (run_planner)
#
# Planner 是三角色闭环中的"战略决策者"——接收 Critic 的质量反馈
# 和当前 DAG 状态，通过一次 LLM 调用产出：
#   - 原语序列（primitives + deleted_nodes + deleted_edges）
#   - 双层结构假设声明（hypothesis_part1 + hypothesis_part2）

from typing import Literal
import re

from pydantic import BaseModel, Field, field_validator
from langchain_core.prompts import ChatPromptTemplate

from .models import DAG, DAGNode, ChunkInfo, EdgeType, NodeStatus, CriticHealth
from .dag_utils import (
    topological_layers,
    get_dependency_answers, get_child_answers,
    get_direct_children, get_dependency_sources,
)
from .operations import (
    InheritPrimitive, InheritAndRelabelPrimitive,
    InitializePrimitive, LinkPrimitive, Primitive,
)
from .formatting import (
    truncate, source_distribution, node_label,
    format_chunk_summary, format_chunk_summary_compact,
    format_source_summary, format_blocking_info,
    format_supporting_chunks, format_non_supporting_chunks,
)
from .structured_output import get_structured_model


# ═══════════════════════════════════════════════════════════════════
# 4.1: Pydantic 输出模型
# ═══════════════════════════════════════════════════════════════════
# 四种原语各自独立的 Pydantic 模型，用 primitive_type Literal 作为
# 类型标识。PlannerOutput 顶层容器聚合所有产出。


class InheritOutput(BaseModel):
    """INHERIT——该事实已被正确确立，原封不动带入新 DAG"""
    primitive_type: Literal["INHERIT"] = Field(
        description="原语类型标识"
    )
    node_id: str = Field(
        description="旧 DAG 中存在的节点 ID。该节点应已被正确确立（SOLVED + Critic 判为 healthy），本轮可直接保留"
    )
    new_rationale: str = Field(
        description="本轮保留此节点的理由——为什么这个事实对回答 Q 仍然必要"
    )


class InheritAndRelabelOutput(BaseModel):
    """INHERIT_AND_RELABEL——同一事实，换探测方式重新搜索"""
    primitive_type: Literal["INHERIT_AND_RELABEL"] = Field(
        description="原语类型标识"
    )
    node_id: str = Field(
        description="旧 DAG 中存在的节点 ID。该事实方向正确，但上一轮探测方式（question）有问题，需要换个角度搜索"
    )
    new_question: str = Field(
        description="新探测问题——换个角度搜索同一个事实。status 将重置为 UNSOLVED，旧观测保留为历史记录"
    )
    new_rationale: str = Field(
        description="换探测方式的理由——为什么原来的 question 不够好（搜索精度不足/措辞不当/需要更具体等）"
    )


class InitializeOutput(BaseModel):
    """INITIALIZE——发现未被 DAG 覆盖的语义需求，新增探针

    系统按 INITIALIZE 在 primitives 列表中的出现顺序分配占位符 $1, $2, $3...。
    在 LINK 中引用新节点时使用这些占位符。
    """
    primitive_type: Literal["INITIALIZE"] = Field(
        description="原语类型标识"
    )
    question: str = Field(
        description="新探测问题——指向 Q 中尚未被任何节点覆盖的语义需求。这将是 Solver 搜索的目标"
    )
    rationale: str = Field(
        description="为什么这个事实对回答 Q 是必要的——它填补了 Q 分解中的哪个缺口"
    )


class LinkOutput(BaseModel):
    """LINK——声明两个事实之间的 decomposition 或 dependency 关系"""
    primitive_type: Literal["LINK"] = Field(
        description="原语类型标识"
    )
    from_id: str = Field(
        description="源节点 ID。旧节点使用实际 ID（在输入 §2-§3 中可见），本轮 INITIALIZE 的新节点使用占位符 $N"
    )
    to_id: str = Field(
        description="目标节点 ID。规则同上——旧节点用实际 ID，新节点用占位符 $N"
    )
    edge_type: Literal["decomposition", "dependency"] = Field(
        description="decomposition = 子事实是父事实的分解产物，求解时汇总为搜索上下文（软约束）；"
        "dependency = 目标 question 含指代词，必须用源 answer 消解后才能搜索（硬约束）"
    )


class DeletedEdgeItem(BaseModel):
    """声明删除一条边——与 deleted_nodes 配合构成显式删除清单"""
    from_id: str = Field(description="被删除边的源节点 ID")
    to_id: str = Field(description="被删除边的目标节点 ID")
    edge_type: Literal["decomposition", "dependency"] = Field(description="被删除边的类型")


# 用于解析 LLM 可能输出的字符串格式边引用（如 "N1_1→N1_2 (dependency)"）。
# LLM 在 prompt 中看到边以此格式展示（见 _format_s2_topology_summary），
# 有时会在 deleted_edges 中原样返回字符串而非结构化对象。
_EDGE_STRING_RE = re.compile(
    r'^(.+?)\s*(?:→|->|=>)\s*(.+?)\s*[\(（](.+?)[\)）]$'
)


class PlannerOutput(BaseModel):
    """Planner 单次调用的完整输出——系统消费并分发至 apply_primitives / Critic / 对话历史"""
    primitives: list[
        InheritOutput | InheritAndRelabelOutput | InitializeOutput | LinkOutput
    ] = Field(
        description=(
            "原语序列——按此顺序应用。这是你对本轮 DAG 结构的完整表达。"
            "旧 DAG 中未被 INHERIT/INHERIT_AND_RELABEL 覆盖的节点会被自动删除，"
            "未被 LINK 覆盖的边会被自动删除——除非你将其列入 deleted_nodes/deleted_edges。"
            "新节点的 ID 按 INITIALIZE 在列表中的出现顺序，"
            "从 §1 中告知的起始编号依次递增分配（第一个 INITIALIZE = N{next}，第二个 = N{next+1}，...）。"
            "在 LINK 中引用新节点时，直接使用这些预期 ID"
        )
    )
    deleted_nodes: list[str] = Field(
        default_factory=list,
        description="显式声明要删除的节点 ID 列表。列入此列表的节点及其关联边将不再出现在新 DAG 中"
    )
    deleted_edges: list[DeletedEdgeItem] = Field(
        default_factory=list,
        description="显式声明要删除的边列表。列入此列表的边将不再出现在新 DAG 中"
    )

    @field_validator('deleted_edges', mode='before')
    @classmethod
    def _parse_edge_strings(cls, v: list) -> list:
        """将字符串格式的边引用（如 "N1_1→N1_2 (dependency)"）转换为 DeletedEdgeItem 兼容的 dict。

        LLM 在 prompt 的 §2 拓扑概览中看到边以 "N1_1 → N1_2 (dependency)" 格式展示，
        有时会原样输出字符串而非结构化对象。此 validator 在 Pydantic 验证之前将其转换。
        """
        result: list = []
        for item in v:
            if isinstance(item, str):
                m = _EDGE_STRING_RE.match(item.strip())
                if m:
                    result.append({
                        "from_id": m.group(1).strip(),
                        "to_id": m.group(2).strip(),
                        "edge_type": m.group(3).strip(),
                    })
                else:
                    # 无法解析的字符串——原样保留，让 Pydantic 的正常验证报告错误
                    result.append(item)
            else:
                result.append(item)
        return result
    hypothesis_part1: str = Field(
        description=(
            "目标状态声明——描述本轮 DAG 的理想结构：哪些事实需要确立（节点）、"
            "它们之间有什么关系（边）。这是你对「世界应该长什么样」的当前假设。"
            "用自然语言写，不要复述原语序列。例：'Q 需要确立以下事实: F1: <question>（理由：<rationale>），"
            "F2: ...，关系: F1 是 F2 的前置条件（dependency），F3 是 F1 的分解（decomposition）'"
        )
    )
    hypothesis_part2: str = Field(
        description=(
            "变化量及理由——与上一轮相比，你做了哪些结构调整，每项附带原因。"
            "这是 Critic 审查你结构决策质量的直接依据。格式："
            "新增: N5（question='...'），原因：... / "
            "删除: N3（question='...'），原因：... / "
            "换标: N2 question 从 '...' 改为 '...'，原因：... / "
            "新增边: N1→N3（dependency），原因：... / "
            "撤销边: N2→N4（decomposition），原因：... / "
            "首轮标注：'本轮为首轮——所有节点为新增，无从上一轮调整的变更'"
        )
    )


# ═══════════════════════════════════════════════════════════════════
# 4.1: 类型转换函数
# ═══════════════════════════════════════════════════════════════════
# 纯类型映射——将 Pydantic 输出模型转换为 operations.py 的 Primitive
# dataclass 类型。零 ID 逻辑——新节点 ID 由 apply_primitives 分配。


def planner_output_to_primitives(
    output: PlannerOutput,
    round_number: int,
) -> tuple[list[Primitive], set[str], set[tuple[str, str, str]]]:
    """将 PlannerOutput 转换为 apply_primitives 期望的类型。

    占位符解析——消除 LLM 预测实际 ID 的需求：
    - Step 1: 统计 INITIALIZE 数量 → 构建 {$1: N{round}_{1}, $2: N{round}_{2}, ...} 映射
    - Step 2: 验证 LINK 中的占位符引用有效（$N 不越界）
    - Step 3: INITIALIZE 补全 node_id；LINK 中占位符替换为实际 ID；
      INHERIT/INHERIT_AND_RELABEL 直通

    占位符越界 → ValueError（Phase 5 捕获，计入 Planner ≤3 次修订）。
    旧节点 ID（不以 $ 开头）原样保留。
    """
    # Step 1: 统计 INITIALIZE 构建占位符 → 实际 ID 映射
    init_count = 0
    placeholder_map: dict[str, str] = {}

    for p in output.primitives:
        if p.primitive_type == "INITIALIZE":
            init_count += 1
            placeholder_map[f"${init_count}"] = f"N{round_number}_{init_count}"

    valid_placeholders = set(placeholder_map.keys())

    # Step 2: 验证 LINK 中的占位符有效性
    for p in output.primitives:
        if p.primitive_type == "LINK":
            for ref in (p.from_id, p.to_id):
                if ref.startswith("$") and ref not in valid_placeholders:
                    raise ValueError(
                        f"无效占位符 {ref}——primitives 中共 {init_count} 个 "
                        f"INITIALIZE，合法占位符: "
                        f"{sorted(valid_placeholders) if valid_placeholders else '(无)'}"
                    )

    # Step 3: 逐条转换为 operations.py Primitive 类型
    primitives: list[Primitive] = []
    init_seq = 0

    for p in output.primitives:
        match p.primitive_type:
            case "INHERIT":
                primitives.append(InheritPrimitive(
                    node_id=p.node_id,
                    new_rationale=p.new_rationale,
                ))
            case "INHERIT_AND_RELABEL":
                primitives.append(InheritAndRelabelPrimitive(
                    node_id=p.node_id,
                    new_question=p.new_question,
                    new_rationale=p.new_rationale,
                ))
            case "INITIALIZE":
                init_seq += 1
                real_id = placeholder_map[f"${init_seq}"]
                primitives.append(InitializePrimitive(
                    question=p.question,
                    rationale=p.rationale,
                    node_id=real_id,
                ))
            case "LINK":
                from_id = placeholder_map.get(p.from_id, p.from_id)
                to_id = placeholder_map.get(p.to_id, p.to_id)
                primitives.append(LinkPrimitive(
                    from_id=from_id,
                    to_id=to_id,
                    edge_type=EdgeType(p.edge_type),
                ))

    deleted_nodes: set[str] = set(output.deleted_nodes)
    deleted_edges: set[tuple[str, str, str]] = {
        (e.from_id, e.to_id, e.edge_type) for e in output.deleted_edges
    }

    return primitives, deleted_nodes, deleted_edges


# ═══════════════════════════════════════════════════════════════════
# 4.2: USER 消息组装
# ═══════════════════════════════════════════════════════════════════
# 五段结构（§1-§5）。首轮兼容——空字符串参数渲染为"(首轮，无历史)"。
# Planner 的节点详情展示与 Critic 的差异：
#   - chunk 仅展示摘要（不含 page_content 全文）
#   - Planner 不做逐字核实，摘要足以支撑结构决策
#   - 如果 Planner 认为 Critic 判断有误，可通过 INHERIT_AND_RELABEL 触发重搜


def build_planner_user_message(
    dag: DAG,
    critic_planner_guidance: str,
    previous_hypothesis_part1: str,
    previous_hypothesis_part2: str,
    detail_level: Literal["full", "summary"] = "full",
) -> str:
    """组装 Planner 的 USER 消息——系统中介通信机制的具体实现。

    FRAMEWORK.md 模块 3 要求：系统从各角色的结构化输出字段提取信息，
    注入另一角色的 USER 消息。此函数是"Critic/DAG → Planner 输入"
    方向的格式化管道。

    detail_level="full" (默认): 当前轮——supporting chunks 展示 page_content 全文
    detail_level="summary": 历史轮——supporting chunks 仅展示摘要
    """
    sections: list[str] = []

    sections.append(_format_s1_question(dag))
    sections.append(_format_s2_topology(dag))
    sections.append(_format_s3_node_details(dag, detail_level))
    sections.append(_format_s4_critic_guidance(critic_planner_guidance))
    sections.append(_format_s5_previous_hypothesis(
        previous_hypothesis_part1, previous_hypothesis_part2
    ))

    return "\n\n".join(sections)


# ── §1: 原始问题与轮次 ──


def _format_s1_question(dag: DAG) -> str:
    """§1 — Q、当前轮次、节点引用规则。

    新节点 ID 由系统自动分配（格式 N{轮次}_{序号}），Planner 使用
    占位符 $1, $2, ... 引用本轮 INITIALIZE 创建的节点。
    """
    return (
        f"═══════════════════════════════════════════\n"
        f"§1 原始问题与轮次\n"
        f"───────────────────────────────────────────\n"
        f"原始问题 Q: {dag.q}\n"
        f"当前轮次: 第 {dag.round_number} 轮 → 正在规划第 {dag.round_number + 1} 轮 DAG\n"
        f"\n"
        f"节点引用规则：\n"
        f"  旧节点（INHERIT/INHERIT_AND_RELABEL 保留的）→ 使用实际 ID（在 §2-§3 中可见）\n"
        f"  新节点（本轮 INITIALIZE 创建的）→ 使用占位符 $1, $2, $3...\n"
        f"    $1 = 你的 primitives 列表中第 1 个 INITIALIZE\n"
        f"    $2 = 你的 primitives 列表中第 2 个 INITIALIZE\n"
        f"    ...以此类推\n"
        f"  系统自动将占位符解析为实际 ID——你永远不需要知道或预测实际 ID"
    )


# ── §2: DAG 拓扑概览 ──


def _format_s2_topology(dag: DAG) -> str:
    """§2 — DAG 骨架：节点按拓扑层排列，边紧凑列出。

    Planner 用此节快速理解当前结构，随后在 §3 深读各节点。
    格式与 Critic §2 保持一致（共享节点标记逻辑）。
    """
    lines = [
        "═══════════════════════════════════════════",
        "§2 DAG 拓扑概览",
        "───────────────────────────────────────────",
    ]

    layers = topological_layers(dag)
    lines.append(f"节点 ({len(dag.nodes)} 个):")
    for layer in layers:
        for nid in sorted(layer):
            node = dag.nodes[nid]
            health_str = node.critic_health.value if node.critic_health else "未审查"
            label = node_label(dag, nid)
            label_str = f"[{label}] " if label else ""
            lines.append(
                f"  {label_str}{nid}: {truncate(node.question, 60)}  "
                f"status={node.status.value}  health={health_str}"
            )
    lines.append("")

    if dag.edges:
        lines.append(f"边 ({len(dag.edges)} 条):")
        for edge in dag.edges:
            lines.append(f"  {edge.from_id} → {edge.to_id} ({edge.edge_type.value})")
    else:
        lines.append("边: (无)")

    return "\n".join(lines)


# ── §3: 逐节点详情 ──


def _format_s3_node_details(dag: DAG, detail_level: str) -> str:
    """§3 — 每个节点的审查数据，按拓扑层排列（根→叶子）。

    detail_level="full": supporting chunks 全文 + non-supporting 摘要（与 Critic 对称）
    detail_level="summary": 所有 chunks 仅摘要

    与 Critic §3 的关键差异：
    - 不区分六种节点形态——用统一格式覆盖所有节点，突出 Planner 做结构决策
      所需的关键字段
    """
    layers = topological_layers(dag)
    sections: list[str] = []

    sections.append(
        "═══════════════════════════════════════════\n"
        "§3 逐节点详情\n"
        "───────────────────────────────────────────"
    )

    for layer in layers:
        for nid in sorted(layer):
            sections.append(_format_single_planner_node(dag, nid, detail_level))

    return "\n\n".join(sections)


def _format_single_planner_node(dag: DAG, nid: str, detail_level: str) -> str:
    """格式化单个节点的 Planner 决策视图。

    所有节点类型使用统一的字段展示格式——Planner 关注的是
    "这个节点的证据是否足够支撑我保留/换标/替换它"，
    而非 Critic 关注的"答案是否精确匹配 chunk 内容"。

    detail_level="full": Planner 可独立审视 supporting chunk 全文，
    不盲信 Critic 的审查判断。
    """
    node = dag.nodes[nid]
    root = dag.root
    is_root = root is not None and nid == root.id
    label = node_label(dag, nid)
    health_str = node.critic_health.value if node.critic_health else "未审查"

    header = f"━━━ {nid}: {node.question}"
    if label:
        header = f"━━━ [{label}] {nid}: {node.question}"
    lines = [header]

    # ── 基础字段 ──
    lines.append(f"状态: {node.status.value} | 健康: {health_str}")
    if node.answer:
        lines.append(f"声称答案: {truncate(node.answer, 100)}")
    else:
        lines.append("声称答案: (未找到)")

    if is_root:
        lines.append("搜索查询: (根节点——跳过 Solver 搜索阶段)")
    elif node.search_query:
        lines.append(f"搜索查询: {node.search_query}")
    else:
        lines.append("搜索查询: (未执行搜索)")

    if node.planner_rationale:
        lines.append(f"存在理由: {node.planner_rationale}")

    if node.solver_judgment:
        lines.append(f"Solver 判断: {node.solver_judgment}")

    # ── Critic 评价 ──
    if node.critic_factual_notes:
        lines.append(f"Critic 事实笔记: {node.critic_factual_notes}")
    if node.critic_normative_advice:
        lines.append(f"Critic 建议: {node.critic_normative_advice}")

    # ── Chunk 详情（detail_level 控制展示级别）──
    _append_chunk_details(lines, node, detail_level)

    # ── 依赖/阻塞信息 ──
    if label == "BLOCKED":
        blocking_text = format_blocking_info(dag, nid)
        if blocking_text:
            lines.append(blocking_text)

    if is_root:
        children = get_direct_children(dag, nid)
        if children:
            child_str = ", ".join(
                f"{cid}({dag.nodes[cid].status.value}, "
                f"answer={truncate(dag.nodes[cid].answer, 30) or '(空)'})"
                for cid in children
            )
            lines.append(f"直接子节点: {child_str}")
        deps = get_dependency_sources(dag, nid)
        if deps:
            dep_str = ", ".join(
                f"{did}(answer={truncate(dag.nodes[did].answer, 30) or '(空)'})"
                for did in deps
            )
            lines.append(f"依赖源节点: {dep_str}")

    # ── 推理依据（对推理型节点）──
    if (
        not is_root
        and node.status == NodeStatus.SOLVED
        and not node.search_query
    ):
        dep_answers = get_dependency_answers(dag, nid)
        child_answers = get_child_answers(dag, nid)
        if dep_answers:
            lines.append(f"推理依据-依赖事实: {', '.join(dep_answers)}")
        if child_answers:
            lines.append(f"推理依据-子事实: {', '.join(child_answers)}")

    return "\n".join(lines)


def _append_chunk_details(lines: list[str], node: DAGNode, detail_level: str) -> None:
    """追加 chunk 展示信息——detail_level 控制粒度。

    detail_level="full": supporting chunks 全文（与 Critic 对称，独立审视证据）
    detail_level="summary": 所有 chunks 仅摘要（历史轮次退化）
    """
    if not node.retrieved_chunks:
        if not node.search_query:
            return  # 未搜索，无 chunk
        lines.append("  ▸ 检索结果: (空——搜索未返回任何结果)")
        return

    # Supporting chunks — detail_level 控制全文/摘要
    supporting_lines = format_supporting_chunks(node, detail_level)
    if supporting_lines:
        lines.extend(supporting_lines)

    # Non-supporting chunks — 始终摘要
    non_supporting_lines = format_non_supporting_chunks(node)
    if non_supporting_lines:
        lines.extend(non_supporting_lines)

    # 来源分布
    source_line = format_source_summary(node.retrieved_chunks)
    if source_line:
        lines.append(source_line)


# ── §4: Critic 结构建议 ──


def _format_s4_critic_guidance(critic_planner_guidance: str) -> str:
    """§4 — Critic 的 planner_guidance 原文照登。

    这是 Critic 给 Planner 的结构层面建议——Q 语义覆盖、dependency 边有效性、
    收敛趋势、冗余缺口。Planner 应将其作为重要的决策参考。

    首轮为空字符串时标注"(首轮，无历史审查)"。
    """
    guidance = critic_planner_guidance.strip()
    if not guidance:
        guidance = "(首轮，无历史审查——请从零开始分解 Q)"

    return (
        f"═══════════════════════════════════════════\n"
        f"§4 Critic 结构建议\n"
        f"───────────────────────────────────────────\n"
        f"以下为 Critic 对本轮 DAG 的结构层面分析。"
        f"请仔细阅读——Critic 指出了可能的结构缺口、无效依赖或冗余。"
        f"你可以不同意 Critic 的判断，但应在 hypothesis_part2 中说明理由。\n"
        f"\n"
        f"{guidance}"
    )


# ── §5: 上一轮结构假设（自参照）──


def _format_s5_previous_hypothesis(
    previous_part1: str,
    previous_part2: str,
) -> str:
    """§5 — Planner 上一轮自己的结构假设声明。

    这使 Planner 能追踪自己的认知演化——对比本轮和上轮的假设，
    判断自己是在收敛（缺口在缩小）还是震荡（反复增删相同节点）。

    首轮两个参数均为空字符串。
    """
    part1 = previous_part1.strip()
    part2 = previous_part2.strip()

    lines = [
        "═══════════════════════════════════════════",
        "§5 上一轮结构假设（自参照）",
        "───────────────────────────────────────────",
        "以下为你上一轮产出的结构假设声明。对比本轮 DAG 的实际状态（§2-§3），"
        "判断你的上一轮假设哪些被验证了、哪些被推翻了。"
        "这有助于你追踪自己的认知演化——避免在相同错误方向上反复震荡。",
        "",
    ]

    lines.append("── 上一轮 Part 1: 目标状态（你上一轮认为世界长这样）──")
    if part1:
        lines.append(part1)
    else:
        lines.append("(首轮，无历史——本轮是从零开始的初始分解)")

    lines.append("")
    lines.append("── 上一轮 Part 2: 变化量及理由（你上一轮的认知调整说明）──")
    if part2:
        lines.append(part2)
    else:
        lines.append("(首轮，无历史)")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# 4.3: SYSTEM prompt
# ═══════════════════════════════════════════════════════════════════


PLANNER_SYSTEM_TEMPLATE = """\
你是规划者（Planner）。你的职责是决定当前轮次 DAG 的结构——有哪些节点（探测方向）、节点之间如何连接（关系假设）。你通过原语序列表达你的结构决策。

## 世界观：DAG 的本质

你操作的 DAG 不是一个搜索任务列表。它是你对「Q 应该如何分解为原子事实」的结构假设。具体而言：

- DAG 是对 Q 的正确分解结构的近似。算法看不到「理想 DAG」——那个完美分解 Q 的客观结构。每一轮，Solver 搜索 KB 获取观测，Critic 审查观测质量和结构质量，而你的任务是综合这些信息，修正 DAG 使其更接近理想 DAG
- 每个节点是一个事实的**认知替代物**——question 是探针（指向你想探测的事实），answer 是从 KB 中提取或推理出的声称值，retrieved_chunks 和 supporting_chunks 是证据，critic_health 和 critic_factual_notes 是 Critic 对证据质量的评估
- 根节点（question = Q）是内部 DAG 与理想 DAG 的唯一锚点——它的 question 必须严格等于 Q。根节点不参与 Solver 搜索——它的答案最终由子节点的推理链聚合而成

## 两种边的操作语义

边表达了你对事实间关系的假设。有两种类型，它们在 Solver 执行时有截然不同的效果：

**decomposition（分解边，父→子）**：
- 子节点的 question 是父节点 question 的分解产物——子事实是父事实的组成部分或背景上下文
- 求解父节点时，已解子节点的 answer 被汇总为搜索上下文
- 这是一种**软约束**——即使子节点未求解成功，父节点仍可搜索（因为父节点的 query 本身是一个合法问题，只是缺少部分上下文）

**dependency（依赖边，源→目标）**：
- 目标节点的 question 包含依赖源节点 answer 的指代词或变量（如 "that person"、"this city"、"that year"）
- 求解目标节点时，源节点的 answer 被注入查询改写以消解指代
- 这是一种**硬约束**——如果源节点未 SOLVED 或其 answer 为空，目标节点**无法被搜索**。因为 query 含未解析指代（如 "In what year was that founded?"），搜索必然返回噪声
- 如果目标节点被标记为 [BLOCKED]，你需要优先修复阻塞源——修正依赖源节点的结构问题，或在无法修复时用 INITIALIZE 创建替代路径

## 你的输入

USER 消息包含五个段落（§1-§5）：
- §1: 原始问题 Q、当前轮次、下一个可用节点 ID
- §2: DAG 拓扑概览——节点列表（含 status/health 快照和 ROOT/BLOCKED 标记）+ 边列表
- §3: 逐节点详情——每个节点的 question、answer、搜索查询、Solver 判断、Critic 评价、支撑 chunk 全文（当前轮，供独立验证 Critic 判断）、其他 chunk 摘要、来源分布。你应重点关注 status、health、Critic 的 factual_notes 和 normative_advice
- §4: Critic 的结构建议——Q 语义覆盖、dependency 边有效性、收敛趋势、冗余缺口
- §5: 你上一轮的结构假设声明——对比本轮 DAG 实际状态，追踪自己的认知演化

首轮时 §4 和 §5 为空——这表示你需要从零开始，利用你的推理能力分析 Q 的结构，产出初始 DAG。

## 你的输出要求

你需要产出三样东西。系统只检查输出完整性。

### (A) 原语序列 → primitives

你通过四种原语表达本轮的全部结构决策。有效的 DAG 必须满足 7 条结构不变式（无环、边端点存在、非根节点有入边、唯一根、ID 一致、根 question=Q、边类型合法）。系统会自动验证——验证失败会在同轮给你最多 3 次修订机会。

你的原语序列应包含本轮所需的所有节点和边。旧 DAG 中未出现在原语序列中的节点/边会被**自动删除**——除非你将其列入 deleted_nodes/deleted_edges。这是有意设计的：你不声明保留也不声明删除的元素，系统会认为你意外遗漏并要求确认。

四种原语的适用场景：

**INHERIT**——该事实已被正确确立，本轮原封不动保留：
- 何时使用：上一轮此节点已 SOLVED + Critic 判为 healthy，你认同 Critic 的判断
- **根节点**：每轮必须 INHERIT——根节点永远存在，你只需更新其 rationale
- 效果：保留所有字段（question、answer、chunks、critic_*），仅更新 rationale
- node_id 必须引用旧 DAG 中存在的节点

**INHERIT_AND_RELABEL**——同一事实方向正确，但探测方式需要调整：
- 何时使用：事实对，但上一轮的 question 导致搜索精度不足或返回了无关内容（Critic 判为 needs_verification 或 unreliable，但问题不在事实本身而在搜索策略）
- 效果：除 id 和 round_created（槽位身份）外，不保留任何旧数据。question 和 planner_rationale 替换为新值，status 重置为 UNSOLVED。所有 Solver 字段（answer、chunks、judgment、search_query）和 Critic 字段（health、notes、advice）清空——旧搜索已被证明完全无效，从零开始
- node_id 必须引用旧 DAG 中存在的节点

**INITIALIZE**——发现 Q 中尚未被覆盖的语义需求，新增探针：
- 何时使用：Q 的某个语义方面（谁、在哪、什么时候、哪一个、什么关系）尚未被任何节点的 question 覆盖；或 Critic 指出结构缺口；或 [BLOCKED] 节点的阻塞源无法修复，需要从不同角度建立替代路径
- 效果：创建全新节点，所有 Solver/Critic 字段为空，status = UNSOLVED
- 在 LINK 中引用新节点时使用占位符 $1, $2, ...（见下方「节点引用规则」）

**LINK**——声明两个事实之间的关系假设：
- 何时使用：你认为一个节点的 answer 是另一个节点求解所需的上下文（decomposition）或消解指代所需的前置信息（dependency）
- edge_type 选择指南：
  · decomposition：子事实是父事实的分解产物、组成成分或背景上下文。大部分节点间关系属于此类
  · dependency：目标 question 的措辞本身依赖源 answer——含"that"、"this"、"the"等指代词，不注入源 answer 就无法构造有意义的搜索查询
- 同一节点对可以同时有两条边（某个子节点既是父节点的分解产物，父节点求解时又需要其 answer 来消解指代）
- from_id 和 to_id：旧节点使用实际 ID（在 §2-§3 中可见），新节点使用占位符 $1, $2, ...（见下方「节点引用规则」）

### (B) 删除声明 → deleted_nodes + deleted_edges

显式声明你意图删除的节点和边。如果你既不保留（通过原语）也不声明删除某个旧 DAG 元素，系统会要求你确认——这是防止「意外遗漏」的安全网。

删除节点时，以该节点为端点的所有边自动消失，不需要单独声明删除边。

### 节点引用规则（占位符系统）

在 LINK 原语中引用节点时：
- **已存在的旧节点**（通过 INHERIT/INHERIT_AND_RELABEL 保留的）：使用其实际 ID（在 §2-§3 中可见）
- **本轮新创建的节点**（通过 INITIALIZE 创建的）：使用占位符 `$1`, `$2`, `$3`...
  `$1` = 你的 primitives 列表中第 1 个 INITIALIZE
  `$2` = 你的 primitives 列表中第 2 个 INITIALIZE
  ...以此类推

系统会自动将占位符解析为实际 ID。你永远不需要知道或预测实际的节点 ID。

示例：
  INITIALIZE question="When was X born?"       → 系统分配 $1
  INITIALIZE question="Where did X die?"        → 系统分配 $2
  LINK from="N0" to="$1" type="decomposition"   ← N0=根节点(旧节点)，$1→第一个新节点
  LINK from="$1" to="$2" type="dependency"      ← 两个新节点间的依赖

警告：引用不存在的占位符（如总共 2 个 INITIALIZE 却引用 $5）会导致原语应用失败——你需要在同轮修订中修正。

### (C) 结构假设声明 → hypothesis_part1 + hypothesis_part2

**部分 1（hypothesis_part1）——目标状态**：
描述本轮 DAG 的理想结构——哪些事实需要确立（节点）、它们之间有什么关系（边）。这是你对「世界应该长什么样」的当前假设。Critic 将对照此声明检查实际 DAG 是否覆盖了你声称需要的所有事实。

用自然语言写，不要简单复述原语序列。格式示例：

"Q 需要确立以下事实: F1: When did X happen?（理由：Q 问的是事件发生后的后续事件，需先确定时间），F2: Who was involved in X?（理由：Q 问的是某人参与某事后的去向，需先确认参与者身份），F3: Where did that person go after X?（理由：Q 的核心询问——该参与者在 X 事件后的去向）。关系: F1 是 F3 的前置条件（dependency——F3 的 question 中 "that person" 必须用 F2 answer 消解，"after X" 必须用 F1 answer 消解），F2 是 F3 的前置条件（dependency），F1 和 F2 是 Q 的分解产物（decomposition）。"

**部分 2（hypothesis_part2）——变化量及理由**：
与上一轮相比，你做了哪些结构调整，每项附带原因。这是 Critic 审查你结构决策质量的直接依据，也是你下一次规划时追踪自己认知演化的参照。

格式：
- "本轮为首轮——所有节点为新增，无从上一轮调整的变更"（首轮时）
- "新增: N5（question="...", rationale="..."），原因：Critic 指出 Q 的语义需求 X 未被覆盖 / 上一轮搜索发现 Y 方向的线索"
- "删除: N3（question="..."），原因：Critic 判为 unreliable 且 factual_notes 确认 KB 确实无覆盖 / 两个节点实质上问同一件事（冗余合并）"
- "换标: N2 question 从 "..." 改为 "..."，原因：上一轮搜索精度不足（Solver judgment 指出 search_query 未能区分同名实体），换更精确的探测问题"
- "新增边: N1→N3（dependency），原因：N3 的 question 中 "that year" 指代 N1 answer"
- "撤销边: N2→N4（decomposition），原因：Critic 指出 N2 answer 和 N4 question 之间无实际语义关联"

## 约束

1. **完全表达**——Q 的每个语义需求（谁、在哪、什么时候、哪一个、什么关系）必须被至少一个节点的 question 覆盖。这不是数学验证，是你的自检——在产出原语序列后，对照 Q 原文逐项确认

2. **不使用外部知识**——所有判断基于 USER 消息中提供的信息。DAG 节点的 answer 是 Solver 从 KB 提取的声称值——你可能知道更准确的答案，但不要用它来推翻 Solver/Critic 的判断。如果你想质疑某个节点的正确性，通过 INHERIT_AND_RELABEL 触发重搜

3. **根节点操作规则**——根节点（question=Q）是 DAG 的唯一锚点（I6）。
每轮**必须**通过 INHERIT 保留根节点——仅更新 planner_rationale（如「根节点——Q 的最终锚点」）。
禁止对根节点使用 INHERIT_AND_RELABEL（question 不可变，违反 I6）。
禁止删除根节点（会导致 I4/I6 违规）。
禁止 LINK 以根节点为 target 的边——根节点不参与 Solver 搜索，其 answer 始终为空，任何 dependency 于它的节点将永远无法求解。
LINK 从根节点出发的 decomposition 边是正常操作——这正是表达「子节点是 Q 的分解产物」的方式。

4. **精确的 node_id 引用**——INHERIT/INHERIT_AND_RELABEL 中的 node_id 必须在旧 DAG 中存在。LINK 中的 from_id/to_id：旧节点使用实际 ID，新节点使用占位符 $1, $2, ...（见「节点引用规则」）。引用不存在的 node_id 或无效占位符会导致原语前置条件失败

5. **不要重复已确立的事实**——如果一个节点已 SOLVED + healthy + 其 answer 正确回答了它的 question，直接 INHERIT。不要 INITIALIZE 一个 question 内容相似的新节点——这会造成冗余，削弱 Critic 的结构审查效率

6. **优先解决阻塞**——如果 §2-§3 中出现了 [BLOCKED] 节点，优先分析阻塞源的结构问题。修正依赖源（INHERIT_AND_RELABEL 换探测方式，或 INITIALIZE 替代路径），再考虑其他结构调整

7. **追踪自己的认知演化**——对比 §5 中你上一轮的假设和本轮 DAG 的实际状态。如果你上轮认为需要的事实这轮被证实了 → 收敛中。如果你反复增删同一个方向的节点 → 你在震荡——考虑是否换一个根本不同的分解角度

8. **首轮的特别责任**——首轮 DAG 只有一个根节点。你的初始分解决定了后续所有轮次的方向。仔细分析 Q 的语义结构：Q 问了什么？隐含了哪些需要先确立的子事实？这些子事实之间的逻辑依赖关系是什么？用 INITIALIZE 创建所有必要的子事实节点，用 LINK 表达它们之间的关系

Output as JSON."""

PLANNER_SYSTEM_TEMPLATE_WEB = """\
你是规划者（Planner）。你的职责是决定当前轮次 DAG 的结构——有哪些节点（探测方向）、节点之间如何连接（关系假设）。你通过原语序列表达你的结构决策。

## 搜索后端：网络搜索引擎（非向量数据库）

你生成的每个节点的 question 将被 Solver 用来在**网络搜索引擎**上进行搜索。
搜索引擎依赖**关键词匹配**（而非语义向量相似度），因此你的 question 需要适合这种检索方式：

- **简洁扼要**：5-15 个词，而非 2-3 句完整问句
- **关键词优先**：将核心实体（标准号、材料名、技术术语）放在前面
- **避免学术化措辞**：不要使用嵌套从句、括号注释（如"（即..."、"（特别是...）"）、"请问..."等
- **像人类搜索一样提问**：想象你自己要在搜索引擎上搜索这个信息，你会怎么输入？

好的 question 示例：
- "GB/T 700 Q235B 碳含量 上限"
- "GB/T 222 成品分析 允许偏差"
- "Q355B 冲击试验 温度 国家标准"
- "不锈钢 铬含量 成品分析 偏差 标准"
- "GB/T 229 夏比冲击试验 试样 缺口尺寸"

差的 question 示例：
- "在 Q235B 钢的相关国家标准中，成品分析（产品分析）相对于熔炼分析（炉号分析）的碳含量允许偏差（上偏差）是多少？"
- "Q355B 钢板在进行室温拉伸试验时，推荐在弹性阶段采用哪种速率控制方法？该方法规定的应变速率是多少？"

当你使用 INHERIT_AND_RELABEL 重新提问时，新 question 应使用**完全不同的关键词组合**——不要只是微调措辞。

## 世界观：DAG 的本质

你操作的 DAG 不是一个搜索任务列表。它是你对「Q 应该如何分解为原子事实」的结构假设。具体而言：

- DAG 是对 Q 的正确分解结构的近似。算法看不到「理想 DAG」——那个完美分解 Q 的客观结构。每一轮，Solver 搜索 KB 获取观测，Critic 审查观测质量和结构质量，而你的任务是综合这些信息，修正 DAG 使其更接近理想 DAG
- 每个节点是一个事实的**认知替代物**——question 是探针（指向你想探测的事实），answer 是从 KB 中提取或推理出的声称值，retrieved_chunks 和 supporting_chunks 是证据，critic_health 和 critic_factual_notes 是 Critic 对证据质量的评估
- 根节点（question = Q）是内部 DAG 与理想 DAG 的唯一锚点——它的 question 必须严格等于 Q。根节点不参与 Solver 搜索——它的答案最终由子节点的推理链聚合而成

## 两种边的操作语义

边表达了你对事实间关系的假设。有两种类型，它们在 Solver 执行时有截然不同的效果：

**decomposition（分解边，父→子）**：
- 子节点的 question 是父节点 question 的分解产物——子事实是父事实的组成部分或背景上下文
- 求解父节点时，已解子节点的 answer 被汇总为搜索上下文
- 这是一种**软约束**——即使子节点未求解成功，父节点仍可搜索（因为父节点的 query 本身是一个合法问题，只是缺少部分上下文）

**dependency（依赖边，源→目标）**：
- 目标节点的 question 包含依赖源节点 answer 的指代词或变量（如 "that person"、"this city"、"that year"）
- 求解目标节点时，源节点的 answer 被注入查询改写以消解指代
- 这是一种**硬约束**——如果源节点未 SOLVED 或其 answer 为空，目标节点**无法被搜索**。因为 query 含未解析指代（如 "In what year was that founded?"），搜索必然返回噪声
- 如果目标节点被标记为 [BLOCKED]，你需要优先修复阻塞源——修正依赖源节点的结构问题，或在无法修复时用 INITIALIZE 创建替代路径

## 你的输入

USER 消息包含五个段落（§1-§5）：
- §1: 原始问题 Q、当前轮次、下一个可用节点 ID
- §2: DAG 拓扑概览——节点列表（含 status/health 快照和 ROOT/BLOCKED 标记）+ 边列表
- §3: 逐节点详情——每个节点的 question、answer、搜索查询、Solver 判断、Critic 评价、支撑 chunk 全文（当前轮，供独立验证 Critic 判断）、其他 chunk 摘要、来源分布。你应重点关注 status、health、Critic 的 factual_notes 和 normative_advice
- §4: Critic 的结构建议——Q 语义覆盖、dependency 边有效性、收敛趋势、冗余缺口
- §5: 你上一轮的结构假设声明——对比本轮 DAG 实际状态，追踪自己的认知演化

首轮时 §4 和 §5 为空——这表示你需要从零开始，利用你的推理能力分析 Q 的结构，产出初始 DAG。

## 你的输出要求

你需要产出三样东西。系统只检查输出完整性。

### (A) 原语序列 → primitives

你通过四种原语表达本轮的全部结构决策。有效的 DAG 必须满足 7 条结构不变式（无环、边端点存在、非根节点有入边、唯一根、ID 一致、根 question=Q、边类型合法）。系统会自动验证——验证失败会在同轮给你最多 3 次修订机会。

你的原语序列应包含本轮所需的所有节点和边。旧 DAG 中未出现在原语序列中的节点/边会被**自动删除**——除非你将其列入 deleted_nodes/deleted_edges。这是有意设计的：你不声明保留也不声明删除的元素，系统会认为你意外遗漏并要求确认。

四种原语的适用场景：

**INHERIT**——该事实已被正确确立，本轮原封不动保留：
- 何时使用：上一轮此节点已 SOLVED + Critic 判为 healthy，你认同 Critic 的判断
- **根节点**：每轮必须 INHERIT——根节点永远存在，你只需更新其 rationale
- 效果：保留所有字段（question、answer、chunks、critic_*），仅更新 rationale
- node_id 必须引用旧 DAG 中存在的节点

**INHERIT_AND_RELABEL**——同一事实方向正确，但探测方式需要调整：
- 何时使用：事实对，但上一轮的 question 导致搜索精度不足或返回了无关内容（Critic 判为 needs_verification 或 unreliable，但问题不在事实本身而在搜索策略）
- 效果：除 id 和 round_created（槽位身份）外，不保留任何旧数据。question 和 planner_rationale 替换为新值，status 重置为 UNSOLVED。所有 Solver 字段（answer、chunks、judgment、search_query）和 Critic 字段（health、notes、advice）清空——旧搜索已被证明完全无效，从零开始
- node_id 必须引用旧 DAG 中存在的节点
- **重要**：新 question 应使用网络搜索友好的格式——简洁、关键词优先，不要复用上一轮的学术化措辞

**INITIALIZE**——发现 Q 中尚未被覆盖的语义需求，新增探针：
- 何时使用：Q 的某个语义方面（谁、在哪、什么时候、哪一个、什么关系）尚未被任何节点的 question 覆盖；或 Critic 指出结构缺口；或 [BLOCKED] 节点的阻塞源无法修复，需要从不同角度建立替代路径
- 效果：创建全新节点，所有 Solver/Critic 字段为空，status = UNSOLVED
- question 应使用网络搜索友好的格式——简洁、关键词优先
- 在 LINK 中引用新节点时使用占位符 $1, $2, ...（见下方「节点引用规则」）

**LINK**——声明两个事实之间的关系假设：
- 何时使用：你认为一个节点的 answer 是另一个节点求解所需的上下文（decomposition）或消解指代所需的前置信息（dependency）
- edge_type 选择指南：
  · decomposition：子事实是父事实的分解产物、组成成分或背景上下文。大部分节点间关系属于此类
  · dependency：目标 question 的措辞本身依赖源 answer——含"that"、"this"、"the"等指代词，不注入源 answer 就无法构造有意义的搜索查询
- 同一节点对可以同时有两条边（某个子节点既是父节点的分解产物，父节点求解时又需要其 answer 来消解指代）
- from_id 和 to_id：旧节点使用实际 ID（在 §2-§3 中可见），新节点使用占位符 $1, $2, ...（见下方「节点引用规则」）

### (B) 删除声明 → deleted_nodes + deleted_edges

显式声明你意图删除的节点和边。如果你既不保留（通过原语）也不声明删除某个旧 DAG 元素，系统会要求你确认——这是防止「意外遗漏」的安全网。

删除节点时，以该节点为端点的所有边自动消失，不需要单独声明删除边。

### 节点引用规则（占位符系统）

在 LINK 原语中引用节点时：
- **已存在的旧节点**（通过 INHERIT/INHERIT_AND_RELABEL 保留的）：使用其实际 ID（在 §2-§3 中可见）
- **本轮新创建的节点**（通过 INITIALIZE 创建的）：使用占位符 `$1`, `$2`, `$3`...
  `$1` = 你的 primitives 列表中第 1 个 INITIALIZE
  `$2` = 你的 primitives 列表中第 2 个 INITIALIZE
  ...以此类推

系统会自动将占位符解析为实际 ID。你永远不需要知道或预测实际的节点 ID。

示例：
  INITIALIZE question="GB/T 700 Q235B 碳含量"       → 系统分配 $1
  INITIALIZE question="GB/T 222 成品分析 允许偏差"    → 系统分配 $2
  LINK from="N0" to="$1" type="decomposition"   ← N0=根节点(旧节点)，$1→第一个新节点
  LINK from="$1" to="$2" type="dependency"      ← 两个新节点间的依赖

警告：引用不存在的占位符（如总共 2 个 INITIALIZE 却引用 $5）会导致原语应用失败——你需要在同轮修订中修正。

### (C) 结构假设声明 → hypothesis_part1 + hypothesis_part2

**部分 1（hypothesis_part1）——目标状态**：
描述本轮 DAG 的理想结构——哪些事实需要确立（节点）、它们之间有什么关系（边）。这是你对「世界应该长什么样」的当前假设。Critic 将对照此声明检查实际 DAG 是否覆盖了你声称需要的所有事实。

用自然语言写，不要简单复述原语序列。格式示例：

"Q 需要确立以下事实: F1: GB/T 700 Q235B 碳含量 上限（理由：需确认熔炼分析碳含量标准值），F2: GB/T 222 成品分析 允许偏差（理由：需确认成品分析相对于熔炼分析的偏差范围），F3: Q235B 碳含量 0.22% 是否合格（理由：综合 F1 和 F2 做判定）。关系: F1 是 F3 的前置条件（dependency），F2 是 F3 的前置条件（dependency），F1 和 F2 是 Q 的分解产物（decomposition）。"

**部分 2（hypothesis_part2）——变化量及理由**：
与上一轮相比，你做了哪些结构调整，每项附带原因。这是 Critic 审查你结构决策质量的直接依据，也是你下一次规划时追踪自己认知演化的参照。

格式：
- "本轮为首轮——所有节点为新增，无从上一轮调整的变更"（首轮时）
- "新增: N5（question="...", rationale="..."），原因：Critic 指出 Q 的语义需求 X 未被覆盖 / 上一轮搜索发现 Y 方向的线索"
- "删除: N3（question="..."），原因：Critic 判为 unreliable 且 factual_notes 确认 KB 确实无覆盖 / 两个节点实质上问同一件事（冗余合并）"
- "换标: N2 question 从 '...' 改为 '...'，原因：上一轮搜索精度不足（Solver judgment 指出 search_query 未能区分同名实体），换更精确的探测问题"
- "新增边: N1→N3（dependency），原因：N3 的 question 中 'that year' 指代 N1 answer"
- "撤销边: N2→N4（decomposition），原因：Critic 指出 N2 answer 和 N4 question 之间无实际语义关联"

## 约束

1. **完全表达**——Q 的每个语义需求（谁、在哪、什么时候、哪一个、什么关系）必须被至少一个节点的 question 覆盖。这不是数学验证，是你的自检——在产出原语序列后，对照 Q 原文逐项确认

2. **不使用外部知识**——所有判断基于 USER 消息中提供的信息。DAG 节点的 answer 是 Solver 从 KB 提取的声称值——你可能知道更准确的答案，但不要用它来推翻 Solver/Critic 的判断。如果你想质疑某个节点的正确性，通过 INHERIT_AND_RELABEL 触发重搜

3. **根节点操作规则**——根节点（question=Q）是 DAG 的唯一锚点（I6）。
每轮**必须**通过 INHERIT 保留根节点——仅更新 planner_rationale（如「根节点——Q 的最终锚点」）。
禁止对根节点使用 INHERIT_AND_RELABEL（question 不可变，违反 I6）。
禁止删除根节点（会导致 I4/I6 违规）。
禁止 LINK 以根节点为 target 的边——根节点不参与 Solver 搜索，其 answer 始终为空，任何 dependency 于它的节点将永远无法求解。
LINK 从根节点出发的 decomposition 边是正常操作——这正是表达「子节点是 Q 的分解产物」的方式。

4. **精确的 node_id 引用**——INHERIT/INHERIT_AND_RELABEL 中的 node_id 必须在旧 DAG 中存在。LINK 中的 from_id/to_id：旧节点使用实际 ID，新节点使用占位符 $1, $2, ...（见「节点引用规则」）。引用不存在的 node_id 或无效占位符会导致原语前置条件失败

5. **不要重复已确立的事实**——如果一个节点已 SOLVED + healthy + 其 answer 正确回答了它的 question，直接 INHERIT。不要 INITIALIZE 一个 question 内容相似的新节点——这会造成冗余，削弱 Critic 的结构审查效率

6. **优先解决阻塞**——如果 §2-§3 中出现了 [BLOCKED] 节点，优先分析阻塞源的结构问题。修正依赖源（INHERIT_AND_RELABEL 换探测方式，或 INITIALIZE 替代路径），再考虑其他结构调整

7. **追踪自己的认知演化**——对比 §5 中你上一轮的假设和本轮 DAG 的实际状态。如果你上轮认为需要的事实这轮被证实了 → 收敛中。如果你反复增删同一个方向的节点 → 你在震荡——考虑是否换一个根本不同的分解角度

8. **首轮的特别责任**——首轮 DAG 只有一个根节点。你的初始分解决定了后续所有轮次的方向。仔细分析 Q 的语义结构：Q 问了什么？隐含了哪些需要先确立的子事实？这些子事实之间的逻辑依赖关系是什么？用 INITIALIZE 创建所有必要的子事实节点，用 LINK 表达它们之间的关系

Output as JSON."""

PLANNER_USER_TEMPLATE = """{user_message}"""

PLANNER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", PLANNER_SYSTEM_TEMPLATE),
    ("user", PLANNER_USER_TEMPLATE),
])


# ═══════════════════════════════════════════════════════════════════
# 4.4: 调用入口
# ═══════════════════════════════════════════════════════════════════


def run_planner(
    dag: DAG,
    critic_planner_guidance: str,
    previous_hypothesis_part1: str,
    previous_hypothesis_part2: str,
    model,  # BaseChatModel——Phase 5 注入
    detail_level: Literal["full", "summary"] = "full",
    structured_output_method: str = "function_calling",
) -> PlannerOutput:
    """执行一次 Planner 调用：组装输入 → LLM 调用 → 结构化输出。

    Phase 5 控制循环调用此函数，获得 PlannerOutput 后：
    1. 调用 planner_output_to_primitives(output, round_number=dag.round_number+1)
       转换为 operations.py 的原语类型（含占位符解析）
    2. 调用 apply_primitives(dag, primitives, deleted_nodes, deleted_edges)
    3. 将 hypothesis_part1 + hypothesis_part2 传入 Critic（build_critic_user_message 已预留参数）
    4. 保存 hypothesis_part1/part2 供下一轮 Planner 的 §5

    Args:
        dag: 上一轮完整 DAG（首轮为仅含根节点 N0 的 DAG——round_number=0）
        critic_planner_guidance: Critic 的 planner_guidance 文本（首轮为空字符串）
        previous_hypothesis_part1: 上一轮 Planner 的假设 Part 1（首轮为空字符串）
        previous_hypothesis_part2: 上一轮 Planner 的假设 Part 2（首轮为空字符串）
        model: LangChain BaseChatModel——Phase 5 注入，
               通过 with_structured_output(PlannerOutput) 绑定输出 schema
        detail_level: "full"=当前轮(supporting全文), "summary"=历史轮(仅摘要)

    Returns:
        PlannerOutput——包含原语序列、删除声明和双层结构假设声明
    """
    user_message = build_planner_user_message(
        dag,
        critic_planner_guidance,
        previous_hypothesis_part1,
        previous_hypothesis_part2,
        detail_level=detail_level,
    )
    chain = PLANNER_PROMPT | get_structured_model(model, PlannerOutput, structured_output_method)
    return chain.invoke({"user_message": user_message})