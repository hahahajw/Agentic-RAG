# ═══════════════════════════════════════════════════════════════════
# Critic 组件
# ═══════════════════════════════════════════════════════════════════
# 3.1 输入组装: 将 DAG + Planner 假设 + 系统检查 → 格式化 USER 消息
# 3.2 逐节点答案质量审查: 独立性/一致性检查、health 赋值、factual_notes
# 3.3 结构质量审查: Q 覆盖、依赖有效性、收敛趋势、冗余缺口
# 3.4 终止判断 + 输出回填: 4 条件裁决、DAGNode.critic_* 写入
# 3.2-3.4 合并为一次 LLM 调用——Critic 自行决定审查顺序。

from typing import Literal

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from .models import DAG, DAGNode, ChunkInfo, EdgeType, NodeStatus, CriticHealth
from .dag_utils import (
    get_dependency_answers, get_child_answers,
    get_direct_children, get_dependency_sources,
    topological_layers, find_leaves, find_all_paths,
)
from .formatting import (
    truncate, source_distribution, node_label,
    format_chunk_summary, format_chunk_full,
    format_supporting_chunks, format_non_supporting_chunks,
    format_source_summary, format_blocking_info,
)
from .structured_output import get_structured_model


# ═══════════════════════════════════════════════════════════════════
# Pydantic 输出模型 (3.2-3.4)
# ═══════════════════════════════════════════════════════════════════


class NodeReview(BaseModel):
    """Critic 对单个 DAG 节点的审查结果——回填至 DAGNode.critic_* 字段"""
    node_id: str = Field(
        description="节点 ID——必须与 DAG 中的一个节点精确对应。系统会验证 node_reviews 覆盖了全部节点"
    )
    critic_health: CriticHealth = Field(
        description=(
            "健康判词：healthy(证据充分，≥2独立来源一致或单来源直接具体) / "
            "needs_verification(有提示但不充分，单来源模糊或推理不自洽) / "
            "unreliable(来源矛盾或声称与chunk内容不匹配) / "
            "blocked(搜索无返回、返回完全无关、或依赖未满足未搜索)"
        )
    )
    critic_factual_notes: str = Field(
        description=(
            "客观事实描述——来源数量、独立性（不同文档=独立）、"
            "一致性（一致/粒度差异/矛盾）、交叉验证状态（通过/不充分/存在冲突）。"
            "推理型节点标注'无独立来源——由已知事实推理得出'。不评价，只陈述"
        )
    )
    critic_normative_advice: str = Field(
        description=(
            "基于事实观察的具体建议——建议重搜/换搜索方向/拆分/放弃此方向/INHERIT保留。"
            "引用具体节点ID和chunk ID。ROOT节点的advice应是结构层面的"
            "（如'建议新增节点覆盖缺失的语义X'）"
        )
    )


class TerminationJudgment(BaseModel):
    """终止判断——综合系统条件①④和LLM判断条件②③的最终裁决"""
    should_terminate: bool = Field(
        description="是否应终止探索——仅当四条件（①②③④）全部满足时为 True"
    )
    condition_2_passed: bool = Field(
        description="推理链语义匹配——从根到所有叶子的每个 dependency 环节是否语义有效"
    )
    condition_3_passed: bool = Field(
        description="全部节点 healthy——node_reviews 中所有节点的 critic_health 是否均为 healthy"
    )
    termination_reason: str = Field(
        description=(
            "终止/不终止的原因。不终止时明确指出阻碍条件（如'条件②未通过：N2→N4依赖链语义断裂'）。"
            "终止时说明置信评估（如'四条件全满足，推理链完整，所有节点证据充分'）"
        )
    )


class CriticOutput(BaseModel):
    """Critic 单次审查的完整输出——系统消费并分发至 DAGNode / Planner / 控制循环"""
    node_reviews: list[NodeReview] = Field(
        description="逐节点审查结果——必须覆盖 DAG 中的每个节点（包括 ROOT）。系统验证完整性"
    )
    planner_guidance: str = Field(
        description=(
            "给 Planner 的结构分析——Q语义覆盖、dependency边有效性、收敛趋势、冗余缺口。"
            "直接、具体、可行动。引用具体节点ID。让Planner知道'结构层面应该怎么调整'"
        )
    )
    termination: TerminationJudgment = Field(
        description="终止判断——综合系统条件①④和LLM判断条件②③的最终裁决。Phase 5控制循环消费"
    )


# ═══════════════════════════════════════════════════════════════════
# 3.1 输入组装
# ═══════════════════════════════════════════════════════════════════


def build_critic_user_message(
    dag: DAG,
    planner_hypothesis_part1: str,
    planner_hypothesis_part2: str,
    previous_dag: DAG | None,
    detail_level: Literal["full", "summary"] = "full",
) -> str:
    """组装 Critic 的 USER 消息——系统中介通信机制的具体实现。

    FRAMEWORK.md 模块 3 要求：系统从各角色的结构化输出字段提取信息，
    注入另一角色的 USER 消息。此函数是"Solver/Critic 输出 → Critic 输入"
    方向的格式化管道。

    detail_level="full" (默认): 当前轮——supporting chunks 展示 page_content 全文
    detail_level="summary": 历史轮——supporting chunks 仅展示摘要

    五段结构：
      §1 原始问题与轮次
      §2 DAG 拓扑概览（节点列表 + 边列表）
      §3 逐节点详情（拓扑层排序，chunk 分层展示）
      §4 Planner 结构假设（原文照登）
      §5 系统自动检查（条件①/④ 结果）
    """
    sections: list[str] = []

    sections.append(_format_section1_question(dag))
    sections.append(_format_section2_topology(dag))
    sections.append(_format_section3_node_details(dag, detail_level))
    sections.append(_format_section4_planner_hypothesis(
        planner_hypothesis_part1, planner_hypothesis_part2
    ))
    sections.append(_format_section5_system_checks(dag, previous_dag))

    return "\n\n".join(sections)


# ═══════════════════════════════════════════════════════════════════
# §1: 原始问题与轮次
# ═══════════════════════════════════════════════════════════════════


def _format_section1_question(dag: DAG) -> str:
    """§1 — 最简段落：本轮在回答什么问题、第几轮。"""
    return (
        f"═══════════════════════════════════════════\n"
        f"§1 原始问题与轮次\n"
        f"───────────────────────────────────────────\n"
        f"原始问题 Q: {dag.q}\n"
        f"当前轮次: 第 {dag.round_number} 轮"
    )


# ═══════════════════════════════════════════════════════════════════
# §2: DAG 拓扑概览
# ═══════════════════════════════════════════════════════════════════


def _format_section2_topology(dag: DAG) -> str:
    """§2 — DAG 骨架：节点一行一个，边紧凑列出。

    Critic 用此节快速理解拓扑结构，随后在 §3 深读各节点。
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
            lines.append(
                f"  [{node_label(dag, nid)}] {nid}: {truncate(node.question, 60)}  "
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


# ═══════════════════════════════════════════════════════════════════
# §3: 逐节点详情
# ═══════════════════════════════════════════════════════════════════


def _format_section3_node_details(dag: DAG, detail_level: str) -> str:
    """§3 — 每个节点的完整审查数据，按拓扑层排列。

    Chunk 分层展示（detail_level 控制）：
    - "full": Supporting chunks page_content 全文，non-supporting 摘要
    - "summary": 所有 chunks 仅摘要

    节点按六种形态分发格式：
    - ROOT, BLOCKED, REASONING, SOLVED, FAILED_SEARCH, EMPTY_SEARCH
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
            sections.append(_format_single_node(dag, nid, detail_level))

    return "\n\n".join(sections)


def _format_single_node(dag: DAG, nid: str, detail_level: str) -> str:
    """格式化单个节点的审查数据——分发到具体的形态格式化函数。

    分发基于两个关键信号：
    - was_searched: search_query 非空 ⇔ Solver 调用了 search_fn
    - node.answer: 非空 ⇔ 提取/推理成功

    六种形态：
    - ROOT: 根节点（跳过搜索）
    - BLOCKED: 未搜索 + UNSOLVED（dependency 硬阻塞）
    - REASONING: 未搜索 + SOLVED（prepare_node_query 直接推理）
    - SOLVED: 已搜索 + 提取成功
    - FAILED_SEARCH: 已搜索 + 有 chunk 但提取失败
    - EMPTY_SEARCH: 已搜索 + 搜索返回空列表
    """
    node = dag.nodes[nid]
    root = dag.root
    is_root = root is not None and nid == root.id

    if is_root:
        return _format_root_node(dag, nid)

    was_searched = bool(node.search_query)

    if not was_searched:
        if node.status == NodeStatus.UNSOLVED:
            return _format_blocked_node(dag, nid)
        else:
            return _format_reasoning_node(dag, nid)

    if node.answer:
        return _format_solved_node(dag, nid, detail_level)
    elif node.retrieved_chunks:
        return _format_failed_search_node(dag, nid, detail_level)
    else:
        return _format_empty_search_node(dag, nid, detail_level)


# ── 节点形态格式化辅助函数 ──


def _format_root_node(dag: DAG, nid: str) -> str:
    """根节点——跳过 Solver 搜索，展示子节点推理链。

    FRAMEWORK.md 模块 2: 根节点 question = Q，不参与搜索阶段。
    Critic 审查根节点的子节点推理链完整性（而非自身 chunk 支撑）。
    """
    node = dag.nodes[nid]
    health_str = node.critic_health.value if node.critic_health else "未审查"

    lines = [
        f"━━━ [ROOT] {nid}: {node.question} ━━━",
        f"(根节点——跳过 Solver 搜索阶段。答案由子节点推理链合成，填入由答案生成器完成)",
        f"状态: {node.status.value} | 健康: {health_str}",
    ]

    if node.planner_rationale:
        lines.append(f"存在理由: {node.planner_rationale}")

    children = get_direct_children(dag, nid)
    if children:
        child_str = ", ".join(
            f"{cid}({dag.nodes[cid].status.value}, "
            f"answer={truncate(dag.nodes[cid].answer, 40) or '无'})"
            for cid in children
        )
        lines.append(f"直接子节点: {child_str}")
    else:
        lines.append("直接子节点: (无)")

    deps = get_dependency_sources(dag, nid)
    if deps:
        dep_str = ", ".join(
            f"{did}(answer={truncate(dag.nodes[did].answer, 40) or '(空)'})"
            for did in deps
        )
        lines.append(f"依赖源节点: {dep_str}")

    if node.critic_factual_notes:
        lines.append(f"事实笔记: {node.critic_factual_notes}")
    if node.critic_normative_advice:
        lines.append(f"建议: {node.critic_normative_advice}")

    return "\n".join(lines)


def _format_solved_node(dag: DAG, nid: str, detail_level: str) -> str:
    """正常 SOLVED 节点——chunk 分层展示。

    detail_level="full": supporting chunks 全文 + non-supporting 摘要
    detail_level="summary": 所有 chunks 仅摘要
    """
    node = dag.nodes[nid]
    health_str = node.critic_health.value if node.critic_health else "未审查"

    lines = [
        f"━━━ {nid}: {node.question} ━━━",
        f"状态: {node.status.value} | 健康: {health_str}",
        f"声称答案: {node.answer}",
        f"搜索查询: {node.search_query or '(未记录)'}",
        f"存在理由: {node.planner_rationale}",
        f"Solver 判断: {node.solver_judgment or '(无)'}",
    ]

    if node.retrieved_chunks:
        lines.append("")
        # Supporting chunks
        supporting_lines = format_supporting_chunks(node, detail_level)
        if supporting_lines:
            lines.extend(supporting_lines)
        # Non-supporting chunks (always summary)
        non_supporting_lines = format_non_supporting_chunks(node)
        if non_supporting_lines:
            lines.extend(non_supporting_lines)
        # Source distribution
        source_line = format_source_summary(node.retrieved_chunks)
        if source_line:
            lines.append(source_line)
    else:
        lines.append("  ▸ (无检索结果)")

    if node.critic_factual_notes:
        lines.append(f"  ▸ 上轮事实笔记: {node.critic_factual_notes}")
    if node.critic_normative_advice:
        lines.append(f"  ▸ 上轮建议: {node.critic_normative_advice}")

    return "\n".join(lines)


def _format_reasoning_node(dag: DAG, nid: str) -> str:
    """推理 SOLVED 节点——prepare_node_query 直接从已知事实推理出答案。

    无 chunk、无 search_query、solver_judgment 标记推理来源。
    """
    node = dag.nodes[nid]
    health_str = node.critic_health.value if node.critic_health else "未审查"

    lines = [
        f"━━━ {nid}: {node.question} ━━━",
        f"状态: {node.status.value} | 健康: {health_str}",
        f"声称答案: {node.answer}",
        f"搜索查询: (未执行搜索——从已知事实推理得出)",
        f"存在理由: {node.planner_rationale}",
        f"Solver 判断: {node.solver_judgment or '(无)'}",
    ]

    dep_answers = get_dependency_answers(dag, nid)
    child_answers = get_child_answers(dag, nid)
    if dep_answers:
        lines.append(f"推理依据-依赖事实: {', '.join(dep_answers)}")
    if child_answers:
        lines.append(f"推理依据-子事实: {', '.join(child_answers)}")

    lines.append("  ▸ (无检索结果——由推理得出)")

    return "\n".join(lines)


def _format_failed_search_node(dag: DAG, nid: str, detail_level: str) -> str:
    """已搜索但提取失败的节点——有 chunks 但 answer 为空。

    所有 chunk 展示为摘要（无支撑 chunk——因为没有找到答案）。
    Critic 需审查：是搜索方向错误还是 KB 中确实无答案。
    """
    node = dag.nodes[nid]
    health_str = node.critic_health.value if node.critic_health else "未审查"

    lines = [
        f"━━━ {nid}: {node.question} ━━━",
        f"状态: {node.status.value} | 健康: {health_str}",
        f"声称答案: (未找到)",
        f"搜索查询: {node.search_query or '(未记录)'}",
        f"存在理由: {node.planner_rationale}",
        f"Solver 判断: {node.solver_judgment or '(无)'}",
    ]

    if node.retrieved_chunks:
        lines.append(f"  ▸ 检索结果 (摘要, {len(node.retrieved_chunks)} 个):")
        for chunk in node.retrieved_chunks:
            lines.append(f"    {format_chunk_summary(chunk)}")

        source_line = format_source_summary(node.retrieved_chunks)
        if source_line:
            lines.append(source_line)

    return "\n".join(lines)


def _format_blocked_node(dag: DAG, nid: str) -> str:
    """被硬阻塞节点——dependency 源未解导致无法构造查询。

    无 chunk、无 search_query。展示阻塞原因以帮助 Critic
    给 Planner 提供结构性修复建议。
    """
    node = dag.nodes[nid]

    blocking_sources: list[str] = []
    for edge in dag.edges:
        if edge.to_id == nid and edge.edge_type == EdgeType.DEPENDENCY:
            src = dag.nodes.get(edge.from_id)
            if src is None:
                continue
            if src.status != NodeStatus.SOLVED or not src.answer:
                blocking_sources.append(
                    f"{edge.from_id}(status={src.status.value}, answer="
                    f"{truncate(src.answer, 30) or '(空)'})"
                )

    reason = (
        f"依赖源未满足，无法构造有效搜索查询: {', '.join(blocking_sources)}"
        if blocking_sources
        else "未知原因"
    )

    lines = [
        f"━━━ [BLOCKED] {nid}: {node.question} ━━━",
        f"阻塞原因: {reason}",
        f"状态: {node.status.value}",
        f"存在理由: {node.planner_rationale}",
        f"  ▸ (未执行搜索——被 dependency 硬阻塞)",
    ]

    return "\n".join(lines)


def _format_empty_search_node(dag: DAG, nid: str, detail_level: str) -> str:
    """搜索返回空结果的节点——search 调用返回了空列表。

    与 BLOCKED 的区别：此节点成功构造了查询并调用了 search_fn，
    但 KB 中无匹配结果。问题可能在搜索方向或 KB 覆盖度——
    而非 Planner 的依赖结构设计。

    与 FAILED_SEARCH 的区别：没有 chunk 可供提取——不是"提取失败"，
    而是"无内容可提取"。
    """
    node = dag.nodes[nid]
    health_str = node.critic_health.value if node.critic_health else "未审查"

    lines = [
        f"━━━ {nid}: {node.question} ━━━",
        f"状态: {node.status.value} | 健康: {health_str}",
        f"声称答案: (未找到)",
        f"搜索查询: {node.search_query}",
        f"存在理由: {node.planner_rationale}",
        f"Solver 判断: {node.solver_judgment or '(无)'}",
        f"  ▸ 搜索未返回任何结果——KB 中可能无匹配文档，或搜索方向需要调整",
    ]

    if node.critic_factual_notes:
        lines.append(f"  ▸ 上轮事实笔记: {node.critic_factual_notes}")
    if node.critic_normative_advice:
        lines.append(f"  ▸ 上轮建议: {node.critic_normative_advice}")

    return "\n".join(lines)


# ── 来源分布统计 ──


def _source_distribution(chunks: list[ChunkInfo]) -> dict[str, int]:
    """统计各 context_title 的 chunk 数量——Critic 独立性检查的直接输入。

    两个 chunk 来自同一 context_title → 同一文档的不同段落 → 非独立。
    不同 context_title → 不同文档 → 独立来源。
    """
    counts: dict[str, int] = {}
    for chunk in chunks:
        counts[chunk.context_title] = counts.get(chunk.context_title, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: -x[1]))


# ═══════════════════════════════════════════════════════════════════
# §4: Planner 结构假设
# ═══════════════════════════════════════════════════════════════════


def _format_section4_planner_hypothesis(
    part1: str,
    part2: str,
) -> str:
    """§4 — Planner 结构假设原文照登，不做系统改写。

    Part 1: 目标状态——"本轮 DAG 应该长什么样"
    Part 2: 变化量及理由——"从上一轮调整到现在的原因"

    Critic 对照 part 1 检查实际 DAG 是否覆盖了 Planner 声称需要的所有事实。
    Part 2 是 Critic 审查 Planner 结构决策质量的直接依据。
    """
    lines = [
        "═══════════════════════════════════════════",
        "§4 Planner 结构假设声明",
        "───────────────────────────────────────────",
        "以下为 Planner 对本轮 DAG 结构的声明。",
        "Critic 请对照「部分 1（目标状态）」检查实际 DAG（§2-§3）"
        "是否覆盖了声称需要的所有事实。",
        "「部分 2（变化量及理由）」是审查 Planner 结构决策质量的依据——"
        "Planner 从上一轮调整到现在的理由是否合理？",
        "",
        "── 部分 1: 目标状态（Planner 认为世界应该长这样）──",
        part1 if part1.strip() else "(Planner 未提供目标状态声明)",
        "",
        "── 部分 2: 变化量及理由（Planner 的认知调整说明）──",
        part2 if part2.strip() else "(Planner 未提供变化说明)",
    ]

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# §5: 系统自动检查
# ═══════════════════════════════════════════════════════════════════


def _format_section5_system_checks(
    dag: DAG,
    previous_dag: DAG | None,
) -> str:
    """§5 — 终止条件①和④的系统自动计算结果。

    这两个条件是确定性检查——不需要 LLM 语义理解。
    注入为 Critic 的输入信息：Critic 基于这些确定性结果 + §3-§4 的内容，
    集中 LLM 能力在条件②（推理链语义匹配）和条件③（全部 healthy）的判断上。
    """
    lines = [
        "═══════════════════════════════════════════",
        "§5 系统自动检查（终止条件 ①④——确定性计算，非 LLM 判断）",
        "───────────────────────────────────────────",
    ]

    lines.append(_check_condition_1(dag))
    lines.append("")
    lines.append(_check_condition_4(dag, previous_dag))
    lines.append("")
    lines.append(_compute_reasoning_chains(dag))

    return "\n".join(lines)


def _check_condition_1(dag: DAG) -> str:
    """终止条件①：除根节点外所有节点 status = SOLVED。

    系统自动判断（FRAMEWORK.md 模块 5 终止条件①）。
    """
    root = dag.root
    if root is None:
        return "条件① 无法检查：DAG 没有唯一根节点（I4 违规）。"

    non_root = [
        (nid, node) for nid, node in dag.nodes.items() if nid != root.id
    ]
    unsolved = [
        (nid, node) for nid, node in non_root
        if node.status != NodeStatus.SOLVED
    ]

    total = len(non_root)

    if total == 0:
        return "条件① 满足（DAG 仅含根节点，无非根节点）。"
    if not unsolved:
        return f"条件① 满足：所有 {total} 个非根节点均已 SOLVED。"

    solved_count = total - len(unsolved)
    result_lines = [
        f"条件① 未满足：{solved_count}/{total} 个非根节点已 SOLVED。",
        "未 SOLVED 的节点:",
    ]
    for nid, node in unsolved:
        if not node.search_query and not node.retrieved_chunks:
            cause = "被硬阻塞（dependency 源未满足，未执行搜索）"
        elif node.retrieved_chunks and not node.answer:
            cause = "已搜索但未提取到答案"
        elif not node.retrieved_chunks and node.answer:
            cause = "从已知事实推理得出（未搜索）——但 status 未设为 SOLVED（异常）"
        else:
            cause = f"原因未知（status={node.status.value}）"
        short_q = truncate(node.question, 50)
        result_lines.append(f"  - {nid} ({short_q}): {cause}")

    return "\n".join(result_lines)


def _check_condition_4(dag: DAG, previous_dag: DAG | None) -> str:
    """终止条件④：DAG 拓扑连续两轮无变化。

    系统自动判断（FRAMEWORK.md 模块 5 终止条件④）。
    比较节点集（ID + question）和边集（from, to, type）。

    previous_dag 为 None 时为首轮，无条件④历史数据。

    注意：此处只提供单轮比较数据（本轮 vs 上一轮）。
    条件④要求连续 2 轮拓扑快照相同——由控制循环（Phase 5）
    跨轮追踪 topology_history 列表完成。
    """
    if previous_dag is None:
        return "条件④ 不适用：首轮，无历史 DAG 可比对。"

    curr_nodes = {(nid, node.question) for nid, node in dag.nodes.items()}
    prev_nodes = {(nid, node.question) for nid, node in previous_dag.nodes.items()}
    curr_edges = {(e.from_id, e.to_id, e.edge_type.value) for e in dag.edges}
    prev_edges = {(e.from_id, e.to_id, e.edge_type.value) for e in previous_dag.edges}

    nodes_added = curr_nodes - prev_nodes
    nodes_removed = prev_nodes - curr_nodes
    edges_added = curr_edges - prev_edges
    edges_removed = prev_edges - curr_edges

    unchanged = not (nodes_added or nodes_removed or edges_added or edges_removed)

    if unchanged:
        return (
            "条件④ 本轮通过：本轮拓扑与上一轮一致"
            "（节点集、questions、边集均未变化）。\n"
            "注意：条件④要求连续 2 轮拓扑快照相同——"
            "由控制循环（Phase 5）跨轮追踪 topology_history 列表。\n"
            "如这是首次出现拓扑不变（即上一轮有变化），"
            "本轮通过不意味着条件④已满足——还需下一轮确认。"
        )

    change_lines = ["条件④ 本轮未通过：本轮拓扑与上一轮相比有变化。"]
    for (nid, q) in sorted(nodes_added, key=lambda x: x[0]):
        change_lines.append(f"  + 新增节点: {nid} (question=\"{truncate(q, 50)}\")")
    for (nid, q) in sorted(nodes_removed, key=lambda x: x[0]):
        change_lines.append(f"  - 删除节点: {nid} (question=\"{truncate(q, 50)}\")")
    for (fid, tid, etype) in sorted(edges_added):
        change_lines.append(f"  + 新增边: {fid} → {tid} ({etype})")
    for (fid, tid, etype) in sorted(edges_removed):
        change_lines.append(f"  - 删除边: {fid} → {tid} ({etype})")

    return "\n".join(change_lines)


# ═══════════════════════════════════════════════════════════════════
# 推理链路径追踪 —— 系统预计算，供条件②审查
# ═══════════════════════════════════════════════════════════════════


def _compute_reasoning_chains(dag: DAG) -> str:
    """预计算根→每片叶子的完整路径，供 Critic 逐环节判断语义匹配性。

    输出格式：
      路径 1 (根→N2→N4):
        步骤 1: N2 answer="Moscow State University"
                --[dependency]--> N3 question="In what year was that founded?"
        步骤 2: N3 answer="1755"
                --[dependency]--> N4 question="What event happened in that year?"

    Critic 对照每个步骤的上游 answer 和下游 question，判断依赖链是否语义有效。
    系统做图遍历（确定性），Critic 做语义判断（LLM），分工明确。
    """
    root = dag.root
    if root is None:
        return "── 推理链路径追踪 ──\n无法计算：DAG 无唯一根节点（I4 违规）。"

    leaves = find_leaves(dag)
    if not leaves:
        return "── 推理链路径追踪 ──\nDAG 无叶子节点（所有节点均有出边——可能存在环，I1 违规）。"

    lines = [
        "── 推理链路径追踪（供条件②审查）──",
        "以下为系统预计算的从根到每片叶子的完整路径。",
        "Critic 请逐环节判断：",
        "  (a) 上游 answer 是否语义上消解了下游 question 中的指代/变量？",
        "  (b) 分解链的答案聚合后是否能支撑父节点的推理？",
        "",
    ]

    for i, leaf in enumerate(sorted(leaves), 1):
        paths = find_all_paths(dag, root.id, leaf)
        if not paths:
            lines.append(f"叶子 {leaf}: 无从根可达的路径（I3 违规——此节点可能孤立）")
            lines.append("")
            continue

        for j, path in enumerate(paths):
            if len(paths) > 1:
                lines.append(f"路径 {i}.{j+1}: {' → '.join(path)}")
            else:
                lines.append(f"路径 {i}: {' → '.join(path)}")

            for k in range(len(path) - 1):
                src_id = path[k]
                tgt_id = path[k + 1]
                src_node = dag.nodes[src_id]
                tgt_node = dag.nodes[tgt_id]

                # 找到连接这两个节点的边类型
                edge_types: list[str] = []
                for edge in dag.edges:
                    if edge.from_id == src_id and edge.to_id == tgt_id:
                        edge_types.append(edge.edge_type.value)
                edge_label = "+".join(edge_types) if edge_types else "?"

                lines.append(
                    f"  步骤 {k+1}: {src_id} answer=\"{truncate(src_node.answer, 60) or '(空)'}\""
                    f"  --[{edge_label}]-->  {tgt_id} question=\"{truncate(tgt_node.question, 60)}\""
                )
            lines.append("")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# Critic SYSTEM Prompt + LLM 调用 (3.2-3.4)
# ═══════════════════════════════════════════════════════════════════


CRITIC_SYSTEM_TEMPLATE = """\
你是严格的质量审查者。你的职责：
1. 审查 Solver 对每个节点的搜索结果——判断答案质量的可靠性
2. 审查 Planner 的 DAG 结构设计——判断结构是否合理、是否存在缺口或冗余
3. 做出终止判断——当前 DAG 是否已达到可生成最终答案的状态

## 世界观：你在审查什么

你审查的 DAG 不是搜索任务清单——它是 Planner 对「Q 应该如何分解为原子事实」的结构假设。具体而言：

- DAG 是对 Q 的正确分解结构（理想 DAG）的认知近似。算法看不到理想 DAG——那个完美分解 Q 的客观结构。每一轮，Solver 搜索 KB 获取观测，Planner 根据你的审查反馈修正 DAG 结构，使其更接近理想 DAG
- 每个节点是一个事实的**认知替代物**——question 是探针（指向想探测的事实），answer 是从 KB 提取或推理出的声称值，retrieved_chunks 和 supporting_chunks 是证据
- 你的审查有两个维度：
  (a) **答案质量**——证据是否充分支撑声称？独立来源是否一致？
  (b) **结构质量**——Planner 的分解假设是否合理？DAG 是否在逼近理想结构？
- 结构问题通过症状暴露：语义缺口 → 覆盖率不完整；无效依赖 → 搜索结果与源 answer 不匹配；反复震荡 → 拓扑不收敛（同方向反复增删）。你不需要看到理想 DAG——你通过症状诊断结构问题
- 你的所有判断基于 DAG 内部可得信息——不与隐藏的真实世界比较。最终正确性由离线评估用 ground truth 验证

## 输入格式

你的 USER 消息包含五个段落（§1-§5）：
- §1: 原始问题 Q 和当前轮次
- §2: DAG 拓扑概览——节点列表（含 status/health 快照）+ 边列表
- §3: 逐节点详情——每个节点的完整审查数据。支撑 chunk 展示全文，
  非支撑 chunk 仅展示摘要。"来源分布"统计了各文档的 chunk 数量
- §4: Planner 的结构假设声明——"我认为世界应该长这样"（部分 1）
  和"从上一轮调整到现在的理由"（部分 2）
- §5: 系统自动计算结果——终止条件①（节点是否全 SOLVED）、
  条件④（拓扑是否变化）、推理链路径追踪（根→每片叶子的完整路径）

## 你的任务

你需要产出三样东西。你可以按任意顺序组织推理过程。系统只检查输出完整性。

### (A) 逐节点审查 → node_reviews

对 DAG 中的**每一个节点**（包括 ROOT）产出一条 NodeReview。

**critic_health 赋值准则**：

healthy（健康）：
- 答案有充分证据支撑。≥2 个独立来源（不同文档）一致 → 强正向信号。
  单来源但直接、具体地陈述了答案 → 可接受（需在 factual_notes 中标注"单来源"）
- ROOT 节点的 healthy 取决于其子节点推理链的完整性和一致性——
  ROOT 自身不搜索，其"证据"是子节点推理链的聚合

needs_verification（待验证）：
- 有提示性证据但不充分——单来源、表述含糊、或 Solver judgment
  与 chunk 内容之间存在不自洽
- 推理型节点（标注"从已知事实推理得出"）：无独立来源可验证，
  检查推理是否逻辑自洽、是否过度推断。在 factual_notes 中标注推理依据

unreliable（不可靠）：
- 不同来源之间存在明显矛盾——对照 supporting chunk 全文逐字比对
- Solver 声称的答案与支撑 chunk 内容不匹配——答案不在 chunk 中
- 来源明显不相关（chunk 讨论的是不同实体/主题）

blocked（阻塞）：
- 搜索未返回任何结果（EMPTY_SEARCH）或返回内容完全无关（FAILED_SEARCH）
- [BLOCKED] 标注的节点——因依赖未满足而未能执行搜索

**critic_factual_notes（事实笔记）**：
客观描述你在数据中看到的——不评价，只陈述：
- 来源数量和分布："3 个来源，其中 2 个独立（文档A，文档B）"
- 一致性："两个独立来源对核心事实表述一致" / "来源A和B存在粒度差异"
- 交叉验证状态："通过" / "不充分（仅 1 个来源）" / "存在冲突（来源A和B矛盾）"
- 对于推理型节点："无独立来源——由已知事实推理得出。推理依据: {{依赖事实}} + {{子事实}}"
- 对于 BLOCKED 节点："无法求解——依赖源 Nx answer 为空，未执行搜索"

**critic_normative_advice（规范性建议）**：
基于事实观察的**具体**建议——Planner 和 Solver 应据此行动：
- "建议重搜——当前 chunk 不相关，尝试更精确的查询"
- "建议拆分——此问题包含多个子问题，宜分解为独立节点"
- "建议换搜索方向——当前方向反复无改善，考虑不同角度"
- "建议放弃此方向——三无（无结果/无关/矛盾），且 KB 确实无覆盖"
- "建议 INHERIT——证据充分，下轮直接保留"
- ROOT 节点的 advice 应是结构层面的——"建议新增节点覆盖缺失的语义 X"
  或"依赖边 N1→N3 语义不匹配，建议修正"

**重要**：你必须为 DAG 中的每一个节点输出审查结果。系统会验证 node_reviews
是否覆盖了全部的节点 ID。遗漏节点 = 审查不完整。

### (B) 结构质量审查 → planner_guidance

这是一段直接写给 Planner 的文字。分析当前 DAG 的全局结构质量：

1. **Q 语义覆盖**：对照 §4 部分 1（Planner 声称需要的事实）和实际 DAG（§2-§3），
   Q 的所有语义需求是否都有节点覆盖？Planner 声称需要的但 DAG 中缺失的 → 指出缺口。
   DAG 中有但 Planner 未声明的 → 询问意图。

2. **Dependency 边有效性**：对每条 dependency 边，源节点的 answer 是否
   语义上使目标节点的 question 可被执行？
   例：源 answer="Moscow State University" + 目标 question="In what year was that founded?" → 有效
   例：源 answer="1755" + 目标 question="Who won the award?" → 无效——年份不能消解"who"

3. **收敛趋势**：对照 §5 条件④的结果（拓扑变化明细），判断 Planner 的认知是否在收敛：
   - 新增节点是在填补缺口（正向）还是试探性震荡（摇摆）
   - 删除的节点是合理放弃（正向）还是反复（上一轮新增的这轮就删了）
   - 新一轮的调整理由（§4 部分 2）是否合理

4. **冗余与缺口**：是否有多个节点在问实质上相同的问题（冗余）？
   是否有 Q 的某个语义方面完全没有被覆盖（缺口）？

写作风格：直接、具体、可行动。引用具体节点 ID。不说"有问题"，
说"N3 的依赖源 N2 给出的答案是年份，但 N3 的 question 要求的是人名，
这个依赖链在语义上不成立"。

### (C) 终止判断 → termination

综合 §5 系统计算结果和你的 (A)(B) 分析：

- condition_2_passed（推理链语义匹配）：
  对照 §5 的路径追踪，逐环节判断每个 dependency 步骤：
  上游 answer 是否语义上消解了下游 question 中的指代/变量？
  所有路径的所有步骤都匹配 → True

- condition_3_passed（全部节点 healthy）：
  你在 (A) 中给出的所有节点（包括 ROOT）的 critic_health 是否均为 healthy？

- should_terminate：
  True ⇔ §5 的条件①已满足 AND 条件④已通过
         AND condition_2_passed=True AND condition_3_passed=True

- termination_reason：
  不终止时：明确指出阻碍终止的具体条件（如"条件②未通过：N2→N4 的依赖链语义断裂"）
  终止时：说明对答案的置信评估（如"四条件全满足，推理链完整，所有节点证据充分"）

## 约束

- 所有判断严格基于 USER 消息中提供的信息。不使用外部知识
- 阅读 supporting chunk 全文时逐字比对——Solver 声称的答案是否真的出现在 chunk 中？
- 阅读 non-supporting chunk 摘要时快速扫描——是否有 Solver 明显遗漏的答案？
- 对每个节点独立审查后再做全局判断——不要因为多数节点 healthy 就放宽对个别节点的审查

Output as JSON."""

CRITIC_USER_TEMPLATE = """{user_message}"""

CRITIC_PROMPT = ChatPromptTemplate.from_messages([
    ("system", CRITIC_SYSTEM_TEMPLATE),
    ("user", CRITIC_USER_TEMPLATE),
])


# ═══════════════════════════════════════════════════════════════════
# Critic 调用入口 + 输出回填 (3.2-3.4)
# ═══════════════════════════════════════════════════════════════════


def run_critic(
    dag: DAG,
    planner_hypothesis_part1: str,
    planner_hypothesis_part2: str,
    previous_dag: DAG | None,
    model,  # BaseChatModel——Phase 5 注入
    detail_level: Literal["full", "summary"] = "full",
    structured_output_method: str = "function_calling",
) -> CriticOutput:
    """执行完整 Critic 审查：组装输入 → LLM 调用 → 结构化输出。

    这是 3.1（输入组装）和 3.2-3.4（审查+终止+回填管道）之间的桥梁。
    一次 LLM 调用完成逐节点审查、结构审查和终止判断三项产出。

    Phase 5 控制循环调用此函数，获得 CriticOutput 后：
    1. 调用 apply_critic_output(dag, output) 回填 DAGNode.critic_* 字段
    2. 读取 output.termination 决定终止或进入下一轮
    3. 将 output.planner_guidance 注入下一轮 Planner 的 USER 消息

    Args:
        dag: Solver 刚更新完的当前 DAG（包含完整的 Solver 字段）
        planner_hypothesis_part1: Planner 结构假设部分 1——"本轮目标状态"
        planner_hypothesis_part2: Planner 结构假设部分 2——"变化量及理由"
        previous_dag: 上一轮 DAG（首轮为 None），用于 §5 拓扑比较
        model: LangChain BaseChatModel——Phase 5 注入，
               通过 with_structured_output(CriticOutput) 绑定输出 schema
        detail_level: "full"=当前轮(supporting全文), "summary"=历史轮(仅摘要)

    Returns:
        CriticOutput——包含逐节点审查、结构建议和终止判断
    """
    user_message = build_critic_user_message(
        dag, planner_hypothesis_part1, planner_hypothesis_part2, previous_dag,
        detail_level=detail_level,
    )
    chain = CRITIC_PROMPT | get_structured_model(model, CriticOutput, structured_output_method)
    return chain.invoke({"user_message": user_message})


def apply_critic_output(dag: DAG, output: CriticOutput) -> None:
    """将 CriticOutput.node_reviews 逐条回填到 DAGNode.critic_* 字段。

    系统完整性验证：
    - Critic 必须审查 DAG 中的每个节点（包括 ROOT）
    - 缺失节点 → ValueError（LLM 输出异常，Phase 5 负责重试或降级处理）
    - 多余节点（不在 DAG 中的 node_id）→ 忽略并发出警告

    不修改 output 本身——planner_guidance 和 termination 由 Phase 5 消费。

    Args:
        dag: 当前 DAG（将被原地修改——各节点的 critic_* 字段被填充）
        output: Critic 的完整输出

    Raises:
        ValueError: node_reviews 未覆盖 DAG 中的某些节点
    """
    reviewed_ids = {r.node_id for r in output.node_reviews}
    dag_ids = set(dag.nodes.keys())

    missing = dag_ids - reviewed_ids
    if missing:
        raise ValueError(
            f"Critic 未审查以下节点: {sorted(missing)}。"
            f"node_reviews 必须覆盖 DAG 中的每个节点（包括 ROOT）。"
            f"当前 DAG 有 {len(dag_ids)} 个节点，node_reviews 覆盖了 {len(reviewed_ids)} 个。"
        )

    extra = reviewed_ids - dag_ids
    if extra:
        import warnings
        warnings.warn(
            f"Critic 输出了不存在节点的审查: {sorted(extra)}。"
            f"这些 node_id 不在当前 DAG 中——可能是 LLM 幻觉。已忽略。"
        )

    for review in output.node_reviews:
        if review.node_id not in dag.nodes:
            continue  # 已在上面警告
        node = dag.nodes[review.node_id]
        node.critic_health = review.critic_health
        node.critic_factual_notes = review.critic_factual_notes
        node.critic_normative_advice = review.critic_normative_advice