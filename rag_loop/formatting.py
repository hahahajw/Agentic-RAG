# ═══════════════════════════════════════════════════════════════════
# 格式化工具
# ═══════════════════════════════════════════════════════════════════
# 消费者: Planner (planner.py)、Critic (critic.py)、Pipeline (pipeline.py)
# 从 planner.py 和 critic.py 提取的纯函数——不修改状态，不调用 LLM。

from .models import DAG, DAGNode, ChunkInfo, EdgeType, NodeStatus


# ═══════════════════════════════════════════════════════════════════
# 文本截断与统计
# ═══════════════════════════════════════════════════════════════════


def truncate(text: str, max_len: int) -> str:
    """截断长文本用于紧凑展示。"""
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."


def source_distribution(chunks: list[ChunkInfo]) -> dict[str, int]:
    """统计各 context_title 的 chunk 数量。

    两个 chunk 来自同一 context_title → 同一文档的不同段落 → 非独立来源。
    Critic 独立性检查的核心依据。
    """
    counts: dict[str, int] = {}
    for chunk in chunks:
        counts[chunk.context_title] = counts.get(chunk.context_title, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: -x[1]))


# ═══════════════════════════════════════════════════════════════════
# 节点标记
# ═══════════════════════════════════════════════════════════════════


def node_label(dag: DAG, nid: str) -> str:
    """节点的角色标记——ROOT / BLOCKED / 空字符串。

    ROOT: 根节点（入度为 0 的唯一节点，I4）
    BLOCKED: UNSOLVED + 未执行搜索 + 存在未满足的 dependency 源
    """
    root = dag.root
    if root and nid == root.id:
        return "ROOT"
    node = dag.nodes[nid]
    if (
        node.status == NodeStatus.UNSOLVED
        and not node.search_query
    ):
        for edge in dag.edges:
            if edge.to_id == nid and edge.edge_type == EdgeType.DEPENDENCY:
                src = dag.nodes.get(edge.from_id)
                if src is None:
                    continue
                if src.status != NodeStatus.SOLVED or not src.answer:
                    return "BLOCKED"
    return ""


# ═══════════════════════════════════════════════════════════════════
# Chunk 格式化——两种粒度
# ═══════════════════════════════════════════════════════════════════


def format_chunk_summary(chunk: ChunkInfo) -> str:
    """摘要级展示——chunk_id + context_title + chunk_title + chunk_summary。

    不含 page_content 全文。Planner 结构决策和 Critic 历史退化使用此级别。
    """
    return (
        f"[{chunk.chunk_id}] | 来源: {chunk.context_title} | "
        f"标题: {chunk.chunk_title} | 摘要: {chunk.chunk_summary}"
    )


def format_chunk_summary_compact(chunk: ChunkInfo) -> str:
    """紧凑摘要——chunk_id + context_title + chunk_summary（不含标题）。

    用于 non-supporting chunk 和空间敏感场景。
    """
    return (
        f"[{chunk.chunk_id}] | 来源: {chunk.context_title} | "
        f"摘要: {chunk.chunk_summary}"
    )


def format_chunk_full(chunk: ChunkInfo) -> str:
    """全文级展示——chunk 的完整内容，供逐字核实。

    Critic 当前轮和 Planner 当前轮（对称设计）使用此级别。
    包含 page_content 全文——Critic 逐字核实 Solver 声称所需，
    Planner 独立验证 Critic 判断所需。
    """
    return (
        f"[{chunk.chunk_id}] | 来源: {chunk.context_title} | "
        f"标题: {chunk.chunk_title}\n"
        f"摘要: {chunk.chunk_summary}\n"
        f"全文: {chunk.page_content}"
    )


# ═══════════════════════════════════════════════════════════════════
# Chunk 列表格式化——区分 supporting / non-supporting
# ═══════════════════════════════════════════════════════════════════


def format_supporting_chunks(
    node: DAGNode,
    detail_level: str,  # "full" | "summary"
) -> list[str]:
    """格式化节点的 supporting chunks 展示行列表。

    detail_level="full"  → format_chunk_full (page_content 全文)
    detail_level="summary" → format_chunk_summary (仅摘要)
    """
    supporting_ids = set(node.supporting_chunks)
    supporting = [c for c in node.retrieved_chunks if c.chunk_id in supporting_ids]

    if not supporting:
        return []

    lines = [f"  ▸ 支撑 chunk ({len(supporting)} 个):"]
    for chunk in supporting:
        if detail_level == "full":
            lines.append(f"    {format_chunk_full(chunk)}")
        else:
            lines.append(f"    {format_chunk_summary(chunk)}")
    return lines


def format_non_supporting_chunks(node: DAGNode) -> list[str]:
    """格式化节点的 non-supporting chunks 展示行列表。

    始终使用紧凑摘要——非支撑 chunk 只需判断"是否遗漏了答案"，
    不需要逐字核实。
    """
    supporting_ids = set(node.supporting_chunks)
    non_supporting = [c for c in node.retrieved_chunks if c.chunk_id not in supporting_ids]

    if not non_supporting:
        return []

    lines = [f"  ▸ 其他检索结果 ({len(non_supporting)} 个):"]
    for chunk in non_supporting:
        lines.append(f"    {format_chunk_summary_compact(chunk)}")
    return lines


def format_source_summary(chunks: list[ChunkInfo]) -> str:
    """格式化来源分布摘要——独立来源数 + 各文档分布。

    例: "来源分布: docA (3), docB (2) = 2 个独立来源"
    """
    if not chunks:
        return ""
    counts = source_distribution(chunks)
    dist_str = ", ".join(f"{src} ({cnt})" for src, cnt in counts.items())
    independent = len(counts)
    return f"  ▸ 来源分布: {dist_str}  = {independent} 个独立来源"


# ═══════════════════════════════════════════════════════════════════
# 阻塞信息
# ═══════════════════════════════════════════════════════════════════


def format_blocking_info(dag: DAG, nid: str) -> str:
    """格式化节点的阻塞原因——哪些 dependency 源导致了硬阻塞。

    返回空字符串表示此节点未被阻塞。
    """
    blocking_sources: list[str] = []
    for edge in dag.edges:
        if edge.to_id == nid and edge.edge_type == EdgeType.DEPENDENCY:
            src = dag.nodes.get(edge.from_id)
            if src is None:
                continue
            if src.status != NodeStatus.SOLVED or not src.answer:
                blocking_sources.append(
                    f"{edge.from_id}(status={src.status.value}, "
                    f"answer={truncate(src.answer, 30) or '(空)'})"
                )

    if not blocking_sources:
        return ""

    return (
        f"阻塞原因: 依赖源未满足——{', '.join(blocking_sources)}。"
        f"此节点未执行搜索（query 含未解析指代）"
    )