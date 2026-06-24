# ═══════════════════════════════════════════════════════════════════
# Solver 组件
# ═══════════════════════════════════════════════════════════════════
# 节点准备(推理+查询构造) + 搜索调度(波次并行) + LLM 提取 + 字段回填。
# Solver 是算法接触真实世界的唯一窗口。

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from .models import DAG, DAGNode, ChunkInfo, EdgeType, NodeStatus
from .interfaces import SearchFn
from .dag_utils import get_dependency_answers, get_child_answers
from .structured_output import get_structured_model


# ═══════════════════════════════════════════════════════════════════
# Pydantic 输出模型 (2.1 + 2.3 共用)
# ═══════════════════════════════════════════════════════════════════


class NodePreparation(BaseModel):
    """prepare_node_query 的结构化输出——一次调用完成推理+查询构造。

    can_answer=True  → answer 已填入，跳过搜索
    can_answer=False → search_query 已构造，继续搜索
    """
    can_answer: bool = Field(
        description="已知事实是否足以直接回答 question。True=可直接推理，False=需要搜索"
    )
    answer: str = Field(
        default="",
        description="声称的事实值（名字/日期/数字，简洁短语）。can_answer=False 时为空字符串"
    )
    search_query: str = Field(
        default="",
        description="自然语言搜索查询，融合已知实体名。can_answer=True 时为空字符串"
    )


class ExtractionResult(BaseModel):
    """extract_from_chunks 的结构化输出——从 chunk 中提取答案。

    三个字段对应 FRAMEWORK.md 模块 2 Solver 的三项产出。
    """
    answer: str = Field(
        default="",
        description="声称的事实值（名字/日期/数字，简洁短语）。找不到时为空字符串"
    )
    supporting_chunks: list[str] = Field(
        default_factory=list,
        description="直接支撑答案的 chunk_id 列表。找不到时为空列表"
    )
    judgment: str = Field(
        default="",
        description="判断理由——必须引用具体 chunk 内容"
    )


# ═══════════════════════════════════════════════════════════════════
# 节点准备 (2.1) —— ChatPromptTemplate + with_structured_output
# ═══════════════════════════════════════════════════════════════════


PREPARE_SYSTEM_TEMPLATE = """\
你是精确的信息处理器。根据已消解的实体和已知背景信息，判断能否直接回答问题，如不能则构造搜索查询。

## 输入说明
- 问题（question）：需要回答的目标问题。其中的指代词已被消解——如果原问题含 "that"、"this" 等词，dependency_facts 中已提供了对应实体
- 依赖事实（dependency_facts）：来自前置节点的答案，即问题中指代词对应的实体值。这些节点均已被求解，其实体值确定。例：question="In what year was that founded?" → dependency_fact="Moscow State University"
- 子事实（child_facts）：来自子节点的答案，作为回答问题的背景信息。可能为空——表示没有子节点，或子节点尚未解出答案

## 输出要求
can_answer（布尔值）：
- true = 已知事实中直接包含问题的答案 → 填写 answer，search_query 留空
- false = 已知事实不足以回答 → 填写 search_query，answer 留空

answer（字符串，仅 can_answer=true 时填写）：
- 简洁的声称值：名字、日期、数字、地名
- 例："1755"、"Megan Thee Stallion"、"2020"
- 仅基于提供的已知事实，不使用外部知识

search_query（字符串，仅 can_answer=false 时填写）：
- 将 dependency_facts 中的实体名自然融入问题，消除任何残留歧义
  例：question="In what year was that founded?" + dep="Moscow State University"
  → "Moscow State University founding year"
- 如果仅有 child_facts 而无 dependency_facts，以 question 为基，融入子事实中的关键实体
- 使用自然语言，非关键词堆砌

Output as JSON."""

PREPARE_SYSTEM_TEMPLATE_WEB = """\
你是精确的信息处理器——为网络搜索引擎（DuckDuckGo）构造查询。

搜索后端是关键词匹配的网络搜索引擎，不是语义向量数据库。你需要生成适合搜索引擎的查询。

## 输入说明
- 问题（question）：需要回答的目标问题
- 依赖事实（dependency_facts）：来自前置节点的答案
- 子事实（child_facts）：来自子节点的答案，作为背景信息

## 输出要求
can_answer（布尔值）：
- true = 已知事实中直接包含问题的答案 → 填写 answer，search_query 留空
- false = 已知事实不足以回答 → 填写 search_query，answer 留空

answer（字符串，仅 can_answer=true 时填写）：
- 简洁的声称值：名字、日期、数字、地名
- 仅基于提供的已知事实，不使用外部知识

search_query（字符串，仅 can_answer=false 时填写）：
- **关键：这是网络搜索引擎查询，不是自然语言问题**
- 简洁扼要（5-15 词），关键词优先，而非完整自然语言问句
- 将核心实体（标准号、材料名称、技术术语）放在查询开头
- 避免使用问句格式（如"什么是..."、"请问..."），改用关键词组合
- 避免嵌套从句、括号注释、学术化措辞
- 将 dependency_facts 中的实体名融入查询
  例：question="成品分析碳含量允许偏差是多少？" + dep="GB/T 700"
  → "GB/T 700 成品分析 碳含量 允许偏差"
  例：question="In what year was that founded?" + dep="Moscow State University"
  → "Moscow State University founded year"

Output as JSON."""

PREPARE_USER_TEMPLATE = """\
问题: {question}

依赖事实（已消解的实体）:
{dependency_facts}

子事实（背景信息）:
{child_facts}

请判断能否直接回答，或构造搜索查询。"""

PREPARE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", PREPARE_SYSTEM_TEMPLATE),
    ("user", PREPARE_USER_TEMPLATE),
])

PREPARE_PROMPT_WEB = ChatPromptTemplate.from_messages([
    ("system", PREPARE_SYSTEM_TEMPLATE_WEB),
    ("user", PREPARE_USER_TEMPLATE),
])


def _format_facts(answers: list[str]) -> str:
    """将 answer 列表格式化为编号列表。空列表时返回 '(无)'。"""
    if not answers:
        return "(无)"
    return "\n".join(f"- {a}" for a in answers)


def prepare_node_query(
    question: str,
    dependency_answers: list[str],
    child_answers: list[str],
    model,  # BaseChatModel——Phase 5 注入
    structured_output_method: str = "function_calling",
    custom_prompt=None,  # 可选的自定义 ChatPromptTemplate（如 PREPARE_PROMPT_WEB）
) -> NodePreparation:
    """一次 LLM 调用：尝试从已知事实推理，或构造搜索查询。

    无上下文（dependency_answers 和 child_answers 均为空）时：
    跳过 LLM 调用，直接返回 can_answer=False, search_query=question。

    当 custom_prompt 提供时（如网络搜索场景），使用自定义提示词模板；
    否则使用默认的 PREPARE_PROMPT（面向 Milvus 语义搜索）。
    """
    if not dependency_answers and not child_answers:
        return NodePreparation(
            can_answer=False,
            answer="",
            search_query=question,
        )

    prompt = custom_prompt if custom_prompt is not None else PREPARE_PROMPT
    chain = prompt | get_structured_model(model, NodePreparation, structured_output_method)
    return chain.invoke({
        "question": question,
        "dependency_facts": _format_facts(dependency_answers),
        "child_facts": _format_facts(child_answers),
    })


# ═══════════════════════════════════════════════════════════════════
# 搜索调度 (2.2)
# ═══════════════════════════════════════════════════════════════════


def find_ready_nodes(
    dag: DAG,
    root_id: str,
    attempted: set[str],
) -> list[str]:
    """找出本轮的就绪节点——仅依赖 dependency 硬约束。

    就绪条件:
    1. 非根节点（根节点跳过 Solver 搜索）
    2. status = UNSOLVED（已解节点不重复搜索）
    3. 未在 attempted 中（本轮已求解的跳过）
    4. 所有 dependency 边源节点满足: status = SOLVED AND answer ≠ ""
    """
    ready: list[str] = []

    for nid, node in dag.nodes.items():
        if nid == root_id:
            continue
        if node.status != NodeStatus.UNSOLVED:
            continue
        if nid in attempted:
            continue
        if not _all_deps_solved(dag, nid):
            continue
        ready.append(nid)

    return ready


def _all_deps_solved(dag: DAG, node_id: str) -> bool:
    """检查 node_id 的所有 dependency 源是否均已 SOLVED 且 answer 非空。

    dependency 边的语义（FRAMEWORK.md 模块 2）:
    求解目标节点时需要参考源节点的 answer 来消解指代。
    源节点未 SOLVED 或其 answer 为空 → 无法消解 → query 含未解析代词
    → 不应搜索。
    """
    for edge in dag.edges:
        if edge.to_id == node_id and edge.edge_type == EdgeType.DEPENDENCY:
            src = dag.nodes.get(edge.from_id)
            if src is None:
                continue
            if src.status != NodeStatus.SOLVED or not src.answer:
                return False
    return True


def solve_dag(dag: DAG, search_fn: SearchFn, model,
              rewrite_model=None, solver_model=None,
              rewrite_method: str = "function_calling",
              solver_method: str = "function_calling",
              custom_prepare_prompt=None) -> int:
    """Solver 阶段主入口——波次求解，推理优先，返回搜索调用次数。

    每节点三步: prepare_node_query(推理+查询构造) → search(如需) → extract_from_chunks
    波内节点可并行（Phase 5 实现），波间串行——前一波 answer 解锁后续波次。
    根节点始终跳过。

    Args:
        rewrite_model: 用于 prepare_node_query，None 则回退到 model
        solver_model: 用于 extract_from_chunks，None 则回退到 model
        custom_prepare_prompt: 可选的自定义 ChatPromptTemplate（如 PREPARE_PROMPT_WEB）
    """
    root = dag.root
    if root is None:
        return 0

    _rm = rewrite_model or model
    _sm = solver_model or model

    root_id = root.id
    attempted: set[str] = set()
    search_count = 0

    while True:
        ready = find_ready_nodes(dag, root_id, attempted)
        if not ready:
            break

        for nid in ready:
            node = dag.nodes[nid]

            dep_answers = get_dependency_answers(dag, nid)
            child_answers = get_child_answers(dag, nid)

            # Step 1: 节点准备——尝试推理 OR 构造搜索查询（一次 LLM 调用）
            prep = prepare_node_query(node.question, dep_answers, child_answers, _rm,
                                       structured_output_method=rewrite_method,
                                       custom_prompt=custom_prepare_prompt)

            if prep.can_answer:
                node.answer = prep.answer
                node.search_query = ""
                node.status = NodeStatus.SOLVED
                node.solver_judgment = "从已知事实推理得出，未执行搜索"
                node.supporting_chunks = []
                node.round_last_updated = dag.round_number
                attempted.add(nid)
                continue

            # Step 2: 搜索
            node.search_query = prep.search_query
            node.retrieved_chunks = search_fn(prep.search_query)
            search_count += 1

            # Step 3: LLM 提取
            extract_from_chunks(node, _sm, structured_output_method=solver_method)

            node.round_last_updated = dag.round_number
            attempted.add(nid)

    return search_count


# ═══════════════════════════════════════════════════════════════════
# LLM 提取 (2.3) —— ChatPromptTemplate + with_structured_output
# ═══════════════════════════════════════════════════════════════════


EXTRACT_SYSTEM_TEMPLATE = """\
你是精确的信息提取器。你的首要任务是从检索结果中提取问题的答案。你的次要任务是在判断中评估检索质量。

## 输入说明
- 问题（question）：需要回答的目标问题。这是你提取答案的唯一锚点——你必须回答这个问题，而非其他问题
- 搜索查询（search_query）：实际用于检索的查询字符串。仅供你在写判断（judgment）时参考——用来评估检索是否命中了正确信息。不要用它替代 question 作为提取目标
- 检索结果（chunks）：搜索返回的文本片段。每个有唯一 chunk_id

## 提取答案（以 question 为目标）
answer（字符串）：
- 如果 chunks 中包含 question 的答案 → 简洁的声称值（名字、日期、数字、地名）
- 如果 chunks 中不包含 → 空字符串 ""
- 严格依据 chunk 内容，不使用外部知识

supporting_chunks（字符串列表）：
- 直接支撑 answer 的 chunk_id 列表
- 仅包含实际依据的 chunk
- answer 为空时 → 空列表 []

## 判断检索质量（仅用于 judgment）
judgment（字符串）：
- 第一部分：提取结果。找到了什么、从哪个 chunk 找到的、引用原文摘录
- 第二部分：检索质量评估。对照 question 需要什么信息、search_query 搜索了什么、chunks 实际返回了什么——这三者之间是否存在缺口
  例："question 要求确认'that year'指代的具体年份的获奖者。search_query 搜索了'2020 Grammy Best New Artist winner'。chunks 返回了 2020 年 Grammy 获奖名单，其中 chunk_X 记载'Megan Thee Stallion won Best New Artist'，直接回答了 question。检索精准，信息完整。"
  例："question 要求 Moscow State University 的成立年份。search_query 搜索了'Moscow State University founding year'。但 chunks 返回的是 University of Moscow（不同机构）和 Moscow city（不同实体类型）的信息——检索未能区分 Moscow State University 与其他 Moscow 相关实体。chunks 中无目标信息，无法提取答案。"
- 不要只说"找不到"——说明 question 需要什么、search_query 搜了什么、chunks 给了什么、缺口在哪

Output as JSON."""

EXTRACT_USER_TEMPLATE = """\
问题: {question}
（搜索时使用的查询: {search_query}——供判断检索质量时参考）

{chunks_text}

从上述检索结果中提取能回答「{question}」的信息。"""

EXTRACT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", EXTRACT_SYSTEM_TEMPLATE),
    ("user", EXTRACT_USER_TEMPLATE),
])


def _format_chunks(chunks: list[ChunkInfo]) -> str:
    """将 ChunkInfo 列表格式化为 LLM prompt 文本。

    每 chunk 显示 chunk_id、标题、来源文档、摘要、全文内容。
    """
    if not chunks:
        return "(无检索结果)"

    parts: list[str] = []
    for chunk in chunks:
        parts.append(f"[CHUNK: {chunk.chunk_id}]")
        parts.append(f"标题: {chunk.chunk_title}")
        parts.append(f"来源文档: {chunk.context_title}")
        parts.append(f"摘要: {chunk.chunk_summary}")
        parts.append(f"内容: {chunk.page_content}")
        parts.append("")

    return "\n".join(parts)


def extract_from_chunks(
    node: DAGNode,
    model,  # BaseChatModel——Phase 5 注入
    structured_output_method: str = "function_calling",
) -> None:
    """从 node.retrieved_chunks 中提取答案并回填 Solver 字段。

    一次 LLM 调用，with_structured_output(ExtractionResult) 保证类型安全输出。
    prompt 包含 question + search_query + chunks——search_query 标注为搜索参考，
    LLM 以 question 为提取锚点，用 search_query 辅助判断检索质量。

    Side effects: 修改 node.answer, node.supporting_chunks, node.solver_judgment, node.status
    """
    chunks_text = _format_chunks(node.retrieved_chunks)
    search_query = node.search_query or "(未记录)"

    chain = EXTRACT_PROMPT | get_structured_model(model, ExtractionResult, structured_output_method)
    result: ExtractionResult = chain.invoke({
        "question": node.question,
        "search_query": search_query,
        "chunks_text": chunks_text,
    })

    _validate_and_fill(node, result)


def _validate_and_fill(node: DAGNode, result: ExtractionResult) -> None:
    """验证 ExtractionResult 并回填 node。

    验证项目:
    - answer: trim 空白，非空 → SOLVED，空 → 保持 UNSOLVED
    - supporting_chunks: 滤除不存在于 retrieved_chunks 的幻觉 ID
    - judgment: 空 → 填默认值
    """
    node.answer = result.answer.strip()

    valid_ids = {c.chunk_id for c in node.retrieved_chunks}
    node.supporting_chunks = [
        cid for cid in result.supporting_chunks
        if isinstance(cid, str) and cid in valid_ids
    ]

    if result.judgment.strip():
        node.solver_judgment = result.judgment.strip()
    else:
        node.solver_judgment = "LLM 未提供判断理由"

    if node.answer:
        node.status = NodeStatus.SOLVED