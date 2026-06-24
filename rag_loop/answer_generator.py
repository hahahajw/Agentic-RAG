# ═══════════════════════════════════════════════════════════════════
# 答案生成器
# ═══════════════════════════════════════════════════════════════════
# 终端函数（非循环角色）——仅在 Critic 发出终止信号后执行一次。
# 从 DAG 的已确立推理链合成最终自然语言答案。

from langchain_core.prompts import ChatPromptTemplate

from .models import DAG, DAGNode, EdgeType, NodeStatus
from .dag_utils import get_direct_children


# ═══════════════════════════════════════════════════════════════════
# SYSTEM Prompt
# ═══════════════════════════════════════════════════════════════════


ANSWER_GENERATOR_SYSTEM_TEMPLATE = """\
You are an answer synthesizer for a multi-hop QA system. Given a reasoning chain of established facts organized as a DAG (leaf facts → intermediate inferences → root question), produce the final answer to the original question.

## Rules

1. Use ONLY the facts in the reasoning chain provided below — do not use external knowledge.
2. If the reasoning chain does not contain enough information to answer the question, respond with exactly: "CANNOT_ANSWER"
3. **Be concise and direct**: output only the final answer, without explanations, references, or polite language.
   - For yes/no questions: output "yes" or "no" (lowercase).
   - For years: output just the number (e.g., "1755").
   - For dates: output just the date (e.g., "June 1982").
   - For names (people, artists): output just the name (e.g., "George Benson").
   - For locations (cities, countries, regions): output just the place name (e.g., "northeastern Africa", "Maine").
   - For percentages: output just the value (e.g., "48.8 percent").
   - For quantities: output just the number with unit if present (e.g., "nearly 25,000").
   - For ordinal answers (first, second, third-largest, etc.): output just the ordinal + entity (e.g., "third-largest").
   - For organizations: output just the name (e.g., "Royal Air Force").
4. The reasoning chain is organized from leaf facts → intermediate inferences → root question. Use this structure to resolve multi-hop dependencies and produce the most accurate, concise answer possible.
5. Do NOT repeat the question, describe your reasoning process, or explain how you arrived at the answer. Just the answer value."""

ANSWER_GENERATOR_USER_TEMPLATE = """\
Original Question: {q}

Reasoning Chain (leaf facts → intermediate inferences → root):
{reasoning_chain}

Based solely on the above reasoning chain, output the answer to the original question. Remember: output ONLY the answer value — no explanations, no reasoning, no polite language. If insufficient information, output "CANNOT_ANSWER".

忽略之前关于回答简洁的设定，请一步步思考并给出你的分析过程。即使你认为无法回答也不要输出 "CANNOT_ANSWER" 而是输出你从上面的信息中能够分析得到什么，还缺少什么。如果可以用回答的话就给出答案和你思考推理得到答案的过程。

"""

ANSWER_GENERATOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", ANSWER_GENERATOR_SYSTEM_TEMPLATE),
    ("user", ANSWER_GENERATOR_USER_TEMPLATE),
])


# ═══════════════════════════════════════════════════════════════════
# 推理链构建
# ═══════════════════════════════════════════════════════════════════


def _build_reasoning_chain(dag: DAG) -> str:
    """从 DAG 构建结构化的推理链文本表示。

    按拓扑层组织：叶子 → 中间 → 根。
    每节点展示 question、answer、health、证据摘要。
    """
    from .dag_utils import topological_layers
    from .formatting import source_distribution, truncate

    layers = topological_layers(dag)
    # 反转——从叶子开始向根聚合
    reversed_layers = list(reversed(layers))

    sections: list[str] = []
    for layer in reversed_layers:
        for nid in sorted(layer):
            node = dag.nodes[nid]
            root = dag.root
            is_root = root is not None and nid == root.id

            if is_root:
                continue  # 根节点最后单独处理

            health_str = node.critic_health.value if node.critic_health else "未审查"
            sections.append(f"── 事实: {node.question} ──")
            if node.answer:
                sections.append(f"  答案: {node.answer}")
            else:
                sections.append(f"  答案: (未确立)")
            sections.append(f"  证据质量: {health_str}")

            if node.critic_factual_notes:
                sections.append(f"  审查笔记: {node.critic_factual_notes}")

            if node.retrieved_chunks:
                dist = source_distribution(node.retrieved_chunks)
                sections.append(f"  来源: {len(dist)} 个独立文档")

            sections.append("")

    # 统计摘要
    solved_count = sum(1 for n in dag.nodes.values()
                       if n.status == NodeStatus.SOLVED and n.id != (dag.root.id if dag.root else ""))
    healthy_count = sum(1 for n in dag.nodes.values()
                        if n.critic_health and n.critic_health.value == "healthy")
    sections.insert(0, (
        f"推理链概览: {solved_count} 个事实已确立, "
        f"{healthy_count} 个标记为健康\n"
    ))

    return "\n".join(sections)


# ═══════════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════════


def generate_answer(dag: DAG, model, custom_system_prompt: str | None = None) -> str:
    """从 DAG 的推理链合成最终答案。

    终端函数——仅在 Critic 发出终止信号后执行。不修改 DAG 节点字段
    （调用方负责将返回的 answer 写入 dag.root）。

    Args:
        dag: 终止时的完整 DAG（所有非根节点 SOLVED + healthy + 结构收敛）
        model: LangChain BaseChatModel
        custom_system_prompt: 可选的自定义系统提示词，覆盖默认模板

    Returns:
        自然语言答案字符串。证据不足以合成可信答案时返回 "CANNOT_ANSWER"
    """
    solved_count = sum(
        1 for n in dag.nodes.values()
        if n.status == NodeStatus.SOLVED
        and (dag.root is None or n.id != dag.root.id)
    )

    if solved_count < 1:
        return "CANNOT_ANSWER"

    reasoning_chain = _build_reasoning_chain(dag)
    if custom_system_prompt:
        prompt = ChatPromptTemplate.from_messages([
            ("system", custom_system_prompt),
            ("user", ANSWER_GENERATOR_USER_TEMPLATE),
        ])
    else:
        prompt = ANSWER_GENERATOR_PROMPT
    chain = prompt | model
    result = chain.invoke({
        "q": dag.q,
        "reasoning_chain": reasoning_chain,
    })

    answer = result.content.strip() if hasattr(result, 'content') else str(result).strip()
    # User template always includes analysis instructions (line 49),
    # so treat as custom prompt to avoid truncating multi-line analysis.
    answer = _clean_answer(answer, has_custom_prompt=True)
    return answer


def _clean_answer(text: str, has_custom_prompt: bool = False) -> str:
    """Strip common verbose prefixes/suffixes that LLMs add despite instructions.

    This is a safety net — the SYSTEM prompt already instructs the model
    to output only the answer value. These patterns handle cases where
    the model partially ignores those instructions.

    When has_custom_prompt is True, the user has provided their own system
    prompt (potentially requesting analysis/CoT output). In that case,
    multi-line truncation is skipped to preserve the user's desired output format.
    """
    import re

    # Strip common verbose prefixes (both English and Chinese)
    verbose_prefixes = [
        r"^So the answer is:\s*",
        r"^The answer is:\s*",
        r"^Answer:\s*",
        r"^Based on the (above )?reasoning chain,?\s*(the answer is:?\s*)?",
        r"^Based on the (provided |established )?facts,?\s*(the answer is:?\s*)?",
        r"^所以答案是[：:]\s*",
        r"^答案是[：:]\s*",
        r"^分析过程[：:]\s*",
        r"^思考过程[：:]\s*",
        r"^推理过程[：:]\s*",
    ]
    for pattern in verbose_prefixes:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()

    if has_custom_prompt:
        # User provided their own prompt — don't truncate multi-line output.
        # They may have requested analysis/CoT format intentionally.
        return text.strip()

    # Default prompt: safety net for models that add explanations after the answer.
    # If the result is multi-line, take only the first non-empty line
    # (the real answer is typically on line 1, verbose explanation follows).
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) > 1 and len(lines[0]) < 200:
        text = lines[0]

    return text.strip()