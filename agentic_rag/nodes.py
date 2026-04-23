"""Agentic RAG — 节点函数与入口

节点函数（供 LangGraph StateGraph 调用）：
- plan_node: 生成探索计划（假设 + 目标 + 优先级）
- execute_node: 执行检索（重写查询 + 并行检索 + RRF 融合）
- evaluate_node: 评估当前轮次进度
- reflect_node: 当连续 stuck 时反思并生成 pivot 策略
- generate_answer_node: 基于完整探索历史生成最终答案

入口：
- run_agentic_rag(): 调用编译好的 StateGraph，执行闭环探索
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from agentic_rag.state import AgenticRAGState
from agentic_rag.prompts import (
    get_planner_prompt,
    get_evaluator_prompt,
    get_reflector_prompt,
    get_answer_prompt,
)

logger = logging.getLogger(__name__)


# ─── 辅助函数：从 config 获取 LLM 实例 ─────────────────────────

def _get_planner_llm(config: RunnableConfig):
    return config["configurable"].get("planner_llm") or _get_llm(config)


def _get_evaluator_llm(config: RunnableConfig):
    return config["configurable"].get("evaluator_llm") or _get_llm(config)


def _get_reflector_llm(config: RunnableConfig):
    return config["configurable"].get("reflector_llm") or _get_llm(config)


def _get_rewrite_llm(config: RunnableConfig):
    return config["configurable"].get("rewrite_llm") or _get_llm(config)


def _get_answer_llm(config: RunnableConfig):
    return config["configurable"].get("answer_llm") or _get_llm(config)


def _get_llm(config: RunnableConfig):
    return config["configurable"]["llm"]


def _get_retriever(config: RunnableConfig):
    return config["configurable"]["retriever"]


def _get_max_chunks(config: RunnableConfig) -> int:
    return config["configurable"].get("max_chunks", 8)


def _get_max_rounds(config: RunnableConfig) -> int:
    return config["configurable"].get("max_rounds", 5)


# ─── LangGraph 节点函数 ────────────────────────────────────────

def plan_node(state: AgenticRAGState, config: RunnableConfig) -> dict:
    """节点：生成探索计划。

    综合当前问题、探索历史、上一轮反馈，制定检索策略。
    如果上一轮 Evaluator 是 "stuck" 且 Reflector 已执行，
    则 Reflector 的 pivot_queries 会覆盖 Planner 的 targets。

    返回: {"plan": {"hypotheses": [...], "targets": [...], "priorities": [...]}}
    """
    query = state["query"]
    history = state.get("exploration_history", [])
    llm = _get_planner_llm(config)
    round_num = len(history) + 1

    # 构建历史上下文
    history_context = _format_history_for_planner(history)
    feedback_instruction = ""
    if history:
        last_eval = history[-1].get("evaluation", {})
        if last_eval.get("status") == "stuck":
            feedback_instruction = (
                "\n\n**Tip**: The previous round were stuck and did not yield useful information. "
                "Try using different terminology or exploring related but different angles."
            )
        elif last_eval.get("status") == "progressing":
            gaps = last_eval.get("knowledge_gaps", [])
            if gaps:
                gaps_text = "; ".join(gaps[:3])
                feedback_instruction = (
                    f"\n\nFocus on these knowledge gaps: {gaps_text}"
                )

    prompt = get_planner_prompt(
        question=query,
        round_num=round_num,
        history_context=history_context,
        feedback_instruction=feedback_instruction,
    )

    response = llm.invoke([prompt])
    content = response.content.strip()

    # 解析 JSON
    plan = _parse_json_response(content)
    assert "hypotheses" in plan and "targets" in plan, f"Invalid plan: {plan}"

    # 确保 targets 是列表
    if isinstance(plan.get("targets"), str):
        plan["targets"] = [plan["targets"]]

    logger.info(
        "[round=%d] 探索计划生成: hypotheses=%d, targets=%d",
        round_num, len(plan["hypotheses"]), len(plan["targets"])
    )
    return {"plan": plan}


def execute_node(state: AgenticRAGState, config: RunnableConfig) -> dict:
    """节点：执行检索。

    对每个检索 query：
    1. 调用 retriever.get_similar_chunks_with_rewrite() 完成 重写→检索→RRF 融合
    2. 按优先级分配配额，跨组去重合并

    返回: {"rewritten_queries": [...], "chunks": [...]}
    """
    query = state["query"]
    plan = state.get("plan", {})
    retriever = _get_retriever(config)
    max_chunks = _get_max_chunks(config)
    llm = _get_rewrite_llm(config)

    targets = plan.get("targets", [])
    priorities = plan.get("priorities", [])

    # 原始问题加入检索列表，赋予最高优先级
    if targets:
        all_queries = [query] + targets
        all_priorities = [1] + priorities
    else:
        all_queries = [query]
        all_priorities = [1]

    # 去重（保持顺序）
    seen = set()
    unique_queries = []
    unique_priorities = []
    for q, p in zip(all_queries, all_priorities):
        if q not in seen:
            seen.add(q)
            unique_queries.append(q)
            unique_priorities.append(p)

    n_queries = len(unique_queries)

    # 按优先级分配检索配额
    if n_queries == 1:
        quotas = [max_chunks]
    else:
        weights = [1.0 / max(p, 1) for p in unique_priorities]
        total_weight = sum(weights)
        raw_slots = [w / total_weight * max_chunks for w in weights]
        quotas = [max(1, int(s)) for s in raw_slots]
        # 补齐：确保总数 = max_chunks
        allocated = sum(quotas)
        remainder = max_chunks - allocated
        if remainder > 0:
            fractions = sorted(
                range(n_queries),
                key=lambda i: raw_slots[i] - quotas[i],
                reverse=True,
            )
            for i in fractions[:remainder]:
                quotas[i] += 1

    # 并行检索（每个 query 内部完成 rewrite + 独立检索 + RRF 融合）
    with ThreadPoolExecutor(max_workers=n_queries) as executor:
        futures = {
            executor.submit(retriever.get_similar_chunks_with_rewrite, q, llm): idx
            for idx, q in enumerate(unique_queries)
        }
        results_by_idx: dict[int, list] = {}
        for f in as_completed(futures):
            results_by_idx[futures[f]] = f.result()

    # 按 query 顺序合并（高优先级优先），去重 chunk_id
    seen_ids = set()
    chunks_out = []
    for idx in range(n_queries):
        for doc, score in results_by_idx.get(idx, []):
            cid = doc.metadata.get("chunk_id")
            if not cid or cid in seen_ids:
                continue
            seen_ids.add(cid)
            chunks_out.append({
                "chunk_id": cid,
                "chunk_title": doc.metadata.get("chunk_title", ""),
                "chunk_summary": doc.metadata.get("chunk_summary", ""),
                "context_title": doc.metadata.get("context_title", ""),
                "page_content": doc.page_content,
                "score": round(score, 6),
                "source_query": unique_queries[idx],
            })
            if len(chunks_out) >= max_chunks:
                break
        if len(chunks_out) >= max_chunks:
            break

    logger.info(
        "检索完成，query 数=%d，配额=%s，合并后 chunks=%d（上限 %d）",
        n_queries, quotas, len(chunks_out), max_chunks
    )
    return {"rewritten_queries": unique_queries, "chunks": chunks_out}


def evaluate_node(state: AgenticRAGState, config: RunnableConfig) -> dict:
    """节点：评估当前轮次进度。

    综合当前检索结果和历史探索，判断：
    - status: "answered" / "progressing" / "stuck"
    - 如果 answered，直接提取答案
    - 如果 stuck，提供反馈
    - 如果 progressing，指出知识缺口

    返回: {"evaluation": {"status", "confidence", "answer", "feedback", "knowledge_gaps", "suggested_actions"}}
    """
    from pydantic import BaseModel, Field
    from typing import List

    query = state["query"]
    chunks = state.get("chunks", [])
    history = state.get("exploration_history", [])
    llm = _get_evaluator_llm(config)
    round_num = len(history) + 1

    # 格式化 chunks
    chunks_text = _format_chunks_for_evaluator(chunks)

    # 构建历史上下文
    history_context = ""
    if history:
        parts = []
        for h in history[-3:]:  # 最近 3 轮
            ev = h.get("evaluation", {})
            parts.append(
                f"  Round {h['round']}: status={ev.get('status', '?')}, "
                f"confidence={ev.get('confidence', 0):.2f}, "
                f"feedback={ev.get('feedback', '')[:200]}"
            )
        history_context = "### Previous Rounds\n" + "\n".join(parts)

    prompt = get_evaluator_prompt(
        question=query,
        chunks_text=chunks_text,
        round_num=round_num,
        history_context=history_context,
    )

    # 结构化输出
    class EvaluationResult(BaseModel):
        status: str = Field(description="One of: answered, progressing, stuck")
        confidence: float = Field(description="Confidence 0.0-1.0")
        answer: str = Field(description="If answered: ONLY the direct answer (e.g., 'yes', 'no', '1755', 'Paris'). No explanations, no sentences. If not answered: empty string.")
        feedback: str = Field(description="What worked or went wrong")
        knowledge_gaps: List[str] = Field(description="Missing facts")
        suggested_actions: List[str] = Field(description="Suggestions for next round")

    structured_llm = llm.with_structured_output(EvaluationResult)
    result = structured_llm.invoke([prompt])

    # 验证 status
    assert result.status in ("answered", "progressing", "stuck"), f"Invalid status: {result.status}"

    logger.info(
        "[round=%d] 评估结果: status=%s, confidence=%.2f, gaps=%d",
        round_num, result.status, result.confidence, len(result.knowledge_gaps)
    )

    # 构建当前轮次的完整记录
    current_round = {
        "round": round_num,
        "plan": state.get("plan", {}),
        "rewritten_queries": state.get("rewritten_queries", []),
        "chunks": chunks,
        "evaluation": {
            "status": result.status,
            "confidence": result.confidence,
            "answer": result.answer,
            "feedback": result.feedback,
            "knowledge_gaps": result.knowledge_gaps,
            "suggested_actions": result.suggested_actions,
        },
    }

    return {
        "evaluation": current_round["evaluation"],
        # 通过 reducer 追加到 exploration_history
        "exploration_history": [current_round],
    }


def reflect_node(state: AgenticRAGState, config: RunnableConfig) -> dict:
    """节点：反思失败原因并生成 pivot 策略。

    当连续 >= 2 轮 stuck 时触发（由 route_after_evaluate 控制）。
    分析失败原因，生成 pivot 检索角度。
    返回: {"plan": {"hypotheses": [...], "targets": pivot_queries, "priorities": [...]}}
    路由到 execute_node 直接使用 pivot plan，跳过 plan_node。
    """
    query = state["query"]
    history = state.get("exploration_history", [])
    llm = _get_reflector_llm(config)

    # 由 route_after_evaluate 保证只有 stuck_count >= 2 时才会路由到这里
    stuck_rounds = 0
    for h in reversed(history):
        if h.get("evaluation", {}).get("status") == "stuck":
            stuck_rounds += 1
        else:
            break

    history_text = _format_history_full(history)
    prompt = get_reflector_prompt(
        question=query,
        history_text=history_text,
        stuck_rounds=stuck_rounds,
    )

    response = llm.invoke([prompt])
    content = response.content.strip()
    reflection = _parse_json_response(content)

    # 将 pivot_queries 转换为 plan 格式
    pivot_queries = reflection.get("pivot_queries", [])
    plan = {
        "hypotheses": [reflection.get("diagnosis", "Strategy needs adjustment")],
        "targets": pivot_queries,
        "priorities": list(range(1, len(pivot_queries) + 1)),
        "reflection": reflection,  # 保留完整反思结果
    }

    logger.info(
        "Reflector 反思完成: diagnosis='%s...', pivot_queries=%d",
        reflection.get("diagnosis", "")[:50], len(pivot_queries)
    )
    return {"plan": plan}


def generate_answer_node(state: AgenticRAGState, config: RunnableConfig) -> dict:
    """节点：基于完整探索历史生成最终答案。

    返回: {"final_answer": str, "done": True}
    """
    llm = _get_answer_llm(config)
    history = state.get("exploration_history", [])
    query = state["query"]

    # 为了与 RAG with Judge 进行公平对比（控制回答路径一致），
    # 不直接使用 Evaluator 的答案，而是统一由 Answer LLM 基于完整探索历史生成。
    # Evaluator 的 answer 字段仍保留在 history 中，仅供参考。
    #
    # for h in reversed(history):
    #     ev = h.get("evaluation", {})
    #     if ev.get("status") == "answered" and ev.get("answer"):
    #         logger.info("使用 Evaluator 的直接答案")
    #         return {"final_answer": ev["answer"].strip(), "done": True}

    # 基于探索历史生成综合答案
    history_text = _format_history_for_answer(history)
    messages = get_answer_prompt(history_text, query)

    response = llm.invoke(messages)
    answer = response.content.strip()

    logger.info("综合答案生成完成 (len=%d)", len(answer))
    return {"final_answer": answer, "done": True}


def generate_answer_stream(state: AgenticRAGState, config: RunnableConfig):
    """流式版本：基于完整探索历史逐 token 生成最终答案。

    返回: generator，逐个产出 token 内容。
    """
    llm = _get_answer_llm(config)
    history = state.get("exploration_history", [])
    query = state["query"]

    history_text = _format_history_for_answer(history)
    messages = get_answer_prompt(history_text, query)

    for chunk in llm.stream(messages):
        content = chunk.content if hasattr(chunk, 'content') else str(chunk)
        if content:
            yield content


# ─── 路由函数 ──────────────────────────────────────────────────

def route_after_evaluate(state: AgenticRAGState, config: RunnableConfig) -> str:
    """条件路由：根据 Evaluator 的 status 决定下一步。"""
    evaluation = state.get("evaluation", {})
    status = evaluation.get("status", "progressing")
    history = state.get("exploration_history", [])
    max_rounds = _get_max_rounds(config)
    completed_rounds = len(history)  # evaluate_node 已通过 reducer 追加

    if status == "answered":
        return "generate_answer"

    # 检查是否达到最大轮次
    if completed_rounds >= max_rounds:
        logger.info("达到最大轮次限制 (%d)，强制结束", max_rounds)
        return "generate_answer"

    # 检查连续 stuck 轮数
    stuck_count = 0
    for h in reversed(history):
        if h.get("evaluation", {}).get("status") == "stuck":
            stuck_count += 1
        else:
            break

    if stuck_count >= 2:
        return "reflect"

    return "plan"


def route_after_reflect(state: AgenticRAGState, config: RunnableConfig) -> str:
    """Reflector 之后直接 execute，使用 Reflector 生成的 pivot plan。"""
    max_rounds = _get_max_rounds(config)
    history = state.get("exploration_history", [])
    completed_rounds = len(history)

    if completed_rounds >= max_rounds:
        return "generate_answer"
    return "execute"


# ─── 入口函数 ──────────────────────────────────────────────────

def run_agentic_rag(
    query: str,
    app,  # CompiledStateGraph
    config: RunnableConfig,
    max_rounds: int = 5,
) -> dict:
    """运行 Agentic RAG 闭环探索。

    Args:
        query: 问题文本
        app: 编译好的 LangGraph StateGraph
        config: LangGraph 运行时配置
        max_rounds: 最大探索轮次

    Returns:
        {
            "answer": str,
            "exploration_history": list[dict],
            "total_rounds": int,
            "all_chunks": list[dict],  # unique chunks across all rounds
        }
    """
    # 将 max_rounds 注入 config
    config = dict(config)
    config.setdefault("configurable", {})
    config["configurable"]["max_rounds"] = max_rounds

    start = time.perf_counter()
    logger.info("Agentic RAG 开始探索: query='%s...', max_rounds=%d", query[:80], max_rounds)

    result = app.invoke(
        {
            "query": query,
            "exploration_history": [],
            "plan": {},
            "rewritten_queries": [],
            "chunks": [],
            "evaluation": {},
            "final_answer": "",
            "done": False,
            "messages": [],
        },
        config=config,
    )

    latency = (time.perf_counter() - start) * 1000
    logger.info("Agentic RAG 探索完成: %.0fms, rounds=%d", latency, len(result.get("exploration_history", [])))

    # 收集所有 unique chunks
    all_chunks = _collect_unique_chunks(result.get("exploration_history", []))

    return {
        "answer": result.get("final_answer", ""),
        "exploration_history": result.get("exploration_history", []),
        "total_rounds": len(result.get("exploration_history", [])),
        "all_chunks": all_chunks,
    }


# ─── 格式化与工具函数 ─────────────────────────────────────────

def _format_chunks_for_evaluator(chunks: list[dict]) -> str:
    """将 chunks 格式化为 Evaluator 可读的文本。"""
    if not chunks:
        return "(no knowledge retrieved)"

    parts = []
    for i, c in enumerate(chunks, 1):
        title = c.get("chunk_title", "Unknown")
        summary = c.get("chunk_summary", "")
        content = c.get("page_content", "")
        context = c.get("context_title", "")

        part = f"--- Chunk {i} ---\n"
        part += f"Title: {title}\n"
        if context:
            part += f"Context: {context}\n"
        if summary:
            part += f"Summary: {summary}\n"
        part += f"Content: {content}\n"
        parts.append(part)

    return "\n".join(parts)


def _format_history_for_planner(history: list[dict]) -> str:
    """压缩探索历史为 Planner 可用的简短上下文。"""
    if not history:
        return ""

    parts = []
    for h in history[-3:]:  # 最近 3 轮
        round_num = h.get("round", "?")
        plan_summary = f"targets: {h.get('plan', {}).get('targets', [])[:3]}"
        ev = h.get("evaluation", {})
        ev_summary = f"status={ev.get('status', '?')}, gaps={ev.get('knowledge_gaps', [])[:2]}"
        parts.append(f"Round {round_num}: {plan_summary} | {ev_summary}")

    return "### Exploration History\n" + "\n".join(parts)


def _format_history_full(history: list[dict]) -> str:
    """完整格式化探索历史，供 Reflector 详细分析。"""
    if not history:
        return "(no history)"

    parts = []
    for h in history:
        round_num = h.get("round", "?")
        parts.append(f"=== Round {round_num} ===")

        plan = h.get("plan", {})
        parts.append(f"Plan: {json.dumps(plan, ensure_ascii=False, indent=2)}")

        ev = h.get("evaluation", {})
        parts.append(f"Status: {ev.get('status', '?')}")
        parts.append(f"Feedback: {ev.get('feedback', '')}")
        parts.append(f"Gaps: {ev.get('knowledge_gaps', [])}")
        parts.append("")

    return "\n".join(parts)


def _format_history_for_answer(history: list[dict]) -> str:
    """将探索历史格式化为 LLM 生成答案的输入。"""
    if not history:
        return "(no exploration)"

    lines = []
    for h in history:
        round_num = h.get("round", "?")
        lines.append(f"## Round {round_num}")

        # Chunks
        chunks = h.get("chunks", [])
        if chunks:
            lines.append("  Retrieved Knowledge:")
            for i, c in enumerate(chunks, 1):
                title = c.get("chunk_title", "Unknown")
                summary = c.get("chunk_summary", "")
                content = c.get("page_content", "")
                entry = f"    [{i}] {title}"
                lines.append(entry)
                if summary:
                    lines.append(f"        Summary: {summary}")
                lines.append(f"        Content: {content[:500]}")
                if len(content) > 500:
                    lines.append("        ...(truncated)")

        # Evaluation
        ev = h.get("evaluation", {})
        lines.append(f"  Evaluation: status={ev.get('status', '?')}")
        # 不展示 Evaluator 的中间答案给 Answer LLM，以保证与 RAG with Judge 的公平对比
        # （RAG with Judge 的 Judge 节点不返回答案，Answer LLM 只看检索知识）
        # if ev.get("answer"):
        #     lines.append(f"  Intermediate Answer: {ev['answer']}")

        lines.append("")

    return "\n".join(lines)


def _collect_unique_chunks(history: list[dict]) -> list[dict]:
    """BFS 收集探索历史中所有 unique chunks（按 chunk_id 去重）。"""
    all_chunks = []
    seen_ids = set()

    for h in history:
        for chunk in h.get("chunks", []):
            cid = chunk.get("chunk_id")
            if cid and cid not in seen_ids:
                seen_ids.add(cid)
                all_chunks.append(chunk)

    return all_chunks


def _parse_json_response(content: str) -> dict:
    """解析 LLM 的 JSON 输出（含 markdown code block fallback）。"""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        return json.loads(content.strip())
