"""Agentic RAG v2 — 节点函数与入口

节点函数（供 LangGraph StateGraph 调用）：
- plan_node: 生成需要解决的子问题（带优先级和 rationale）
- solve_sub_questions: 逐一检索 + 评估每个子问题
- synthesize_node: 检查推理链完整性
- reflect_node: 连续 stuck 时反思并生成 pivot 子问题
- generate_answer_node: 基于推理链生成最终答案

入口：
- run_agentic_rag_v2(): 调用编译好的 StateGraph，执行闭环探索
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field
from typing import List

from agentic_rag_v2.state import AgenticRAGV2State
from agentic_rag_v2.prompts import (
    get_planner_prompt,
    get_sq_judgment_prompt,
    get_sq_answer_prompt,
    get_synthesizer_prompt,
    get_reflector_prompt,
    get_answer_prompt,
)

logger = logging.getLogger(__name__)


# ─── 辅助函数 ───────────────────────────────────────────────────

def _get_planner_llm(config: RunnableConfig):
    return config["configurable"].get("planner_llm") or _get_llm(config)


def _get_evaluator_llm(config: RunnableConfig):
    return config["configurable"].get("evaluator_llm") or _get_llm(config)


def _get_reflector_llm(config: RunnableConfig):
    return config["configurable"].get("reflector_llm") or _get_llm(config)


def _get_answer_llm(config: RunnableConfig):
    return config["configurable"].get("answer_llm") or _get_llm(config)


def _get_rewrite_llm(config: RunnableConfig):
    return config["configurable"].get("rewrite_llm") or _get_llm(config)


def _get_synthesizer_llm(config: RunnableConfig):
    return config["configurable"].get("synthesizer_llm") or _get_llm(config)


def _get_llm(config: RunnableConfig):
    return config["configurable"]["llm"]


def _get_retriever(config: RunnableConfig):
    return config["configurable"]["retriever"]


def _get_max_rounds(config: RunnableConfig) -> int:
    return config["configurable"].get("max_rounds", 5)


# ─── LangGraph 节点函数 ────────────────────────────────────────

def plan_node(state: AgenticRAGV2State, config: RunnableConfig) -> dict:
    """节点：生成需要解决的子问题。

    综合当前问题、探索历史、上一轮反馈，制定子问题列表。
    reflect → solve 路径绕过 plan_node，所以 plan_node 只在正常循环中被调用。

    返回: {"plan": {"hypotheses": [...], "sub_questions": [...]}}
    """
    query = state["query"]
    history = state.get("exploration_history", [])
    llm = _get_planner_llm(config)
    round_num = len(history) + 1

    # 正常规划：调用 Planner LLM
    history_context = _format_history_for_planner(history)
    feedback_instruction = _build_feedback_instruction(history)

    prompt = get_planner_prompt(
        question=query,
        round_num=round_num,
        history_context=history_context,
        feedback_instruction=feedback_instruction,
    )

    response = llm.invoke([prompt])
    content = response.content.strip()
    plan = _parse_json_response(content)

    assert "hypotheses" in plan and "sub_questions" in plan, f"Invalid plan: {plan}"

    # 确保 sub_questions 是列表且每个元素有 question 字段
    if not isinstance(plan["sub_questions"], list):
        plan["sub_questions"] = []
    for sq in plan["sub_questions"]:
        if isinstance(sq, str):
            plan["sub_questions"] = [
                {"question": q, "priority": i + 1, "rationale": ""}
                for i, q in enumerate(plan["sub_questions"])
            ]
            break
        if "question" not in sq:
            sq["question"] = str(sq)
        if "priority" not in sq:
            sq["priority"] = 1
        if "rationale" not in sq:
            sq["rationale"] = ""

    logger.info(
        "[round=%d] 探索计划生成: hypotheses=%d, sub_questions=%d",
        round_num, len(plan["hypotheses"]), len(plan["sub_questions"])
    )
    return {"plan": plan}


def solve_sub_questions(state: AgenticRAGV2State, config: RunnableConfig) -> dict:
    """节点：逐一解决每个子问题。

    对 plan 中的每个子问题：
    1. LLM 生成 4 个重写变体 + 原始子问题
    2. 每个变体独立检索，RRF 融合
    3. 调用 evaluator_llm 判断能否从融合结果中回答
    4. 若可回答，调用主 LLM 提取直接答案
    5. 记录状态：solved / unsolved / stuck

    返回: {"sub_questions": [...]}
    """
    plan = state.get("plan", {})
    sub_questions = plan.get("sub_questions", [])
    retriever = _get_retriever(config)
    evaluator_llm = _get_evaluator_llm(config)
    llm = _get_llm(config)
    rewrite_llm = _get_rewrite_llm(config)

    if not sub_questions:
        logger.warning("Plan 中没有子问题，跳过求解")
        return {"sub_questions": []}

    # 并行检索 + 评估每个子问题
    with ThreadPoolExecutor(max_workers=len(sub_questions)) as executor:
        futures = {
            executor.submit(_solve_single_sub_question, sq, retriever, evaluator_llm, llm, rewrite_llm): sq
            for sq in sub_questions
        }
        results = []
        for f in as_completed(futures):
            results.append(f.result())

    # 按优先级排序
    results.sort(key=lambda x: x.get("_priority", 999))
    for r in results:
        r.pop("_priority", None)

    logger.info(
        "子问题求解完成: solved=%d, unsolved=%d, stuck=%d",
        sum(1 for r in results if r["status"] == "solved"),
        sum(1 for r in results if r["status"] == "unsolved"),
        sum(1 for r in results if r["status"] == "stuck"),
    )
    return {"sub_questions": results}


def synthesize_node(state: AgenticRAGV2State, config: RunnableConfig) -> dict:
    """节点：合成推理链，评估完整性。

    1. 收集所有已解决的子问题及其答案
    2. 判断推理链是否完整
    3. 如果不完整/ stuck：记录缺失

    返回: {"evaluation": {"status", "reasoning_chain", "missing"}}
    """
    query = state["query"]
    history = state.get("exploration_history", [])
    all_sub_questions = state.get("sub_questions", [])
    llm = _get_synthesizer_llm(config)

    # 按问题去重，保留最新版本（后出现的覆盖先出现的）
    seen_questions = {}
    for sq in all_sub_questions:
        q = sq.get("question", "")
        seen_questions[q] = sq
    deduped_sub_questions = list(seen_questions.values())

    # 分类已解决和未解决的子问题
    solved = [sq for sq in deduped_sub_questions if sq.get("status") == "solved"]
    unsolved = [sq for sq in deduped_sub_questions if sq.get("status") in ("unsolved", "stuck")]

    solved_text = _format_solved_sub_questions(solved)
    unsolved_text = _format_unsolved_sub_questions(unsolved)

    prompt = get_synthesizer_prompt(
        question=query,
        solved_text=solved_text,
        unsolved_text=unsolved_text,
    )

    # 结构化输出
    class SynthesisResult(BaseModel):
        status: str = Field(description="One of: complete, incomplete, stuck")
        reasoning_chain: str = Field(description="Summary of solved facts in logical order")
        missing: List[str] = Field(description="Missing facts or sub-questions")

    structured_llm = llm.with_structured_output(SynthesisResult)
    result = structured_llm.invoke([prompt])

    assert result.status in ("complete", "incomplete", "stuck"), f"Invalid status: {result.status}"

    logger.info(
        "[round=%d] 合成结果: status=%s, solved=%d, missing=%d",
        len(history) + 1, result.status, len(solved), len(result.missing)
    )

    # 构建当前轮次的完整记录
    current_round = {
        "round": len(history) + 1,
        "plan": state.get("plan", {}),
        "sub_questions": all_sub_questions,
        "evaluation": {
            "status": result.status,
            "reasoning_chain": result.reasoning_chain,
            "missing": result.missing,
        },
    }

    return {
        "evaluation": {
            "status": result.status,
            "reasoning_chain": result.reasoning_chain,
            "missing": result.missing,
        },
        "exploration_history": [current_round],
    }


def reflect_node(state: AgenticRAGV2State, config: RunnableConfig) -> dict:
    """节点：反思失败原因并生成 pivot 子问题。

    当连续 >= 2 轮 stuck 时触发（由 route_after_synthesize 控制）。
    分析失败原因，生成新的检索角度。

    返回: {"plan": {"hypotheses": [...], "sub_questions": [...], "reflection": {...}}}
    路由到 solve_sub_questions 后，该节点直接使用 Reflector 生成的 pivot 子问题。
    """
    query = state["query"]
    history = state.get("exploration_history", [])
    llm = _get_reflector_llm(config)

    # 由 router 保证只有 stuck_count >= 2 时才会路由到这里
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

    # 将 pivot 子问题转换为 plan 格式
    pivot_questions = reflection.get("pivot_sub_questions", [])
    plan = {
        "hypotheses": [reflection.get("diagnosis", "Strategy needs adjustment")],
        "sub_questions": [
            {"question": q, "priority": i + 1, "rationale": "Pivot query from reflection"}
            for i, q in enumerate(pivot_questions)
        ],
        "reflection": reflection,
    }

    logger.info(
        "Reflector 反思完成: diagnosis='%s...', pivot_sub_questions=%d",
        reflection.get("diagnosis", "")[:50], len(pivot_questions)
    )
    return {"plan": plan}


def generate_answer_node(state: AgenticRAGV2State, config: RunnableConfig) -> dict:
    """节点：基于推理链生成最终答案。

    使用 Synthesizer 评估时产生的 reasoning_chain 作为输入。
    与 RAG with Judge 公平：不使用中间评估答案，只基于推理链合成。

    返回: {"final_answer": str, "done": True}
    """
    llm = _get_answer_llm(config)
    history = state.get("exploration_history", [])
    query = state["query"]
    all_sub_questions = state.get("sub_questions", [])

    # 按问题去重，保留最新版本
    seen_questions = {}
    for sq in all_sub_questions:
        q = sq.get("question", "")
        seen_questions[q] = sq
    deduped_sub_questions = list(seen_questions.values())

    # 构建推理链文本
    reasoning_chain_text = _format_reasoning_chain(deduped_sub_questions)

    # 调用 Answer LLM 基于推理链生成
    messages = get_answer_prompt(reasoning_chain_text, query)
    response = llm.invoke(messages)
    answer = response.content.strip()

    logger.info("综合答案生成完成 (len=%d)", len(answer))
    return {"final_answer": answer, "done": True}


def generate_answer_stream(state: AgenticRAGV2State, config: RunnableConfig):
    """流式版本：基于推理链逐 token 生成最终答案。

    返回: generator，逐个产出 token 内容。
    """
    llm = _get_answer_llm(config)
    query = state["query"]
    all_sub_questions = state.get("sub_questions", [])

    # 去重保留最新版本
    seen_questions = {}
    for sq in all_sub_questions:
        q = sq.get("question", "")
        seen_questions[q] = sq
    deduped_sub_questions = list(seen_questions.values())

    reasoning_chain_text = _format_reasoning_chain(deduped_sub_questions)

    # 调用 Answer LLM 流式生成
    messages = get_answer_prompt(reasoning_chain_text, query)
    for chunk in llm.stream(messages):
        content = chunk.content if hasattr(chunk, 'content') else str(chunk)
        if content:
            yield content


# ─── 路由函数 ──────────────────────────────────────────────────

def route_after_synthesize(state: AgenticRAGV2State, config: RunnableConfig) -> str:
    """条件路由：根据 Synthesizer 的 status 决定下一步。"""
    evaluation = state.get("evaluation", {})
    status = evaluation.get("status", "incomplete")
    history = state.get("exploration_history", [])
    max_rounds = _get_max_rounds(config)
    completed_rounds = len(history)

    if status == "complete":
        return "generate_answer"

    # 达到最大轮次，强制结束
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

    # incomplete → 回到 plan 补充子问题
    return "plan"


def route_after_reflect(state: AgenticRAGV2State, config: RunnableConfig) -> str:
    """Reflector 之后直接执行 solve_sub_questions，使用 Reflector 生成的 pivot plan。"""
    max_rounds = _get_max_rounds(config)
    history = state.get("exploration_history", [])
    completed_rounds = len(history)

    if completed_rounds >= max_rounds:
        return "generate_answer"
    return "solve_sub_questions"


# ─── 入口函数 ──────────────────────────────────────────────────

def run_agentic_rag_v2(
    query: str,
    app,  # CompiledStateGraph
    config: RunnableConfig,
    max_rounds: int = 5,
) -> dict:
    """运行 Agentic RAG v2 闭环探索。

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
            "sub_questions": list[dict],  # 所有子问题的解决状态
            "all_chunks": list[dict],     # 所有 unique chunks
        }
    """
    config = dict(config)
    config.setdefault("configurable", {})
    config["configurable"]["max_rounds"] = max_rounds

    start = time.perf_counter()
    logger.info("Agentic RAG v2 开始探索: query='%s...', max_rounds=%d", query[:80], max_rounds)

    result = app.invoke(
        {
            "query": query,
            "sub_questions": [],
            "exploration_history": [],
            "plan": {},
            "evaluation": {},
            "final_answer": "",
            "done": False,
            "messages": [],
        },
        config=config,
    )

    latency = (time.perf_counter() - start) * 1000
    logger.info("Agentic RAG v2 探索完成: %.0fms, rounds=%d", latency, len(result.get("exploration_history", [])))

    # 收集所有 unique chunks
    all_chunks = _collect_unique_chunks(result.get("exploration_history", []))

    return {
        "answer": result.get("final_answer", ""),
        "exploration_history": result.get("exploration_history", []),
        "total_rounds": len(result.get("exploration_history", [])),
        "sub_questions": result.get("sub_questions", []),
        "all_chunks": all_chunks,
    }


# ─── 子问题求解辅助函数 ─────────────────────────────────────────

def _solve_single_sub_question(
    sub_question: dict,
    retriever,
    evaluator_llm,
    llm,
    rewrite_llm,
) -> dict:
    """求解单个子问题。

    1. 问题重写 → 多路检索 → RRF 融合
    2. 判断（evaluator_llm）：能否从检索结果中回答
    3. 提取（llm，仅当 answerable）：从 chunks 中抽取直接答案
    4. 记录
    """
    question = sub_question.get("question", "")
    priority = sub_question.get("priority", 999)

    # 重写 + 检索（get_similar_chunks_with_rewrite 内部完成：重写→扇出检索→RRF→截断）
    results = retriever.get_similar_chunks_with_rewrite(query=question, rewrite_llm=rewrite_llm)

    # 格式化 chunks
    chunks_text = _format_chunks_for_evaluator(results)

    # ── 步骤 1：判断（evaluator_llm，可启用 thinking）
    prompt_judge = get_sq_judgment_prompt(
        sub_question=question,
        chunks_text=chunks_text,
    )

    class SQJudgment(BaseModel):
        answerable: bool = Field(description="Whether the sub-question can be answered")
        reason: str = Field(description="Brief explanation of your judgment")

    try:
        structured_llm = evaluator_llm.with_structured_output(SQJudgment)
        judgment = structured_llm.invoke([prompt_judge])
        answerable = judgment.answerable
    except Exception as e:
        logger.warning("子问题判断失败: %s", e)
        answerable = False

    # ── 步骤 2：提取答案（仅当 answerable，使用主 LLM）
    answer = ""
    if answerable:
        prompt_answer = get_sq_answer_prompt(
            sub_question=question,
            chunks_text=chunks_text,
        )

        class SQAnswer(BaseModel):
            answer: str = Field(description="The direct answer, or empty if not found")

        try:
            structured_llm = llm.with_structured_output(SQAnswer)
            answer_result = structured_llm.invoke([prompt_answer])
            answer = answer_result.answer.strip()
        except Exception as e:
            logger.warning("子问题答案提取失败: %s", e)
            answer = ""

    status = "solved" if answerable else "unsolved"

    # 组装 chunks 输出
    chunks_out = []
    for rank, (doc, _) in enumerate(results, 1):
        chunks_out.append({
            "chunk_id": doc.metadata.get("chunk_id", ""),
            "chunk_title": doc.metadata.get("chunk_title", ""),
            "chunk_summary": doc.metadata.get("chunk_summary", ""),
            "context_title": doc.metadata.get("context_title", ""),
            "page_content": doc.page_content,
        })

    logger.info(
        "子问题: '%s' → status=%s, answer='%s', chunks=%d",
        question[:60], status, answer[:50], len(chunks_out)
    )

    return {
        "question": question,
        "status": status,
        "answer": answer,
        "retrieved_chunks": chunks_out,
        "priority": priority,
    }


# ─── 格式化与工具函数 ─────────────────────────────────────────

def _format_chunks_for_evaluator(results: list) -> str:
    """将检索结果格式化为 Evaluator 可读文本。"""
    if not results:
        return "(no knowledge retrieved)"

    parts = []
    for i, (doc, score) in enumerate(results, 1):
        title = doc.metadata.get("chunk_title", "Unknown")
        summary = doc.metadata.get("chunk_summary", "")
        content = doc.page_content
        context = doc.metadata.get("context_title", "")

        part = f"--- Chunk {i} (score: {score:.4f}) ---\n"
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
    for h in history[-3:]:
        round_num = h.get("round", "?")
        plan_summary = f"sub_questions: {len(h.get('plan', {}).get('sub_questions', []))}"
        ev = h.get("evaluation", {})
        ev_summary = f"status={ev.get('status', '?')}, reasoning={ev.get('reasoning_chain', '')}"   # 传递完整信息，不截断
        parts.append(f"Round {round_num}: {plan_summary} | {ev_summary}")

    return "### Exploration History\n" + "\n".join(parts)


def _build_feedback_instruction(history: list[dict]) -> str:
    """根据历史构建 Planner 的反馈引导。"""
    if not history:
        return ""

    last_eval = history[-1].get("evaluation", {})
    if last_eval.get("status") == "stuck":
        return (
            "\n\n**Tip**: The previous round did not yield useful information. "
            "Try using different terminology or exploring related but different angles."
        )
    elif last_eval.get("status") == "incomplete":
        missing = last_eval.get("missing", [])
        if missing:
            missing_text = "; ".join(missing[:3])
            return f"\n\nFocus on these missing facts: {missing_text}"
    return ""


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
        parts.append(f"Reasoning Chain: {ev.get('reasoning_chain', '')}")
        parts.append(f"Missing: {ev.get('missing', [])}")
        parts.append("")

    return "\n".join(parts)


def _format_solved_sub_questions(solved: list[dict]) -> str:
    """格式化已解决的子问题为推理链文本。"""
    if not solved:
        return "(none solved)"

    parts = []
    for sq in solved:
        parts.append(f"- Q: {sq['question']}")
        parts.append(f"  A: {sq.get('answer', '(no answer)')}")

    return "\n".join(parts)


def _format_unsolved_sub_questions(unsolved: list[dict]) -> str:
    """格式化未解决的子问题。"""
    if not unsolved:
        return "(all solved)"

    parts = []
    for sq in unsolved:
        parts.append(f"- Q: {sq['question']} (status: {sq.get('status', '?')})")

    return "\n".join(parts)


def _format_reasoning_chain(all_sub_questions: list[dict]) -> str:
    """构建完整的推理链文本（用于 Answer LLM 输入）。"""
    if not all_sub_questions:
        return "(no exploration)"

    parts = []
    for sq in all_sub_questions:
        status = sq.get("status", "?")
        parts.append(f"[{status.upper()}] {sq['question']}")
        if sq.get("answer"):
            parts.append(f"  Answer: {sq['answer']}")
        else:
            parts.append("  Answer: (not found)")

    return "\n".join(parts)


def _collect_unique_chunks(history: list[dict]) -> list[dict]:
    """收集探索历史中所有 unique chunks（按 chunk_id 去重）。"""
    all_chunks = []
    seen_ids = set()

    for h in history:
        for sq in h.get("sub_questions", []):
            for chunk in sq.get("retrieved_chunks", []):
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
