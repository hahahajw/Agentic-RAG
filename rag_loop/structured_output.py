# ═══════════════════════════════════════════════════════════════════
# 结构化输出工具
# ═══════════════════════════════════════════════════════════════════
# 提供 get_structured_model() 作为 model.with_structured_output() 的
# 直接替代品。支持三种方法：
#
#   - "function_calling": 委托给 with_structured_output（默认）。
#     会设置 tool_choice，在 Qwen 上不兼容 enable_thinking。
#   - "json_mode": 委托给 with_structured_output。
#     兼容 thinking，但 schema 仅通过 prompt 文本传递，无 API 级约束。
#   - "thinking_tool": 使用 bind_tools 不设 tool_choice。
#     保留 schema 定义（tool definition）的同时允许 enable_thinking。
#     模型 think 后自愿调用工具；回退解析 content 中的 JSON。
#
# 关键洞察：Qwen3 API 禁止 enable_thinking + tool_choice 共存，
# 但 enable_thinking + tools（无 tool_choice）是支持的。
# function_calling 的问题在于 LangChain 总是强制设置 tool_choice，
# thinking_tool 绕过这一点。

from __future__ import annotations

import json
import logging
import re
from typing import Any

from pydantic import BaseModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.utils.function_calling import convert_to_openai_tool

logger = logging.getLogger(__name__)

# 模块级开关：True 时输出原始 JSON（调试用），False 时输出聚焦摘要
verbose_raw = False


def _extract_json_from_content(content: str) -> dict[str, Any] | None:
    """从 LLM 文本输出中提取 JSON 对象。

    处理常见的 LLM 输出格式：
    - 纯 JSON: '{"key": "value"}'
    - Markdown 代码块: '```json\\n{...}\\n```'
    - 无语言标记的代码块: '```\\n{...}\\n```'
    """
    text = content.strip()

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    fence_pattern = r"```(?:json)?\s*\n?(.*?)\n?```"
    matches = re.findall(fence_pattern, text, re.DOTALL)
    for match in matches:
        try:
            data = json.loads(match.strip())
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            continue

    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        try:
            data = json.loads(text[first_brace : last_brace + 1])
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

    return None


# ═══════════════════════════════════════════════════════════════════
# 聚焦摘要输出（默认模式）—— 证明各角色正确工作
# ═══════════════════════════════════════════════════════════════════

def _summarize_planner(instance: dict) -> None:
    """Planner 输出摘要：DAG 结构（节点 + 边，标注 dependency）。"""
    primitives = instance.get("primitives", [])
    inherited: list[str] = []
    initialized: list[tuple[str, str]] = []  # (node_id_placeholder, question)
    links: list[tuple[str, str, str]] = []   # (from, to, edge_type)

    for p in primitives:
        pt = p.get("primitive_type", "")
        if pt == "INHERIT":
            inherited.append(p.get("node_id", "?"))
        elif pt == "INHERIT_AND_RELABEL":
            nid = p.get("node_id", "?")
            q = p.get("new_question", "?")[:70]
            initialized.append((f"{nid}(relabel)", q))
        elif pt == "INITIALIZE":
            q = p.get("question", "?")[:80]
            initialized.append(("NEW", q))
        elif pt == "LINK":
            links.append((
                p.get("from_id", "?"),
                p.get("to_id", "?"),
                p.get("edge_type", "?"),
            ))

    lines = ["DAG Plan:"]
    for nid in inherited:
        lines.append(f"  INHERIT {nid}")
    for placeholder, q in initialized:
        lines.append(f"  INIT   {placeholder}: \"{q}\"")
    if links:
        lines.append("Edges:")
        for src, dst, etype in links:
            marker = " [dependency]" if etype == "dependency" else ""
            lines.append(f"  {src} -> {dst} ({etype}){marker}")
    logger.info("\n".join(lines))


def _summarize_critic(instance: dict) -> None:
    """Critic 输出摘要：节点 health + 终止判断。"""
    reviews = instance.get("node_reviews", [])
    termination = instance.get("termination", {})

    health_parts = []
    for r in reviews:
        nid = r.get("node_id", "?")
        h = r.get("critic_health", "?")
        health_parts.append(f"{nid}={h}")

    logger.info("Critic: %s", "  ".join(health_parts))
    logger.info(
        "  terminate=%s  cond_2=%s  cond_3=%s  reason=%s",
        termination.get("should_terminate", "?"),
        termination.get("condition_2_passed", "?"),
        termination.get("condition_3_passed", "?"),
        (termination.get("termination_reason", "") or "")[:120],
    )


def _summarize_node_prep(instance: dict) -> None:
    """Solver:Query 摘要：推理优先 → 搜索查询构造。"""
    can_answer = instance.get("can_answer", False)
    answer = instance.get("answer", "")
    search_query = instance.get("search_query", "")

    if can_answer:
        logger.info("Solver:Query -> INFER (no search needed): answer=\"%s\"", answer[:100])
    else:
        logger.info("Solver:Query -> SEARCH: \"%s\"", search_query[:120])


def _summarize_extraction(instance: dict) -> None:
    """Solver:Extract 摘要：答案 + 证据。"""
    answer = instance.get("answer", "")
    chunks = instance.get("supporting_chunks", [])
    judgment = instance.get("judgment", "")

    logger.info("Solver:Extract -> answer=\"%s\"", answer[:120])
    logger.info("  evidence: %d chunks %s", len(chunks), chunks[:5])
    if judgment:
        logger.info("  judgment: %s", judgment[:150])


def _log_structured_summary(instance: BaseModel, schema_name: str) -> None:
    """按 schema 类型输出聚焦摘要。"""
    # 转为 dict（无论 Pydantic 实例还是 dict 都兼容）
    data = instance if isinstance(instance, dict) else instance.model_dump()

    if schema_name == "PlannerOutput":
        _summarize_planner(data)
    elif schema_name == "CriticOutput":
        _summarize_critic(data)
    elif schema_name == "NodePreparation":
        _summarize_node_prep(data)
    elif schema_name == "ExtractionResult":
        _summarize_extraction(data)
    else:
        # 未知类型：简要 JSON 摘要
        j = json.dumps(data, ensure_ascii=False)
        logger.info("%s: %s", schema_name, j[:300])


# ═══════════════════════════════════════════════════════════════════
# 原始输出模式（--verbose）—— 调试用
# ═══════════════════════════════════════════════════════════════════

def _log_raw_response(response: AIMessage, schema_name: str, used_tool: bool) -> None:
    """记录模型的原始输出内容。"""
    reasoning = ""
    if hasattr(response, "additional_kwargs"):
        reasoning = response.additional_kwargs.get("reasoning_content", "") or ""
    if not reasoning and hasattr(response, "reasoning_content"):
        reasoning = getattr(response, "reasoning_content", "") or ""

    content = response.content or ""
    path_label = "tool_calls" if used_tool else "content JSON"
    logger.info("══════ [%s -> %s] RAW ══════", schema_name, path_label)

    if reasoning:
        rp = reasoning if len(reasoning) <= 800 else reasoning[:400] + f"\n... [{len(reasoning)-800} chars] ...\n" + reasoning[-400:]
        logger.info("[reasoning_content]\n%s", rp)
    else:
        logger.info("[reasoning] (not captured by LangChain)")

    if used_tool and response.tool_calls:
        for i, tc in enumerate(response.tool_calls):
            args_str = json.dumps(tc.get("args", {}), ensure_ascii=False, indent=2)
            if len(args_str) > 1500:
                args_str = args_str[:1000] + f"\n... [{len(args_str)-1500} chars] ...\n" + args_str[-500:]
            logger.info("[tool_call #%d: %s]\n%s", i + 1, tc.get("name", "?"), args_str)

    if content:
        cs = content if isinstance(content, str) else str(content)
        if len(cs) > 1500:
            cs = cs[:1000] + f"\n... [{len(cs)-1500} chars] ...\n" + cs[-500:]
        logger.info("[content]\n%s", cs)

    logger.info("══════ END [%s] ══════", schema_name)


# ═══════════════════════════════════════════════════════════════════
# Token 用量日志
# ═══════════════════════════════════════════════════════════════════

def _log_token_usage(response: AIMessage, schema_name: str) -> None:
    """记录本次 LLM 调用的 token 消耗。"""
    um = getattr(response, "usage_metadata", None) or {}
    inp = um.get("input_tokens", 0)
    out = um.get("output_tokens", 0)
    total = um.get("total_tokens", 0)
    reasoning = (um.get("output_token_details", {}) or {}).get("reasoning", 0)
    answer_tokens = max(0, out - reasoning)

    if total > 0:
        parts = [f"in={inp}"]
        if reasoning > 0:
            parts.append(f"think={reasoning}")
        parts.append(f"out={answer_tokens}")
        parts.append(f"total={total}")
        logger.info("[%s tokens] %s", schema_name, "  ".join(parts))


# ═══════════════════════════════════════════════════════════════════
# thinking_tool 链构建
# ═══════════════════════════════════════════════════════════════════

def _create_thinking_tool_chain(model, schema: type[BaseModel]) -> Runnable:
    """构建 thinking_tool 链：bind_tools（无 tool_choice）→ 解析响应。"""
    tool_def = convert_to_openai_tool(schema)
    bound = model.bind_tools([tool_def])

    def _parse(response: AIMessage) -> schema:
        # 检查是否被 max_completion_tokens 截断
        finish_reason = ""
        if hasattr(response, "response_metadata"):
            finish_reason = (response.response_metadata or {}).get("finish_reason", "") or ""

        if finish_reason == "length":
            raise ValueError(
                f"thinking_tool: 模型输出被 max_completion_tokens 截断"
                f" (schema={schema.__name__})。"
                f"请增加 max_completion_tokens 或精简输入。"
                f" 已有 content 预览: {str(response.content or '')[:200]}"
            )

        # 主路径：模型在 thinking 之后调用了工具
        if response.tool_calls and len(response.tool_calls) > 0:
            args: dict[str, Any] = response.tool_calls[0].get("args", {})
            if not isinstance(args, dict):
                raise ValueError(
                    f"thinking_tool: tool_calls[0]['args'] 不是 dict 类型: "
                    f"{type(args).__name__}: {args!r}"
                )
            result = schema.model_validate(args)
            if verbose_raw:
                _log_raw_response(response, schema.__name__, used_tool=True)
            _log_structured_summary(result, schema.__name__)
            _log_token_usage(response, schema.__name__)
            return result

        # 回退路径：模型没有调用工具，但在 content 中输出了 JSON
        content = response.content or ""
        if isinstance(content, str) and content.strip():
            data = _extract_json_from_content(content)
            if data is not None:
                result = schema.model_validate(data)
                if verbose_raw:
                    _log_raw_response(response, schema.__name__, used_tool=False)
                _log_structured_summary(result, schema.__name__)
                _log_token_usage(response, schema.__name__)
                return result

        preview = str(content)[:300]
        raise ValueError(
            f"thinking_tool: 模型既没有调用工具，也没有在 content 中输出合法 JSON。"
            f"目标 schema: {schema.__name__}。"
            f"Content 预览: {preview}"
        )

    return bound | RunnableLambda(_parse, name=f"parse_{schema.__name__}")


def get_structured_model(
    model,
    schema: type[BaseModel],
    method: str = "function_calling",
) -> Runnable:
    """返回产生 schema 实例的 Runnable，直接替代 model.with_structured_output()。

    Args:
        model: BaseChatModel 实例（如 ChatOpenAI）
        schema: Pydantic BaseModel 子类，定义输出结构
        method: "function_calling" | "json_mode" | "thinking_tool"

    Returns:
        Runnable[LanguageModelInput, schema_instance]
    """
    if method == "thinking_tool":
        return _create_thinking_tool_chain(model, schema)
    return model.with_structured_output(schema, method=method)