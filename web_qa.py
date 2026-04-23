"""Web QA — Streamlit Web UI for RAG systems with web search

用户输入任意问题，选择 RAG 系统（Naive RAG / RAG with Judge / Agentic RAG V3），
系统从 DuckDuckGo 网络搜索获取信息并生成答案。
"""

import logging
import os
import time
from typing import Any, Dict

import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from Retrieval.web_retriever import WebRetriever

# 导入默认 prompt 模板
from naive_rag.prompts import QUERY_REWRITE_PROMPT
from naive_rag.nodes import RAG_SYS_PROMPT
from rag_with_judge.prompts import (
    JUDGE_PROMPT_B_TEMPLATE,
    ANSWER_SYSTEM_PROMPT as JUDGE_ANSWER_PROMPT,
)
from agentic_rag_v3.prompts import (
    PLANNER_PROMPT_TEMPLATE,
    SQ_JUDGMENT_TEMPLATE,
    SQ_ANSWER_TEMPLATE,
    SYNTHESIZER_PROMPT_TEMPLATE,
    REFLECTOR_PROMPT_TEMPLATE,
    ANSWER_SYSTEM_PROMPT as AGENTIC_ANSWER_PROMPT,
)

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Page Config ────────────────────────────────────────────────

st.set_page_config(
    page_title="Web QA — RAG Systems",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Web QA — 多系统网络搜索")
st.caption("输入问题，选择 RAG 系统，从 DuckDuckGo 网络搜索获取信息并生成答案")


# ─── Visualization: RAG with Judge Search Tree ─────────────────

def render_search_tree(search_path: Dict[str, Any], max_depth: int = 3, indent: int = 0):
    """递归渲染 SEARCH_PATH 搜索树。

    节点颜色:
      - 绿色: answerable=True (知识足够)
      - 橙色: answerable=False, depth < max_depth (需要进一步探索)
      - 灰色: depth >= max_depth (达到最大深度)
    """
    if not search_path:
        st.warning("无搜索树数据")
        return

    question = search_path.get("question", "Unknown")
    answerable = search_path.get("answerable", False)
    chunks = search_path.get("chunks", [])
    next_queries = search_path.get("next_queries", [])
    answer = search_path.get("answer", "")
    children_count = len([c for c in next_queries if isinstance(c, dict)])

    # 当前深度
    if indent == 0:
        depth = 0
    else:
        # 递归深度由嵌套层级推断
        depth = indent

    # 节点颜色
    if answerable:
        color = "#22c55e"
        status_icon = "✅"
        status_text = "可回答"
    elif depth >= max_depth:
        color = "#94a3b8"
        status_icon = "🚫"
        status_text = "达到最大深度"
    else:
        color = "#f59e0b"
        status_icon = "🔄"
        status_text = "需要更多知识"

    # 截断问题文本
    q_short = question[:80] + ("…" if len(question) > 80 else "")

    # 渲染节点
    cols = st.columns([1, 5, 2])

    with cols[0]:
        st.markdown(f"""
        <div style="background:{color};color:white;border-radius:8px;padding:4px 8px;
                     text-align:center;font-weight:bold;font-size:14px;margin-top:4px;">
            {status_icon}
        </div>
        """, unsafe_allow_html=True)

    with cols[1]:
        st.markdown(f"**Q:** {q_short}")
        meta = f"{status_text}"
        meta += f" | Chunks: {len(chunks)}"
        if children_count > 0:
            meta += f" | 子问题: {children_count}"
        if answer:
            meta += f" | 答案: {answer[:60]}{'…' if len(answer) > 60 else ''}"
        st.caption(meta)

    with cols[2]:
        if chunks:
            with st.expander(f"📄 {len(chunks)} 来源", expanded=False):
                for i, c in enumerate(chunks, 1):
                    title = c.get("chunk_title", "Unknown")
                    content = c.get("page_content", "")[:200]
                    st.markdown(f"**{i}. {title}**\n\n{content}")

    # 递归渲染子节点
    for child in next_queries:
        if isinstance(child, dict):
            render_search_tree(child, max_depth=max_depth, indent=indent + 1)
            st.markdown(f"""
            <div style="border-left:2px solid {color};margin-left:20px;padding:4px;">
            </div>
            """, unsafe_allow_html=True)


def render_search_tree_treeview(search_path: Dict[str, Any], max_depth: int = 3):
    """使用 Excalidraw 风格的树形图渲染 SEARCH_PATH。

    生成 Mermaid 图，Streamlit 会自动渲染。
    """
    if not search_path:
        return

    # 递归收集节点和边
    nodes = []
    edges = []

    def _collect(path: dict, parent_id: str = None, depth: int = 0):
        q = path.get("question", "")
        q_short = q[:40].replace('"', "'").replace("\n", " ")
        answerable = path.get("answerable", False)
        n_chunks = len(path.get("chunks", []))
        children = [c for c in path.get("next_queries", []) if isinstance(c, dict)]

        node_id = f"n_{hash(q[:50]) & 0xFFFF}_{depth}"
        status = "YES" if answerable else "NO"
        if depth >= max_depth and not answerable:
            status = "MAX"

        label = f"{q_short}<br/>({'✅' if answerable else '❌'} {status}, {n_chunks} chunks)"
        nodes.append((node_id, label, answerable, depth))

        if parent_id:
            edges.append((parent_id, node_id))

        for i, child in enumerate(children):
            _collect(child, node_id, depth + 1)

    _collect(search_path, max_depth=max_depth)

    # 生成 Mermaid
    lines = ["graph TD"]
    for nid, label, ans, d in nodes:
        color = "#22c55e" if ans else "#f59e0b" if d < max_depth else "#94a3b8"
        lines.append(f'    {nid}["{label}"]')
        lines.append(f'    style {nid} fill:{color},stroke:#333,stroke-width:1px')

    for src, dst in edges:
        lines.append(f"    {src} --> {dst}")

    mermaid_code = "\n".join(lines)
    st.markdown(mermaid_code, unsafe_allow_html=True)


def render_search_tree_viz(search_path: Dict[str, Any], max_depth: int = 3):
    """渲染 RAG with Judge 的搜索树。

    使用 Mermaid 生成有向图，展示递归探索路径。
    """
    if not search_path:
        return

    nodes = []
    edges = []
    counter = [0]

    def _collect(path: dict, parent_id: str = None, depth: int = 0):
        q = path.get("question", "")
        q_short = q[:35].replace('"', "'").replace("\n", " ")
        answerable = path.get("answerable", False)
        n_chunks = len(path.get("chunks", []))
        children = [c for c in path.get("next_queries", []) if isinstance(c, dict)]
        answer = path.get("answer", "")
        ans_short = answer[:25].replace('"', "'") if answer else ""

        node_id = f"n{counter[0]}"
        counter[0] += 1

        # 构建节点标签
        lines_label = [f"Q{depth+1}: {q_short}"]
        lines_label.append(f"Chunks: {n_chunks}")
        if answerable:
            lines_label.append("Judge: 可回答")
        else:
            lines_label.append("Judge: 不可回答")
        if children:
            lines_label.append(f"Follow-ups: {len(children)}")
        if ans_short:
            lines_label.append(f"Ans: {ans_short}")

        label = "<br/>".join(lines_label)
        nodes.append((node_id, label, answerable, depth))

        if parent_id:
            edges.append((parent_id, node_id))

        for child in children:
            _collect(child, node_id, depth + 1)

    _collect(search_path, max_depth=max_depth)

    # 生成 Mermaid 图
    m_lines = ["graph TD"]
    for nid, label, ans, d in nodes:
        if ans:
            color = "#22c55e"
            text_color = "white"
        elif d >= max_depth:
            color = "#94a3b8"
            text_color = "white"
        else:
            color = "#f59e0b"
            text_color = "black"

        escaped_label = label.replace('"', "&quot;")
        m_lines.append(f'    {nid}["{escaped_label}"]')
        m_lines.append(f'    style {nid} fill:{color},stroke:#333,stroke-width:2px,color:{text_color}')

    for src, dst in edges:
        m_lines.append(f"    {src} --> {dst}")

    mermaid_code = "\n".join(m_lines)
    st.html(f"""
    <div style="background:#f8f9fa;border-radius:12px;padding:16px;margin:8px 0;">
        <h4 style="margin:0 0 8px 0;">🌳 递归探索树 (SEARCH_PATH)</h4>
        <p style="color:#666;font-size:12px;margin:0 0 12px 0;">
            绿色 = Judge 认为知识足够 | 橙色 = 需要更多知识 | 灰色 = 达到最大深度
        </p>
        <pre class="mermaid" style="background:white;padding:12px;border-radius:8px;">
{mermaid_code}
        </pre>
    </div>
    """, height=400 + len(nodes) * 60)

    st.markdown(
        f'<script type="module">import "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";</script>',
        unsafe_allow_html=True,
    )


def render_search_tree_detail(search_path: Dict[str, Any], max_depth: int = 3):
    """以折叠式 Streamlit 组件渲染搜索树的详细视图。"""
    if not search_path:
        return

    st.markdown("""
    <div style="background:#f8f9fa;border-radius:12px;padding:16px;margin:8px 0;">
        <h3 style="margin:0 0 4px 0;">🌳 递归探索树 (SEARCH_PATH)</h3>
        <p style="color:#666;font-size:13px;margin:0;">
            每次 Judge 判断后，如果知识不足，生成 follow-up 问题并递归探索。
            绿色节点表示 Judge 认为当前知识已足够回答问题。
        </p>
    </div>
    """, unsafe_allow_html=True)

    _render_tree_node(search_path, depth=0, max_depth=max_depth)


def _render_tree_node(node: dict, depth: int, max_depth: int):
    """递归渲染单个树节点。"""
    question = node.get("question", "Unknown")
    answerable = node.get("answerable", False)
    chunks = node.get("chunks", [])
    next_queries = node.get("next_queries", [])
    answer = node.get("answer", "")
    reason = node.get("judgement_reason", "")
    children = [c for c in next_queries if isinstance(c, dict)]

    # 状态标签
    if answerable:
        badge = "✅ 可回答"
        color = "#22c55e"
    elif depth >= max_depth:
        badge = "🚫 达到最大深度"
        color = "#94a3b8"
    else:
        badge = "🔄 需要更多知识"
        color = "#f59e0b"

    # 缩进
    indent = "  " * depth
    arrow = "└─" if depth > 0 else ""

    with st.container(border=True):
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"**{arrow}{indent}Q:** {question}")
            st.caption(f"{badge} | Depth {depth}/{max_depth} | Chunks: {len(chunks)} | Follow-ups: {len(children)}")
            if reason:
                st.caption(f"**Judge 理由:** {reason[:150]}{'…' if len(reason) > 150 else ''}")
            if answer:
                st.caption(f"**中间答案:** {answer[:120]}{'…' if len(answer) > 120 else ''}")
        with col2:
            if chunks:
                with st.expander(f"📄{len(chunks)}", expanded=False):
                    for i, c in enumerate(chunks, 1):
                        title = c.get("chunk_title", "Unknown")
                        content = c.get("page_content", "")[:200]
                        st.markdown(f"**{i}. {title}**\n\n{content}")

    # 递归渲染子节点
    for child in children:
        _render_tree_node(child, depth=depth + 1, max_depth=max_depth)


# ─── Visualization: Agentic RAG V3 Timeline ────────────────────

def render_timeline(exploration_history: list, sub_questions_all: list, final_status: str):
    """渲染 Agentic RAG V3 的多角色智能体讨论时间线。

    时间线展示:
    Round 1: Planner → Solve Sub-Questions → Synthesize (incomplete/stuck)
    Round 2: Planner (补充子问题) → Solve → Synthesize (stuck)
    Round 3: Reflect (诊断) → Solve (pivot) → Synthesize (complete)
    Final: Generate Answer
    """
    if not exploration_history:
        return

    st.markdown("""
    <div style="background:#f8f9fa;border-radius:12px;padding:16px;margin:8px 0;">
        <h3 style="margin:0 0 4px 0;">🤖 多角色智能体讨论时间线</h3>
        <p style="color:#666;font-size:13px;margin:0;">
            Planner 规划子问题 → Solver 检索并评估 → Synthesizer 检查推理链完整性
            → 若 stuck，Reflector 诊断根因并生成新的检索角度
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 角色配色
    COLORS = {
        "planner": "#6366f1",     # 靛蓝
        "solver": "#0ea5e9",      # 天蓝
        "synthesizer": "#f59e0b", # 琥珀
        "reflector": "#ef4444",   # 红
        "answer": "#22c55e",      # 绿
    }

    for round_data in exploration_history:
        round_num = round_data.get("round", "?")
        plan = round_data.get("plan", {})
        sub_qs = round_data.get("sub_questions", [])
        evaluation = round_data.get("evaluation", {})
        status = evaluation.get("status", "?")
        reasoning_chain = evaluation.get("reasoning_chain", "")
        missing = evaluation.get("missing", [])

        # 判断是否触发了 Reflect
        is_reflect = False
        for h in exploration_history[:round_num - 1] if isinstance(round_num, int) else []:
            if h.get("evaluation", {}).get("status") == "stuck":
                is_reflect = True

        # 检查连续 stuck
        if isinstance(round_num, int) and round_num >= 2:
            prev_statuses = []
            for h in exploration_history:
                if h.get("round", 0) < round_num:
                    prev_statuses.append(h.get("evaluation", {}).get("status", ""))
            stuck_count = 0
            for s in reversed(prev_statuses):
                if s == "stuck":
                    stuck_count += 1
                else:
                    break
            is_reflect = stuck_count >= 2

        with st.container(border=True):
            st.markdown(f"### 📋 Round {round_num}")

            # Step 1: Planner
            sub_questions = plan.get("sub_questions", [])
            hypotheses = plan.get("hypotheses", [])
            with st.container():
                col1, col2 = st.columns([0.15, 0.85])
                with col1:
                    st.markdown(f"""
                    <div style="background:{COLORS['planner']};color:white;border-radius:8px;
                                padding:8px;text-align:center;font-weight:bold;">
                        📐<br/>Planner
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"**生成了 {len(sub_questions)} 个子问题，{len(hypotheses)} 个假设**")
                    if hypotheses:
                        for h in hypotheses:
                            st.caption(f"- {h[:100]}{'…' if len(h) > 100 else ''}")
                    if sub_questions:
                        for sq in sub_questions[:3]:
                            q = sq.get("question", "") if isinstance(sq, dict) else str(sq)
                            pri = sq.get("priority", "") if isinstance(sq, dict) else ""
                            st.caption(f"  Q: {q[:80]}{'…' if len(q) > 80 else ''} (Priority: {pri})")
                        if len(sub_questions) > 3:
                            st.caption(f"  ... 还有 {len(sub_questions) - 3} 个子问题")

            # Step 2: Solver
            with st.container():
                col1, col2 = st.columns([0.15, 0.85])
                with col1:
                    st.markdown(f"""
                    <div style="background:{COLORS['solver']};color:white;border-radius:8px;
                                padding:8px;text-align:center;font-weight:bold;">
                        🔍<br/>Solver
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    solved = sum(1 for sq in sub_qs if sq.get("status") == "solved")
                    unsolved = sum(1 for sq in sub_qs if sq.get("status") == "unsolved")
                    stuck = sum(1 for sq in sub_qs if sq.get("status") == "stuck")
                    st.markdown(f"**求解完成: {solved} solved, {unsolved} unsolved, {stuck} stuck**")
                    # 展示每个子问题的求解结果
                    for sq in sub_qs:
                        q = sq.get("question", "?")[:60]
                        s = sq.get("status", "?")
                        a = sq.get("answer", "")
                        icon = "✅" if s == "solved" else ("❌" if s == "stuck" else "⚠️")
                        st.caption(f"{icon} {q} → {s}" + (f" | {a[:40]}" if a else ""))

            # Step 3: Reflect (if triggered)
            if is_reflect and "reflection" in plan:
                reflection = plan.get("reflection", {})
                with st.container():
                    col1, col2 = st.columns([0.15, 0.85])
                    with col1:
                        st.markdown(f"""
                        <div style="background:{COLORS['reflector']};color:white;border-radius:8px;
                                    padding:8px;text-align:center;font-weight:bold;">
                            🪞<br/>Reflect
                        </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        diagnosis = reflection.get("diagnosis", "")
                        st.markdown(f"**诊断:** {diagnosis[:150]}{'…' if len(diagnosis) > 150 else ''}")
                        pivots = reflection.get("pivot_sub_questions", [])
                        if pivots:
                            st.markdown(f"**Pivot 子问题:** {', '.join(p[:40] for p in pivots[:3])}")

            # Step 4: Synthesizer
            with st.container():
                col1, col2 = st.columns([0.15, 0.85])
                with col1:
                    status_color = COLORS.get("synthesizer", "#f59e0b")
                    if status == "complete":
                        status_color = COLORS["answer"]
                    elif status == "stuck":
                        status_color = COLORS["reflector"]
                    st.markdown(f"""
                    <div style="background:{status_color};color:white;border-radius:8px;
                                padding:8px;text-align:center;font-weight:bold;">
                        🔗<br/>Synth
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    status_icon = "✅" if status == "complete" else ("❌" if status == "stuck" else "⚠️")
                    st.markdown(f"**{status_icon} {status.upper()}**")
                    if reasoning_chain:
                        st.caption(f"**推理链:** {reasoning_chain[:200]}{'…' if len(reasoning_chain) > 200 else ''}")
                    if missing:
                        st.caption(f"**缺失:** {', '.join(m[:50] for m in missing[:3])}")

    # 最终答案
    st.markdown(f"""
    <div style="background:{COLORS['answer']};color:white;border-radius:12px;
                padding:16px;margin:8px 0;text-align:center;">
        <h3 style="margin:0;">✅ 最终答案生成</h3>
        <p style="margin:4px 0 0 0;opacity:0.9;">Status: {final_status}</p>
    </div>
    """, unsafe_allow_html=True)


def render_timeline_mermaid(exploration_history: list, final_status: str):
    """用 Mermaid 图渲染 Agentic RAG V3 的多角色讨论流程。"""
    if not exploration_history:
        return

    m_lines = ["graph LR"]

    # 构建节点
    for round_data in exploration_history:
        round_num = round_data.get("round", "?")
        evaluation = round_data.get("evaluation", {})
        status = evaluation.get("status", "?")
        plan = round_data.get("plan", {})
        sub_qs = round_data.get("sub_questions", [])

        # 判断是否触发 reflect
        is_reflect = False
        if isinstance(round_num, int) and round_num >= 2:
            prev = [h.get("evaluation", {}).get("status", "") for h in exploration_history if h.get("round", 0) < round_num]
            stuck_count = 0
            for s in reversed(prev):
                if s == "stuck":
                    stuck_count += 1
                else:
                    break
            is_reflect = stuck_count >= 2

        # 节点
        sub_questions = plan.get("sub_questions", [])
        solved = sum(1 for sq in sub_qs if sq.get("status") == "solved")

        m_lines.append(f'    R{round_num}P["R{round_num} Planner<br/>{len(sub_questions)} sub-Qs"]')
        m_lines.append(f'    style R{round_num}P fill:#6366f1,stroke:#333,color:white')

        m_lines.append(f'    R{round_num}S["R{round_num} Solver<br/>{solved} solved"]')
        m_lines.append(f'    style R{round_num}S fill:#0ea5e9,stroke:#333,color:white')

        if is_reflect:
            m_lines.append(f'    R{round_num}R["R{round_num} Reflect"]')
            m_lines.append(f'    style R{round_num}R fill:#ef4444,stroke:#333,color:white')
            m_lines.append(f'    R{round_num}S --> R{round_num}R')

        synth_color = "#22c55e" if status == "complete" else ("#ef4444" if status == "stuck" else "#f59e0b")
        m_lines.append(f'    R{round_num}Syn["R{round_num} Synth<br/>{status}"]')
        m_lines.append(f'    style R{round_num}Syn fill:{synth_color},stroke:#333,color:white')

        # 边
        m_lines.append(f'    R{round_num}P --> R{round_num}S')
        if not is_reflect:
            m_lines.append(f'    R{round_num}S --> R{round_num}Syn')
        else:
            m_lines.append(f'    R{round_num}R --> R{round_num + 1}P')
            m_lines.append(f'    R{round_num}Syn --> R{round_num}R')

    # 最终答案
    m_lines.append(f'    ANSWER["Final Answer<br/>{final_status}"]')
    m_lines.append(f'    style ANSWER fill:#22c55e,stroke:#333,color:white')
    last_round = exploration_history[-1].get("round", 1)
    m_lines.append(f'    R{last_round}Syn --> ANSWER')

    mermaid_code = "\n".join(m_lines)
    st.markdown(
        f'<script type="module">import "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";</script>',
        unsafe_allow_html=True,
    )
    st.html(f"""
    <div style="background:#f8f9fa;border-radius:12px;padding:16px;margin:8px 0;">
        <h4 style="margin:0 0 8px 0;">🤖 多角色智能体讨论流程 (Mermaid)</h4>
        <pre class="mermaid" style="background:white;padding:12px;border-radius:8px;">
{mermaid_code}
        </pre>
    </div>
    """, height=300 + len(exploration_history) * 150)


# ─── Prompt 默认模板映射 ────────────────────────────────────────

PROMPT_DEFAULTS = {
    "Naive RAG (Schema A)": {
        "Rewrite": QUERY_REWRITE_PROMPT,
        "Answer": RAG_SYS_PROMPT,
    },
    "RAG with Judge": {
        "Judge (Variant B)": JUDGE_PROMPT_B_TEMPLATE,
        "Answer": JUDGE_ANSWER_PROMPT,
    },
    "Agentic RAG V3": {
        "Planner": PLANNER_PROMPT_TEMPLATE,
        "SQ Judge": SQ_JUDGMENT_TEMPLATE,
        "SQ Answer": SQ_ANSWER_TEMPLATE,
        "Synthesizer": SYNTHESIZER_PROMPT_TEMPLATE,
        "Reflector": REFLECTOR_PROMPT_TEMPLATE,
        "Answer": AGENTIC_ANSWER_PROMPT,
    },
}

# config key 映射
PROMPT_CONFIG_KEYS = {
    "Rewrite": "custom_rewrite_prompt",
    "Answer": "custom_answer_prompt",
    "Judge (Variant B)": "custom_judge_prompt",
    "Planner": "custom_planner_prompt",
    "SQ Judge": "custom_sq_judge_prompt",
    "SQ Answer": "custom_sq_answer_prompt",
    "Synthesizer": "custom_synthesizer_prompt",
    "Reflector": "custom_reflector_prompt",
}


def _get_prompt_state_key(system: str, role: str) -> str:
    return f"prompt_{system}_{role}"


def render_prompt_editor(rag_system: str):
    """在 sidebar 渲染可折叠的 prompt 编辑器。"""
    st.divider()
    with st.expander("✏️ Prompt 模板配置", expanded=False):
        roles = PROMPT_DEFAULTS.get(rag_system, {})
        for role, default in roles.items():
            state_key = _get_prompt_state_key(rag_system, role)
            if state_key not in st.session_state:
                st.session_state[state_key] = default

            with st.container(border=True):
                st.caption(f"**{role}**")
                new_val = st.text_area(
                    f"编辑 {role} prompt",
                    value=st.session_state[state_key],
                    height=150,
                    key=f"ta_{state_key}",
                    label_visibility="collapsed",
                )
                st.session_state[state_key] = new_val
                cols = st.columns([1, 1])
                if cols[0].button("重置默认", key=f"reset_{state_key}"):
                    st.session_state[state_key] = default
                    st.rerun()


def get_custom_prompts_config(rag_system: str) -> dict:
    """从 session state 读取自定义 prompt，返回 config 字典。"""
    config = {}
    roles = PROMPT_DEFAULTS.get(rag_system, {})
    for role in roles:
        state_key = _get_prompt_state_key(rag_system, role)
        val = st.session_state.get(state_key, "")
        config_key = PROMPT_CONFIG_KEYS.get(role)
        if config_key and val and val != PROMPT_DEFAULTS[rag_system][role]:
            config[config_key] = val
    return config


# ─── Sidebar: Configuration ─────────────────────────────────────

with st.sidebar:
    st.header("系统配置")

    rag_system = st.selectbox(
        "选择 RAG 系统",
        ["Naive RAG (Schema A)", "RAG with Judge", "Agentic RAG V3"],
        help="选择使用的 RAG 架构",
    )

    st.divider()
    st.header("模型配置")

    api_key = st.text_input("API Key", value=os.getenv("BL_API_KEY", ""), type="password")
    base_url = st.text_input("Base URL", value=os.getenv("BL_BASE_URL", ""))

    # 根据系统显示不同的模型配置
    if rag_system == "Naive RAG (Schema A)":
        rewrite_model = st.text_input("重写模型", value="qwen-plus")
        answer_model = st.text_input("回答模型", value="qwen-max")
    elif rag_system == "RAG with Judge":
        rewrite_model = st.text_input("重写模型", value="qwen-plus")
        judge_model = st.text_input("Judge 模型", value="qwen-plus")
        answer_model = st.text_input("回答模型", value="qwen-max")
        max_depth = st.slider("最大递归深度", 1, 5, 3)
    else:  # Agentic RAG V3
        planner_model = st.text_input("Planner 模型", value="qwen-plus")
        evaluator_model = st.text_input("Evaluator 模型", value="qwen-max")
        answer_model = st.text_input("回答模型", value="qwen-max")
        reflector_model = st.text_input("Reflector 模型", value="qwen-max")
        max_rounds = st.slider("最大探索轮次", 1, 10, 5)

    st.divider()
    st.header("检索参数")
    max_chunks = st.slider("最大 Chunk 数", 3, 20, 8)

    if st.button("🗑️ 清除对话历史", type="secondary"):
        st.session_state.messages = []
        st.session_state.last_chunks = []

    # Prompt 模板编辑器
    render_prompt_editor(rag_system)


# ─── Helper: Create LLM ────────────────────────────────────────

def create_llm(model: str) -> ChatOpenAI:
    return ChatOpenAI(
        api_key=api_key or os.getenv("BL_API_KEY"),
        base_url=base_url or os.getenv("BL_BASE_URL"),
        model=model,
        temperature=0.0,
    )


# ─── Helper: Run Naive RAG ─────────────────────────────────────

def run_naive_rag(query: str, max_chunks: int, custom_prompts: dict):
    from naive_rag.workflow import get_workflow

    app = get_workflow(scheme="a", skip_suggest=True)
    retriever = WebRetriever(max_chunks=max_chunks)

    config = {
        "configurable": {
            "llm": create_llm(answer_model),
            "retriever": retriever,
            "rewrite_llm": create_llm(rewrite_model),
            "answer_llm": create_llm(answer_model),
            **custom_prompts,
        }
    }

    result = app.invoke(
        {"original_query": query, "messages": []},
        config=config,
    )

    return {
        "answer": result.get("answer", ""),
        "chunks": [doc for doc, _ in result.get("fused_chunks", [])],
        "rewritten_queries": result.get("rewritten_queries", []),
    }


# ─── Helper: Run RAG with Judge ─────────────────────────────────

def run_rag_with_judge(query: str, max_chunks: int, custom_prompts: dict):
    from rag_with_judge.nodes import rag_with_judge
    from rag_with_judge.workflow import build_judge_rag_graph

    app = build_judge_rag_graph()
    retriever = WebRetriever(max_chunks=max_chunks)

    config = {
        "configurable": {
            "llm": create_llm(answer_model),
            "retriever": retriever,
            "rewrite_llm": create_llm(rewrite_model),
            "judge_llm": create_llm(judge_model),
            "answer_llm": create_llm(answer_model),
            "max_chunks": max_chunks,
            "judge_variant": "B",
            **custom_prompts,
        }
    }

    search_path = {}
    answer = rag_with_judge(
        query=query,
        path=search_path,
        visited=set(),
        depth=0,
        max_depth=max_depth,
        app=app,
        config=config,
    )

    all_chunks = _collect_chunks_from_search_path(search_path)

    return {
        "answer": answer,
        "chunks": all_chunks,
        "search_path": search_path,
    }


def _collect_chunks_from_search_path(path: dict) -> list:
    """递归收集 SEARCH_PATH 中的所有 chunks。"""
    chunks = []
    for c in path.get("chunks", []):
        chunk_id = c.get("chunk_id", "")
        chunks.append({
            "chunk_id": chunk_id,
            "chunk_title": c.get("chunk_title", "Unknown"),
            "page_content": c.get("page_content", ""),
            "source_url": "",
        })

    children = path.get("next_queries", [])
    if isinstance(children, list):
        for child in children:
            if isinstance(child, dict):
                chunks.extend(_collect_chunks_from_search_path(child))

    return chunks


# ─── Helper: Run Agentic RAG V3 ─────────────────────────────────

def run_agentic_rag_v3(query: str, max_chunks: int, custom_prompts: dict):
    from agentic_rag_v3.nodes import run_agentic_rag_v3
    from agentic_rag_v3.workflow import build_agentic_rag_v3_graph

    app = build_agentic_rag_v3_graph()
    retriever = WebRetriever(max_chunks=max_chunks)

    config = {
        "configurable": {
            "llm": create_llm(evaluator_model),
            "retriever": retriever,
            "planner_llm": create_llm(planner_model),
            "evaluator_llm": create_llm(evaluator_model),
            "answer_llm": create_llm(answer_model),
            "reflector_llm": create_llm(reflector_model),
            "rewrite_llm": create_llm(answer_model),
            "synthesizer_llm": create_llm(answer_model),
            "max_chunks": max_chunks,
            "max_rounds": max_rounds,
            **custom_prompts,
        }
    }

    result = run_agentic_rag_v3(
        query=query,
        app=app,
        config=config,
        max_rounds=max_rounds,
    )

    formatted_chunks = []
    for chunk in result.get("all_chunks", []):
        formatted_chunks.append({
            "chunk_id": chunk.get("chunk_id", ""),
            "chunk_title": chunk.get("chunk_title", "Unknown"),
            "page_content": chunk.get("page_content", ""),
            "source_url": chunk.get("source_url", ""),
        })

    return {
        "answer": result.get("answer", ""),
        "chunks": formatted_chunks,
        "exploration_history": result.get("exploration_history", []),
        "sub_questions": result.get("sub_questions", []),
    }


# ─── Main: Chat Interface ──────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_chunks" not in st.session_state:
    st.session_state.last_chunks = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "chunks" in msg:
            with st.expander(f"📄 检索到的 {len(msg['chunks'])} 个来源", expanded=False):
                for i, chunk in enumerate(msg["chunks"], 1):
                    with st.container(border=True):
                        st.write(f"**{i}. {chunk.get('chunk_title', 'Unknown')}**")
                        if chunk.get("source_url"):
                            st.caption(f"[来源]({chunk['source_url']})")
                        st.write(chunk.get("page_content", "")[:500])

# Chat input
if prompt := st.chat_input("输入你的问题..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        viz_placeholder = st.empty()
        chunks_placeholder = st.empty()
        answer_placeholder = st.empty()

        start_time = time.time()

        try:
            status_placeholder.info("🔍 正在执行网络搜索...")

            # 获取自定义 prompt
            custom_prompts = get_custom_prompts_config(rag_system)

            # 根据选择的系统运行
            if rag_system == "Naive RAG (Schema A)":
                result = run_naive_rag(prompt, max_chunks, custom_prompts)
            elif rag_system == "RAG with Judge":
                status_placeholder.info("🔍 正在执行递归网络探索...")
                result = run_rag_with_judge(prompt, max_chunks, custom_prompts)
            else:
                status_placeholder.info("🔍 正在执行规划-执行-反馈闭环...")
                result = run_agentic_rag_v3(prompt, max_chunks, custom_prompts)

            elapsed = time.time() - start_time

            # 格式化 chunks
            chunks = []
            for chunk in result.get("chunks", []):
                if isinstance(chunk, dict):
                    chunks.append({
                        "chunk_id": chunk.get("chunk_id", ""),
                        "chunk_title": chunk.get("chunk_title", "Unknown"),
                        "page_content": chunk.get("page_content", ""),
                        "source_url": chunk.get("source_url", ""),
                    })

            status_placeholder.success(
                f"✅ 完成 ({elapsed:.1f}s) — 检索到 {len(chunks)} 个来源"
            )

            # ─── 渲染可视化 (Judge 搜索树 / Agentic 时间线) ───
            if rag_system == "RAG with Judge" and result.get("search_path"):
                viz_placeholder.markdown("### 递归探索可视化")
                render_search_tree_detail(result["search_path"], max_depth=max_depth)

            elif rag_system == "Agentic RAG V3" and result.get("exploration_history"):
                viz_placeholder.markdown("### 多角色智能体讨论时间线")
                final_status = result.get("exploration_history", [])[-1].get("evaluation", {}).get("status", "?")
                render_timeline(result["exploration_history"], result.get("sub_questions", []), final_status)

            # 显示检索来源
            if chunks:
                with chunks_placeholder.expander(f"📄 查看 {len(chunks)} 个检索来源", expanded=False):
                    for i, chunk in enumerate(chunks, 1):
                        with st.container(border=True):
                            st.write(f"**{i}. {chunk['chunk_title']}**")
                            if chunk.get("source_url"):
                                st.caption(f"[{chunk['source_url']}]({chunk['source_url']})")
                            st.write(chunk["page_content"][:500])

            # 显示答案
            answer = result.get("answer", "")
            answer_placeholder.markdown(f"**答案：**\n\n{answer}")

            # 保存消息
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "chunks": chunks,
            })
            st.session_state.last_chunks = chunks

        except Exception as e:
            status_placeholder.error(f"❌ 执行失败: {e}")
            logger.exception("RAG execution failed")


# ─── Footer ─────────────────────────────────────────────────────
st.divider()
st.caption("Powered by DuckDuckGo Search + LangGraph RAG Systems")
