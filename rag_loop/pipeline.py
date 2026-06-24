# ═══════════════════════════════════════════════════════════════════
# 主控制循环
# ═══════════════════════════════════════════════════════════════════
# 集成 Planner、Solver、Critic、答案生成器为一个完整的闭环控制循环。
#
# 控制流（FRAMEWORK.md 模块 5）:
#   Planner → 验证器(≤3修订) → Solver → Critic → 终止判断或下一轮
#
# 对话线程管理（FRAMEWORK.md 模块 3）:
#   Planner 和 Critic 各自独立 LLM 对话线程。
#   系统作为中介，从结构化字段提取信息注入 USER 消息。
#   当前轮 full detail，历史轮 summary，保留最近 3 轮。

from dataclasses import dataclass, field
import copy
from datetime import datetime, timezone

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage

from .models import DAG, DAGNode, ChunkInfo, NodeStatus
from .interfaces import SearchFn
from .operations import ApplyResult, InvariantViolation, PrimitiveError
from .operations import apply_primitives
from .solver import solve_dag
from .critic import CriticOutput, run_critic, apply_critic_output, CRITIC_SYSTEM_TEMPLATE
from .critic import build_critic_user_message
from .planner import PlannerOutput, run_planner, planner_output_to_primitives, PLANNER_SYSTEM_TEMPLATE
from .planner import build_planner_user_message
from .answer_generator import generate_answer
from .structured_output import get_structured_model


# ═══════════════════════════════════════════════════════════════════
# 实验记录数据结构
# ═══════════════════════════════════════════════════════════════════


@dataclass
class SearchCallRecord:
    """单次搜索调用的完整记录——供离线分析。"""
    query: str
    chunk_ids: list[str]
    chunk_count: int


@dataclass
class RoundRecord:
    """单轮 Pipeline 执行的完整记录。"""
    round_number: int
    planner_input_dag: dict | None = None       # DAG 快照（Planner 输入）
    planner_output: dict | None = None           # PlannerOutput.model_dump()
    planner_revisions: int = 0
    planner_errors: list[dict] = field(default_factory=list)
    planner_success: bool = False
    solver_search_count: int = 0
    solver_search_calls: list[SearchCallRecord] = field(default_factory=list)
    critic_output: dict | None = None            # CriticOutput.model_dump()
    result_dag: dict | None = None               # DAG 快照（Critic 完成后）
    topology_snapshot: dict | None = None


@dataclass
class ExperimentRecord:
    """完整实验运行记录——包含离线分析所需的所有数据。"""
    question: str
    config: dict = field(default_factory=dict)
    started_at: str = ""
    rounds: list[RoundRecord] = field(default_factory=list)
    # 完整对话线程（不截断，保存所有轮次——供分析用）
    planner_full_thread: list[dict] = field(default_factory=list)
    critic_full_thread: list[dict] = field(default_factory=list)
    # 所有搜索调用（跨轮次汇总）
    all_search_calls: list[SearchCallRecord] = field(default_factory=list)
    # 最终结果
    final_answer: str = ""
    termination_reason: str = ""
    total_rounds: int = 0
    total_search_calls: int = 0

    def to_json(self, path: str) -> None:
        """将实验记录序列化为 JSON 文件。"""
        import json
        record_dict = {
            "question": self.question,
            "config": self.config,
            "started_at": self.started_at,
            "total_rounds": self.total_rounds,
            "total_search_calls": self.total_search_calls,
            "termination_reason": self.termination_reason,
            "final_answer": self.final_answer,
            "rounds": _serialize_rounds(self.rounds),
            "planner_full_thread": self.planner_full_thread,
            "critic_full_thread": self.critic_full_thread,
            "all_search_calls": [
                {"query": c.query, "chunk_ids": c.chunk_ids, "chunk_count": c.chunk_count}
                for c in self.all_search_calls
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record_dict, f, ensure_ascii=False, indent=2)


class TracedSearchFn:
    """包装 SearchFn，记录每次搜索调用的 query 和返回的 chunk IDs。

    同时将调用记录追加到 ExperimentRecord.all_search_calls
    和当前 RoundRecord.solver_search_calls。
    """

    def __init__(self, inner: SearchFn, record: ExperimentRecord):
        self._inner = inner
        self._record = record

    def __call__(self, query: str) -> list[ChunkInfo]:
        chunks = self._inner(query)
        call = SearchCallRecord(
            query=query,
            chunk_ids=[c.chunk_id for c in chunks],
            chunk_count=len(chunks),
        )
        self._record.all_search_calls.append(call)
        # 追加到当前轮次记录
        if self._record.rounds:
            self._record.rounds[-1].solver_search_calls.append(call)
        return chunks


def _serialize_rounds(rounds: list[RoundRecord]) -> list[dict]:
    """将 RoundRecord 列表序列化为 JSON 兼容的 dict 列表。"""
    result = []
    for r in rounds:
        result.append({
            "round_number": r.round_number,
            "planner_success": r.planner_success,
            "planner_revisions": r.planner_revisions,
            "planner_errors": r.planner_errors,
            "solver_search_count": r.solver_search_count,
            "solver_search_calls": [
                {"query": c.query, "chunk_ids": c.chunk_ids, "chunk_count": c.chunk_count}
                for c in r.solver_search_calls
            ],
            "planner_input_dag": r.planner_input_dag,
            "planner_output": r.planner_output,
            "critic_output": r.critic_output,
            "result_dag": r.result_dag,
            "topology_snapshot": r.topology_snapshot,
        })
    return result


# ═══════════════════════════════════════════════════════════════════
# 配置与结果
# ═══════════════════════════════════════════════════════════════════


@dataclass
class PipelineConfig:
    """Pipeline 运行配置——所有参数有合理默认值。"""
    max_rounds: int = 5
    max_revisions: int = 3
    max_history_rounds: int = 3
    max_consecutive_planner_failures: int = 2


@dataclass
class PipelineResult:
    """完整 Pipeline 运行结果。"""
    answer: str
    final_dag: DAG
    total_rounds: int
    total_search_calls: int
    termination_reason: str  # "all_conditions_met" | "max_rounds" | "planner_failure"
    planner_thread: list[BaseMessage] = field(default_factory=list)
    critic_thread: list[BaseMessage] = field(default_factory=list)
    round_dags: list[DAG] = field(default_factory=list)  # 每轮 Critic 完成后的 DAG 快照


# ═══════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════


def _format_validation_errors(
    errors: list[InvariantViolation | PrimitiveError],
    revision: int,
) -> str:
    """将验证错误列表格式化为 Planner 可理解的自然语言修订请求。

    错误分为三类，每类有不同的修正指引：
    - 前置条件失败：引用不存在的节点 → 检查 node_id 拼写
    - 一致性检查失败：节点/边未被覆盖 → 补上 INHERIT/LINK 或加入删除列表
    - 不变式违规：结构逻辑问题 → 重新设计拓扑
    """
    lines = [
        f"═══ 验证失败（第 {revision} 次修订） ═══",
        "",
        "你的原语序列未通过系统验证。请根据以下具体违规修正后重新输出。",
        "",
    ]

    precondition_errors = [
        e for e in errors
        if isinstance(e, PrimitiveError) and e.primitive_index >= 0
    ]
    consistency_errors = [
        e for e in errors
        if isinstance(e, PrimitiveError) and e.primitive_index == -1
    ]
    invariant_errors = [
        e for e in errors
        if isinstance(e, InvariantViolation)
    ]

    if precondition_errors:
        lines.append("【前置条件失败——引用或参数错误】")
        for e in precondition_errors:
            lines.append(f"  • {e.description}")
        lines.append("  → 修正方向：检查原语中引用的 node_id 是否正确，参数是否合法。")
        lines.append("")

    if consistency_errors:
        lines.append("【一致性检查失败——旧 DAG 元素未被处理】")
        for e in consistency_errors:
            lines.append(f"  • {e.description}")
        lines.append("  → 修正方向：将遗漏的节点/边加入原语序列（INHERIT/INHERIT_AND_RELABEL/LINK）或删除列表（deleted_nodes/deleted_edges）。")
        lines.append("")

    if invariant_errors:
        lines.append("【结构不变式违规——DAG 拓扑逻辑问题】")
        for e in invariant_errors:
            lines.append(f"  • [{e.invariant}] {e.description}")
        lines.append("  → 修正方向：重新设计 DAG 结构以满足 I1-I7 全部不变式。特别注意环检测和根节点约束。")
        lines.append("")

    lines.append("请修正后重新输出完整的 PlannerOutput（包含所有字段，不只是修正部分）。")

    return "\n".join(lines)


def _format_planner_failure(
    errors: list[InvariantViolation | PrimitiveError],
    round_number: int,
    max_revisions: int,
) -> str:
    """Planner 全部修订失败后的摘要，注入 planner_thread。"""
    error_text = _format_validation_errors(errors, revision=max_revisions)
    return (
        f"第 {round_number} 轮规划未通过验证（{max_revisions} 次修订耗尽）。\n\n"
        f"最终违规：\n{error_text}\n\n"
        f"DAG 回退至第 {round_number - 1} 轮状态。"
        f"下一轮 Planner 需要采用不同的分解策略——当前结构方向存在逻辑问题。"
    )


def _compute_topology_snapshot(dag: DAG) -> tuple[frozenset, frozenset]:
    """计算 DAG 的拓扑快照——用于条件④（结构收敛）判断。

    快照包含：
    - 节点: (node_id, question) 对
    - 边: (from_id, to_id, edge_type) 三元组

    不包含 answer、chunks、critic_*、planner_rationale、元数据——
    这些是"观测"和"解释"，不是"结构认知"。
    """
    nodes = frozenset((nid, node.question) for nid, node in dag.nodes.items())
    edges = frozenset(
        (e.from_id, e.to_id, e.edge_type.value) for e in dag.edges
    )
    return (nodes, edges)


def _dag_to_dict(dag: DAG) -> dict:
    """将 DAG 序列化为 JSON 兼容的 dict——供 ExperimentRecord 使用。"""
    return {
        "q": dag.q,
        "round_number": dag.round_number,
        "nodes": {
            nid: {
                "id": node.id,
                "question": node.question,
                "answer": node.answer,
                "status": node.status.value,
                "health": node.critic_health.value if node.critic_health else None,
                "search_query": node.search_query,
                "planner_rationale": node.planner_rationale,
                "solver_judgment": node.solver_judgment,
                "critic_factual_notes": node.critic_factual_notes,
                "critic_normative_advice": node.critic_normative_advice,
                "round_created": node.round_created,
                "round_last_updated": node.round_last_updated,
                "supporting_chunks_count": len(node.supporting_chunks),
                "retrieved_chunks_count": len(node.retrieved_chunks),
                "supporting_chunks": node.supporting_chunks,
                "retrieved_chunks_summary": [
                    {
                        "chunk_id": c.chunk_id,
                        "chunk_title": c.chunk_title,
                        "chunk_summary": c.chunk_summary,
                        "context_title": c.context_title,
                        "page_content": c.page_content,
                    }
                    for c in node.retrieved_chunks
                ],
            }
            for nid, node in dag.nodes.items()
        },
        "edges": [
            {"from": e.from_id, "to": e.to_id, "type": e.edge_type.value}
            for e in dag.edges
        ],
    }


def _check_condition_1(dag: DAG) -> bool:
    """终止条件①：除根节点外所有节点 status == SOLVED。"""
    root = dag.root
    root_id = root.id if root else None
    for nid, node in dag.nodes.items():
        if nid == root_id:
            continue
        if node.status != NodeStatus.SOLVED:
            return False
    return True


def _check_condition_4(
    topology_history: list[tuple[frozenset, frozenset]],
) -> bool:
    """终止条件④：DAG 拓扑连续两轮无变化。

    需要 2 个连续快照 → 1 次比较确认无变化。
    """
    if len(topology_history) < 2:
        return False
    return topology_history[-1] == topology_history[-2]


def _trim_thread(
    thread: list[BaseMessage],
    max_history_rounds: int,
) -> list[BaseMessage]:
    """保留 SystemMessage + 最近 N 轮的 USER/AI 消息对。

    每轮 = 1 HumanMessage + 1 AIMessage = 2 条消息。
    """
    system_msg = thread[0]  # SystemMessage 始终保留
    pairs = thread[1:]       # USER/AI 对
    recent_pairs = pairs[-(max_history_rounds * 2):]
    return [system_msg] + recent_pairs


# ═══════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════


def run_pipeline(
    q: str,
    search_fn: SearchFn,
    model,  # BaseChatModel — fallback for all roles
    config: PipelineConfig = PipelineConfig(),
    record: ExperimentRecord | None = None,
    role_models: dict | None = None,
    role_methods: dict[str, str] | None = None,
    custom_system_prompts: dict[str, str] | None = None,
    solver_prepare_prompt=None,  # 可选的自定义 ChatPromptTemplate（如网络搜索场景）
) -> PipelineResult:
    """运行完整的 Agentic RAG Pipeline 闭环。

    Args:
        q: 原始问题
        search_fn: 搜索接口——MilvusAdapter 或 mock lambda
        model: LangChain BaseChatModel——所有角色的默认模型
        config: 运行配置
        record: 可选的实验记录——填充完整的离线分析数据
        role_models: 可选的分角色模型覆盖，key ∈ {planner, critic, rewrite, solver, answer}。
                     未指定的角色回退到 model。
        role_methods: 可选的分角色 structured_output method，
                      key ∈ {planner, critic, rewrite, solver, answer}，
                      value ∈ {"function_calling", "json_mode"}。
                      未指定的角色回退到 "function_calling"。
                      当角色启用 enable_thinking 时需设为 "json_mode"。
        custom_system_prompts: 可选的自定义系统提示词，
                      key ∈ {planner, critic, answer}。
                      未指定的角色使用默认模板。

    Returns:
        PipelineResult——最终答案、DAG、统计、对话线程
    """

    # ── 模型解析辅助函数 ──
    def _m(role: str):
        """Resolve model for role, falling back to default model."""
        if role_models and role in role_models:
            return role_models[role]
        return model

    def _method(role: str) -> str:
        """Resolve structured_output method for role.

        rewrite/solver 默认使用 json_mode 而非 function_calling:
        - function_calling 设置 tool_choice，Qwen3 默认 thinking 模式与之冲突
        - json_mode 使用 response_format (json_object)，兼容 thinking 模式
        - Eval 通过 role_methods 显式覆盖所有角色，不受默认值影响
        - planner/critic/answer 保留 function_calling 默认（schema 约束更重要）
        """
        if role_methods and role in role_methods:
            return role_methods[role]
        if role in ("rewrite", "solver"):
            return "json_mode"
        return "function_calling"

    # ═══════════════════════════════════════════════════════════════
    # 初始化
    # ═══════════════════════════════════════════════════════════════

    dag = DAG(
        nodes={
            "N0": DAGNode(
                id="N0",
                question=q,
                planner_rationale="根节点——Q 的最终锚点",
                round_created=0,
                round_last_updated=0,
            )
        },
        edges=[],
        round_number=0,
        q=q,
    )

    planner_thread: list[BaseMessage] = [
        SystemMessage(content=custom_system_prompts.get("planner", PLANNER_SYSTEM_TEMPLATE)
                      if custom_system_prompts else PLANNER_SYSTEM_TEMPLATE)
    ]
    critic_thread: list[BaseMessage] = [
        SystemMessage(content=custom_system_prompts.get("critic", CRITIC_SYSTEM_TEMPLATE)
                      if custom_system_prompts else CRITIC_SYSTEM_TEMPLATE)
    ]

    # 完整对话线程——永不被截断，供离线分析
    planner_full: list[dict] = []
    critic_full: list[dict] = []

    # 实验记录初始化
    if record is not None:
        record.question = q
        record.started_at = datetime.now(timezone.utc).isoformat()
        record.config = {
            "max_rounds": config.max_rounds,
            "max_revisions": config.max_revisions,
            "max_history_rounds": config.max_history_rounds,
            "max_consecutive_planner_failures": config.max_consecutive_planner_failures,
        }

    previous_hypothesis_part1 = ""
    previous_hypothesis_part2 = ""
    critic_planner_guidance = ""
    previous_dag: DAG | None = None
    topology_history: list[tuple[frozenset, frozenset]] = []
    round_dags: list[DAG] = []

    # 搜索追踪——包装 search_fn
    if record is not None:
        search_fn = TracedSearchFn(search_fn, record)

    total_search_calls = 0
    round_number = 1
    consecutive_planner_failures = 0
    planner_output: PlannerOutput | None = None

    # ═══════════════════════════════════════════════════════════════
    # 主循环
    # ═══════════════════════════════════════════════════════════════

    while round_number <= config.max_rounds:
        # ── 记录：本轮开始 ──
        round_record = RoundRecord(round_number=round_number)
        if record is not None:
            round_record.planner_input_dag = _dag_to_dict(dag)
            record.rounds.append(round_record)

        # ── 1. PLANNER PHASE ──
        current_user_msg = build_planner_user_message(
            dag, critic_planner_guidance,
            previous_hypothesis_part1, previous_hypothesis_part2,
            detail_level="full",
        )

        phase_messages: list[BaseMessage] = [
            HumanMessage(content=current_user_msg)
        ]

        planner_output = run_planner(
            dag, critic_planner_guidance,
            previous_hypothesis_part1, previous_hypothesis_part2,
            _m("planner"), detail_level="full",
            structured_output_method=_method("planner"),
        )
        phase_messages.append(AIMessage(content=planner_output.model_dump_json()))

        # 验证 + 修订循环
        validation_passed = False
        for revision in range(1, config.max_revisions + 1):
            try:
                primitives, del_nodes, del_edges = planner_output_to_primitives(
                    planner_output, round_number=round_number
                )
            except ValueError as e:
                # 占位符错误——特殊处理
                error_text = (
                    f"═══ 验证失败（第 {revision} 次修订） ═══\n\n"
                    f"【占位符引用错误】\n"
                    f"  • {e}\n\n"
                    f"请修正占位符引用后重新输出。"
                )
                if revision == config.max_revisions:
                    break
                phase_messages.append(HumanMessage(content=error_text))
                planner_output = get_structured_model(_m("planner"), PlannerOutput, _method("planner")).invoke(
                    phase_messages
                )
                phase_messages.append(AIMessage(content=planner_output.model_dump_json()))
                continue

            result: ApplyResult = apply_primitives(
                dag, primitives, del_nodes, del_edges
            )

            if not result.errors:
                validation_passed = True
                break

            if revision == config.max_revisions:
                break

            error_text = _format_validation_errors(result.errors, revision)
            phase_messages.append(HumanMessage(content=error_text))
            planner_output = get_structured_model(_m("planner"), PlannerOutput, _method("planner")).invoke(
                phase_messages
            )
            phase_messages.append(AIMessage(content=planner_output.model_dump_json()))

        if not validation_passed:
            # Planner 失败
            consecutive_planner_failures += 1
            failure_msg = _format_planner_failure(
                result.errors if 'result' in dir() and result.errors
                else [PrimitiveError(-1, None, "所有修订耗尽")],
                round_number, config.max_revisions,
            )
            planner_thread.append(HumanMessage(content=failure_msg))

            if consecutive_planner_failures >= config.max_consecutive_planner_failures:
                if record is not None:
                    record.final_answer = "CANNOT_ANSWER"
                    record.termination_reason = "planner_failure"
                    record.total_rounds = round_number
                    record.total_search_calls = total_search_calls
                    record.planner_full_thread = planner_full
                    record.critic_full_thread = critic_full
                return PipelineResult(
                    answer="CANNOT_ANSWER",
                    final_dag=dag,
                    total_rounds=round_number,
                    total_search_calls=total_search_calls,
                    termination_reason="planner_failure",
                    planner_thread=planner_thread,
                    critic_thread=critic_thread,
                    round_dags=round_dags,
                )

            round_number += 1
            continue

        # Planner 成功
        consecutive_planner_failures = 0
        new_dag = result.dag

        # ── 记录：Planner 输出 ──
        if record is not None:
            round_record.planner_output = planner_output.model_dump(mode="json")
            round_record.planner_success = True
            # 保存完整对话（非退化）到 full thread
            planner_full.append({"role": "user", "content": current_user_msg})
            planner_full.append({"role": "assistant", "content": planner_output.model_dump_json()})

        # 存储退化消息到 planner_thread
        degraded_user_msg = build_planner_user_message(
            dag, critic_planner_guidance,
            previous_hypothesis_part1, previous_hypothesis_part2,
            detail_level="summary",
        )
        planner_thread.append(HumanMessage(content=degraded_user_msg))
        planner_thread.append(AIMessage(content=planner_output.model_dump_json()))
        planner_thread = _trim_thread(planner_thread, config.max_history_rounds)

        # ── 2. SOLVER PHASE ──
        search_count = solve_dag(new_dag, search_fn, model,
                               rewrite_model=_m("rewrite"), solver_model=_m("solver"),
                               rewrite_method=_method("rewrite"), solver_method=_method("solver"),
                               custom_prepare_prompt=solver_prepare_prompt)
        total_search_calls += search_count

        # ── 3. CRITIC PHASE ──
        critic_user_msg = build_critic_user_message(
            new_dag,
            planner_output.hypothesis_part1,
            planner_output.hypothesis_part2,
            previous_dag,
            detail_level="full",
        )
        critic_output = run_critic(
            new_dag,
            planner_output.hypothesis_part1,
            planner_output.hypothesis_part2,
            previous_dag,
            _m("critic"),
            detail_level="full",
            structured_output_method=_method("critic"),
        )
        apply_critic_output(new_dag, critic_output)

        # 存储退化消息到 critic_thread
        degraded_critic_msg = build_critic_user_message(
            new_dag,
            planner_output.hypothesis_part1,
            planner_output.hypothesis_part2,
            previous_dag,
            detail_level="summary",
        )
        critic_thread.append(HumanMessage(content=degraded_critic_msg))
        critic_thread.append(AIMessage(content=critic_output.model_dump_json()))
        critic_thread = _trim_thread(critic_thread, config.max_history_rounds)

        # ── 记录：Critic 输出 + 本轮完成 ──
        if record is not None:
            critic_full.append({"role": "user", "content": critic_user_msg})
            critic_full.append({"role": "assistant", "content": critic_output.model_dump_json()})
            round_record.critic_output = critic_output.model_dump(mode="json")
            round_record.solver_search_count = search_count
            round_record.result_dag = _dag_to_dict(new_dag)
            round_record.topology_snapshot = {
                "nodes": sorted(list(topology_history[-1][0])) if topology_history else [],
                "edges": sorted(list(topology_history[-1][1])) if topology_history else [],
            }

        # ── 更新跨轮状态 ──
        previous_hypothesis_part1 = planner_output.hypothesis_part1
        previous_hypothesis_part2 = planner_output.hypothesis_part2
        critic_planner_guidance = critic_output.planner_guidance
        previous_dag = new_dag

        # 保存本轮 DAG 快照
        round_dags.append(copy.deepcopy(new_dag))

        topology_history.append(_compute_topology_snapshot(new_dag))
        if len(topology_history) > 2:
            topology_history.pop(0)

        # ── 4. TERMINATION CHECK ──
        condition_1 = _check_condition_1(new_dag)
        condition_2 = critic_output.termination.condition_2_passed
        condition_3 = critic_output.termination.condition_3_passed
        condition_4 = _check_condition_4(topology_history)

        if condition_1 and condition_2 and condition_3 and condition_4:
            answer = generate_answer(new_dag, _m("answer"),
                                     custom_system_prompt=custom_system_prompts.get("answer")
                                     if custom_system_prompts else None)
            if answer:
                new_dag.root.answer = answer
                new_dag.root.status = NodeStatus.SOLVED
            if record is not None:
                record.final_answer = answer
                record.termination_reason = "all_conditions_met"
                record.total_rounds = round_number
                record.total_search_calls = total_search_calls
                record.planner_full_thread = planner_full
                record.critic_full_thread = critic_full
            return PipelineResult(
                answer=answer,
                final_dag=new_dag,
                total_rounds=round_number,
                total_search_calls=total_search_calls,
                termination_reason="all_conditions_met",
                planner_thread=planner_thread,
                critic_thread=critic_thread,
                round_dags=round_dags,
            )

        dag = new_dag
        round_number += 1

    # ═══════════════════════════════════════════════════════════════
    # 强制终止（round > max_rounds）
    # ═══════════════════════════════════════════════════════════════

    # 最后一次 Critic 审查——异常保护：超时不应丢失已积累的全部工作
    if planner_output is not None:
        try:
            critic_output = run_critic(
                dag,
                planner_output.hypothesis_part1,
                planner_output.hypothesis_part2,
                previous_dag,
                _m("critic"),
                detail_level="full",
                structured_output_method=_method("critic"),
            )
            apply_critic_output(dag, critic_output)
        except Exception:
            logger.warning("强制终止 Critic 调用失败，跳过，直接生成最终答案",
                           exc_info=True)

    answer = generate_answer(dag, _m("answer"),
                         custom_system_prompt=custom_system_prompts.get("answer")
                         if custom_system_prompts else None)
    if answer and answer != "CANNOT_ANSWER":
        root = dag.root
        if root:
            root.answer = answer
            root.status = NodeStatus.SOLVED

    if record is not None:
        record.final_answer = answer
        record.termination_reason = "max_rounds"
        record.total_rounds = round_number - 1
        record.total_search_calls = total_search_calls
        record.planner_full_thread = planner_full
        record.critic_full_thread = critic_full

    return PipelineResult(
        answer=answer,
        final_dag=dag,
        total_rounds=round_number - 1,
        total_search_calls=total_search_calls,
        termination_reason="max_rounds",
        planner_thread=planner_thread,
        critic_thread=critic_thread,
        round_dags=round_dags,
    )