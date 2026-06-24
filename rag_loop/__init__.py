# rag_loop — DAG-based plan-execute-feedback loop RAG algorithm
#
# 基于 DAG 世界模型的升级版闭环 RAG 算法（Path A）。
# 核心控制流: Planner → Solver → Critic → 终止判断或下一轮。
#
# 与 agentic_rag_v3 的关键差异:
#   - DAG 作为"世界镜像"，逼近问题的理想分解结构
#   - Critic 双重审查: 答案质量 + 结构质量
#   - Planner 声明式结构假设（目标状态 + delta）
#   - 4 条件终止（含结构收敛条件）
#   - 系统中介对话线程（Planner/Critic 独立 LLM 对话）
#   - 拓扑波次求解（依赖硬约束，分解软约束）
#
# 主要入口:
#   run_pipeline(q, search_fn, model) -> PipelineResult
#
# 外部适配:
#   MilvusAdapter 将 MilvusRetriever 适配为 SearchFn Protocol

from rag_loop.models import (
    DAG,
    DAGNode,
    DAGEdge,
    ChunkInfo,
    NodeStatus,
    CriticHealth,
    EdgeType,
)
from rag_loop.interfaces import SearchFn
from rag_loop.invariants import check_invariants, InvariantViolation
from rag_loop.operations import (
    apply_primitives,
    ApplyResult,
    InheritPrimitive,
    InheritAndRelabelPrimitive,
    InitializePrimitive,
    LinkPrimitive,
)
from rag_loop.solver import solve_dag, find_ready_nodes, prepare_node_query
from rag_loop.critic import (
    run_critic,
    apply_critic_output,
    CriticOutput,
    NodeReview,
    TerminationJudgment,
)
from rag_loop.planner import (
    run_planner,
    planner_output_to_primitives,
    PlannerOutput,
)
from rag_loop.answer_generator import generate_answer
from rag_loop.adapters import MilvusAdapter, WebAdapter
from rag_loop.pipeline import (
    run_pipeline,
    PipelineConfig,
    PipelineResult,
    ExperimentRecord,
    RoundRecord,
    SearchCallRecord,
)
from rag_loop.structured_output import get_structured_model