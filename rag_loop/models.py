# ═══════════════════════════════════════════════════════════════════
# 核心数据类型
# ═══════════════════════════════════════════════════════════════════
# 枚举、ChunkInfo、DAGNode、DAGEdge、DAG —— 算法的基础数据结构。
# 无外部依赖，所有其他模块依赖此文件。

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ═══════════════════════════════════════════════════════════════════
# 枚举
# ═══════════════════════════════════════════════════════════════════


class NodeStatus(Enum):
    """节点的求解状态——Solver 负责写入"""
    UNSOLVED = "unsolved"
    SOLVED = "solved"


class CriticHealth(Enum):
    """节点的健康判词——Critic 负责写入

    HEALTHY 不要求硬性 ≥2 独立来源。Critic 综合判断证据充分性：
    多独立来源一致 = 强正向信号；单来源但直接具体陈述 = 可接受（需标注）。
    """
    HEALTHY = "healthy"
    NEEDS_VERIFICATION = "needs_verification"
    UNRELIABLE = "unreliable"
    BLOCKED = "blocked"


class EdgeType(Enum):
    """边的类型——Planner 通过 LINK 原语声明

    decomposition: Y 是 X 的子事实，求解 X 时 Y 的 answer 汇总为搜索上下文
    dependency:   求解 X 需要 Y 的 answer 消解指代（注入查询改写）
    同一节点对可同时有两条边。
    """
    DECOMPOSITION = "decomposition"
    DEPENDENCY = "dependency"


# ═══════════════════════════════════════════════════════════════════
# 数据类
# ═══════════════════════════════════════════════════════════════════


@dataclass
class ChunkInfo:
    """搜索返回的单个 chunk——与 MilvusRetriever 返回结构对齐

    仅保留算法有操作意义的 5 个字段。context_index 有意排除——
    算法通过 search 获取 chunk，无法得知相邻 chunk 内容，
    故位置索引没有操作意义。aggregated_propositions 有意排除——
    反映的是离线索引的语义聚合策略，而非检索结果的质量。
    """
    chunk_id: str
    chunk_title: str
    chunk_summary: str          # 100-200 字摘要，供 Planner/Critic 快速浏览
    context_title: str          # 所属文档标题——Critic 独立性检查的核心依据
    page_content: str           # chunk 全文


@dataclass
class DAGNode:
    """内部 DAG 节点——算法对理想 DAG 中一个事实的认知替代物

    ID 命名规则: N{round}_{seq} 格式（如 N1_1, N2_3）。INITIALIZE 分配新 ID，
    由 planner_output_to_primitives 按占位符 $1,$2,... → N{round}_{seq} 解析。
    INHERIT / INHERIT_AND_RELABEL 保留原 ID。首轮初始根节点为 "N0"。
    round 标识首次创建的轮次，seq 为该轮内 INITIALIZE 的顺序号。

    字段按负责角色分组，与 FRAMEWORK.md 模块 2 规格严格对应：
    - Planner: id, question, planner_rationale
    - Solver:  status, retrieved_chunks, supporting_chunks, answer, solver_judgment, search_query
    - Critic:  critic_health, critic_factual_notes, critic_normative_advice
    - 系统:    round_created, round_last_updated
    """
    # ── Planner 负责 ──
    id: str
    question: str
    planner_rationale: str

    # ── Solver 负责（初始为空/默认值）──
    status: NodeStatus = NodeStatus.UNSOLVED
    retrieved_chunks: list[ChunkInfo] = field(default_factory=list)
    supporting_chunks: list[str] = field(default_factory=list)
    answer: str = ""
    solver_judgment: str = ""
    search_query: str = ""

    # ── Critic 负责（初始为空/None）──
    critic_health: Optional[CriticHealth] = None
    critic_factual_notes: str = ""
    critic_normative_advice: str = ""

    # ── 系统维护 ──
    round_created: int = 0
    round_last_updated: int = 0


@dataclass
class DAGEdge:
    """有向边——Planner 对事实间关系的假设

    方向: from_id → to_id
    """
    from_id: str
    to_id: str
    edge_type: EdgeType
    round_created: int = 0


@dataclass
class DAG:
    """完整的内部 DAG——算法的世界画像

    纯数据容器。不包含验证方法——所有验证逻辑集中在 invariants.py
    和 operations.py 中。

    同时承载三重职能（FRAMEWORK.md 模块 2）：
    1. 搜索策略——拓扑 = 下一步搜什么方向
    2. 观测记录——每个节点内的 chunks + answer + judgment
    3. 审查基础——Critic 基于完整内容和结构做判断
    """
    nodes: dict[str, DAGNode] = field(default_factory=dict)
    edges: list[DAGEdge] = field(default_factory=list)
    round_number: int = 0
    q: str = ""

    @property
    def root(self) -> Optional[DAGNode]:
        """唯一根节点（入度为 0 的节点，I4 要求恰好一个）

        多个组件需要此查询——不变式检查器(I4/I6)、Critic(终止条件②)、
        答案生成器(合成起点)。
        返回 None 当且仅当根节点不唯一（0 或 ≥2 个）。
        """
        indegrees = {nid: 0 for nid in self.nodes}
        for edge in self.edges:
            indegrees[edge.to_id] = indegrees.get(edge.to_id, 0) + 1
        roots = [nid for nid, deg in indegrees.items() if deg == 0]
        if len(roots) == 1:
            return self.nodes[roots[0]]
        return None