# rag_loop — 评估器
# 接入 Eval 的 checkpoint 系统，继承 BaseEvaluator。
# 每个问题独立运行 rag_loop DAG 闭环（Planner → Solver → Critic），收集检索指标和效率指标。

import logging
import os

from Eval.base import BaseEvaluator, NormalizedQuestion
from Eval.metrics import (
    extract_supporting_titles,
    compute_context_recall,
    compute_hit,
    compute_mrr,
    compute_retrieval_precision,
)

logger = logging.getLogger(__name__)


class RAGLoopEvaluator(BaseEvaluator):
    """rag_loop DAG 闭环评估器。

    每个问题独立创建 Pipeline 实例，避免状态污染。
    支持 checkpoint 断点续跑，默认启用分题 JSON 存储。
    """

    def __init__(
        self,
        llm,
        dataset_type: str,
        batch_size: int = 20,
        max_workers: int = 5,
        max_retries: int = 2,
        topk: int = 50,
        max_chunks: int = 8,
        max_rounds: int = 5,
        max_revisions: int = 3,
        use_reranker: bool = False,
        # 自定义 Milvus collection 支持
        custom_collection: str | None = None,
        custom_dense_field: str = "embedding",
        custom_text_field: str = "proposition_text",
        custom_sparse_field: str | None = None,
        # 分角色模型名称（None = 使用默认 llm）
        planner_model: str | None = None,
        critic_model: str | None = None,
        rewrite_model: str | None = None,
        solver_model: str | None = None,
        answer_model: str | None = None,
        # 全局 + 分角色 ChatOpenAI 参数
        model_params: dict | None = None,
        role_params: dict | None = None,
        # structured output 方式
        structured_output_method: str | None = None,
        # thinking 模式角色（CLI --enable-thinking 传入）
        thinking_roles: set[str] | None = None,
        # 搜索源（默认 Milvus，可选 Tavily）
        search_source: str = "milvus",
    ):
        super().__init__(
            eval_mode="rag_loop",
            llm=llm,
            dataset_type=dataset_type,
            batch_size=batch_size,
            max_workers=max_workers,
            max_retries=max_retries,
            per_question_results=True,
        )

        self.topk = topk
        self.max_chunks = max_chunks
        self.max_rounds = max_rounds
        self.max_revisions = max_revisions
        self.use_reranker = use_reranker

        # 自定义 collection 配置
        self.custom_collection = custom_collection
        self.custom_dense_field = custom_dense_field
        self.custom_text_field = custom_text_field
        self.custom_sparse_field = custom_sparse_field

        # 分角色模型配置
        self.planner_model = planner_model
        self.critic_model = critic_model
        self.rewrite_model = rewrite_model
        self.solver_model = solver_model
        self.answer_model = answer_model
        self.model_params = model_params
        self.role_params = role_params
        self.structured_output_method = structured_output_method
        self.thinking_roles = thinking_roles or set()
        self.search_source = search_source

    def _make_search_fn(self, rewrite_model=None):
        """为当前评估创建 SearchFn 实例。

        根据 search_source 选择适配器：
        - "tavily": TavilyAdapter（Tavily 网络搜索）
        - "web": WebAdapter（DuckDuckGo 免费搜索）
        - 默认: MilvusAdapter（Milvus 向量库）

        如果配置了自定义 collection，使用 CustomMilvusRetriever；
        否则使用标准的 MilvusRetriever。

        Args:
            rewrite_model: 可选的多查询重写模型。传入后启用
                           get_similar_chunks_with_rewrite（5 查询 + RRF 融合）。
        """
        if self.search_source == "tavily":
            from rag_loop.adapters import TavilyAdapter
            return TavilyAdapter(
                max_chunks=self.max_chunks,
                rewrite_model=rewrite_model,
            )

        if self.search_source == "web":
            from rag_loop.adapters import WebAdapter
            return WebAdapter(
                max_chunks=self.max_chunks,
                rewrite_model=rewrite_model,
            )

        from rag_loop.adapters import MilvusAdapter

        return MilvusAdapter(
            dataset_type=self.dataset_type,
            topk_propositions=self.topk,
            max_chunks=self.max_chunks,
            use_reranker=self.use_reranker,
            custom_collection=self.custom_collection,
            custom_dense_field=self.custom_dense_field,
            custom_text_field=self.custom_text_field,
            custom_sparse_field=self.custom_sparse_field,
            rewrite_model=rewrite_model,
        )

    _BASE_DEFAULTS = {
        "temperature": 0.0,
        "extra_body": {"enable_thinking": False, "enable_search": False},
        "max_retries": 1,
        # "timeout": 120.0,
    }

    def _create_llm(self, model_name: str | None, role: str):
        """创建角色专用 ChatOpenAI 实例——4 层合并（照搬 agentic_rag_v3 模式）。

        Layer 1: _BASE_DEFAULTS
        Layer 2: self.model_params（全局参数，应用于所有角色）
        Layer 3: self.role_params[role]（角色特定覆盖）
        Layer 4: model=model_name

        当 role 在 self.thinking_roles 中但 model_name 未指定时，
        自动使用 self.llm.model_name 作为模型名，确保 thinking 角色
        有独立模型实例（携带 thinking_budget + 长 timeout）。

        返回 None 表示该角色未配置，应使用默认 llm。

        自动检测：当 enable_thinking=True 时，自动追加 thinking_budget 和更长 timeout。
        """
        # thinking 角色未指定模型名时，使用全局默认模型名
        if model_name is None and role in self.thinking_roles:
            model_name = self.llm.model_name
        if model_name is None:
            return None
        from langchain_openai import ChatOpenAI
        kwargs = {**self._BASE_DEFAULTS, **(self.model_params or {}),
                  **(self.role_params or {}).get(role, {}), "model": model_name}
        kwargs["api_key"] = os.getenv("BL_API_KEY")
        kwargs["base_url"] = os.getenv("BL_BASE_URL")

        # 自动检测 thinking 角色，追加 thinking_budget + 延长 timeout
        extra = (kwargs.get("extra_body") or {}).copy()
        # 来源 1: CLI --enable-thinking 显式指定 → 强制启用
        # 来源 2: model_params/role_params 中已有 enable_thinking=True
        if role in self.thinking_roles:
            extra["enable_thinking"] = True
            extra["enable_search"] = False
        if extra.get("enable_thinking"):
            extra.setdefault("thinking_budget", 8192)
            kwargs["extra_body"] = extra
            # 确保 thinking 模型有足够的超时时间（至少 580s，尊重用户显式设置）
            if kwargs.get("timeout", 0) < 580.0:
                kwargs["timeout"] = 580.0

        return ChatOpenAI(**kwargs)

    def _make_role_models(self) -> dict | None:
        """构建 role_models 字典——仅包含已配置角色的模型实例。

        返回 None 表示无任何角色覆盖，run_pipeline 将全部使用默认 model。
        """
        models = {}
        for role, model_arg in [
            ("planner", self.planner_model),
            ("critic", self.critic_model),
            ("rewrite", self.rewrite_model),
            ("solver", self.solver_model),
            ("answer", self.answer_model),
        ]:
            m = self._create_llm(model_arg, role)
            if m is not None:
                models[role] = m
        return models or None

    def _detect_thinking_roles(self) -> set[str]:
        """检测哪些角色的 extra_body 中启用了 enable_thinking。

        三个来源（合并）：
        1. self.thinking_roles — CLI --enable-thinking 显式指定
        2. self.model_params（全局）— extra_body.enable_thinking
        3. self.role_params（分角色）— extra_body.enable_thinking

        返回需要切换到 thinking_tool 的角色名集合。
        """
        thinking_roles: set[str] = set(self.thinking_roles)
        all_roles = ["planner", "critic", "rewrite", "solver", "answer"]

        def _has_thinking(params: dict | None) -> bool:
            if not params:
                return False
            extra = params.get("extra_body", {})
            if isinstance(extra, dict):
                return extra.get("enable_thinking", False)
            return False

        global_thinking = _has_thinking(self.model_params)

        for role in all_roles:
            role_thinking = _has_thinking((self.role_params or {}).get(role, {}))
            if role_thinking or (global_thinking and self._model_for_role_is_set(role)):
                thinking_roles.add(role)

        return thinking_roles

    def _model_for_role_is_set(self, role: str) -> bool:
        """检查指定角色是否有独立模型配置（即用户有意为该角色自定义）。"""
        role_model_attr = f"{role}_model"
        return bool(getattr(self, role_model_attr, None))

    def _make_role_methods(self) -> dict | None:
        """构建 role_methods 字典——为启用 thinking 的角色自动切换 thinking_tool。

        显式传入 structured_output_method 时，所有角色使用该值（覆盖自动检测）。
        否则自动检测 enable_thinking 并切换受影响角色。

        同时输出 warning 供用户确认。
        """
        # 显式覆盖：用户明确指定了 method
        if self.structured_output_method and self.structured_output_method != "function_calling":
            all_roles = ["planner", "critic", "rewrite", "solver", "answer"]
            logger.info(
                "structured_output_method 显式设为 '%s'，所有角色将使用此方式。",
                self.structured_output_method,
            )
            return {role: self.structured_output_method for role in all_roles}

        # 自动检测：检查哪些角色启用了 enable_thinking
        thinking_roles = self._detect_thinking_roles()
        if not thinking_roles:
            return None

        logger.warning(
            "检测到以下角色启用了 enable_thinking: %s。"
            "将自动切换 structured_output method 为 'thinking_tool'（使用 bind_tools "
            "无 tool_choice，允许 Qwen3 enable_thinking 与 function calling 共存）。",
            ", ".join(sorted(thinking_roles)),
        )

        methods = {}
        for role in thinking_roles:
            methods[role] = "thinking_tool"
        return methods

    def _make_custom_prompts(self) -> dict[str, str] | None:
        """构建 custom_system_prompts——为网络搜索场景注入优化提示词。

        当 search_source == "web" 时，返回包含 web 优化 Planner 提示词的字典。
        否则返回 None（使用默认提示词）。
        """
        if self.search_source == "web":
            from rag_loop.planner import PLANNER_SYSTEM_TEMPLATE_WEB
            return {"planner": PLANNER_SYSTEM_TEMPLATE_WEB}
        return None

    def _make_solver_prepare_prompt(self):
        """构建 Solver prepare 提示词——为网络搜索场景注入优化查询构造。

        当 search_source == "web" 时，返回 PREPARE_PROMPT_WEB。
        否则返回 None（使用默认的 PREPARE_PROMPT）。
        """
        if self.search_source == "web":
            from rag_loop.solver import PREPARE_PROMPT_WEB
            return PREPARE_PROMPT_WEB
        return None

    def evaluate_single(self, question: NormalizedQuestion) -> dict:
        """处理单个问题。

        Returns:
            {
                "prediction": str,
                "error": None | str,
                "chunks": [...],
                "context_recall": float, "hit": int, "mrr": float, "retrieval_precision": float,
                "retrieval_count": int, "total_chunks": int, "total_distinct_titles": int,
                "search_depth": int, "total_rounds": int, "termination_reason": str,
                "dag_nodes": int, "dag_edges": int,
                # rag_loop 特有:
                "dag_snapshots": [...],       # 每轮 DAG 结构快照
                "search_trace": [...],        # 所有搜索调用记录
                "round_summaries": [...],     # 每轮 Planner/Critic 摘要
            }
        """
        from rag_loop.pipeline import run_pipeline, PipelineConfig, ExperimentRecord
        from rag_loop.pipeline import _dag_to_dict

        # 先构建角色模型（需要在创建 search_fn 之前，以便将 rewrite model 注入适配器）
        role_models = self._make_role_models()
        role_methods = self._make_role_methods()

        # 提取 rewrite model：优先使用角色模型中的 rewrite，否则回退到默认 llm
        rewrite_model = None
        if role_models and "rewrite" in role_models:
            rewrite_model = role_models["rewrite"]
        else:
            rewrite_model = self.llm  # fallback：始终使用默认模型做多查询重写

        search_fn = self._make_search_fn(rewrite_model=rewrite_model)
        config = PipelineConfig(
            max_rounds=self.max_rounds,
            max_revisions=self.max_revisions,
            max_history_rounds=3,
            max_consecutive_planner_failures=2,
        )

        # 创建实验记录以捕获结构化数据
        # 不含完整消息线程（太大），仅含每轮摘要和搜索追踪
        record = ExperimentRecord(question=question.question)

        result = run_pipeline(
            q=question.question,
            search_fn=search_fn,
            model=self.llm,
            config=config,
            record=record,
            role_models=role_models,
            role_methods=role_methods,
            custom_system_prompts=self._make_custom_prompts(),
            solver_prepare_prompt=self._make_solver_prepare_prompt(),
        )

        # ── DAG 快照：每轮 Critic 完成后的 DAG 结构 ──
        dag_snapshots = [_dag_to_dict(rd) for rd in result.round_dags]

        # ── 搜索追踪：所有搜索调用（跨轮次汇总） ──
        search_trace = [
            {
                "round": r.round_number,
                "query": c.query,
                "chunks_returned": c.chunk_count,
                "chunk_ids": c.chunk_ids,
            }
            for r in record.rounds
            for c in r.solver_search_calls
        ]

        # ── 每轮摘要：Planner/Critic 的关键决策信息 ──
        round_summaries = []
        for r in record.rounds:
            rs: dict = {
                "round": r.round_number,
                "planner_ok": r.planner_success,
                "planner_revisions": r.planner_revisions,
                "search_calls": r.solver_search_count,
            }
            # Planner 输出摘要（仅假设文本，不含完整 primitive 序列）
            if r.planner_output:
                rs["hypothesis_target"] = r.planner_output.get("hypothesis_part1", "")[:500]
                rs["hypothesis_delta"] = r.planner_output.get("hypothesis_part2", "")[:500]
                rs["planner_errors"] = [
                    e.get("description", str(e)) for e in r.planner_errors[:3]
                ]
            # Critic 终止判断摘要
            if r.critic_output:
                term = r.critic_output.get("termination", {})
                rs["termination_check"] = {
                    "should_terminate": term.get("should_terminate"),
                    "condition_2_passed": term.get("condition_2_passed"),
                    "condition_3_passed": term.get("condition_3_passed"),
                    "reason": term.get("termination_reason", "")[:200],
                }
            round_summaries.append(rs)

        # ── 收集所有 round DAG 中的所有 chunks ──
        all_chunks: list[dict] = []
        seen_chunk_ids: set[str] = set()
        for rd in result.round_dags:
            for node in rd.nodes.values():
                for c in node.retrieved_chunks:
                    cid = c.chunk_id
                    if cid not in seen_chunk_ids:
                        seen_chunk_ids.add(cid)
                        all_chunks.append({
                            "chunk_id": c.chunk_id,
                            "chunk_title": c.chunk_title,
                            "chunk_summary": c.chunk_summary,
                            "context_title": c.context_title,
                            "page_content": c.page_content,
                        })

        retrieved_titles = [c.get("context_title", "") for c in all_chunks]
        supporting = extract_supporting_titles(question.raw, self.dataset_type)

        final_dag = result.final_dag
        dag_node_count = len(final_dag.nodes) if final_dag else 0
        dag_edge_count = len(final_dag.edges) if final_dag else 0

        prediction = result.answer.strip() if result.answer else ""

        return {
            "prediction": prediction,
            "error": None,
            "chunks": all_chunks,
            "context_recall": compute_context_recall(retrieved_titles, supporting),
            "hit": compute_hit(retrieved_titles, supporting),
            "mrr": compute_mrr(retrieved_titles, supporting),
            "retrieval_precision": compute_retrieval_precision(retrieved_titles, supporting),
            # 效率指标
            "retrieval_count": result.total_search_calls,
            "total_chunks": len(all_chunks),
            "total_distinct_titles": len(set(retrieved_titles)),
            "search_depth": result.total_rounds,
            # rag_loop 特有指标
            "total_rounds": result.total_rounds,
            "termination_reason": result.termination_reason,
            "dag_nodes": dag_node_count,
            "dag_edges": dag_edge_count,
            # rag_loop 演化轨迹
            "dag_snapshots": dag_snapshots,
            "search_trace": search_trace,
            "round_summaries": round_summaries,
        }