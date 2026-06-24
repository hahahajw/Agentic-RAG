"""统一 RAG 服务层——封装所有 RAG 算法为统一接口。

外部（Frontend/Eval）只需调用 UnifiedRAGService.query()，
不直接依赖任何具体算法模块。

算法:
  - LLM Only: 直接调用 LLM，不检索
  - 模块化 RAG: naive_rag/ — 多查询重写 + RRF 融合
  - 递归检索 RAG: rag_with_judge/ — Judge 判断 + 递归搜索树
  - 规划-执行-反馈闭环 RAG: rag_loop/ — Planner-Solver-Critic DAG 闭环

检索源:
  - Milvus: MilvusRetriever (向量数据库)
  - Web: WebRetriever (DuckDuckGo 网络搜索)
  - Tavily: TavilyRetriever (Tavily Search API，专为 AI Agent 设计)
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from .result_types import AlgorithmType, SearchSource, UnifiedResult

logger = logging.getLogger(__name__)


class UnifiedRAGService:
    """统一的 RAG 服务——所有算法对外暴露同一入口。"""

    def query(
        self,
        question: str,
        algorithm: AlgorithmType,
        search_source: SearchSource | None = None,
        *,
        # 模型（ChatOpenAI 实例）
        llm: ChatOpenAI | None = None,
        rewrite_llm: ChatOpenAI | None = None,
        judge_llm: ChatOpenAI | None = None,
        answer_llm: ChatOpenAI | None = None,
        planner_llm: ChatOpenAI | None = None,
        critic_llm: ChatOpenAI | None = None,
        solver_llm: ChatOpenAI | None = None,
        # 检索参数
        dataset_type: str = "hotpotqa",
        max_chunks: int = 8,
        topk_propositions: int = 50,
        use_reranker: bool = False,
        # RAG with Judge 参数
        max_depth: int = 3,
        # rag_loop 参数
        max_rounds: int = 5,
        max_revisions: int = 3,
        # Prompt 覆盖
        custom_rewrite_prompt: str | None = None,
        custom_answer_prompt: str | None = None,
        custom_judge_prompt: str | None = None,
        custom_planner_prompt: str | None = None,
        custom_critic_prompt: str | None = None,
        # Structured output 方法覆盖（Thinking 模式用）
        role_methods: dict[str, str] | None = None,
    ) -> UnifiedResult:
        """执行一次 RAG 查询。

        Args:
            question: 用户问题
            algorithm: 算法类型
            search_source: 检索源（LLM Only 时为 None）
            llm: 默认 LLM（未指定角色模型时的回退）
            *_llm: 各角色专用模型
            dataset_type: Milvus 数据集（仅 search_source=MILVUS 时有效）
            max_chunks: 最大返回 chunk 数
            topk_propositions: 检索 top-K 命题数
            use_reranker: 是否启用重排序
            max_depth: RAG with Judge 最大递归深度
            max_rounds: rag_loop 最大轮次
            max_revisions: rag_loop Planner 每轮最大修订次数
            custom_*_prompt: 自定义 prompt 模板

        Returns:
            UnifiedResult——归一化结果
        """
        start = time.time()

        try:
            if algorithm == AlgorithmType.LLM_ONLY:
                return self._run_llm_only(question, llm or answer_llm, start)

            # 构建检索器 / SearchFn
            retriever = self._create_retriever(
                search_source, dataset_type, max_chunks,
                topk_propositions, use_reranker,
            )

            if algorithm == AlgorithmType.NAIVE_RAG:
                return self._run_naive_rag(
                    question, retriever, max_chunks, search_source,
                    llm, rewrite_llm, answer_llm,
                    custom_rewrite_prompt, custom_answer_prompt,
                    start,
                )
            elif algorithm == AlgorithmType.RAG_WITH_JUDGE:
                return self._run_rag_with_judge(
                    question, retriever, max_chunks, max_depth, search_source,
                    llm, rewrite_llm, judge_llm, answer_llm,
                    custom_rewrite_prompt, custom_judge_prompt, custom_answer_prompt,
                    role_methods,
                    start,
                )
            elif algorithm == AlgorithmType.RAG_LOOP:
                return self._run_rag_loop(
                    question, search_source, dataset_type,
                    max_chunks, max_rounds, max_revisions,
                    topk_propositions, use_reranker,
                    llm, rewrite_llm, planner_llm, critic_llm, solver_llm, answer_llm,
                    custom_planner_prompt, custom_critic_prompt, custom_answer_prompt,
                    role_methods,
                    start,
                )
        except Exception as e:
            logger.exception("RAG 查询失败")
            elapsed = time.time() - start
            return UnifiedResult(
                answer="",
                algorithm=algorithm,
                search_source=search_source,
                elapsed=elapsed,
                error=f"{type(e).__name__}: {e}",
            )

    # ═══════════════════════════════════════════════════════════════
    # 算法实现
    # ═══════════════════════════════════════════════════════════════

    def _run_llm_only(
        self, question: str, llm: ChatOpenAI | None, start: float,
    ) -> UnifiedResult:
        """直接调用 LLM 回答，无检索。"""
        if llm is None:
            elapsed = time.time() - start
            return UnifiedResult(
                answer="",
                algorithm=AlgorithmType.LLM_ONLY,
                search_source=None,
                elapsed=elapsed,
                error="未配置 LLM 模型——请提供 API Key 并选择模型",
            )
        response = llm.invoke([HumanMessage(content=question)])
        elapsed = time.time() - start
        return UnifiedResult(
            answer=response.content if hasattr(response, "content") else str(response),
            algorithm=AlgorithmType.LLM_ONLY,
            search_source=None,
            elapsed=elapsed,
        )

    def _run_naive_rag(
        self,
        question: str,
        retriever,
        max_chunks: int,
        search_source: SearchSource,
        llm: ChatOpenAI | None,
        rewrite_llm: ChatOpenAI | None,
        answer_llm: ChatOpenAI | None,
        custom_rewrite_prompt: str | None,
        custom_answer_prompt: str | None,
        start: float,
    ) -> UnifiedResult:
        """执行模块化 RAG (naive_rag/)。"""
        from naive_rag.workflow import get_workflow

        _llm = llm or answer_llm
        if _llm is None:
            elapsed = time.time() - start
            return UnifiedResult(
                answer="", algorithm=AlgorithmType.NAIVE_RAG,
                search_source=search_source, elapsed=elapsed,
                error="未配置 LLM 模型",
            )
        app = get_workflow(scheme="a", skip_suggest=True)

        config: dict[str, Any] = {
            "configurable": {
                "llm": _llm,
                "retriever": retriever,
                "max_chunks": max_chunks,
            }
        }
        if rewrite_llm:
            config["configurable"]["rewrite_llm"] = rewrite_llm
        if answer_llm:
            config["configurable"]["answer_llm"] = answer_llm
        if custom_rewrite_prompt:
            config["configurable"]["custom_rewrite_prompt"] = custom_rewrite_prompt
        if custom_answer_prompt:
            config["configurable"]["custom_answer_prompt"] = custom_answer_prompt

        result = app.invoke(
            {"original_query": question, "messages": []},
            config=config,
        )

        # 提取 chunks
        chunks = _docs_to_chunk_dicts(result.get("fused_chunks", []))

        elapsed = time.time() - start
        return UnifiedResult(
            answer=result.get("answer", ""),
            chunks=chunks,
            algorithm=AlgorithmType.NAIVE_RAG,
            search_source=search_source,
            elapsed=elapsed,
            rewritten_queries=result.get("rewritten_queries", []),
        )

    def _run_rag_with_judge(
        self,
        question: str,
        retriever,
        max_chunks: int,
        max_depth: int,
        search_source: SearchSource,
        llm: ChatOpenAI | None,
        rewrite_llm: ChatOpenAI | None,
        judge_llm: ChatOpenAI | None,
        answer_llm: ChatOpenAI | None,
        custom_rewrite_prompt: str | None,
        custom_judge_prompt: str | None,
        custom_answer_prompt: str | None,
        role_methods: dict[str, str] | None,
        start: float,
    ) -> UnifiedResult:
        """执行递归检索 RAG (rag_with_judge/)。"""
        from rag_with_judge.nodes import rag_with_judge
        from rag_with_judge.workflow import build_judge_rag_graph

        _llm = llm or answer_llm
        if _llm is None:
            elapsed = time.time() - start
            return UnifiedResult(
                answer="", algorithm=AlgorithmType.RAG_WITH_JUDGE,
                search_source=search_source, elapsed=elapsed,
                error="未配置 LLM 模型",
            )
        app = build_judge_rag_graph()

        config: dict[str, Any] = {
            "configurable": {
                "llm": _llm,
                "retriever": retriever,
                "max_chunks": max_chunks,
                "judge_variant": "B",
                "judge_method": (role_methods or {}).get("judge", "function_calling"),
            }
        }
        if rewrite_llm:
            config["configurable"]["rewrite_llm"] = rewrite_llm
        if judge_llm:
            config["configurable"]["judge_llm"] = judge_llm
        if answer_llm:
            config["configurable"]["answer_llm"] = answer_llm
        if custom_rewrite_prompt:
            config["configurable"]["custom_rewrite_prompt"] = custom_rewrite_prompt
        if custom_judge_prompt:
            config["configurable"]["custom_judge_prompt"] = custom_judge_prompt
        if custom_answer_prompt:
            config["configurable"]["custom_answer_prompt"] = custom_answer_prompt

        search_path: dict = {}
        answer = rag_with_judge(
            query=question,
            path=search_path,
            visited=set(),
            depth=0,
            max_depth=max_depth,
            app=app,
            config=config,
        )

        # 递归收集所有 chunks
        chunks = _collect_chunks_from_search_path(search_path)

        # 从根节点提取重写查询
        rewritten_queries = search_path.get("rewritten_queries", [])

        elapsed = time.time() - start
        return UnifiedResult(
            answer=answer,
            chunks=chunks,
            algorithm=AlgorithmType.RAG_WITH_JUDGE,
            search_source=search_source,
            elapsed=elapsed,
            search_path=search_path,
            rewritten_queries=rewritten_queries if rewritten_queries else None,
        )

    def _run_rag_loop(
        self,
        question: str,
        search_source: SearchSource | None,
        dataset_type: str,
        max_chunks: int,
        max_rounds: int,
        max_revisions: int,
        topk_propositions: int,
        use_reranker: bool,
        llm: ChatOpenAI | None,
        rewrite_llm: ChatOpenAI | None,
        planner_llm: ChatOpenAI | None,
        critic_llm: ChatOpenAI | None,
        solver_llm: ChatOpenAI | None,
        answer_llm: ChatOpenAI | None,
        custom_planner_prompt: str | None,
        custom_critic_prompt: str | None,
        custom_answer_prompt: str | None,
        role_methods: dict[str, str] | None,
        start: float,
    ) -> UnifiedResult:
        """执行规划-执行-反馈闭环 RAG (rag_loop/)。"""
        from rag_loop.pipeline import run_pipeline, PipelineConfig
        from rag_loop.adapters import MilvusAdapter
        from rag_loop.adapters import WebAdapter as RLWebAdapter
        from rag_loop.adapters import TavilyAdapter

        # 创建 SearchFn
        if search_source == SearchSource.TAVILY:
            search_fn = TavilyAdapter(max_chunks=max_chunks,
                                      rewrite_model=rewrite_llm or llm or answer_llm)
        elif search_source == SearchSource.WEB:
            search_fn = RLWebAdapter(max_chunks=max_chunks,
                                     rewrite_model=rewrite_llm or llm or answer_llm)
        else:
            search_fn = MilvusAdapter(
                dataset_type=dataset_type,
                max_chunks=max_chunks,
                topk_propositions=topk_propositions,
                use_reranker=use_reranker,
                rewrite_model=rewrite_llm or llm or answer_llm,
            )

        _llm = llm or answer_llm
        pipeline_config = PipelineConfig(
            max_rounds=max_rounds,
            max_revisions=max_revisions,
        )

        role_models = {}
        if planner_llm:
            role_models["planner"] = planner_llm
        if critic_llm:
            role_models["critic"] = critic_llm
        if solver_llm:
            role_models["solver"] = solver_llm
        if answer_llm:
            role_models["answer"] = answer_llm

        # 构建自定义系统提示词（仅传递非 None 值）
        custom_system_prompts: dict[str, str] = {}
        if custom_planner_prompt:
            custom_system_prompts["planner"] = custom_planner_prompt
        if custom_critic_prompt:
            custom_system_prompts["critic"] = custom_critic_prompt
        if custom_answer_prompt:
            custom_system_prompts["answer"] = custom_answer_prompt

        result = run_pipeline(
            q=question,
            search_fn=search_fn,
            model=_llm,
            config=pipeline_config,
            role_models=role_models if role_models else None,
            role_methods=role_methods,
            custom_system_prompts=custom_system_prompts if custom_system_prompts else None,
        )

        # 收集 DAG 中的所有 chunks
        chunks = _collect_chunks_from_dag(result.final_dag)

        # PipelineResult → dict
        pipeline_dict = _pipeline_result_to_dict(result)

        # 答案回退: generate_answer() 可能返回空 → 从根节点提取
        answer = result.answer
        if not answer:
            root = result.final_dag.root
            if root and root.answer:
                answer = root.answer

        elapsed = time.time() - start
        return UnifiedResult(
            answer=answer,
            chunks=chunks,
            algorithm=AlgorithmType.RAG_LOOP,
            search_source=search_source,
            elapsed=elapsed,
            pipeline_result=pipeline_dict,
        )

    # ═══════════════════════════════════════════════════════════════
    # 检索器工厂
    # ═══════════════════════════════════════════════════════════════

    def _create_retriever(
        self,
        source: SearchSource | None,
        dataset_type: str,
        max_chunks: int,
        topk_propositions: int = 50,
        use_reranker: bool = False,
    ):
        """创建检索器实例。

        Returns:
            MilvusRetriever 或 WebRetriever——两者都实现了
            get_similar_chunk_with_score(query) 和
            get_similar_chunks_with_rewrite(query, rewrite_llm) 接口。
        """
        if source == SearchSource.TAVILY:
            from Retrieval.tavily_retriever import TavilyRetriever
            return TavilyRetriever(max_chunks=max_chunks)

        if source == SearchSource.WEB:
            from Retrieval.web_retriever import WebRetriever
            return WebRetriever(max_chunks=max_chunks)

        # 默认: Milvus
        from Retrieval.milvus_retriever import MilvusRetriever
        return MilvusRetriever(
            dataset_type=dataset_type,
            topk_propositions=topk_propositions,
            max_chunks=max_chunks,
            use_reranker=use_reranker,
        )


# ═══════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════


def _docs_to_chunk_dicts(fused_chunks: list) -> list[dict]:
    """将 (Document, score) 列表转换为统一 chunk dict 格式。"""
    chunks = []
    for item in fused_chunks:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            doc, score = item
        else:
            doc = item
            score = 0.0
        meta = getattr(doc, "metadata", {})
        chunks.append({
            "chunk_id": meta.get("chunk_id", ""),
            "chunk_title": meta.get("chunk_title", "Unknown"),
            "chunk_summary": meta.get("chunk_summary", ""),
            "context_title": meta.get("context_title", ""),
            "source_url": meta.get("source_url", ""),
            "page_content": getattr(doc, "page_content", ""),
            "score": score,
        })
    return chunks


def _collect_chunks_from_search_path(path: dict) -> list[dict]:
    """递归收集 SEARCH_PATH 中的所有 chunks（去重）。"""
    chunks: list[dict] = []
    seen_ids: set[str] = set()

    def _collect(p: dict) -> None:
        for c in p.get("chunks", []):
            cid = c.get("chunk_id", "")
            if cid and cid in seen_ids:
                continue
            if cid:
                seen_ids.add(cid)
            chunks.append({
                "chunk_id": cid,
                "chunk_title": c.get("chunk_title", "Unknown"),
                "chunk_summary": c.get("chunk_summary", ""),
                "context_title": c.get("context_title", ""),
                "source_url": c.get("source_url", ""),
                "page_content": c.get("page_content", ""),
                "score": c.get("score", 0.0),
            })

        for child in p.get("next_queries", []):
            if isinstance(child, dict):
                _collect(child)

    _collect(path)
    return chunks


def _collect_chunks_from_dag(dag) -> list[dict]:
    """从 rag_loop DAG 的所有节点收集 ChunkInfo。"""
    chunks = []
    seen_ids = set()
    for node in dag.nodes.values():
        for chunk in node.retrieved_chunks:
            cid = chunk.chunk_id
            if cid not in seen_ids:
                seen_ids.add(cid)
                chunks.append({
                    "chunk_id": chunk.chunk_id,
                    "chunk_title": chunk.chunk_title,
                    "chunk_summary": chunk.chunk_summary,
                    "context_title": chunk.context_title,
                    "source_url": "",
                    "page_content": chunk.page_content,
                })
    return chunks


def _pipeline_result_to_dict(result) -> dict:
    """将 PipelineResult 关键字段转为前端可用的 dict。

    注意: _dag_to_dict 是 rag_loop.pipeline 的模块级函数（非公开 API），
    但它是 DAG → dict 序列化的唯一实现，直接复用避免重复代码。
    """
    from rag_loop.pipeline import _dag_to_dict

    return {
        "answer": result.answer,
        "total_rounds": result.total_rounds,
        "total_search_calls": result.total_search_calls,
        "termination_reason": result.termination_reason,
        "final_dag": _dag_to_dict(result.final_dag),
        "round_dags": [_dag_to_dict(d) for d in result.round_dags],
    }


