"""统一结果类型定义——Service 层与 Frontend 层之间的数据契约。

所有算法执行结果统一归一化为 UnifiedResult，前端只依赖此结构，
不感知具体算法实现的差异。
"""

from dataclasses import dataclass, field
from enum import Enum


class AlgorithmType(str, Enum):
    """系统支持的 RAG 算法"""
    LLM_ONLY = "llm-only"
    NAIVE_RAG = "naive-rag"
    RAG_WITH_JUDGE = "rag-with-judge"
    RAG_LOOP = "rag-loop"


class SearchSource(str, Enum):
    """检索源"""
    MILVUS = "milvus"
    WEB = "web"
    TAVILY = "tavily"


@dataclass
class UnifiedResult:
    """统一的 RAG 执行结果——所有算法归一化到此结构。

    通用字段（所有算法都有）:
      - answer: 最终答案文本
      - chunks: 检索到的来源列表
      - algorithm: 使用的算法类型
      - search_source: 检索源（LLM Only 时为 None）
      - elapsed: 执行耗时（秒）
      - error: 错误信息（成功时为 None）

    算法特有字段（按需填充）:
      - rewritten_queries: Naive RAG 的重写查询列表
      - search_path: RAG with Judge 的递归搜索树
      - pipeline_result: rag_loop 的 PipelineResult 序列化
    """
    answer: str
    chunks: list[dict] = field(default_factory=list)
    algorithm: AlgorithmType = AlgorithmType.LLM_ONLY
    search_source: SearchSource | None = None
    elapsed: float = 0.0
    error: str | None = None

    # Naive RAG 特有
    rewritten_queries: list[str] | None = None

    # RAG with Judge 特有
    search_path: dict | None = None

    # rag_loop 特有
    pipeline_result: dict | None = None  # PipelineResult 关键字段的 dict