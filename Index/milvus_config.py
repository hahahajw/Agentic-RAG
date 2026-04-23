"""Milvus 连接、嵌入函数、路径等共享配置"""

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# ── Milvus 连接 ──────────────────────────────────────────────
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "")  # 如果有认证


def get_collection_name(dataset_type: str) -> str:
    """返回指定数据集对应的 collection 名称"""
    return f"agentic_propositions_{dataset_type.lower()}"

# ── Embedding ────────────────────────────────────────────────
_EMBEDDING_CACHE: dict[str, Any] = {}


def get_embedding_function() -> Any:
    """返回嵌入函数（单例缓存）"""
    key = "default"
    if key not in _EMBEDDING_CACHE:
        from langchain_openai import OpenAIEmbeddings

        _EMBEDDING_CACHE[key] = OpenAIEmbeddings(
            api_key=os.getenv("BL_API_KEY"),
            base_url=os.getenv("BL_BASE_URL"),
            model="text-embedding-v4",
            dimensions=1024,
            check_embedding_ctx_length=False,
        )
    return _EMBEDDING_CACHE[key]


# ── 嵌入批处理参数 ────────────────────────────────────────────
EMBED_BATCH_SIZE = 10

# ── 检索默认参数 ─────────────────────────────────────────────
RETRIEVER_TOPK_PROPOSITIONS = 50
RETRIEVER_MAX_CHUNKS = 8

# ── 数据路径 ─────────────────────────────────────────────────
_BENCHMARK_DIR = Path(__file__).resolve().parent.parent / "Data" / "benchmark"

_DATASET_FILES = {
    "hotpotqa": "HotpotQA_500_benchmark_chunked.json",
    "2wikimultihopqa": "2WikiMultihopQA_500_benchmark_chunked.json",
    "musique": "MuSiQue_500_benchmark_chunked.json",
}


def get_chunked_path(dataset_type: str) -> Path:
    """返回指定数据集的 chunked JSON 路径"""
    ds = dataset_type.lower()
    if ds not in _DATASET_FILES:
        raise ValueError(f"不支持的数据集: {ds}，可选: {list(_DATASET_FILES.keys())}")
    return _BENCHMARK_DIR / _DATASET_FILES[ds]


def get_all_dataset_types() -> list[str]:
    return list(_DATASET_FILES.keys())
