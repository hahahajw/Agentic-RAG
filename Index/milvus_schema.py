"""Milvus Collection Schema 定义、集合创建/删除、索引创建 — 每个数据集独立 collection"""

import logging
import sys
from pathlib import Path
from typing import Optional

# 确保项目根目录在 sys.path 中
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pymilvus import (
    MilvusClient,
    CollectionSchema,
    FieldSchema,
    DataType,
    Function,
    FunctionType,
)
from pymilvus.milvus_client.index import IndexParams

from Index.milvus_config import (
    MILVUS_URI,
    MILVUS_TOKEN,
    get_collection_name,
    get_all_dataset_types,
)

logger = logging.getLogger(__name__)

_DIM = 1024  # text-embedding-v3 输出维度


def _get_client() -> MilvusClient:
    """获取 MilvusClient 实例"""
    connect_kwargs = {"uri": MILVUS_URI}
    if MILVUS_TOKEN:
        connect_kwargs["token"] = MILVUS_TOKEN
    return MilvusClient(**connect_kwargs)


def build_schema() -> CollectionSchema:
    """构建 Collection Schema — dataset-agnostic"""
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
        FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="question_id", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="context_index", dtype=DataType.INT32),
        FieldSchema(name="context_title", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="chunk_title", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="chunk_summary", dtype=DataType.VARCHAR, max_length=2048),
        FieldSchema(
            name="proposition_text",
            dtype=DataType.VARCHAR,
            max_length=4096,
            enable_analyzer=True,
        ),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=_DIM),
        FieldSchema(name="sparse_embedding", dtype=DataType.SPARSE_FLOAT_VECTOR),
    ]

    bm25_fn = Function(
        name="bm25_fn",
        function_type=FunctionType.BM25,
        input_field_names=["proposition_text"],
        output_field_names=["sparse_embedding"],
    )

    return CollectionSchema(fields=fields, functions=[bm25_fn])


def _build_index_params() -> list:
    """返回索引参数列表"""
    dense_ip = IndexParams()
    dense_ip.add_index(
        field_name="embedding",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 32, "efConstruction": 512},
    )

    sparse_ip = IndexParams()
    sparse_ip.add_index(
        field_name="sparse_embedding",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25",
        params={"bm25_k1": 1.2, "bm25_b": 0.75},
    )

    return [dense_ip, sparse_ip]


def collection_exists(dataset_type: str, client: Optional[MilvusClient] = None) -> bool:
    """检查指定数据集的 collection 是否存在"""
    c = client or _get_client()
    return c.has_collection(get_collection_name(dataset_type))


def create_collection(
    dataset_type: str,
    drop_old: bool = False,
    client: Optional[MilvusClient] = None,
) -> MilvusClient:
    """为指定数据集创建 Collection（含索引），返回 client 实例"""
    c = client or _get_client()
    coll_name = get_collection_name(dataset_type)

    if drop_old and c.has_collection(coll_name):
        logger.info("Dropping existing collection: %s", coll_name)
        c.drop_collection(coll_name)

    if c.has_collection(coll_name):
        logger.info("Collection already exists: %s", coll_name)
        return c

    schema = build_schema()
    logger.info("Creating collection: %s", coll_name)
    c.create_collection(collection_name=coll_name, schema=schema)

    for ip in _build_index_params():
        c.create_index(collection_name=coll_name, index_params=ip)

    c.load_collection(coll_name)
    logger.info("Collection created and loaded: %s", coll_name)
    return c


def drop_collection(dataset_type: str, client: Optional[MilvusClient] = None) -> None:
    """删除指定数据集的 Collection"""
    c = client or _get_client()
    coll_name = get_collection_name(dataset_type)
    if c.has_collection(coll_name):
        c.drop_collection(coll_name)
        logger.info("Collection dropped: %s", coll_name)


def get_collection_stats() -> dict:
    """获取所有数据集 collection 的统计信息"""
    c = _get_client()
    result = {}

    for ds in get_all_dataset_types():
        coll_name = get_collection_name(ds)
        if c.has_collection(coll_name):
            stats = c.get_collection_stats(coll_name)
            result[ds] = {"collection": coll_name, "row_count": stats.get("row_count", 0)}
        else:
            result[ds] = {"collection": coll_name, "row_count": 0, "status": "not_created"}

    return result
