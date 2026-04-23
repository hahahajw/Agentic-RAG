"""离线索引：展平 chunked JSON → 计算嵌入 → 批量插入 Milvus

每个数据集独立 collection，互不影响。

两阶段流程：
  阶段一：计算所有命题的嵌入并持久化到 embeddings 文件
  阶段二：从 embeddings 文件读取，批量 upsert 到 Milvus
"""

import hashlib
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# 确保项目根目录在 sys.path 中
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from Index.milvus_config import (
    get_embedding_function,
    get_chunked_path,
    get_all_dataset_types,
    get_collection_name,
    EMBED_BATCH_SIZE,
)
from Index.milvus_schema import _get_client, build_schema, create_collection

logger = logging.getLogger(__name__)


# ── 数据记录 ─────────────────────────────────────────────────


@dataclass
class PropositionRecord:
    """Milvus 集合中的一行"""

    id: str
    chunk_id: str
    question_id: str
    context_index: int
    context_title: str
    chunk_title: str
    chunk_summary: str
    proposition_text: str


# ── Embedding 状态管理 ──────────────────────────────────────


class EmbeddingState:
    """管理命题嵌入的持久化与恢复

    状态文件包含所有命题的完整信息（含嵌入），
    通过 save_interval 控制写入频率，确保 crash 时不丢失已完成工作。
    """

    EMBEDDING_FIELD = "proposition_text_embedding"
    STATE_FIELD = "embedding_state"

    def __init__(self, dataset_type: str, save_interval: int = 1):
        self.filepath = get_chunked_path(dataset_type).parent / f"{dataset_type}_embeddings.json"
        self.save_interval = save_interval
        self._batch_counter = 0

    def load(self) -> List[Dict[str, Any]]:
        """加载状态文件，返回命题列表"""
        if not self.filepath.exists():
            return []

        with open(self.filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        logger.info("从状态文件恢复: %d 个命题", len(data))
        return data

    def save(self, propositions: List[Dict[str, Any]]) -> None:
        """原子写入状态文件"""
        tmp_path = self.filepath.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(propositions, f, ensure_ascii=False)
            f.flush()
        tmp_path.replace(self.filepath)

    @staticmethod
    def to_dict(record: PropositionRecord) -> Dict[str, Any]:
        """PropositionRecord → 可序列化的状态字典"""
        d = asdict(record)
        d[EmbeddingState.EMBEDDING_FIELD] = None
        d[EmbeddingState.STATE_FIELD] = "pending"
        return d

    @classmethod
    def from_records(cls, records: List[PropositionRecord]) -> List[Dict[str, Any]]:
        """批量转换 PropositionRecord 为状态字典"""
        return [cls.to_dict(r) for r in records]

    def filter_by_state(self, propositions: List[Dict[str, Any]], state: str) -> List[Dict[str, Any]]:
        """筛选指定状态的命题"""
        return [p for p in propositions if p[self.STATE_FIELD] == state]

    def update_batch(
        self,
        propositions: List[Dict[str, Any]],
        indices: List[int],
        embeddings: Optional[List[List[float]]] = None,
        state: str = "success",
        error: str = "",
    ) -> None:
        """更新一批命题的嵌入结果

        Args:
            propositions: 完整命题列表（会被原地修改）
            indices: 要更新的命题在列表中的索引
            embeddings: 嵌入向量列表（None 时为失败情况）
            state: 新状态 "success" | "failed"
            error: 失败时的错误信息
        """
        for i, idx in enumerate(indices):
            p = propositions[idx]
            p[self.STATE_FIELD] = state
            if embeddings and i < len(embeddings):
                p[self.EMBEDDING_FIELD] = embeddings[i]
            if error:
                p["embedding_error"] = error

        if self._should_save():
            self.save(propositions)

    def exists(self) -> bool:
        return self.filepath.exists()

    def clear(self) -> None:
        if self.filepath.exists():
            self.filepath.unlink()
            logger.info("已清除状态文件: %s", self.filepath)

    def _should_save(self) -> bool:
        """判断是否需要写入磁盘"""
        if self.save_interval <= 0:
            return False
        self._batch_counter += 1
        return self._batch_counter % self.save_interval == 0

    def force_save(self, propositions: List[Dict[str, Any]]) -> None:
        """强制写入磁盘，无视 save_interval"""
        self.save(propositions)


# ── 数据展平 ─────────────────────────────────────────────────


class DataFlattener:
    """将嵌套的 chunked JSON 展平为 PropositionRecord 列表"""

    @staticmethod
    def _make_prop_id(chunk_id: str, prop_text: str, chunk_index: int) -> str:
        """生成确定性的 proposition ID"""
        h = hashlib.md5(f"{chunk_id}_{chunk_index}_{prop_text}".encode()).hexdigest()[:16]
        return f"p_{h}"

    @staticmethod
    def flatten_hotpotqa(filepath: str, dataset_type: str) -> List[PropositionRecord]:
        """展平 HotpotQA / 2WikiMultihopQA 格式（context 为列表）"""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        records: List[PropositionRecord] = []
        for item in data:
            question_id = item.get("_id", "")
            for ctx_idx, ctx in enumerate(item.get("context", [])):
                if len(ctx) < 3 or not isinstance(ctx[2], dict):
                    continue
                title = ctx[0]
                meta = ctx[2]
                chunks = meta.get("chunks", {})
                for chunk_id, chunk_data in chunks.items():
                    props = chunk_data.get("propositions", [])
                    chunk_title = chunk_data.get("title", "")
                    chunk_summary = chunk_data.get("summary", "")
                    chunk_index_val = chunk_data.get("chunk_index", 0)
                    for prop_text in props:
                        records.append(PropositionRecord(
                            id=DataFlattener._make_prop_id(chunk_id, prop_text, chunk_index_val),
                            chunk_id=chunk_id,
                            question_id=question_id,
                            context_index=ctx_idx,
                            context_title=title,
                            chunk_title=chunk_title,
                            chunk_summary=chunk_summary,
                            proposition_text=prop_text,
                        ))
        return records

    @staticmethod
    def flatten_musique(filepath: str) -> List[PropositionRecord]:
        """展平 MuSiQue 格式（paragraphs 为列表）"""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        records: List[PropositionRecord] = []
        for item in data:
            question_id = item.get("id", "")
            for para_idx, para in enumerate(item.get("paragraphs", [])):
                title = para.get("title", "")
                chunks = para.get("chunks", {})
                for chunk_id, chunk_data in chunks.items():
                    props = chunk_data.get("propositions", [])
                    chunk_title = chunk_data.get("title", "")
                    chunk_summary = chunk_data.get("summary", "")
                    chunk_index_val = chunk_data.get("chunk_index", 0)
                    for prop_text in props:
                        records.append(PropositionRecord(
                            id=DataFlattener._make_prop_id(chunk_id, prop_text, chunk_index_val),
                            chunk_id=chunk_id,
                            question_id=question_id,
                            context_index=para_idx,
                            context_title=title,
                            chunk_title=chunk_title,
                            chunk_summary=chunk_summary,
                            proposition_text=prop_text,
                        ))
        return records

    @classmethod
    def flatten(cls, dataset_type: str, filepath: Optional[str] = None) -> List[PropositionRecord]:
        """统一入口"""
        ds = dataset_type.lower()
        if filepath is None:
            filepath = str(get_chunked_path(ds))
        if ds == "musique":
            return cls.flatten_musique(filepath)
        else:
            return cls.flatten_hotpotqa(filepath, dataset_type=ds)


# ── 嵌入计算 ─────────────────────────────────────────────────


class EmbeddingComputer:
    """批量计算命题的密集嵌入，支持重试和断点续算"""

    def __init__(
        self,
        batch_size: int = EMBED_BATCH_SIZE,
        max_retries: int = 3,
    ):
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.embedding_func = get_embedding_function()

    def _embed_with_retry(self, texts: List[str]) -> Optional[List[List[float]]]:
        """带指数退避重试的嵌入计算"""
        for attempt in range(self.max_retries):
            try:
                return self.embedding_func.embed_documents(texts)
            except Exception as e:
                if attempt < self.max_retries - 1:
                    delay = 2 ** (attempt + 1)  # 2s, 4s, 8s
                    logger.warning("嵌入计算失败 (attempt %d/%d)，%d 秒后重试: %s",
                                   attempt + 1, self.max_retries, delay, e)
                    time.sleep(delay)
                else:
                    logger.error("嵌入计算失败，已达最大重试次数 %d: %s",
                                 self.max_retries, e)
                    return None

    def compute(self, propositions: List[Dict[str, Any]], state: EmbeddingState) -> Dict[str, int]:
        """计算所有 pending 命题的嵌入

        Args:
            propositions: 完整命题状态列表（会被原地更新）
            state: EmbeddingState 实例，用于持久化

        Returns:
            统计信息 {"success": N, "failed": N, "skipped": N}
        """
        stats = {"success": 0, "failed": 0, "skipped": 0}

        # 收集待计算的命题索引
        pending = state.filter_by_state(propositions, "pending")
        failed_prev = state.filter_by_state(propositions, "failed")
        to_compute = pending + failed_prev
        stats["skipped"] = len(propositions) - len(to_compute)

        if not to_compute:
            logger.info("所有命题嵌入已完成，跳过计算")
            return stats

        # 构建索引映射: proposition id → 在 propositions 列表中的索引
        id_to_index = {p["id"]: i for i, p in enumerate(propositions)}

        total = len(to_compute)
        logger.info("计算嵌入: %d 个 pending, %d 个之前失败, %d 个已完成",
                     len(pending), len(failed_prev), stats["skipped"])

        start = time.time()
        computed = 0

        for i in range(0, total, self.batch_size):
            batch = to_compute[i:i + self.batch_size]
            batch_indices = [id_to_index[p["id"]] for p in batch]
            batch_texts = [p["proposition_text"] for p in batch]

            result = self._embed_with_retry(batch_texts)

            if result is not None:
                state.update_batch(propositions, batch_indices, embeddings=result, state="success")
                stats["success"] += len(batch)
            else:
                state.update_batch(propositions, batch_indices, state="failed",
                                   error=f"重试 {self.max_retries} 次后仍失败")
                stats["failed"] += len(batch)

            computed += len(batch)
            if (i // self.batch_size + 1) % 50 == 0 or computed >= total:
                elapsed = time.time() - start
                logger.info("嵌入进度: %d/%d (%.1f%%), 耗时 %.1fs",
                            computed, total, computed / total * 100, elapsed)

        elapsed = time.time() - start
        logger.info("嵌入计算完成: 成功 %d, 失败 %d, 跳过 %d, 耗时 %.1fs",
                     stats["success"], stats["failed"], stats["skipped"], elapsed)

        # 确保最后未触发的写入被执行
        state.force_save(propositions)

        return stats


# ── 数据插入 ─────────────────────────────────────────────────


class DataInserter:
    """从 EmbeddingState 读取已完成的嵌入，批量 upsert 到 Milvus"""

    def __init__(self, dataset_type: str, batch_size: int = 500):
        self.batch_size = batch_size
        self.collection_name = get_collection_name(dataset_type)

    def insert_from_state(
        self,
        propositions: List[Dict[str, Any]],
    ) -> Dict[str, int]:
        """从状态列表中提取成功嵌入的命题并 upsert 到 Milvus

        Returns:
            {"upserted": N, "skipped_failed": N, "skipped_pending": N}
        """
        client = _get_client()

        # 只插入状态为 success 的命题
        success = [p for p in propositions if p["embedding_state"] == "success"]
        failed = [p for p in propositions if p["embedding_state"] == "failed"]
        pending = [p for p in propositions if p["embedding_state"] == "pending"]

        if not success:
            logger.warning("没有状态为 success 的命题可以插入")
            return {"upserted": 0, "skipped_failed": len(failed), "skipped_pending": len(pending)}

        total = len(success)
        upserted = 0
        embed_field = EmbeddingState.EMBEDDING_FIELD

        # Milvus 需要的字段（proposition_text_embedding → embedding 映射）
        milvus_fields = [
            "id", "chunk_id", "question_id", "context_index", "context_title",
            "chunk_title", "chunk_summary", "proposition_text", "embedding",
        ]

        for i in range(0, total, self.batch_size):
            end = min(i + self.batch_size, total)
            batch = success[i:end]

            entities = []
            for p in batch:
                entity = {field: p.get(field) for field in milvus_fields[:-1]}
                # 将状态文件中的嵌入字段映射到 Milvus 的 embedding 字段
                entity["embedding"] = p.get(EmbeddingState.EMBEDDING_FIELD)
                entities.append(entity)

            res = client.upsert(collection_name=self.collection_name, data=entities)
            upserted += res.get("upsert_count", len(entities))

            if (i // self.batch_size + 1) % 10 == 0 or end == total:
                logger.info("Upserted %d/%d (%.1f%%)", upserted, total, upserted / total * 100)

        client.flush(collection_name=self.collection_name)

        logger.info("插入完成: upserted %d, 跳过失败 %d, 跳过 pending %d",
                     upserted, len(failed), len(pending))

        return {
            "upserted": upserted,
            "skipped_failed": len(failed),
            "skipped_pending": len(pending),
        }


# ── 单数据集构建 ──────────────────────────────────────────────


def _build_single_dataset_index(
    dataset_type: str,
    limit: Optional[int] = None,
    rebuild: bool = False,
    save_interval: int = 1,
) -> dict:
    """为单个数据集构建索引

    两阶段流程：
      阶段一：展平 → 加载/创建状态文件 → 计算所有嵌入 → 持久化
      阶段二：创建 Milvus collection → 从状态文件读取 → upsert
    """
    ds = dataset_type.lower()
    path = get_chunked_path(ds)

    # ── 阶段一：嵌入计算 ──────────────────────────────

    state_mgr = EmbeddingState(ds, save_interval=save_interval)

    if rebuild and state_mgr.exists():
        # rebuild 仅重置 Milvus collection，保留已有 embedding 结果
        # 只将 failed 命题重置为 pending 以便重新计算
        propositions = state_mgr.load()
        failed = state_mgr.filter_by_state(propositions, "failed")
        if failed:
            for p in failed:
                p[state_mgr.STATE_FIELD] = "pending"
            state_mgr.force_save(propositions)
            logger.info("已重置 %d 个失败的命题为 pending", len(failed))
    elif rebuild:
        # 无状态文件，从零开始
        pass

    # 加载已有状态或创建新状态
    if state_mgr.exists():
        logger.info("加载已有状态文件: %s", state_mgr.filepath)
        propositions = state_mgr.load()
    else:
        logger.info("展平 %s from %s", ds, path)
        records = DataFlattener.flatten(ds, str(path))
        if not records:
            return {"dataset": ds, "error": "No propositions found"}

        if limit:
            seen_qids = set()
            truncated = []
            for r in records:
                if r.question_id not in seen_qids and len(seen_qids) >= limit:
                    break
                seen_qids.add(r.question_id)
                truncated.append(r)
            records = truncated

        logger.info("%s: %d propositions", ds, len(records))
        propositions = EmbeddingState.from_records(records)
        state_mgr.save(propositions)

    # 计算嵌入
    computer = EmbeddingComputer()
    embed_stats = computer.compute(propositions, state_mgr)

    # ── 阶段二：Milvus 插入 ───────────────────────────

    logger.info("创建 collection for %s", ds)
    create_collection(dataset_type=ds, drop_old=rebuild)

    inserter = DataInserter(ds)
    start = time.time()
    insert_stats = inserter.insert_from_state(propositions)
    insert_elapsed = time.time() - start

    result = {
        "dataset": ds,
        "collection": get_collection_name(ds),
        "propositions": len(propositions),
        "embedding_success": embed_stats["success"],
        "embedding_failed": embed_stats["failed"],
        "embedding_skipped": embed_stats["skipped"],
        "upserted": insert_stats["upserted"],
        "skipped_failed": insert_stats["skipped_failed"],
        "skipped_pending": insert_stats["skipped_pending"],
        "insert_time_s": round(insert_elapsed, 1),
    }

    return result


# ── 顶层构建流程 ──────────────────────────────────────────────


def build_index(
    dataset_types: Optional[List[str]] = None,
    limit: Optional[int] = None,
    rebuild: bool = False,
    save_interval: int = 1,
) -> dict:
    """
    完整构建流程：对每个数据集独立执行 展平 → 嵌入 → 插入

    Returns:
        统计信息 dict
    """
    if dataset_types is None:
        dataset_types = get_all_dataset_types()

    overall_start = time.time()
    results = []

    for ds in dataset_types:
        try:
            result = _build_single_dataset_index(ds, limit=limit, rebuild=rebuild, save_interval=save_interval)
            results.append(result)
        except Exception as e:
            logger.exception("Failed to build index for %s", ds)
            results.append({"dataset": ds, "error": str(e)})

    total_time = time.time() - overall_start
    return {
        "datasets": results,
        "total_time_s": round(total_time, 1),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    datasets = None
    limit = None
    rebuild = False
    save_every = 1

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--dataset" and i + 1 < len(args):
            datasets = [args[i + 1]]
            i += 2
        elif args[i] == "--limit" and i + 1 < len(args):
            limit = int(args[i + 1])
            i += 2
        elif args[i] == "--rebuild":
            rebuild = True
            i += 1
        elif args[i] == "--save-every" and i + 1 < len(args):
            save_every = int(args[i + 1])
            i += 2
        elif args[i] == "--all":
            i += 1
        else:
            i += 1

    result = build_index(dataset_types=datasets, limit=limit, rebuild=rebuild, save_interval=save_every)
    print(json.dumps(result, indent=2, ensure_ascii=False))
