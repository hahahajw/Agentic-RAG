"""评估器基础框架 — 数据归一化 + 批顺序处理。

设计：
  - NormalizedQuestion: 跨数据集统一的问题结构
  - load_benchmark(): 读取原始 benchmark JSON，处理 _id/id 差异
  - BaseEvaluator: 组织批顺序处理流程，包含
    ThreadPoolExecutor 并行、瞬态错误重试、checkpoint 持久化
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from langchain_openai import ChatOpenAI

from Eval.checkpoint import EvalCheckpointManager
from Eval.metrics import exact_match_score, f1_score

logger = logging.getLogger(__name__)


# ─── 数据归一化 ──────────────────────────────────────────────────

@dataclass
class NormalizedQuestion:
    """跨数据集统一的问题结构。

    HotpotQA 和 2Wiki 使用 `_id` 字段，MuSiQue 使用 `id` 字段。
    HotpotQA 和 2Wiki 使用 `context` 列表，MuSiQue 使用 `paragraphs` 列表。
    本类将两者归一化，使 checkpoint 系统保持数据集无关。
    """
    index: int          # benchmark 数组索引（0-based）
    question_id: str    # _id 或 id — 稳定的跨数据集标识符
    question: str       # 问题文本
    answer: str         # 标准答案（ground truth）
    raw: dict           # 原始完整条目（供需要 context 的 RAG 模式使用）


def load_benchmark(dataset_type: str, source_dir: str = "Data/benchmark") -> list[NormalizedQuestion]:
    """加载 benchmark JSON 并跨数据集模式归一化。

    处理差异：
      - HotpotQA / 2Wiki: `_id` 字段，`context` 列表
      - MuSiQue: `id` 字段，`paragraphs` 列表
    """
    source_path = Path(source_dir) / f"{dataset_type}_500_benchmark.json"
    with open(source_path, encoding="utf-8") as f:
        data = json.load(f)

    questions = []
    for idx, entry in enumerate(data):
        # ID 字段：HotpotQA/2Wiki 用 "_id"，MuSiQue 用 "id"
        qid = entry.get("_id") or entry.get("id", f"q{idx}")

        questions.append(NormalizedQuestion(
            index=idx,
            question_id=qid,
            question=entry["question"],
            answer=str(entry["answer"]),
            raw=entry,
        ))
    return questions


# ─── 评估器基类 ──────────────────────────────────────────────────

# 可安全自动重试的瞬态错误类型
_TRANSIENT_ERRORS = (
    ConnectionError,
    TimeoutError,
    OSError,  # 包含 socket 错误、管道断裂等
)


class BaseEvaluator(ABC):
    """支持 checkpoint 的批顺序评估器。

    处理流程：
      1. 加载 benchmark 数据
      2. 从 checkpoint 确定待处理索引
      3. 按 batch_size 分组
      4. 对每个 batch：
         a. 用 ThreadPoolExecutor 并行处理（max_workers）
         b. 更新 checkpoint 条目
         c. 原子保存（checkpoint + result）
      5. 返回全部结果

    重试策略：
      - 瞬态错误（网络、超时）：在每个 worker 内自动重试 max_retries 次，
        使用指数退避
      - 持久性错误（模型拒绝、格式错误）：立即标记为 failed
      - --retry-failed：用户手动触发的对所有 failed 条目的重试
    """

    def __init__(
        self,
        eval_mode: str,
        llm: ChatOpenAI,
        dataset_type: str,
        batch_size: int = 20,
        max_workers: int = 5,
        max_retries: int = 2,
        retry_delay: float = 5.0,
    ):
        self.eval_mode = eval_mode
        self.llm = llm
        self.dataset_type = dataset_type
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self.checkpoint = EvalCheckpointManager(eval_mode, dataset_type)

    def run(self, retry_failed: bool = False, force: bool = False) -> list[dict]:
        """主处理循环。

        Args:
            retry_failed: 为 True 时仅处理 state=="failed" 的条目
            force: 为 True 时忽略所有 checkpoint 状态，全部重跑
        """
        benchmark = load_benchmark(self.dataset_type)
        total = len(benchmark)
        logger.info("已加载 %d 个问题（数据集: %s）", total, self.dataset_type)

        if force:
            logger.info("强制模式：忽略已有 checkpoint。")
            existing_entries: dict[str, dict] = {}
        else:
            existing_entries = self.checkpoint.load()

        # 加载已有结果（保留之前运行的答案）
        existing_results = self.checkpoint.load_results()
        results_map = {r["question_index"]: r for r in existing_results}

        # 确定需要处理的索引
        if force:
            pending = list(range(total))
        else:
            pending = self.checkpoint.get_pending_indices(total, retry_failed=retry_failed)

        if not pending:
            logger.info("无需处理。")
            return sorted(results_map.values(), key=lambda r: r["question_index"])

        logger.info("需处理 %d 个问题（总计 %d 个）", len(pending), total)

        # 按 batch 分组
        batches = [pending[i:i + self.batch_size]
                   for i in range(0, len(pending), self.batch_size)]
        logger.info("分为 %d 个 batch（每批 %d 个问题）", len(batches), self.batch_size)

        # 合并已有 checkpoint 条目
        entries = dict(existing_entries)

        for batch_idx, indices in enumerate(batches, 1):
            logger.info("=== Batch %d/%d（%d 个问题）===",
                         batch_idx, len(batches), len(indices))

            batch_questions = [benchmark[i] for i in indices]
            batch_results = self._process_batch(batch_questions)

            # 更新 checkpoint 条目和结果
            for result in batch_results:
                idx = result["question_index"]
                key = str(idx)
                q = benchmark[idx]

                if result["error"] is None:
                    # 处理成功 — 计算逐题指标
                    em = exact_match_score(result["prediction"], q.answer)
                    cur_f1, prec, rec = f1_score(result["prediction"], q.answer)

                    entries[key] = {
                        "state": "success",
                        "error": None,
                        "attempt_count": entries.get(key, {}).get("attempt_count", 0) + 1,
                        "latency_ms": result["latency_ms"],
                        "last_updated": result["timestamp"],
                    }
                    results_map[idx] = {
                        "question_index": idx,
                        "question_id": q.question_id,
                        "question": q.question,
                        "answer": q.answer,
                        "prediction": result["prediction"],
                        "error": None,
                        "latency_ms": result["latency_ms"],
                        "dataset_type": self.dataset_type,
                        "timestamp": result["timestamp"],
                        "chunks": result.get("chunks", []),
                        "em": em,
                        "f1": round(cur_f1, 6),
                        "precision": round(prec, 6),
                        "recall": round(rec, 6),
                        "context_recall": result.get("context_recall"),
                        "hit": result.get("hit"),
                        "mrr": result.get("mrr"),
                        "retrieval_precision": result.get("retrieval_precision"),
                        # Judge RAG 效率指标
                        "retrieval_count": result.get("retrieval_count"),
                        "total_chunks": result.get("total_chunks"),
                        "total_distinct_titles": result.get("total_distinct_titles"),
                        "search_depth": result.get("search_depth"),
                        "search_path": result.get("search_path"),
                        # Agentic RAG v2 特有：子问题解决状态
                        "sub_questions": result.get("sub_questions"),
                    }
                else:
                    # 处理失败
                    entries[key] = {
                        "state": "failed",
                        "error": result["error"],
                        "attempt_count": entries.get(key, {}).get("attempt_count", 0) + 1,
                        "latency_ms": result["latency_ms"],
                        "last_updated": result["timestamp"],
                    }
                    results_map[idx] = {
                        "question_index": idx,
                        "question_id": q.question_id,
                        "question": q.question,
                        "answer": q.answer,
                        "prediction": None,
                        "error": result["error"],
                        "latency_ms": result["latency_ms"],
                        "dataset_type": self.dataset_type,
                        "timestamp": result["timestamp"],
                        "chunks": [],
                        "em": None,
                        "f1": None,
                        "precision": None,
                        "recall": None,
                        "context_recall": None,
                        "hit": None,
                        "mrr": None,
                        "retrieval_precision": None,
                    }

            # 每个 batch 结束后原子保存
            all_results = sorted(results_map.values(), key=lambda r: r["question_index"])
            summary = self._compute_summary(results_map, self.eval_mode, self.dataset_type)
            self.checkpoint.save(entries, all_results, summary=summary)

            # 进度日志
            done = sum(1 for e in entries.values() if e["state"] == "success")
            logger.info("Batch %d 完成。总计：%d/%d 成功，%d 失败，%d 待处理",
                         batch_idx, done, total,
                         sum(1 for e in entries.values() if e["state"] == "failed"),
                         sum(1 for e in entries.values() if e["state"] == "pending"))

        final_results = sorted(results_map.values(), key=lambda r: r["question_index"])
        logger.info("评估完成。已回答：%d/%d",
                     sum(1 for r in final_results if r["prediction"] is not None),
                     total)
        return final_results

    def _process_batch(self, questions: list[NormalizedQuestion]) -> list[dict]:
        """用 ThreadPoolExecutor 并行处理一个 batch 的问题。

        返回 [{question_index, prediction, error, latency_ms, timestamp}, ...]
        """
        results = [None] * len(questions)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {}
            for pos, q in enumerate(questions):
                future = executor.submit(self._process_single_with_retry, q)
                future_to_idx[future] = pos

            for future in as_completed(future_to_idx):
                pos = future_to_idx[future]
                try:
                    results[pos] = future.result()
                except Exception as e:
                    # 不应发生（内部重试已捕获所有异常），但作为安全网
                    q = questions[pos]
                    results[pos] = {
                        "question_index": q.index,
                        "prediction": None,
                        "error": f"未捕获异常: {type(e).__name__}: {e}",
                        "latency_ms": 0.0,
                        "timestamp": self._now_iso(),
                    }
                    logger.error("问题 %d 未捕获异常: %s", q.index, e)

        return results

    def _process_single_with_retry(self, question: NormalizedQuestion) -> dict:
        """处理单个问题，包含瞬态错误重试。

        重试策略：
          - 瞬态错误（ConnectionError, TimeoutError, OSError）：
            使用指数退避重试最多 max_retries 次
          - 其他错误：不重试，直接标记为 failed
        """
        last_error = None
        for attempt in range(1, self.max_retries + 2):  # 1 次初始 + max_retries 次重试
            try:
                start = time.perf_counter()
                outcome = self.evaluate_single(question)
                latency = (time.perf_counter() - start) * 1000

                return {
                    "question_index": question.index,
                    "prediction": outcome.get("prediction"),
                    "error": outcome.get("error"),
                    "latency_ms": latency,
                    "timestamp": self._now_iso(),
                    "chunks": outcome.get("chunks", []),
                    "context_recall": outcome.get("context_recall"),
                    "hit": outcome.get("hit"),
                    "mrr": outcome.get("mrr"),
                    "retrieval_precision": outcome.get("retrieval_precision"),
                    # Judge RAG 效率指标
                    "retrieval_count": outcome.get("retrieval_count"),
                    "total_chunks": outcome.get("total_chunks"),
                    "total_distinct_titles": outcome.get("total_distinct_titles"),
                    "search_depth": outcome.get("search_depth"),
                    "search_path": outcome.get("search_path"),
                    # Agentic RAG v2 特有：子问题解决状态
                    "sub_questions": outcome.get("sub_questions"),
                }

            except _TRANSIENT_ERRORS as e:
                last_error = e
                delay = self.retry_delay * (2 ** (attempt - 1))  # 指数退避
                logger.warning("问题 %d 瞬态错误（第 %d 次尝试）: %s — %.1fs 后重试",
                               question.index, attempt, e, delay)
                time.sleep(delay)

            except Exception as e:
                # 持久性错误 — 不重试
                latency = (time.perf_counter() - start) * 1000 if 'start' in dir() else 0
                return {
                    "question_index": question.index,
                    "prediction": None,
                    "error": f"{type(e).__name__}: {e}",
                    "latency_ms": latency,
                    "timestamp": self._now_iso(),
                }

        # 所有重试耗尽
        return {
            "question_index": question.index,
            "prediction": None,
            "error": f"{type(last_error).__name__}: {last_error}",
            "latency_ms": 0.0,
            "timestamp": self._now_iso(),
        }

    @abstractmethod
    def evaluate_single(self, question: NormalizedQuestion) -> dict:
        """子类必须实现：返回 {prediction: str|None, error: str|None}。"""
        ...

    @staticmethod
    def _compute_summary(results_map: dict, eval_mode: str, dataset_type: str) -> dict:
        """计算聚合指标 summary。仅基于 prediction != null 的问题。"""
        total = len(results_map)
        answered_items = [r for r in results_map.values() if r.get("prediction") is not None]
        answered = len(answered_items)
        if answered:
            aggregate = {
                "em": round(sum(r["em"] for r in answered_items) / answered, 6),
                "f1": round(sum(r["f1"] for r in answered_items) / answered, 6),
                "precision": round(sum(r["precision"] for r in answered_items) / answered, 6),
                "recall": round(sum(r["recall"] for r in answered_items) / answered, 6),
            }
            # 检索指标（可能不存在于 LLM-only 模式）
            retrieval_keys = ["context_recall", "hit", "mrr", "retrieval_precision"]
            for key in retrieval_keys:
                values = [r[key] for r in answered_items if r.get(key) is not None]
                if values:
                    aggregate[key] = round(sum(values) / len(values), 6)
                else:
                    aggregate[key] = None
            # Judge RAG 效率指标
            efficiency_keys = ["retrieval_count", "total_chunks",
                               "total_distinct_titles", "search_depth"]
            for key in efficiency_keys:
                values = [r[key] for r in answered_items if r.get(key) is not None]
                if values:
                    aggregate[key] = round(sum(values) / len(values), 6)
                else:
                    aggregate[key] = None
        else:
            aggregate = {"em": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}
        return {
            "eval_mode": eval_mode,
            "dataset_type": dataset_type,
            "total": total,
            "answered": answered,
            "unanswered": total - answered,
            "aggregate": aggregate,
        }

    @staticmethod
    def _now_iso() -> str:
        """返回当前 UTC 时间的 ISO 8601 格式字符串。"""
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()
