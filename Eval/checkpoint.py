"""EvalCheckpointManager — 轻量状态追踪 + 原子持久化。

双文件设计（参考 Index/benchmark_chunker_v3.py）：
  - checkpoint/{dataset}.json ：仅存元数据（状态、错误、尝试次数）
  - result/{dataset}.json     ：存完整结果（问题、答案、预测等）

通过临时文件 + os.replace 实现原子写入，崩溃时不会损坏原文件。
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


class EvalCheckpointManager:
    """管理单个 (eval_mode, dataset_type) 对的 checkpoint 和 result 文件。

    目录结构:
        Eval/
          {eval_mode}_data/
            checkpoint/{dataset_type}.json   ← 仅状态
            result/{dataset_type}.json       ← 完整结果
    """

    def __init__(self, eval_mode: str, dataset_type: str, base_dir: str = "Eval"):
        self.mode = eval_mode
        self.dataset = dataset_type

        # 数据目录使用 "{mode}_data" 后缀，避免与同名模块文件冲突
        # （例如 Eval/llm_only.py 与 Eval/llm_only/ 同名时的导入冲突）
        data_dir = eval_mode if eval_mode.endswith("_data") else f"{eval_mode}_data"
        base = Path(base_dir)
        self.checkpoint_path = base / data_dir / "checkpoint" / f"{dataset_type}.json"
        self.result_path = base / data_dir / "result" / f"{dataset_type}.json"

        # 确保目录存在
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.result_path.parent.mkdir(parents=True, exist_ok=True)

        self.state: dict = {"entries": {}}

    # ── 加载 ──────────────────────────────────────────────────────

    def load(self) -> dict[str, dict]:
        """加载 checkpoint 文件，返回 entries 字典。

        文件不存在或损坏时返回空字典。
        """
        try:
            with open(self.checkpoint_path, encoding="utf-8") as f:
                data = json.load(f)
            self.state = data
            return data.get("entries", {})
        except FileNotFoundError:
            logger.info("无 checkpoint 文件 %s — 从头开始。", self.checkpoint_path)
            return {}
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("checkpoint 文件损坏 %s: %s — 从头开始。", self.checkpoint_path, e)
            return {}

    def load_results(self) -> list[dict]:
        """加载 result 文件。兼容新旧格式。

        新格式: {"summary": {...}, "results": [...]}
        旧格式: [...]（纯数组）
        """
        try:
            with open(self.result_path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "results" in data:
                return data["results"]
            return data  # 旧格式：纯数组
        except FileNotFoundError:
            logger.info("无 result 文件 %s — 从头开始。", self.result_path)
            return []
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("result 文件损坏 %s: %s — 从头开始。", self.result_path, e)
            return []

    # ── 查询待处理索引 ────────────────────────────────────────────

    def get_pending_indices(self, total: int, retry_failed: bool = False) -> list[int]:
        """返回需要处理的问题索引列表。

        Args:
            total: benchmark 中的问题总数。
            retry_failed: 为 True 时仅返回 state=="failed" 的索引；
                         为 False 时返回所有 state!="success" 的索引。
        """
        pending = []
        for i in range(total):
            key = str(i)
            entry = self.state.get("entries", {}).get(key, {})
            state = entry.get("state", "pending")

            if state == "success":
                continue
            if retry_failed and state != "failed":
                continue
            pending.append(i)
        return pending

    # ── 更新条目 ──────────────────────────────────────────────────

    def update_entry(self, index: int, state: str, error: str | None,
                     attempt: int, latency_ms: float):
        """更新 checkpoint 中单个条目的状态。"""
        key = str(index)
        self.state.setdefault("entries", {})[key] = {
            "state": state,
            "error": error,
            "attempt_count": attempt,
            "latency_ms": round(latency_ms, 2),
            "last_updated": _now_iso(),
        }
        self.state["last_updated"] = _now_iso()

    # ── 保存 ──────────────────────────────────────────────────────

    def save(self, entries: dict[str, dict], results: list[dict], summary: dict | None = None):
        """原子写入 checkpoint 和 result 文件。

        Args:
            entries: 写入 checkpoint 的完整 entries 字典。
            results: 写入 result 文件的结果列表。
            summary: 聚合指标 summary（写入 result 文件顶端）。
        """
        # 按 question_index 排序，保证输出确定性
        results.sort(key=lambda r: r["question_index"])

        self.state["entries"] = entries
        self.state["eval_mode"] = self.mode
        self.state["dataset_type"] = self.dataset
        self.state["last_updated"] = _now_iso()

        # result 使用新格式：{"summary": ..., "results": [...]}
        result_data = {
            "summary": summary or {},
            "results": results,
        }

        self._atomic_write(self.checkpoint_path, self.state)
        self._atomic_write(self.result_path, result_data)
        logger.info("已保存 checkpoint（%d 条目）+ results（%d 项），%s/%s",
                     len(entries), len(results), self.mode, self.dataset)

    def _atomic_write(self, path: Path, data):
        """先写临时文件，再原子重命名。

        先写入 {name}.tmp，内容完整后 os.replace() 替换原文件。
        Python 的 os.replace() 在同一文件系统上是原子操作。
        崩溃发生在写入阶段 → .tmp 可能损坏但原文件完好；
        崩溃发生在重命名阶段 → 原文件完好或已被替换，不会半写半不写。
        """
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        tmp.replace(path)

    # ── 进度摘要 ──────────────────────────────────────────────────

    def get_summary(self, total: int) -> str:
        """返回人类可读的进度摘要。"""
        entries = self.state.get("entries", {})
        counts = {"success": 0, "failed": 0, "pending": 0}
        for i in range(total):
            state = entries.get(str(i), {}).get("state", "pending")
            counts[state] = counts.get(state, 0) + 1

        lines = [
            f"数据集: {self.dataset}",
            f"模式:   {self.mode}",
            f"总数:   {total}",
            f"成功:   {counts['success']}",
            f"失败:   {counts['failed']}",
            f"待处理: {counts['pending']}",
        ]
        return "\n".join(lines)


def _now_iso() -> str:
    """返回当前 UTC 时间的 ISO 8601 格式字符串。"""
    return datetime.now(timezone.utc).isoformat()
