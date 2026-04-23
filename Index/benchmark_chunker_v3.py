"""
Benchmark 分块处理器 V3 - 逐问题处理 + 问题内并行。

本模块是 benchmark_chunker 的 V3 版本，核心设计理念：
1. 逐问题顺序处理 - 天然限制并发量（~10-20 个/批），避免 API rate limit
2. 问题内并行批处理 - 每个问题内部的多个上下文并行处理
3. 每问题原子持久化 - checkpoint 和 benchmark 同步保存，保证一致性
4. 轻量 checkpoint - 只存状态元数据，不存完整 propositions/chunks

数据流向：
  加载数据 + checkpoint
    ↓
  构建 ProcessingUnit 列表 → 按问题分组为 QuestionGroup
    ↓
  For each QuestionGroup (顺序):
    ├─ Stage 1: 并行提取命题 (get_propositions_batch)
    ├─ Stage 2: 并行分块处理 (chunk_batch)
    ├─ 应用结果到 all_data
    └─ 原子保存 checkpoint + benchmark
    ↓
  最终保存

示例用法：
    >>> from langchain_openai import ChatOpenAI
    >>> llm = ChatOpenAI(model='qwen3-max', temperature=0)
    >>> processor = BenchmarkChunkProcessorV3(
    ...     llm=llm,
    ...     dataset_type='hotpotqa',
    ...     input_path='Data/benchmark/HotpotQA_500_benchmark.json'
    ... )
    >>> stats = processor.process(limit=10)
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict

from .agentic_chunk import AgenticChunk, PartialBatchError


# ==================== 日志 ====================

def _log(message: str, level: str = "info"):
    """打印日志消息"""
    prefix = {
        "info": "[INFO]",
        "success": "[OK]",
        "warning": "[WARN]",
        "error": "[ERROR]"
    }.get(level, "[•]")
    print(f"{prefix} {message}")


# ==================== 数据类 ====================

@dataclass
class ProcessingUnit:
    """
    最小处理单元 - 一个段落/上下文。

    封装单个 context/paragraph 的完整状态：
    - 定位：checkpoint_key, item_index, context_index, item_id
    - 内容：title, text
    - 运行时状态：propositions, chunks 及其状态
    """
    # 定位
    checkpoint_key: str       # "q0/c1"
    item_index: int           # 问题在 benchmark 列表中的索引
    context_index: int        # 上下文在问题中的索引
    item_id: str              # 问题唯一 ID
    title: str                # 段落标题

    # 内容
    text: str = ""            # "Title: xxx. Content: xxx."

    # 命题阶段
    propositions_state: str = "pending"  # pending | success | failed
    propositions: Optional[List[str]] = None

    # 分块阶段
    chunks_state: str = "pending"
    chunks: Optional[Dict] = None

    # 元数据
    attempt_count: int = 0
    last_updated: Optional[str] = None


@dataclass
class QuestionGroup:
    """属于同一个问题的处理单元组"""
    item_index: int
    item_id: str
    units: List[ProcessingUnit]


# ==================== 工具函数 ====================

def generate_checkpoint_key(item_index: int, context_index: int) -> str:
    """生成扁平 checkpoint 键："q{item_index}/c{context_index}" """
    return f"q{item_index}/c{context_index}"


def parse_checkpoint_key(key: str) -> Tuple[int, int]:
    """解析 checkpoint 键，返回 (item_index, context_index)"""
    q_part, c_part = key.split('/')
    return int(q_part[1:]), int(c_part[1:])


# ==================== Checkpoint 管理器 ====================

class CheckpointManager:
    """
    轻量 Checkpoint 管理器。

    只存状态元数据（propositions_state, chunks_state, attempt_count），
    不存 propositions/chunks 的完整内容，避免 checkpoint 膨胀。
    完整数据由 benchmark JSON 文件管理。

    Checkpoint 结构：
    {
        "dataset_type": "hotpotqa",
        "last_updated": "2026-04-02T...",
        "entries": {
            "q0/c0": {
                "item_id": "5abc...",
                "title": "Moscow State University",
                "propositions_state": "success",
                "chunks_state": "success",
                "attempt_count": 1,
                "last_updated": "..."
            }
        }
    }
    """

    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = checkpoint_path
        self.state = self._load()

    def _load(self) -> Dict:
        if not self.checkpoint_path.exists():
            return {"dataset_type": None, "last_updated": None, "entries": {}}
        try:
            with open(self.checkpoint_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[WARN] checkpoint 文件损坏，将重新创建：{e}")
            return {"dataset_type": None, "last_updated": None, "entries": {}}

    def save(self):
        """原子写入 checkpoint"""
        self.state["last_updated"] = datetime.now().isoformat()
        temp_path = self.checkpoint_path.with_suffix('.tmp')
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2, default=str)
        temp_path.replace(self.checkpoint_path)

    def set_dataset_type(self, dataset_type: str):
        self.state["dataset_type"] = dataset_type

    def get_entry(self, key: str) -> Optional[Dict]:
        return self.state.get("entries", {}).get(key)

    def update_entry(self, key: str, **updates):
        if "entries" not in self.state:
            self.state["entries"] = {}
        if key not in self.state["entries"]:
            self.state["entries"][key] = {}
        entry = self.state["entries"][key]
        entry.update(updates)
        entry["last_updated"] = datetime.now().isoformat()

    def filter_entries(self, **criteria) -> List[str]:
        """筛选符合条件的条目键"""
        entries = self.state.get("entries", {})
        return [
            key for key, entry in entries.items()
            if all(entry.get(k) == v for k, v in criteria.items())
        ]

    def clear_entries(self):
        """清空所有条目（用于 force 模式）"""
        self.state["entries"] = {}


# ==================== 文本提取器 ====================

class TextExtractor:
    """从数据项中提取文本块，支持 HotpotQA/2WikiMultihopQA/MuSiQue"""

    def __init__(self, dataset_type: str):
        self.dataset_type = dataset_type.lower()

    def extract(self, item: Dict, item_idx: int) -> List[Tuple[str, str, str]]:
        """
        返回 [(item_id, title, text), ...]
        text 格式："Title: xxx. Content: xxx."
        """
        if self.dataset_type in ['hotpotqa', '2wikimultihop', '2wikimultihopqa']:
            return self._extract_list_format(item, item_idx)
        elif self.dataset_type == 'musique':
            return self._extract_dict_format(item, item_idx)
        else:
            raise ValueError(f"不支持的数据集类型：{self.dataset_type}")

    def _extract_list_format(self, item: Dict, item_idx: int) -> List[Tuple[str, str, str]]:
        """HotpotQA / 2WikiMultihopQA: context = [["title", [sentences]], ...]"""
        item_id = item.get('_id', '')
        results = []
        for ctx_idx, ctx_entry in enumerate(item.get('context', [])):
            title = ctx_entry[0]
            sentences = ctx_entry[1]
            text = f'Title: {title}. Content: {" ".join(sentences)}'
            results.append((item_id, title, text))
        return results

    def _extract_dict_format(self, item: Dict, item_idx: int) -> List[Tuple[str, str, str]]:
        """MuSiQue: paragraphs = [{"idx", "title", "paragraph_text"}, ...]"""
        item_id = item.get('id', '')
        results = []
        for para in item.get('paragraphs', []):
            title = para.get('title', '')
            text = f'Title: {title}. Content: {para.get("paragraph_text", "")}'
            results.append((item_id, title, text))
        return results


# ==================== 结果应用器 ====================

class ResultApplier:
    """将分块结果写回原始数据结构"""

    def __init__(self, dataset_type: str):
        self.dataset_type = dataset_type.lower()

    def apply(self, item: Dict, units: List[ProcessingUnit]) -> str:
        """
        应用结果到数据项（原地修改）。

        返回：整体状态 "completed" | "partial" | "failed"
        """
        if self.dataset_type in ['hotpotqa', '2wikimultihop', '2wikimultihopqa']:
            return self._apply_list_format(item, units)
        elif self.dataset_type == 'musique':
            return self._apply_dict_format(item, units)
        else:
            raise ValueError(f"不支持的数据集类型：{self.dataset_type}")

    def _apply_list_format(self, item: Dict, units: List[ProcessingUnit]) -> str:
        all_completed = True
        any_success = False

        for unit in units:
            if unit.context_index >= len(item.get('context', [])):
                continue

            ctx_entry = item['context'][unit.context_index]
            if len(ctx_entry) < 3:
                ctx_entry.append({})

            # 派生状态
            if unit.propositions_state == "success" and unit.chunks_state == "success" and unit.chunks:
                status = "completed"
                any_success = True
            elif unit.propositions_state == "success":
                status = "partial"
                all_completed = False
            else:
                status = "failed"
                all_completed = False

            ctx_entry[2] = {
                'propositions': unit.propositions if unit.propositions else [],
                'chunks': unit.chunks if unit.chunks else {},
                'metadata': {
                    'status': status,
                    'timestamp': datetime.now().isoformat()
                }
            }

        return "completed" if all_completed else ("partial" if any_success else "failed")

    def _apply_dict_format(self, item: Dict, units: List[ProcessingUnit]) -> str:
        all_completed = True
        any_success = False

        for unit in units:
            if unit.context_index >= len(item.get('paragraphs', [])):
                continue

            para = item['paragraphs'][unit.context_index]

            if unit.propositions_state == "success" and unit.chunks_state == "success" and unit.chunks:
                status = "completed"
                any_success = True
            elif unit.propositions_state == "success":
                status = "partial"
                all_completed = False
            else:
                status = "failed"
                all_completed = False

            para['propositions'] = unit.propositions if unit.propositions else []
            para['chunks'] = unit.chunks if unit.chunks else {}
            para['metadata'] = {
                'status': status,
                'timestamp': datetime.now().isoformat()
            }

        return "completed" if all_completed else ("partial" if any_success else "failed")


# ==================== 主处理器 ====================

class BenchmarkChunkProcessorV3:
    """
    Benchmark 数据集分块处理器 V3。

    核心策略：逐问题顺序处理 + 问题内并行批处理。
    天然限制并发量 = 单个问题的上下文数量（~10-20），无需额外 rate limit 控制。
    """

    def __init__(
        self,
        llm: Any,
        dataset_type: str,
        input_path: str,
        output_path: Optional[str] = None,
        chunk_max_workers: int = 3,
        timeout_per_task: float = 300.0
    ):
        """
        参数：
            llm: LangChain 兼容的聊天模型
            dataset_type: 数据集类型
            input_path: 输入文件路径
            output_path: 输出文件路径，默认为 input_path + "_chunked.json"
            chunk_max_workers: chunk_batch 的并发线程数（逐问题处理后批次小，3 足够）
            timeout_per_task: 每个任务超时时间（秒）
        """
        self.llm = llm
        self.agentic_chunk = AgenticChunk(llm=llm)
        self.dataset_type = dataset_type.lower()
        self.input_path = Path(input_path)
        self.output_path = Path(output_path) if output_path else self._default_output_path()
        self.checkpoint_path = self.output_path.with_suffix('.checkpoint.json')

        self.chunk_max_workers = chunk_max_workers
        self.timeout_per_task = timeout_per_task

        self.checkpoint = CheckpointManager(self.checkpoint_path)
        self.checkpoint.set_dataset_type(self.dataset_type)
        self.extractor = TextExtractor(self.dataset_type)
        self.applier = ResultApplier(self.dataset_type)

    def _default_output_path(self) -> Path:
        stem = self.input_path.stem
        return self.input_path.parent / f"{stem}_chunked.json"

    def _load_data(self) -> List[Dict]:
        """
        加载数据。

        优先从已有输出文件加载（如果存在），以保留之前运行的结果。
        这样重启时，之前处理过的问题的数据不会丢失。
        """
        if self.output_path.exists():
            _log(f"从已有输出文件加载：{self.output_path}")
            with open(self.output_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        _log(f"加载原始数据：{self.input_path}")
        with open(self.input_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _save_data(self, data: List[Dict]):
        """原子写入 benchmark JSON"""
        _log(f"保存结果：{self.output_path}")
        temp_path = self.output_path.with_suffix('.tmp')
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        temp_path.replace(self.output_path)

    # ==================== 单元构建 ====================

    def _build_units_for_question(
        self,
        item: Dict,
        item_idx: int,
        pending_context_indices: Optional[List[int]] = None
    ) -> List[ProcessingUnit]:
        """
        为单个问题构建 ProcessingUnit 列表。

        参数：
            item: 问题数据
            item_idx: 问题索引
            pending_context_indices: 只构建这些 context 索引的单元；
                                     None 表示构建所有上下文
        """
        extracted = self.extractor.extract(item, item_idx)
        units = []

        for ctx_idx, (item_id, title, text) in enumerate(extracted):
            if pending_context_indices is not None and ctx_idx not in pending_context_indices:
                continue

            key = generate_checkpoint_key(item_idx, ctx_idx)
            entry = self.checkpoint.get_entry(key)

            if entry is None:
                # 新单元，初始化
                unit = ProcessingUnit(
                    checkpoint_key=key,
                    item_index=item_idx,
                    context_index=ctx_idx,
                    item_id=item_id,
                    title=title,
                    text=text,
                    propositions_state="pending",
                    chunks_state="pending"
                )
                self.checkpoint.update_entry(
                    key,
                    item_id=item_id,
                    title=title,
                    propositions_state="pending",
                    chunks_state="pending",
                    attempt_count=0
                )
            else:
                # 已有状态的单元 — 从 item 中加载已有的 propositions/chunks（如果存在）
                propositions = None
                chunks = None

                if self.dataset_type in ['hotpotqa', '2wikimultihop', '2wikimultihopqa']:
                    ctx_list = item.get('context', [])
                    if ctx_idx < len(ctx_list) and len(ctx_list[ctx_idx]) >= 3:
                        ctx_data = ctx_list[ctx_idx][2]
                        if isinstance(ctx_data, dict):
                            propositions = ctx_data.get('propositions')
                            chunks = ctx_data.get('chunks')
                elif self.dataset_type == 'musique':
                    para_list = item.get('paragraphs', [])
                    if ctx_idx < len(para_list):
                        para = para_list[ctx_idx]
                        propositions = para.get('propositions')
                        chunks = para.get('chunks')

                entry = self.checkpoint.get_entry(key)
                prop_state = entry.get("propositions_state", "pending")

                # 如果 checkpoint 标记命题成功但实际数据为空，
                # 回退到 pending 状态以便 Stage 1 重新提取
                if prop_state == "success" and not propositions:
                    prop_state = "pending"

                unit = ProcessingUnit(
                    checkpoint_key=key,
                    item_index=item_idx,
                    context_index=ctx_idx,
                    item_id=item_id,
                    title=title,
                    text=text,
                    propositions_state=prop_state,
                    propositions=propositions if prop_state == "success" else None,
                    chunks_state=entry.get("chunks_state", "pending"),
                    chunks=chunks,
                    attempt_count=entry.get("attempt_count", 0),
                    last_updated=entry.get("last_updated")
                )

            units.append(unit)

        return units

    # ==================== Stage 1: 命题提取 ====================

    def _stage1_extract_propositions(
        self,
        units: List[ProcessingUnit]
    ) -> Tuple[int, int]:
        """
        并行提取命题。使用 AgenticChunk.get_propositions_batch()。

        返回：(成功数, 失败数)
        """
        if not units:
            return 0, 0

        _log(f"Stage 1: 提取 {len(units)} 个单元的命题...")
        texts = [unit.text for unit in units]

        try:
            all_propositions = self.agentic_chunk.get_propositions_batch(texts)

            success_count = 0
            fail_count = 0

            for unit, props in zip(units, all_propositions):
                if props:  # 非空列表 = 成功
                    unit.propositions = props
                    unit.propositions_state = "success"
                    self.checkpoint.update_entry(
                        unit.checkpoint_key,
                        propositions_state="success"
                    )
                    success_count += 1
                else:
                    unit.propositions_state = "failed"
                    self.checkpoint.update_entry(
                        unit.checkpoint_key,
                        propositions_state="failed"
                    )
                    fail_count += 1

            _log(f"Stage 1 完成：成功 {success_count}/{len(units)} ({success_count/len(units)*100:.1f}%)", "success")
            return success_count, fail_count

        except Exception as e:
            _log(f"Stage 1 批次失败：{e}，降级为逐个处理...", "error")
            return self._stage1_extract_sequential(units)

    def _stage1_extract_sequential(
        self,
        units: List[ProcessingUnit]
    ) -> Tuple[int, int]:
        """命题提取降级：逐个处理"""
        success_count = 0
        fail_count = 0

        for unit in units:
            try:
                props = self.agentic_chunk.get_propositions(unit.text)
                if props:
                    unit.propositions = props
                    unit.propositions_state = "success"
                    self.checkpoint.update_entry(
                        unit.checkpoint_key,
                        propositions_state="success"
                    )
                    success_count += 1
                else:
                    unit.propositions_state = "failed"
                    self.checkpoint.update_entry(
                        unit.checkpoint_key,
                        propositions_state="failed"
                    )
                    fail_count += 1
            except Exception as e:
                _log(f"  {unit.checkpoint_key} 命题提取失败：{e}", "error")
                unit.propositions_state = "failed"
                self.checkpoint.update_entry(
                    unit.checkpoint_key,
                    propositions_state="failed"
                )
                fail_count += 1

        return success_count, fail_count

    # ==================== Stage 2: 分块处理 ====================

    def _stage2_chunk_propositions(
        self,
        units: List[ProcessingUnit]
    ) -> Tuple[int, int]:
        """
        并行分块处理。使用 AgenticChunk.chunk_batch()。

        返回：(成功数, 失败数)
        """
        if not units:
            return 0, 0

        _log(f"Stage 2: 分块处理 {len(units)} 个单元 (max_workers={self.chunk_max_workers})...")
        propositions_list = [unit.propositions for unit in units]

        try:
            all_chunks = self.agentic_chunk.chunk_batch(
                propositions_list,
                max_workers=self.chunk_max_workers,
                timeout_per_task=self.timeout_per_task,
                on_error="raise"
            )

            success_count = 0
            fail_count = 0

            for unit, chunks in zip(units, all_chunks):
                if chunks:  # 非空字典 = 成功
                    unit.chunks = chunks
                    unit.chunks_state = "success"
                    self.checkpoint.update_entry(
                        unit.checkpoint_key,
                        chunks_state="success"
                    )
                    success_count += 1
                else:
                    unit.chunks_state = "failed"
                    self.checkpoint.update_entry(
                        unit.checkpoint_key,
                        chunks_state="failed"
                    )
                    fail_count += 1

            _log(f"Stage 2 完成：成功 {success_count}/{len(units)} ({success_count/len(units)*100:.1f}%)", "success")
            return success_count, fail_count

        except PartialBatchError as e:
            _log(f"Stage 2 部分失败：{len(e.get_failures())}/{len(e.get_successful_results()) + len(e.get_failures())}", "warning")

            success_count = 0
            fail_count = 0

            # 成功的结果：chunk_batch 的 get_successful_results() 返回 {index: result}
            for idx, chunks in e.get_successful_results().items():
                unit = units[idx]
                unit.chunks = chunks
                unit.chunks_state = "success"
                self.checkpoint.update_entry(
                    unit.checkpoint_key,
                    chunks_state="success"
                )
                success_count += 1

            # 失败的任务 — 记录具体错误信息
            for idx, error_msg, _ in e.get_failures():
                _log(f"  {units[idx].checkpoint_key} 分块失败：{error_msg}", "error")
                units[idx].chunks_state = "failed"
                self.checkpoint.update_entry(
                    units[idx].checkpoint_key,
                    chunks_state="failed"
                )
                fail_count += 1

            return success_count, fail_count

        except Exception as e:
            _log(f"Stage 2 失败：{e}，降级为顺序处理...", "error")
            return self._stage2_chunk_sequential(units)

    def _stage2_chunk_sequential(
        self,
        units: List[ProcessingUnit]
    ) -> Tuple[int, int]:
        """分块降级：逐个处理"""
        success_count = 0
        fail_count = 0

        for unit in units:
            try:
                fresh_chunker = AgenticChunk(llm=self.llm)
                chunks = fresh_chunker.chunk(unit.propositions)
                if chunks:
                    unit.chunks = chunks
                    unit.chunks_state = "success"
                    self.checkpoint.update_entry(
                        unit.checkpoint_key,
                        chunks_state="success"
                    )
                    success_count += 1
                else:
                    unit.chunks_state = "failed"
                    self.checkpoint.update_entry(
                        unit.checkpoint_key,
                        chunks_state="failed"
                    )
                    fail_count += 1
            except Exception as e:
                _log(f"  {unit.checkpoint_key} 分块失败：{e}", "error")
                unit.chunks_state = "failed"
                self.checkpoint.update_entry(
                    unit.checkpoint_key,
                    chunks_state="failed"
                )
                fail_count += 1

        return success_count, fail_count

    # ==================== 持久化 ====================

    def _persist_question_results(
        self,
        all_data: List[Dict],
        item_idx: int,
        units: List[ProcessingUnit]
    ):
        """
        持久化单个问题的结果。

        原子写入顺序：
        1. 应用结果到 all_data（内存）
        2. 保存 benchmark JSON（磁盘）
        3. 保存 checkpoint（磁盘）

        保证 checkpoint 状态 <= benchmark 已保存的状态。
        """
        # 1. 应用到内存
        self.applier.apply(all_data[item_idx], units)

        # 2. 保存 benchmark（原子写入）
        self._save_data(all_data)

        # 3. 保存 checkpoint（原子写入）
        self.checkpoint.save()

    # ==================== 主处理流程 ====================

    def process(
        self,
        limit: Optional[int] = None,
        retry_failed: bool = False,
        force: bool = False
    ) -> Dict[str, int]:
        """
        处理数据集。

        核心流程：
        1. 加载数据和 checkpoint
        2. 确定需要处理的问题及上下文
        3. 逐问题处理：命题提取 → 分块 → 持久化
        4. 返回统计信息

        参数：
            limit: 限制处理问题数，用于测试
            retry_failed: 只重试之前失败的单元
            force: 强制重新处理所有（忽略 checkpoint）
        """
        print(f"\n{'='*60}")
        print(f"Benchmark Chunker V3 - {self.dataset_type}")
        print(f"{'='*60}")

        # ========== 加载数据 ==========
        all_data = self._load_data()
        total_items = len(all_data)
        _log(f"数据总数：{total_items} 个问题")

        if force:
            _log("强制模式：清空 checkpoint", "warning")
            self.checkpoint.clear_entries()

        # ========== 确定待处理问题 ==========
        # 收集需要处理的问题：[(item_idx, pending_context_indices), ...]
        questions_to_process: List[Tuple[int, List[int]]] = []

        for item_idx, item in enumerate(all_data):
            if limit and len(questions_to_process) >= limit:
                break

            # 获取该问题的 context 数量
            if self.dataset_type in ['hotpotqa', '2wikimultihop', '2wikimultihopqa']:
                total_contexts = len(item.get('context', []))
            else:
                total_contexts = len(item.get('paragraphs', []))

            if total_contexts == 0:
                continue

            pending_ctx_indices = self._get_pending_contexts_for_item(
                item_idx, total_contexts, retry_failed
            )

            if pending_ctx_indices:
                questions_to_process.append((item_idx, pending_ctx_indices))

        if not questions_to_process:
            _log("没有需要处理的数据。")
            return self._empty_stats(total_items)

        _log(f"待处理问题数：{len(questions_to_process)}")

        # ========== 逐问题处理 ==========
        total_prop_success = 0
        total_prop_failed = 0
        total_chunk_success = 0
        total_chunk_failed = 0
        completed_items = 0
        partial_items = 0
        failed_items = 0

        for q_idx, (item_idx, pending_ctx_indices) in enumerate(questions_to_process):
            item = all_data[item_idx]
            _log(f"\n--- 问题 {q_idx + 1}/{len(questions_to_process)} (item_idx={item_idx}) ---")

            # 构建处理单元
            units = self._build_units_for_question(item, item_idx, pending_ctx_indices)

            # Stage 1: 命题提取（只处理 pending/failed 的单元）
            prop_units = [
                u for u in units if u.propositions_state in ("pending", "failed")
            ]
            if prop_units:
                ps, pf = self._stage1_extract_propositions(prop_units)
                total_prop_success += ps
                total_prop_failed += pf
            else:
                _log("没有需要提取命题的单元。")

            # Stage 2: 分块处理（命题成功 + chunks pending/failed）
            chunk_units = [
                u for u in units
                if u.propositions_state == "success" and u.chunks_state in ("pending", "failed")
            ]
            if chunk_units:
                cs, cf = self._stage2_chunk_propositions(chunk_units)
                total_chunk_success += cs
                total_chunk_failed += cf
            else:
                _log("没有需要分块的单元。")

            # 持久化
            self._persist_question_results(all_data, item_idx, units)

            # 统计问题级别状态
            item_status = self._compute_item_status(units)
            if item_status == "completed":
                completed_items += 1
            elif item_status == "partial":
                partial_items += 1
            else:
                failed_items += 1

        # 最终保存（确保一致）
        self._save_data(all_data)
        self.checkpoint.save()

        # ========== 汇总 ==========
        stats = {
            'processed_items': len(questions_to_process),
            'completed': completed_items,
            'partial': partial_items,
            'failed': failed_items,
            'skipped': total_items - len(questions_to_process),
            'propositions_success': total_prop_success,
            'propositions_failed': total_prop_failed,
            'chunks_success': total_chunk_success,
            'chunks_failed': total_chunk_failed
        }

        print(f"\n{'='*60}")
        print(f"处理完成!")
        print(f"  处理问题数：{stats['processed_items']}")
        print(f"  完成：{stats['completed']} | 部分完成：{stats['partial']} | 失败：{stats['failed']}")
        print(f"  命题提取：成功 {stats['propositions_success']} 个，失败 {stats['propositions_failed']} 个")
        print(f"  分块处理：成功 {stats['chunks_success']} 个，失败 {stats['chunks_failed']} 个")
        print(f"  跳过：{stats['skipped']} 个问题")
        print(f"{'='*60}\n")

        return stats

    # ==================== 辅助方法 ====================

    def _get_pending_contexts_for_item(
        self,
        item_idx: int,
        total_contexts: int,
        retry_failed: bool
    ) -> List[int]:
        """获取某个问题中需要处理的 context 索引列表"""
        pending = []

        for ctx_idx in range(total_contexts):
            key = generate_checkpoint_key(item_idx, ctx_idx)
            entry = self.checkpoint.get_entry(key)

            if entry is None:
                # 从未处理过
                pending.append(ctx_idx)
                # 初始化
                item = None
                # 延迟获取 item_id 和 title（由 _build_units_for_question 处理）
                self.checkpoint.update_entry(
                    key,
                    propositions_state="pending",
                    chunks_state="pending",
                    attempt_count=0
                )
            else:
                prop_state = entry.get("propositions_state", "pending")
                chunk_state = entry.get("chunks_state", "pending")

                if retry_failed:
                    # 只处理失败的
                    if prop_state == "failed" or chunk_state == "failed":
                        pending.append(ctx_idx)
                else:
                    # 处理未完成的
                    if prop_state in ("pending", "failed") or chunk_state in ("pending", "failed"):
                        pending.append(ctx_idx)

        return pending

    def _compute_item_status(self, units: List[ProcessingUnit]) -> str:
        """计算单个问题的整体状态"""
        if not units:
            return "failed"

        all_completed = True
        any_success = False

        for unit in units:
            if unit.propositions_state == "success" and unit.chunks_state == "success" and unit.chunks:
                any_success = True
            else:
                all_completed = False

        return "completed" if all_completed else ("partial" if any_success else "failed")

    def _empty_stats(self, total_items: int) -> Dict[str, int]:
        return {
            'processed_items': 0,
            'completed': 0,
            'partial': 0,
            'failed': 0,
            'skipped': total_items,
            'propositions_success': 0,
            'propositions_failed': 0,
            'chunks_success': 0,
            'chunks_failed': 0
        }


# ==================== 兼容函数 ====================

def process_all_datasets(
    llm: Any,
    limit: Optional[int] = None,
    output_dir: Optional[str] = None,
    retry_failed: bool = False
) -> Dict[str, Dict[str, int]]:
    """处理所有 benchmark 数据集"""
    datasets = [
        ('hotpotqa', 'Data/benchmark/HotpotQA_500_benchmark.json'),
        ('2wikimultihop', 'Data/benchmark/2WikiMultihopQA_500_benchmark.json'),
        ('musique', 'Data/benchmark/MuSiQue_500_benchmark.json')
    ]

    all_stats = {}

    for dataset_type, input_path in datasets:
        output_path = None
        if output_dir:
            output_path = Path(output_dir) / Path(input_path).name.replace('.json', '_chunked.json')

        processor = BenchmarkChunkProcessorV3(
            llm=llm,
            dataset_type=dataset_type,
            input_path=input_path,
            output_path=output_path
        )

        stats = processor.process(limit=limit, retry_failed=retry_failed)
        all_stats[dataset_type] = stats

        print(f"\n{'='*60}\n")

    return all_stats


if __name__ == '__main__':
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI

    load_dotenv()

    llm = ChatOpenAI(
        api_key=os.getenv("BL_API_KEY"),
        base_url=os.getenv("BL_BASE_URL"),
        model='qwen3.5-plus',
        temperature=0.0,
        extra_body={"enable_thinking": False}
    )

    processor = BenchmarkChunkProcessorV3(
        llm=llm,
        dataset_type='hotpotqa',
        input_path='Data/benchmark/HotpotQA_500_benchmark.json'
    )

    stats = processor.process(limit=3)
    print(f"\n测试处理完成：{stats}")
