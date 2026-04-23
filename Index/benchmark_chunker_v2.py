"""
Benchmark 分块处理器 V2 - 重构版本。

本模块是 benchmark_chunker.py 的重构版本，核心改进：
1. 扁平化 Checkpoint 结构 - 使用 "q{item_index}/c{context_index}" 键直接定位最小处理单元
2. ProcessingUnit 数据类 - 集中管理状态，简化数据流
3. 两阶段并行 Pipeline - 命题提取和分块处理完全解耦
4. 跨问题并行 - 利用 AgenticChunk 的 batch 方法处理所有待处理单元
5. BatchResult 统一封装 - 简化的错误处理和重试逻辑

数据流向：
  加载 checkpoint → 构建 ProcessingUnit 列表 → Stage1 提取命题 → Stage2 分块 → 持久化到 benchmark

示例用法：
    >>> from langchain_openai import ChatOpenAI
    >>> llm = ChatOpenAI(model='qwen3-max', temperature=0)
    >>> processor = BenchmarkChunkProcessorV2(
    ...     llm=llm,
    ...     dataset_type='hotpotqa',
    ...     input_path='Data/benchmark/HotpotQA_500_benchmark.json'
    ... )
    >>> processor.process(limit=10)  # 测试模式
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict

from .agentic_chunk import AgenticChunk, PartialBatchError


# ==================== 日志配置 ====================

# 简单打印日志，不使用复杂的 logging 模块
def _log(message: str, level: str = "info"):
    """打印日志消息"""
    # 使用 ASCII 字符避免 Windows 控制台编码问题
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

    封装了单个 context/paragraph 的完整状态，包括：
    - 定位信息（checkpoint_key, item_index, context_index, item_id）
    - 文本内容（title, text）
    - 运行时状态（propositions_state, propositions, chunks_state, chunks）

    所有状态集中在一个对象中，便于在 pipeline 中传递。
    """
    # 定位信息
    checkpoint_key: str       # "q0/c1" - checkpoint 中的扁平键
    item_index: int           # 问题在 benchmark 列表中的索引
    context_index: int        # 上下文在问题中的索引
    item_id: str              # 问题的唯一 ID
    title: str                # 段落标题

    # 文本内容（启动时从原始数据填充）
    text: str = ""            # 完整文本 "Title: xxx. Content: xxx."

    # 运行时状态（从 checkpoint 加载或运行时填充）
    propositions_state: str = "pending"  # success/failed/pending
    propositions: Optional[List[str]] = None
    chunks_state: str = "pending"
    chunks: Optional[Dict] = None

    # 元数据
    attempt_count: int = 0
    last_updated: Optional[str] = None


@dataclass
class BatchResult:
    """
    批量处理结果封装。

    统一封装成功和失败的结果，简化错误处理逻辑。

    示例：
        >>> result = process_batch(units)
        >>> print(f"成功率：{result.success_rate:.1%}")
        >>> for unit in result.successful:
        ...     # 处理成功的单元
        >>> for unit, error in result.failed:
        ...     # 处理失败的单元
    """
    successful: List[ProcessingUnit]  # 成功的单元（已填充结果）
    failed: List[Tuple[ProcessingUnit, str]]  # (单元，错误消息)

    @property
    def total(self) -> int:
        return len(self.successful) + len(self.failed)

    @property
    def success_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return len(self.successful) / self.total

    def get_failed_units(self) -> List[ProcessingUnit]:
        """获取失败的单元列表（用于重试）"""
        return [unit for unit, _ in self.failed]


# ==================== Checkpoint 管理器 ====================

class CheckpointManager:
    """
    扁平化 Checkpoint 管理器。

    Checkpoint 结构：
    {
        "dataset_type": "hotpotqa",
        "last_updated": "2026-04-02T15:30:00",
        "entries": {
            "q0/c0": {
                "item_id": "5abc...",
                "title": "Moscow State University",
                "propositions_state": "success",
                "propositions": [...],
                "chunks_state": "pending",
                "chunks": null,
                "attempt_count": 1,
                "last_updated": "..."
            },
            "q0/c1": {...}
        }
    }

    扁平键格式："q{item_index}/c{context_index}"
    - q = question（问题索引）
    - c = context（上下文索引）
    """

    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = checkpoint_path
        self.state = self._load()

    def _load(self) -> Dict:
        """加载 checkpoint 状态"""
        if not self.checkpoint_path.exists():
            return {
                "dataset_type": None,
                "last_updated": None,
                "entries": {}
            }

        try:
            with open(self.checkpoint_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[WARN] checkpoint 文件损坏，将重新创建：{e}")
            return {
                "dataset_type": None,
                "last_updated": None,
                "entries": {}
            }

    def save(self):
        """保存 checkpoint（原子写入）"""
        self.state["last_updated"] = datetime.now().isoformat()

        # 原子写入：先写临时文件，再 rename
        temp_path = self.checkpoint_path.with_suffix('.tmp')
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2, default=str)
        temp_path.replace(self.checkpoint_path)
        _log(f"Checkpoint 已保存：{len(self.state.get('entries', {}))} 个条目", "success")

    def set_dataset_type(self, dataset_type: str):
        """设置数据集类型"""
        self.state["dataset_type"] = dataset_type

    def get_entry(self, key: str) -> Optional[Dict]:
        """获取单个条目"""
        return self.state.get("entries", {}).get(key)

    def update_entry(self, key: str, **updates):
        """
        更新单个条目。

        参数：
            key: checkpoint 键，如 "q0/c0"
            **updates: 要更新的字段，如 propositions_state="success"
        """
        if "entries" not in self.state:
            self.state["entries"] = {}

        if key not in self.state["entries"]:
            self.state["entries"][key] = {}

        entry = self.state["entries"][key]
        entry.update(updates)
        entry["last_updated"] = datetime.now().isoformat()

    def get_all_entries(self) -> Dict[str, Dict]:
        """获取所有条目"""
        return self.state.get("entries", {})

    def filter_entries(self, **criteria) -> List[str]:
        """
        筛选符合条件的条目键。

        参数：
            **criteria: 筛选条件，如 propositions_state="pending"

        返回：
            符合条件的键列表

        示例：
            >>> keys = checkpoint.filter_entries(propositions_state="pending")
            >>> keys = checkpoint.filter_entries(
            ...     propositions_state="success",
            ...     chunks_state="pending"
            ... )
        """
        entries = self.state.get("entries", {})
        matching_keys = []

        for key, entry in entries.items():
            match = True
            for field, value in criteria.items():
                if entry.get(field) != value:
                    match = False
                    break
            if match:
                matching_keys.append(key)

        return matching_keys

    def get_pending_units(self) -> List[str]:
        """
        获取所有待处理单元的键。

        待处理定义：propositions_state 和 chunks_state 都为 "pending"
        （即从未处理过的单元）

        返回：
            待处理单元的键列表
        """
        return self.filter_entries(
            propositions_state="pending",
            chunks_state="pending"
        )

    def get_proposition_pending_units(self) -> List[str]:
        """
        获取需要提取命题的单元的键。

        包括 propositions_state 为 "pending" 或 "failed" 的单元。

        返回：
            需要提取命题的单元的键列表
        """
        keys = self.filter_entries(propositions_state="pending")
        failed_keys = self.filter_entries(propositions_state="failed")
        return list(set(keys + failed_keys))

    def get_chunk_pending_units(self) -> List[str]:
        """
        获取需要分块的单元的键。

        包括 propositions_state=="success" 且 chunks_state 为 "pending" 或 "failed" 的单元。

        返回：
            需要分块的单元的键列表
        """
        # 先找到所有命题成功的单元
        prop_success_keys = set(self.filter_entries(propositions_state="success"))
        # 再找到 chunks 待处理或失败的单元
        chunk_pending_keys = set(self.filter_entries(chunks_state="pending"))
        chunk_failed_keys = set(self.filter_entries(chunks_state="failed"))

        return list(prop_success_keys & (chunk_pending_keys | chunk_failed_keys))

    def remove_entry(self, key: str):
        """移除单个条目（用于清理）"""
        if "entries" not in self.state:
            return
        self.state["entries"].pop(key, None)


# ==================== 工具函数 ====================

def generate_checkpoint_key(item_index: int, context_index: int) -> str:
    """
    生成扁平 checkpoint 键。

    格式："q{item_index}/c{context_index}"

    示例：
        >>> generate_checkpoint_key(0, 0)
        "q0/c0"
        >>> generate_checkpoint_key(1, 2)
        "q1/c2"
    """
    return f"q{item_index}/c{context_index}"


def parse_checkpoint_key(key: str) -> Tuple[int, int]:
    """
    解析 checkpoint 键。

    参数：
        key: "q0/c0" 格式的键

    返回：
        (item_index, context_index) 元组

    示例：
        >>> parse_checkpoint_key("q0/c1")
        (0, 1)
    """
    q_part, c_part = key.split('/')
    item_index = int(q_part[1:])  # 去掉 'q' 前缀
    context_index = int(c_part[1:])  # 去掉 'c' 前缀
    return item_index, context_index


# ==================== 文本提取器 ====================

class TextExtractor:
    """
    从数据项中提取文本块。

    支持三种数据集格式：
    - HotpotQA: [["title", [sentences]], ...]
    - 2WikiMultihopQA: 同 HotpotQA
    - MuSiQue: [{"idx", "title", "paragraph_text"}, ...]
    """

    def __init__(self, dataset_type: str):
        self.dataset_type = dataset_type.lower()

    def extract(self, item: Dict, item_idx: int) -> List[Tuple[str, str, str]]:
        """
        从数据项中提取所有文本块。

        参数：
            item: 单个问题/数据项
            item_idx: 数据项在列表中的索引

        返回：
            [(item_id, title, text), ...] 列表
        """
        if self.dataset_type in ['hotpotqa', '2wikimultihop', '2wikimultihopqa']:
            return self._extract_list_format(item, item_idx)
        elif self.dataset_type == 'musique':
            return self._extract_dict_format(item, item_idx)
        else:
            raise ValueError(
                f"不支持的数据集类型：{self.dataset_type}。"
                f"支持的类型：hotpotqa, 2wikimultihop, musique"
            )

    def _extract_list_format(self, item: Dict, item_idx: int) -> List[Tuple[str, str, str]]:
        """
        提取 HotpotQA 或 2WikiMultihopQA 的文本块。

        数据结构：
            "context": [
                ["Title A", ["sentence1", "sentence2"]],
                ["Title B", ["sentence3", "sentence4"]]
            ]
        """
        results = []
        item_id = item.get('_id', '')
        context = item.get('context', [])

        for ctx_idx, (title, sentences) in enumerate(context):
            text = f'Title: {title}. Content: {" ".join(sentences)}'
            results.append((item_id, title, text))

        return results

    def _extract_dict_format(self, item: Dict, item_idx: int) -> List[Tuple[str, str, str]]:
        """
        提取 MuSiQue 的文本块。

        数据结构：
            "paragraphs": [
                {"idx": 0, "title": "...", "paragraph_text": "..."},
                ...
            ]
        """
        results = []
        item_id = item.get('id', '')
        paragraphs = item.get('paragraphs', [])

        for para in paragraphs:
            title = para.get('title', '')
            text = f'Title: {title}. Content: {para.get("paragraph_text", "")}'
            results.append((item_id, title, text))

        return results


# ==================== 结果应用器 ====================

class ResultApplier:
    """
    将分块结果应用回原始数据结构。

    根据数据集类型，将 propositions 和 chunks 写回对应的数据结构位置。
    """

    def __init__(self, dataset_type: str):
        self.dataset_type = dataset_type.lower()

    def apply(self, item: Dict, units: List[ProcessingUnit]) -> str:
        """
        应用结果到数据项。

        参数：
            item: 原始数据项（会被原地修改）
            units: 属于该问题的处理单元列表

        返回：
            整体状态："completed" | "partial" | "failed"
        """
        if self.dataset_type in ['hotpotqa', '2wikimultihop', '2wikimultihopqa']:
            return self._apply_list_format(item, units)
        elif self.dataset_type == 'musique':
            return self._apply_dict_format(item, units)
        else:
            raise ValueError(f"不支持的数据集类型：{self.dataset_type}")

    def _apply_list_format(self, item: Dict, units: List[ProcessingUnit]) -> str:
        """应用结果到 HotpotQA / 2WikiMultihopQA 格式"""
        all_completed = True
        any_success = False

        for unit in units:
            if unit.context_index >= len(item.get('context', [])):
                continue

            context_entry = item['context'][unit.context_index]

            # 确保 context 条目有第三个元素（结果字典）
            if len(context_entry) < 3:
                context_entry.append({})

            # 确定状态
            if unit.propositions_state == "success" and unit.chunks_state == "success" and unit.chunks:
                status = "completed"
                any_success = True
            elif unit.propositions_state == "success":
                status = "partial"  # 命题成功但分块失败/为空
                all_completed = False
            else:
                status = "failed"
                all_completed = False

            # 写回结果
            context_entry[2] = {
                'propositions': unit.propositions if unit.propositions else [],
                'chunks': unit.chunks if unit.chunks else {},
                'metadata': {
                    'status': status,
                    'timestamp': datetime.now().isoformat()
                }
            }

        if all_completed:
            return "completed"
        elif any_success:
            return "partial"
        else:
            return "failed"

    def _apply_dict_format(self, item: Dict, units: List[ProcessingUnit]) -> str:
        """应用结果到 MuSiQue 格式"""
        all_completed = True
        any_success = False

        for unit in units:
            if unit.context_index >= len(item.get('paragraphs', [])):
                continue

            para = item['paragraphs'][unit.context_index]

            # 确定状态
            if unit.propositions_state == "success" and unit.chunks_state == "success" and unit.chunks:
                status = "completed"
                any_success = True
            elif unit.propositions_state == "success":
                status = "partial"
                all_completed = False
            else:
                status = "failed"
                all_completed = False

            # 写回结果
            para['propositions'] = unit.propositions if unit.propositions else []
            para['chunks'] = unit.chunks if unit.chunks else {}
            para['metadata'] = {
                'status': status,
                'timestamp': datetime.now().isoformat()
            }

        if all_completed:
            return "completed"
        elif any_success:
            return "partial"
        else:
            return "failed"


# ==================== 主处理器 ====================

class BenchmarkChunkProcessorV2:
    """
    Benchmark 数据集分块处理器 V2（重构版本）。

    核心改进：
    1. 扁平化 Checkpoint 结构 - "q0/c0" 键直接定位
    2. ProcessingUnit 集中管理状态
    3. 两阶段并行 Pipeline（命题提取 → 分块）
    4. 跨问题并行 - 所有待处理单元一起处理
    5. BatchResult 统一封装错误处理

    处理流程：
        1. 加载 checkpoint 和原始数据
        2. 构建 ProcessingUnit 列表
        3. Stage 1: 并行提取所有待处理单元的命题
        4. Stage 2: 并行对所有命题成功的单元进行分块
        5. 持久化结果到 benchmark 文件

    示例用法：
        >>> processor = BenchmarkChunkProcessorV2(
        ...     llm=llm,
        ...     dataset_type='hotpotqa',
        ...     input_path='Data/benchmark/HotpotQA_500_benchmark.json'
        ... )
        >>> stats = processor.process(limit=10)
    """

    def __init__(
        self,
        llm: Any,
        dataset_type: str,
        input_path: str,
        output_path: Optional[str] = None,
        max_workers: int = 3,
        timeout_per_task: float = 300.0
    ):
        """
        初始化处理器。

        参数：
            llm: LangChain 兼容的聊天模型
            dataset_type: 数据集类型 ('hotpotqa', '2wikimultihop', 'musique')
            input_path: 输入文件路径
            output_path: 输出文件路径，默认为 None 时自动生成（输入路径 + "_chunked.json"）
            max_workers: 并行工作线程数（传递给 AgenticChunk.chunk_batch）
            timeout_per_task: 每个任务的超时时间（秒）
        """
        self.llm = llm
        self.agentic_chunk = AgenticChunk(llm=llm)
        self.dataset_type = dataset_type.lower()
        self.input_path = Path(input_path)
        self.output_path = Path(output_path) if output_path else self._default_output_path()
        self.checkpoint_path = self.output_path.with_suffix('.checkpoint.json')

        # 配置参数
        self.max_workers = max_workers
        self.timeout_per_task = timeout_per_task

        # 初始化组件
        self.checkpoint = CheckpointManager(self.checkpoint_path)
        self.checkpoint.set_dataset_type(self.dataset_type)
        self.extractor = TextExtractor(self.dataset_type)
        self.applier = ResultApplier(self.dataset_type)

        # 数据缓存
        self.all_data: Optional[List[Dict]] = None

    def _default_output_path(self) -> Path:
        """生成默认输出路径"""
        stem = self.input_path.stem
        return self.input_path.parent / f"{stem}_chunked.json"

    def _load_data(self) -> List[Dict]:
        """加载输入数据"""
        _log(f"加载数据：{self.input_path}")
        with open(self.input_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _save_data(self, data: List[Dict]):
        """保存数据到输出文件（原子写入）"""
        _log(f"保存结果：{self.output_path}")
        temp_path = self.output_path.with_suffix('.tmp')
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        temp_path.replace(self.output_path)

    def _get_pending_keys(self, all_data: List[Dict]) -> List[str]:
        """
        获取所有待处理单元的键。

        逻辑：
        1. 遍历所有数据项和上下文
        2. 如果 checkpoint 中没有该单元，或者状态是 pending/failed，则加入待处理列表

        参数：
            all_data: 完整的 benchmark 数据

        返回：
            待处理单元的键列表
        """
        pending_keys = []

        for item_idx, item in enumerate(all_data):
            # 提取该 item 的所有文本
            extracted = self.extractor.extract(item, item_idx)

            for ctx_idx, (item_id, title, text) in enumerate(extracted):
                key = generate_checkpoint_key(item_idx, ctx_idx)
                entry = self.checkpoint.get_entry(key)

                if entry is None:
                    # checkpoint 中没有该单元，需要处理
                    pending_keys.append(key)
                    # 初始化 checkpoint 条目
                    self.checkpoint.update_entry(
                        key,
                        item_id=item_id,
                        title=title,
                        text=text,
                        propositions_state="pending",
                        chunks_state="pending",
                        attempt_count=0
                    )
                else:
                    # 检查状态
                    prop_state = entry.get("propositions_state", "pending")
                    chunk_state = entry.get("chunks_state", "pending")

                    if prop_state in ("pending", "failed") or chunk_state in ("pending", "failed"):
                        pending_keys.append(key)
                    # 更新 text（可能变化）
                    self.checkpoint.update_entry(key, text=text)

        return pending_keys

    def _build_processing_units(
        self,
        all_data: List[Dict],
        include_all: bool = False
    ) -> List[ProcessingUnit]:
        """
        构建 ProcessingUnit 列表。

        参数：
            all_data: 完整的 benchmark 数据
            include_all: 如果 True，构建所有单元（不管状态）；如果 False，只构建待处理的

        返回：
            ProcessingUnit 列表
        """
        units = []

        for item_idx, item in enumerate(all_data):
            # 提取该 item 的所有文本
            extracted = self.extractor.extract(item, item_idx)

            for item_id, title, text in extracted:
                # 计算 context 索引（通过 enumerate 或从 extracted 推断）
                context_index = len([u for u in units if u.item_index == item_idx])
                key = generate_checkpoint_key(item_idx, context_index)

                # 从 checkpoint 获取已有状态
                entry = self.checkpoint.get_entry(key)

                if entry is None:
                    # 新单元，初始化状态
                    unit = ProcessingUnit(
                        checkpoint_key=key,
                        item_index=item_idx,
                        context_index=context_index,
                        item_id=item_id,
                        title=title,
                        text=text,
                        propositions_state="pending",
                        chunks_state="pending"
                    )
                    # 初始化 checkpoint 条目
                    self.checkpoint.update_entry(
                        key,
                        item_id=item_id,
                        title=title,
                        text=text,
                        propositions_state="pending",
                        chunks_state="pending",
                        attempt_count=0
                    )
                else:
                    # 已有状态的单元
                    unit = ProcessingUnit(
                        checkpoint_key=key,
                        item_index=item_idx,
                        context_index=context_index,
                        item_id=item_id,
                        title=title,
                        text=text,  # 使用新提取的 text（可能更新）
                        propositions_state=entry.get("propositions_state", "pending"),
                        propositions=entry.get("propositions"),
                        chunks_state=entry.get("chunks_state", "pending"),
                        chunks=entry.get("chunks"),
                        attempt_count=entry.get("attempt_count", 0),
                        last_updated=entry.get("last_updated")
                    )

                units.append(unit)

        if include_all:
            return units

        # 只返回待处理的单元（至少有一个任务是 pending 或 failed）
        pending_units = [
            unit for unit in units
            if unit.propositions_state in ("pending", "failed")
            or unit.chunks_state in ("pending", "failed")
        ]

        return pending_units

    def _stage1_extract_propositions(
        self,
        units: List[ProcessingUnit]
    ) -> BatchResult:
        """
        Stage 1: 并行提取命题。

        使用 AgenticChunk.get_propositions_batch() 进行并行处理。
        该方法保证输出顺序与输入一致，因此可以直接 zip 映射。

        参数：
            units: 需要提取命题的单元列表

        返回：
            BatchResult 包含成功和失败的单元
        """
        if not units:
            return BatchResult(successful=[], failed=[])

        _log(f"Stage 1: 提取 {len(units)} 个单元的命题...", "info")

        texts = [unit.text for unit in units]

        try:
            all_propositions = self.agentic_chunk.get_propositions_batch(texts)

            successful = []
            failed = []

            for unit, props in zip(units, all_propositions):
                if props:  # 非空列表
                    unit.propositions = props
                    unit.propositions_state = "success"
                    # 更新 checkpoint（内存中）
                    self.checkpoint.update_entry(
                        unit.checkpoint_key,
                        propositions=props,
                        propositions_state="success"
                    )
                    successful.append(unit)
                else:
                    failed.append((unit, "命题提取返回空列表"))

            _log(f"Stage 1 完成：成功 {len(successful)}/{len(units)} ({len(successful)/len(units)*100:.1f}%)", "success")
            return BatchResult(successful=successful, failed=failed)

        except Exception as e:
            _log(f"Stage 1 失败：{e}", "error")
            # 整个批次失败
            return BatchResult(
                successful=[],
                failed=[(unit, str(e)) for unit in units]
            )

    def _stage2_chunk_propositions(
        self,
        units: List[ProcessingUnit]
    ) -> BatchResult:
        """
        Stage 2: 并行分块处理。

        使用 AgenticChunk.chunk_batch() 进行并行处理。
        该方法保证输出顺序与输入一致，因此可以直接 zip 映射。

        参数：
            units: 需要分块的单元列表（propositions 已填充）

        返回：
            BatchResult 包含成功和失败的单元
        """
        if not units:
            return BatchResult(successful=[], failed=[])

        _log(f"Stage 2: 分块处理 {len(units)} 个单元 (max_workers={self.max_workers})...", "info")

        propositions_list = [unit.propositions for unit in units]

        try:
            all_chunks = self.agentic_chunk.chunk_batch(
                propositions_list,
                max_workers=self.max_workers,
                timeout_per_task=self.timeout_per_task,
                on_error="raise"  # 抛出 PartialBatchError 以便处理部分失败
            )

            successful = []
            failed = []

            for unit, chunks in zip(units, all_chunks):
                if chunks:  # 非空字典
                    unit.chunks = chunks
                    unit.chunks_state = "success"
                    # 更新 checkpoint（内存中）
                    self.checkpoint.update_entry(
                        unit.checkpoint_key,
                        chunks=chunks,
                        chunks_state="success"
                    )
                    successful.append(unit)
                else:
                    failed.append((unit, "分块结果为空"))

            _log(f"Stage 2 完成：成功 {len(successful)}/{len(units)} ({len(successful)/len(units)*100:.1f}%)", "success")
            return BatchResult(successful=successful, failed=failed)

        except PartialBatchError as e:
            # 部分失败，处理成功和失败的结果
            _log(f"Stage 2 部分失败：{len(e.get_failures())}/{len(e.get_successful_results()) + len(e.get_failures())}", "warning")

            successful_units = []
            failed_units = []

            # 处理成功的结果
            # chunk_batch 的 PartialBatchError 中，get_successful_results() 返回 {index: result}
            for idx, chunks in e.get_successful_results().items():
                unit = units[idx]
                unit.chunks = chunks
                unit.chunks_state = "success"
                self.checkpoint.update_entry(
                    unit.checkpoint_key,
                    chunks=chunks,
                    chunks_state="success"
                )
                successful_units.append(unit)

            # 处理失败的任务
            for idx, error_msg, _ in e.get_failures():
                failed_units.append((units[idx], error_msg))

            return BatchResult(successful=successful_units, failed=failed_units)

        except Exception as e:
            _log(f"Stage 2 失败：{e}，降级为顺序处理...", "error")

            # 降级为顺序处理
            successful = []
            failed = []

            for unit in units:
                try:
                    fresh_chunker = AgenticChunk(llm=self.llm)
                    chunks = fresh_chunker.chunk(unit.propositions)
                    unit.chunks = chunks
                    unit.chunks_state = "success"
                    self.checkpoint.update_entry(
                        unit.checkpoint_key,
                        chunks=chunks,
                        chunks_state="success"
                    )
                    successful.append(unit)
                except Exception as inner_e:
                    failed.append((unit, str(inner_e)))

            return BatchResult(successful=successful, failed=failed)

    def _stage3_persist_results(
        self,
        all_data: List[Dict],
        processed_units: List[ProcessingUnit]
    ) -> Dict[str, int]:
        """
        Stage 3: 持久化结果到 benchmark 文件。

        参数：
            all_data: 完整的 benchmark 数据（会被原地修改）
            processed_units: 所有处理过的单元（包括成功和失败）

        返回：
            处理统计信息
        """
        _log("Stage 3: 持久化结果...", "info")

        # 按 item_index 分组
        units_by_item: Dict[int, List[ProcessingUnit]] = defaultdict(list)
        for unit in processed_units:
            units_by_item[unit.item_index].append(unit)

        # 应用结果到每个 item
        stats = {
            'processed_items': 0,
            'completed': 0,
            'partial': 0,
            'failed': 0
        }

        for item_idx, item_units in units_by_item.items():
            item = all_data[item_idx]
            overall_status = self.applier.apply(item, item_units)

            stats['processed_items'] += 1
            if overall_status == "completed":
                stats['completed'] += 1
            elif overall_status == "partial":
                stats['partial'] += 1
            else:
                stats['failed'] += 1

        return stats

    def process(
        self,
        limit: Optional[int] = None,
        retry_failed: bool = False,
        force: bool = False
    ) -> Dict[str, int]:
        """
        处理数据集。

        核心流程：
        1. 加载 checkpoint 和原始数据
        2. 构建 ProcessingUnit 列表
        3. Stage 1: 并行提取所有待处理单元的命题
        4. Stage 2: 并行对所有命题成功的单元进行分块
        5. 持久化结果到 benchmark 文件

        参数：
            limit: 限制处理数量（按问题计数），用于测试。None 表示处理全部。
            retry_failed: 如果 True，只处理之前失败的单元；如果 False，处理所有待处理单元
            force: 如果 True，强制重新处理所有单元（忽略 checkpoint）

        返回：
            处理统计信息字典
        """
        print(f"\n{'='*60}")
        print(f"Benchmark Chunker V2 - {self.dataset_type}")
        print(f"{'='*60}")

        # ========== Step 1: 加载数据 ==========
        all_data = self._load_data()
        total_items = len(all_data)
        _log(f"数据总数：{total_items} 个问题")

        # ========== Step 2: 确定要处理的单元 ==========
        _log("构建处理单元列表...")

        # 强制重新处理时，清空 checkpoint
        if force:
            _log("强制模式：清空 checkpoint", "warning")
            self.checkpoint.state["entries"] = {}

        # 确定要处理的单元
        if retry_failed and not force:
            # 只处理失败的单元
            keys_to_process = []
            keys_to_process.extend(self.checkpoint.filter_entries(propositions_state="failed"))
            keys_to_process.extend(self.checkpoint.filter_entries(
                propositions_state="success",
                chunks_state="failed"
            ))
            # 去重
            keys_to_process = list(set(keys_to_process))
        else:
            # 获取所有待处理的单元
            keys_to_process = self._get_pending_keys(all_data)

        # 限制数量（按问题计数）
        if limit:
            # 先找出涉及的问题索引
            item_indices = set()
            limited_keys = []
            for key in keys_to_process:
                item_idx, _ = parse_checkpoint_key(key)
                if item_idx not in item_indices:
                    if len(item_indices) >= limit:
                        break
                    item_indices.add(item_idx)
                limited_keys.append(key)
            keys_to_process = limited_keys

        _log(f"待处理单元数：{len(keys_to_process)}", "info")

        if not keys_to_process:
            _log("没有需要处理的数据。", "info")
            return {
                'processed_items': 0,
                'skipped': total_items,
                'propositions_success': 0,
                'propositions_failed': 0,
                'chunks_success': 0,
                'chunks_failed': 0
            }

        # 构建 ProcessingUnit 对象
        units_to_process = []
        for key in keys_to_process:
            entry = self.checkpoint.get_entry(key)
            # entry 此时一定存在（因为_get_pending_keys 已经初始化了）
            item_idx, ctx_idx = parse_checkpoint_key(key)
            unit = ProcessingUnit(
                checkpoint_key=key,
                item_index=item_idx,
                context_index=ctx_idx,
                item_id=entry.get("item_id", ""),
                title=entry.get("title", ""),
                text=entry.get("text", ""),
                propositions_state=entry.get("propositions_state", "pending"),
                propositions=entry.get("propositions"),
                chunks_state=entry.get("chunks_state", "pending"),
                chunks=entry.get("chunks"),
                attempt_count=entry.get("attempt_count", 0)
            )
            units_to_process.append(unit)

        # ========== Stage 1: 提取命题 ==========
        units_needing_prop = [
            unit for unit in units_to_process
            if unit.propositions_state in ("pending", "failed")
        ]

        if units_needing_prop:
            prop_result = self._stage1_extract_propositions(units_needing_prop)
            prop_success_count = len(prop_result.successful)
            prop_failed_count = len(prop_result.failed)
        else:
            prop_success_count = 0
            prop_failed_count = 0
            _log("没有需要提取命题的单元。", "info")

        # ========== Stage 2: 分块处理 ==========
        # 找出需要分块的单元（命题成功且 chunks_state 不是 success）
        units_needing_chunk = [
            unit for unit in units_to_process
            if unit.propositions_state == "success"
            and unit.chunks_state in ("pending", "failed")
        ]

        if units_needing_chunk:
            chunk_result = self._stage2_chunk_propositions(units_needing_chunk)
            chunk_success_count = len(chunk_result.successful)
            chunk_failed_count = len(chunk_result.failed)
        else:
            chunk_success_count = 0
            chunk_failed_count = 0
            _log("没有需要分块的单元。", "info")

        # ========== Stage 3: 持久化结果 ==========
        persist_stats = self._stage3_persist_results(all_data, units_to_process)

        # 保存数据和 checkpoint
        self._save_data(all_data)
        self.checkpoint.save()

        # ========== 汇总统计 ==========
        stats = {
            'processed_items': persist_stats['processed_items'],
            'completed': persist_stats['completed'],
            'partial': persist_stats['partial'],
            'failed': persist_stats['failed'],
            'skipped': total_items - persist_stats['processed_items'],
            'propositions_success': prop_success_count,
            'propositions_failed': prop_failed_count,
            'chunks_success': chunk_success_count,
            'chunks_failed': chunk_failed_count
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


# ==================== 兼容函数（供外部调用） ====================

def process_all_datasets(
    llm: Any,
    limit: Optional[int] = None,
    output_dir: Optional[str] = None,
    retry_failed: bool = False,
    use_v2: bool = True
) -> Dict[str, Dict[str, int]]:
    """
    处理所有 benchmark 数据集。

    参数：
        llm: LangChain 兼容的聊天模型
        limit: 每个数据集限制处理数量
        output_dir: 输出目录，默认为输入文件所在目录
        retry_failed: 是否只重试失败的文本块
        use_v2: 是否使用 V2 处理器（默认 True）

    返回：
        每个数据集的处理统计信息
    """
    if not use_v2:
        # 使用 V1 处理器（导入旧的 benchmark_chunker 模块）
        from .benchmark_chunker import process_all_datasets as v1_process
        return v1_process(llm, limit, output_dir, retry_failed)

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

        processor = BenchmarkChunkProcessorV2(
            llm=llm,
            dataset_type=dataset_type,
            input_path=input_path,
            output_path=output_path
        )

        stats = processor.process(limit=limit, retry_failed=retry_failed)
        all_stats[dataset_type] = stats

        print(f"\n{'='*60}\n")

    return all_stats


# ==================== 主程序（测试入口） ====================

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

    # 测试单个数据集
    processor = BenchmarkChunkProcessorV2(
        llm=llm,
        dataset_type='hotpotqa',
        input_path='Data/benchmark/HotpotQA_500_benchmark.json'
    )

    stats = processor.process(limit=3)
    print(f"\n测试处理完成：{stats}")
