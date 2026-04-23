"""
Benchmark 分块处理器 - 对 benchmark 数据集进行批量分块处理。

本模块提供对不同 benchmark 数据集的分块处理功能：
1. HotpotQA - 嵌套列表结构 [["title", [sentences]], ...]
2. 2WikiMultihopQA - 与 HotpotQA 结构相同
3. MuSiQue - 字典列表结构 [{"idx", "title", "paragraph_text"}, ...]

核心特性：
- 复用 AgenticChunk 的 get_propositions_batch 和 chunk_batch 方法实现并行处理
- Checkpoint 机制支持断点续跑和细粒度重试
- 解耦的架构设计，各组件职责清晰
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

from .agentic_chunk import AgenticChunk, PartialBatchError


# ==================== 数据结构 ====================

@dataclass
class TextMetadata:
    """文本块的元数据，用于结果映射"""
    item_idx: int       # 问题在 benchmark 数据列表中的索引
    context_idx: int    # 文本块在问题 context 中的索引
    item_id: str        # 问题的唯一 ID
    title: str          # 文本块标题

    def __hash__(self):
        return hash((self.item_idx, self.context_idx, self.item_id))


# ==================== Checkpoint 管理器 ====================

class _CheckpointManager:
    """
    管理 checkpoint 的加载、保存和状态追踪。

    Checkpoint 结构：
    {
        "dataset_type": "hotpotqa",
        "last_updated": "2026-04-01T12:00:00",
        "items": {
            "item_id_1": {
                "contexts": {
                    "0": {
                        "propositions": "success" | "failed" | "pending",
                        "chunks": "success" | "failed" | "pending",
                        "error": null | "error message",
                        "attempt_count": 1,
                        "last_updated": "..."
                    }
                }
            }
        }
    }
    """

    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = checkpoint_path
        self.state = self._load()

    def _load(self) -> Dict:
        """加载 checkpoint 状态"""
        if not self.checkpoint_path.exists():
            return {"items": {}, "dataset_type": None, "last_updated": None}

        try:
            with open(self.checkpoint_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"警告：checkpoint 文件损坏，将重新创建：{e}")
            return {"items": {}, "dataset_type": None, "last_updated": None}

    def save(self):
        """保存 checkpoint（原子写入）"""
        self.state["last_updated"] = datetime.now().isoformat()
        temp_path = self.checkpoint_path.with_suffix('.tmp')
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2)
        temp_path.replace(self.checkpoint_path)

    def get_item_status(self, item_id: str) -> str:
        """
        获取问题的处理状态

        返回：
            "completed" - 所有文本块都成功
            "partial"   - 部分文本块失败或待处理
            "pending"   - 从未处理过
        """
        if item_id not in self.state.get("items", {}):
            return "pending"

        item = self.state["items"][item_id]
        contexts = item.get("contexts", {})

        if not contexts:
            return "pending"

        statuses = [c.get("propositions") for c in contexts.values()]
        if all(s == "success" for s in statuses):
            return "completed"
        return "partial"

    def get_failed_contexts(self, item_id: str) -> List[int]:
        """获取失败的 context 索引列表"""
        if item_id not in self.state.get("items", {}):
            return []

        failed = []
        for ctx_idx, ctx in self.state["items"][item_id].get("contexts", {}).items():
            prop_status = ctx.get("propositions")
            chunk_status = ctx.get("chunks")
            if prop_status == "failed" or chunk_status == "failed":
                failed.append(int(ctx_idx))
        return failed

    def get_pending_contexts(self, item_id: str, total_contexts: int) -> List[int]:
        """
        获取待处理的 context 索引列表

        参数：
            item_id: 问题 ID
            total_contexts: 该问题总共有多少个 context

        返回：
            待处理的 context 索引列表
        """
        if item_id not in self.state.get("items", {}):
            # 从未处理过，所有 contexts 都待处理
            return list(range(total_contexts))

        processed = set()
        for ctx_idx, ctx in self.state["items"][item_id].get("contexts", {}).items():
            if ctx.get("propositions") == "success" and ctx.get("chunks") == "success":
                processed.add(int(ctx_idx))

        return [i for i in range(total_contexts) if i not in processed]

    def update_context_status(
        self,
        item_id: str,
        context_idx: int,
        prop_status: Optional[str] = None,
        chunk_status: Optional[str] = None,
        error: Optional[str] = None
    ):
        """更新特定文本块的状态"""
        if "items" not in self.state:
            self.state["items"] = {}

        if item_id not in self.state["items"]:
            self.state["items"][item_id] = {"contexts": {}}

        ctx_key = str(context_idx)
        if ctx_key not in self.state["items"][item_id]["contexts"]:
            self.state["items"][item_id]["contexts"][ctx_key] = {
                "propositions": "pending",
                "chunks": "pending",
                "attempt_count": 0
            }

        ctx = self.state["items"][item_id]["contexts"][ctx_key]
        if prop_status:
            ctx["propositions"] = prop_status
        if chunk_status:
            ctx["chunks"] = chunk_status
        if error:
            ctx["error"] = error
        ctx["attempt_count"] = ctx.get("attempt_count", 0) + 1
        ctx["last_updated"] = datetime.now().isoformat()

    def set_dataset_type(self, dataset_type: str):
        """设置数据集类型"""
        self.state["dataset_type"] = dataset_type


# ==================== 文本提取器 ====================

class _TextExtractor:
    """从数据项中提取文本块"""

    def __init__(self, dataset_type: str):
        self.dataset_type = dataset_type.lower()

    def extract(self, item: Dict, item_idx: int) -> List[Tuple[str, TextMetadata]]:
        """
        从数据项中提取所有文本块

        参数：
            item: 单个问题/数据项
            item_idx: 数据项在列表中的索引

        返回：
            [(text, metadata), ...] 列表
        """
        if self.dataset_type in ['hotpotqa', '2wikimultihop', '2wikimultihopqa']:
            return self._extract_hotpotqa(item, item_idx)
        elif self.dataset_type == 'musique':
            return self._extract_musique(item, item_idx)
        else:
            raise ValueError(
                f"不支持的数据集类型：{self.dataset_type}。"
                f"支持的类型：hotpotqa, 2wikimultihop, musique"
            )

    def _extract_hotpotqa(self, item: Dict, item_idx: int) -> List[Tuple[str, TextMetadata]]:
        """提取 HotpotQA 或 2WikiMultihopQA 的文本块"""
        results = []
        context = item.get('context', [])

        for ctx_idx, (title, sentences) in enumerate(context):
            text = f'Title: {title}. Content: {" ".join(sentences)}'
            metadata = TextMetadata(
                item_idx=item_idx,
                context_idx=ctx_idx,
                item_id=item.get('_id', ''),
                title=title
            )
            results.append((text, metadata))

        return results

    def _extract_musique(self, item: Dict, item_idx: int) -> List[Tuple[str, TextMetadata]]:
        """提取 MuSiQue 的文本块"""
        results = []
        paragraphs = item.get('paragraphs', [])

        for ctx_idx, para in enumerate(paragraphs):
            text = f'Title: {para["title"]}. Content: {para["paragraph_text"]}'
            metadata = TextMetadata(
                item_idx=item_idx,
                context_idx=ctx_idx,
                item_id=item.get('id', ''),
                title=para['title']
            )
            results.append((text, metadata))

        return results


# ==================== 结果应用器 ====================

class _ResultApplier:
    """将分块结果应用回原始数据结构"""

    def __init__(self, dataset_type: str):
        self.dataset_type = dataset_type.lower()

    def apply(
        self,
        item: Dict,
        results: List[Tuple[List[str], Dict]],
        context_statuses: Optional[Dict[Tuple[str, int], str]] = None
    ) -> Dict:
        """
        应用结果到数据项

        参数：
            item: 原始数据项
            results: [(propositions, chunks), ...] 列表
            context_statuses: {(item_id, context_idx): status, ...} 状态映射
        """
        if self.dataset_type in ['hotpotqa', '2wikimultihop', '2wikimultihopqa']:
            return self._apply_hotpotqa(item, results, context_statuses)
        elif self.dataset_type == 'musique':
            return self._apply_musique(item, results, context_statuses)
        else:
            raise ValueError(f"不支持的数据集类型：{self.dataset_type}")

    def _apply_hotpotqa(
        self,
        item: Dict,
        results: List[Tuple[List[str], Dict]],
        context_statuses: Optional[Dict[Tuple[str, int], str]] = None
    ) -> Dict:
        """
        应用结果到 HotpotQA 数据结构。

        参数：
            item: 原始数据项
            results: [(propositions, chunks), ...] 列表
            context_statuses: {(item_id, context_idx): status, ...} 状态映射
        """
        item_id = item.get('_id', '')

        for ctx_idx, (props, chunks) in enumerate(results):
            if ctx_idx < len(item['context']):
                # 确保 context 条目有第三个元素（结果字典）
                if len(item['context'][ctx_idx]) < 3:
                    item['context'][ctx_idx].append({})

                # 从 context_statuses 获取状态，如果没有则使用默认值
                status = 'pending'
                if context_statuses:
                    status = context_statuses.get((item_id, ctx_idx), 'pending')

                item['context'][ctx_idx][2] = {
                    'propositions': props if props else [],
                    'chunks': chunks if chunks else {},
                    'metadata': {
                        'status': status,  # 从 Checkpoint 派生
                        'timestamp': datetime.now().isoformat()
                    }
                }
        return item

    def _apply_musique(
        self,
        item: Dict,
        results: List[Tuple[List[str], Dict]],
        context_statuses: Optional[Dict[Tuple[str, int], str]] = None
    ) -> Dict:
        """
        应用结果到 MuSiQue 数据结构。

        参数：
            item: 原始数据项
            results: [(propositions, chunks), ...] 列表
            context_statuses: {(item_id, context_idx): status, ...} 状态映射
        """
        item_id = item.get('id', '')

        for ctx_idx, (props, chunks) in enumerate(results):
            if ctx_idx < len(item['paragraphs']):
                para = item['paragraphs'][ctx_idx]
                para['propositions'] = props if props else []
                para['chunks'] = chunks if chunks else {}

                # 从 context_statuses 获取状态，如果没有则使用默认值
                status = 'pending'
                if context_statuses:
                    status = context_statuses.get((item_id, ctx_idx), 'pending')

                para['metadata'] = {
                    'status': status,  # 从 Checkpoint 派生
                    'timestamp': datetime.now().isoformat()
                }
        return item


# ==================== 主处理器 ====================

class BenchmarkChunkProcessor:
    """
    Benchmark 数据集分块处理器。

    支持三种数据集：
    - HotpotQA
    - 2WikiMultihopQA
    - MuSiQue

    核心特性：
    - 使用 AgenticChunk 的 batch 方法并行处理
    - Checkpoint 机制支持断点续跑和细粒度重试
    - 解耦的架构设计

    示例用法：
        >>> from langchain_openai import ChatOpenAI
        >>> llm = ChatOpenAI(model='qwen3-max', temperature=0)
        >>> processor = BenchmarkChunkProcessor(
        ...     llm=llm,
        ...     dataset_type='hotpotqa',
        ...     input_path='Data/benchmark/HotpotQA_500_benchmark.json',
        ...     output_path='Data/benchmark/HotpotQA_500_benchmark_chunked.json'
        ... )
        >>> processor.process(limit=10)  # 测试模式处理 10 个问题
    """

    def __init__(
        self,
        llm: Any,
        dataset_type: str,
        input_path: str,
        output_path: Optional[str] = None
    ):
        """
        初始化处理器。

        参数：
            llm: LangChain 兼容的聊天模型
            dataset_type: 数据集类型 ('hotpotqa', '2wikimultihop', 'musique')
            input_path: 输入文件路径
            output_path: 输出文件路径，默认为 None 时自动生成
        """
        self.llm = llm
        self.agentic_chunk = AgenticChunk(llm=llm)
        self.dataset_type = dataset_type.lower()
        self.input_path = Path(input_path)
        self.output_path = Path(output_path) if output_path else self._default_output_path()
        self.checkpoint_path = self.output_path.with_suffix('.checkpoint.json')

        # 初始化组件
        self.checkpoint = _CheckpointManager(self.checkpoint_path)
        self.checkpoint.set_dataset_type(self.dataset_type)
        self.extractor = _TextExtractor(self.dataset_type)
        self.applier = _ResultApplier(self.dataset_type)

        # 批处理配置
        self.max_workers = 5
        self.timeout_per_task = 300.0

    def _default_output_path(self) -> Path:
        """生成默认输出路径"""
        stem = self.input_path.stem
        return self.input_path.parent / f"{stem}_chunked.json"

    def _load_data(self) -> List[Dict]:
        """加载输入数据"""
        with open(self.input_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _save_data(self, data: List[Dict]):
        """保存数据到输出文件（原子写入）"""
        temp_path = self.output_path.with_suffix('.tmp')
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        temp_path.replace(self.output_path)

    def _process_propositions_batch(
        self,
        texts: List[str]
    ) -> Tuple[List[int], List[List[str]], Dict[int, str]]:
        """
        并行提取命题

        参数：
            texts: 文本列表

        返回：
            (成功索引列表，成功结果列表，失败字典)
            失败字典：{index: error_message}
        """
        if not texts:
            return [], [], {}

        successful_indices = []
        successful_results = []
        failures = {}

        try:
            all_propositions = self.agentic_chunk.get_propositions_batch(texts)

            for idx, props in enumerate(all_propositions):
                if props is not None:
                    successful_indices.append(idx)
                    successful_results.append(props)
                else:
                    failures[idx] = "命题提取返回空值"

        except Exception as e:
            # 如果整个批次失败，记录所有索引为失败
            for idx in range(len(texts)):
                failures[idx] = str(e)

        return successful_indices, successful_results, failures

    def _process_chunks_batch(
        self,
        propositions_list: List[List[str]]
    ) -> Tuple[List[int], List[Dict], Dict[int, str]]:
        """
        并行分块处理 - 使用 AgenticChunk.chunk_batch() 实现真正的并行处理

        参数：
            propositions_list: 命题组列表

        返回：
            (成功索引列表，成功结果列表，失败字典)
        """
        if not propositions_list:
            return [], [], {}

        successful_indices = []
        successful_results = []
        failures = {}

        # 过滤空命题组，但保留索引映射
        non_empty_indices = [
            (i, props) for i, props in enumerate(propositions_list) if props
        ]
        empty_indices = [
            i for i, props in enumerate(propositions_list) if not props
        ]

        # 处理空命题组（空命题直接返回空块）
        for i in empty_indices:
            successful_indices.append(i)
            successful_results.append({})

        if not non_empty_indices:
            # 全部为空
            return successful_indices, successful_results, {}

        # 使用 chunk_batch 进行并行处理
        print(f"  使用并行模式处理 {len(non_empty_indices)} 个非空命题组 (max_workers={self.max_workers})...")

        # 提取非空命题组（保持顺序）
        non_empty_props = [props for _, props in non_empty_indices]

        try:
            # chunk_batch 返回的结果顺序与输入顺序一致
            batch_results = self.agentic_chunk.chunk_batch(
                non_empty_props,
                max_workers=self.max_workers,
                timeout_per_task=self.timeout_per_task,
                on_error="raise"  # 抛出 PartialBatchError 以便我们处理部分失败
            )

            # 所有任务都成功
            for task_idx, chunks in enumerate(batch_results):
                orig_idx = non_empty_indices[task_idx][0]
                successful_indices.append(orig_idx)
                successful_results.append(chunks)

        except PartialBatchError as e:
            # 部分任务失败
            print(f"  警告：{len(e.get_failures())}/{len(non_empty_props)} 个任务失败，降级处理")

            # 处理成功的结果
            for task_idx, chunks in e.get_successful_results().items():
                orig_idx = non_empty_indices[task_idx][0]
                successful_indices.append(orig_idx)
                successful_results.append(chunks)

            # 处理失败的任务
            for task_idx, error_msg, _ in e.get_failures():
                orig_idx = non_empty_indices[task_idx][0]
                failures[orig_idx] = error_msg

            # 记录进度
            print(f"  成功：{len(successful_indices) - len(empty_indices)} 个，失败：{len(failures)} 个")

        except Exception as e:
            # 如果整个批次失败（比如超时），降级为顺序处理
            print(f"  并行处理失败：{e}，降级为顺序处理...")

            for task_idx, (orig_idx, props) in enumerate(non_empty_indices):
                try:
                    fresh_chunker = AgenticChunk(llm=self.llm)
                    chunks = fresh_chunker.chunk(props)
                    successful_indices.append(orig_idx)
                    successful_results.append(chunks)
                except Exception as inner_e:
                    failures[orig_idx] = str(inner_e)

            print(f"  顺序处理完成：成功 {len(successful_indices) - len(empty_indices)} 个，失败 {len(failures)} 个")

        # 每处理 10 个打印进度（并行模式下在 batch 调用后打印）
        if len(non_empty_indices) >= 10:
            print(f"  已处理 {len(successful_indices) - len(empty_indices)}/{len(non_empty_indices)} 个非空命题组")

        return successful_indices, successful_results, failures

    def _build_results_map(
        self,
        metadata_list: List[TextMetadata],
        propositions_list: List[List[str]],
        chunks_list: List[Dict]
    ) -> Dict[int, List[Tuple[List[str], Dict]]]:
        """
        按 item 组织结果

        参数：
            metadata_list: 元数据列表
            propositions_list: 命题列表
            chunks_list: 分块结果列表

        返回：
            {item_idx: [(propositions, chunks), ...]}
        """
        results_by_item: Dict[int, List[Tuple[List[str], Dict]]] = {}

        for meta, props, chunks in zip(metadata_list, propositions_list, chunks_list):
            if meta.item_idx not in results_by_item:
                # 需要知道该 item 有多少个 context
                results_by_item[meta.item_idx] = []

            # 扩展到足够的长度
            while len(results_by_item[meta.item_idx]) <= meta.context_idx:
                results_by_item[meta.item_idx].append(([], {}))

            results_by_item[meta.item_idx][meta.context_idx] = (props, chunks)

        return results_by_item

    def process(
        self,
        limit: Optional[int] = None,
        retry_failed: bool = False
    ) -> Dict[str, int]:
        """
        处理数据集。

        参数：
            limit: 限制处理数量，用于测试。None 表示处理全部。
            retry_failed: 是否只重试之前失败的文本块

        返回：
            处理统计信息字典
        """
        print(f"\n{'='*60}")
        print(f"开始处理 {self.dataset_type} 数据集...")
        print(f"{'='*60}")
        print(f"输入文件：{self.input_path}")
        print(f"输出文件：{self.output_path}")

        # ========== Step 1: 加载数据 ==========
        all_data = self._load_data()
        total_count = len(all_data)
        print(f"数据总数：{total_count}")

        # ========== Step 2: 确定要处理的项 ==========
        items_to_process = []  # [(item_idx, item, pending_contexts)]

        for item_idx, item in enumerate(all_data):
            if limit and len(items_to_process) >= limit:
                break

            item_id = item.get('_id') or item.get('id')
            if not item_id:
                print(f"警告：跳过没有 ID 的数据项（索引 {item_idx}）")
                continue

            # 获取该 item 的 context 数量
            if self.dataset_type in ['hotpotqa', '2wikimultihop']:
                total_contexts = len(item.get('context', []))
            else:
                total_contexts = len(item.get('paragraphs', []))

            if retry_failed:
                # 只重试失败的 contexts
                failed_ctx_idxs = self.checkpoint.get_failed_contexts(item_id)
                if failed_ctx_idxs:
                    items_to_process.append((item_idx, item, failed_ctx_idxs))
            else:
                # 处理所有未完成的 contexts
                pending_ctx_idxs = self.checkpoint.get_pending_contexts(item_id, total_contexts)
                if pending_ctx_idxs:
                    items_to_process.append((item_idx, item, pending_ctx_idxs))

        if not items_to_process:
            print("没有需要处理的数据。")
            return {'processed': 0, 'skipped': total_count}

        print(f"待处理问题数：{len(items_to_process)}")

        # ========== Step 3: 提取文本块 ==========
        print("提取文本块...")
        all_texts = []
        metadata_list = []
        text_to_meta_idx = []  # 记录每个文本在 metadata_list 中的索引

        for item_idx, item, pending_ctx_idxs in items_to_process:
            texts_with_meta = self.extractor.extract(item, item_idx)

            for text, meta in texts_with_meta:
                if meta.context_idx in pending_ctx_idxs:
                    text_to_meta_idx.append(len(metadata_list))
                    metadata_list.append(meta)
                    all_texts.append(text)

        print(f"共提取 {len(all_texts)} 个文本块")

        if not all_texts:
            print("没有需要处理的文本块。")
            return {'processed': 0, 'skipped': total_count}

        # ========== Step 4: 并行提取命题 ==========
        print(f"并行提取命题...")
        prop_success_indices, prop_results, prop_failures = self._process_propositions_batch(all_texts)

        print(f"命题提取：成功 {len(prop_success_indices)} 个，失败 {len(prop_failures)} 个")

        # 更新 checkpoint 中的命题状态
        for idx, error_msg in prop_failures.items():
            meta = metadata_list[idx]
            self.checkpoint.update_context_status(
                item_id=meta.item_id,
                context_idx=meta.context_idx,
                prop_status="failed",
                error=error_msg
            )

        # ========== Step 5: 并行分块处理 ==========
        print(f"并行分块处理 (max_workers={self.max_workers})...")

        # 为分块准备输入（只有成功提取命题的文本）
        props_for_chunking = []
        valid_meta_indices = []  # 对应 metadata_list 的索引

        for idx in prop_success_indices:
            props_for_chunking.append(prop_results[prop_success_indices.index(idx)])
            valid_meta_indices.append(idx)

        chunk_success_indices, chunk_results, chunk_failures = self._process_chunks_batch(props_for_chunking)

        print(f"分块处理：成功 {len(chunk_success_indices)} 个，失败 {len(chunk_failures)} 个")

        # 更新 checkpoint 中的分块状态
        for orig_idx, error_msg in chunk_failures.items():
            if orig_idx < len(valid_meta_indices):
                meta = metadata_list[valid_meta_indices[orig_idx]]
                self.checkpoint.update_context_status(
                    item_id=meta.item_id,
                    context_idx=meta.context_idx,
                    chunk_status="failed",
                    error=error_msg
                )

        # ========== Step 6: 组织结果 ==========
        print("组织结果...")

        # 构建完整的结果列表（失败的用空值填充）
        final_props_list = [[] for _ in metadata_list]
        final_chunks_list = [{} for _ in metadata_list]

        for success_idx, props in zip(prop_success_indices, prop_results):
            final_props_list[success_idx] = props

        for chunk_idx, orig_idx in enumerate(chunk_success_indices):
            if orig_idx < len(valid_meta_indices):
                meta_idx = valid_meta_indices[orig_idx]
                final_chunks_list[meta_idx] = chunk_results[chunk_idx]

        # ========== Step 7: 应用结果到数据 ==========
        results_by_item = self._build_results_map(
            metadata_list,
            final_props_list,
            final_chunks_list
        )

        # 构建上下文状态映射 {(item_id, context_idx): status}
        # 状态定义：
        #   - "completed" - 命题 AND 分块都成功（分块非空）
        #   - "partial"   - 命题成功但分块失败/为空
        #   - "failed"    - 命题提取失败
        context_statuses: Dict[Tuple[str, int], str] = {}

        # 更新成功的状态
        for idx, meta in enumerate(metadata_list):
            item_id = meta.item_id
            context_idx = meta.context_idx

            if idx in prop_success_indices:
                # 只有命题 AND 分块都成功才算完成
                chunk_result = final_chunks_list[idx]
                if chunk_result:  # 非空字典
                    overall_status = "completed"
                    chunk_status = "success"
                else:
                    overall_status = "partial"  # 命题成功但分块失败/为空
                    chunk_status = "failed"

                # 更新 checkpoint
                self.checkpoint.update_context_status(
                    item_id=item_id,
                    context_idx=context_idx,
                    prop_status="success",
                    chunk_status=chunk_status
                )
            else:
                # 命题提取失败
                overall_status = "failed"

            # 记录状态用于输出 JSON
            context_statuses[(item_id, context_idx)] = overall_status

        # 应用结果到原始数据（传入状态映射）
        processed_count = 0
        for item_idx, item, _ in items_to_process:
            if item_idx in results_by_item:
                results = results_by_item[item_idx]
                self.applier.apply(item, results, context_statuses)
                processed_count += 1

        # ========== Step 8: 保存结果 ==========
        print(f"保存结果...")
        self._save_data(all_data)
        self.checkpoint.save()

        stats = {
            'processed': processed_count,
            'propositions_success': len(prop_success_indices),
            'propositions_failed': len(prop_failures),
            'chunks_success': len(chunk_success_indices),
            'chunks_failed': len(chunk_failures),
            'total': total_count
        }

        print(f"\n处理完成!")
        print(f"  处理问题数：{stats['processed']}")
        print(f"  命题提取：成功 {stats['propositions_success']} 个，失败 {stats['propositions_failed']} 个")
        print(f"  分块处理：成功 {stats['chunks_success']} 个，失败 {stats['chunks_failed']} 个")
        print(f"  总计：{stats['total']} 条")

        return stats


def process_all_datasets(
    llm: Any,
    limit: Optional[int] = None,
    output_dir: Optional[str] = None,
    retry_failed: bool = False
) -> Dict[str, Dict[str, int]]:
    """
    处理所有 benchmark 数据集。

    参数：
        llm: LangChain 兼容的聊天模型
        limit: 每个数据集限制处理数量
        output_dir: 输出目录，默认为输入文件所在目录
        retry_failed: 是否只重试失败的文本块

    返回：
        每个数据集的处理统计信息
    """
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

        processor = BenchmarkChunkProcessor(
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
    # 测试运行示例
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

    # 处理单个数据集进行测试
    processor = BenchmarkChunkProcessor(
        llm=llm,
        dataset_type='hotpotqa',
        input_path='Data/benchmark/HotpotQA_500_benchmark.json'
    )

    stats = processor.process(limit=3)
    print(f"\n测试处理完成：{stats}")
