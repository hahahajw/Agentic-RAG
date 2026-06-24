# ═══════════════════════════════════════════════════════════════════
# 外部接口 Protocol
# ═══════════════════════════════════════════════════════════════════
# SearchFn 是算法与外部世界（KB）之间的边界契约。
# 算法的所有其余部分只依赖此 Protocol，不依赖任何具体实现。

from typing import Protocol

from .models import ChunkInfo


class SearchFn(Protocol):
    """搜索接口——算法接触真实世界（KB）的唯一窗口。

    FRAMEWORK.md 模块 1 定义: 算法只能通过 search(query) 与 KB 交互。
    此 Protocol 是该交互的精确接口契约。

    输入:
      query: 搜索查询字符串。Solver 在调用前已完成:
        - 依赖注入——dependency 边源节点的 answer 替换指代词
        - 上下文聚合——decomposition 子节点的 answer 汇总为搜索上下文

    返回:
      ChunkInfo 列表。可能为空——搜索无结果或底层检索失败。
      失败时返回空列表而非抛异常——空列表让 Solver 正常记录空结果，
      Critic 随后标记节点 BLOCKED，Planner 在下一轮决定方向。

    实现者:
      - MilvusAdapter（Phase 5）: 适配到现有 Retrieval/milvus_retriever.py
      - 测试 mock: `lambda q: [mock_chunk1, mock_chunk2]`
        Protocol 支持结构化鸭子类型。

    同步接口——搜索并行化是 Solver 调度策略（Phase 2），不是接口的事。
    """
    def __call__(self, query: str) -> list[ChunkInfo]: ...