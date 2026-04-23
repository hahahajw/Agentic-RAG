"""
语义分块器 - 支持普通语义分块和递归语义分块，递归语义分块可以在语义和 chunk 长度上都有比较好的表现

语义分块的基本假设是：当两个 *相邻句子（句子窗口）* 在嵌入空间的距离较大时，前后两个句子可能表达了不同的主题。可以看到，即使是语义分块也无法摆脱句子位置的影响
借鉴 LangChain (https://github.com/langchain-ai/langchain-experimental/blob/main/libs/experimental/langchain_experimental/text_splitter.py) 和 B 站 UP 的实现 (https://github.com/blackinkkkxi/RAG_langchain/blob/main/chunsize/visual_semantic_chunking.ipynb)，我实现的语义分割模块的功能包括：
  1. 可以通过 buffer_size 参数控制句子窗口的大小
  2. 实例可以在「普通语义分割」和「递归语义分割」间切换；可以动态的切换「句子窗口的长度」、「递归分割时 chunk 大小」
  3. 可以可视化分割结果（虽然现在我还不知道怎么看）
  模块中的关键方法 chunk 方法（输入为处理好的按句划分的句子，输出为最终划分好的 chunk）的逻辑为：
  1. 首先将语料库以句划分
  2. 根据 buffer_size 参数，调用 _combine_sentences() 方法获得句子窗口
  3. 调用嵌入模型获得句子窗口的嵌入向量
  4. 调用 _calculate_cosine_distances() 方法获得相邻句子窗口在嵌入空间的距离
  5. 根据距离、分割方式和阈值完成分块
  6. 常规语义分块到这里就结束了，递归分块在此更近一步，对于当前获得的 chunk A，如果 chunk A 的长度超过 length_threshold，则直接复用步骤 d 预计算的全局距离数组中 chunk A 对应句子范围的切片进行递归分割（无需重新计算嵌入），直至所有分块长度均满足阈值要求 (代码实现在 371 - 382 行，chunk 方法的 for 循环中)

一个使用实例：
```python
sentences = [
          "Hans Barthold Andresen Butenschøn( 27 December 1877 – 28 November 1971) was a Norwegian businessperson.",
          "He was born in Kristiania as a son of Nils August Andresen Butenschøn and Hanna Butenschøn, and grandson of Nicolay Andresen.",
          "Together with Mabel Anette Plahte( 1877 – 1973, a daughter of Frithjof M. Plahte) he had the son Hans Barthold Andresen Butenschøn Jr. and was through him the father- in- law of Ragnhild Butenschøn and grandfather of Peter Butenschøn.",
          "Through his daughter Marie Claudine he was the father- in- law of Joakim Lehmkuhl, through his daughter Mabel Anette he was the father- in- law of Harald Astrup( a son of Sigurd Astrup) and through his daughter Nini Augusta he was the father- in- law of Ernst Torp.",
          "He took commerce school and agricultural school.",
          "He was hired in the family company N. A. Andresen& Co, and became a co-owner in 1910.",
          "He eventually became chief executive officer.",
          "The bank changed its name to Andresens Bank in 1913 and merged with Bergens Kreditbank in 1920.",
          "The merger was dissolved later in the 1920s.",
          "He was also a landowner, owning Nedre Skøyen farm and a lot of land in Enebakk.",
          "He chaired the board of Nydalens Compagnie from 1926, having not been a board member before that.",
          "He also chaired the supervisory council of Forsikringsselskapet Viking and Nedre Glommen salgsforening, and was a supervisory council member of Filharmonisk Selskap.",
          "He was a member of the gentlemen's club SK Fram since 1890, and was proclaimed a lifetime member in 1964.",
          "He was buried in Enebakk."
        ]

# 定义嵌入模型
import os
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings

load_dotenv()

e_m =  OpenAIEmbeddings(
        api_key=os.getenv("BL_API_KEY"),
        base_url=os.getenv("BL_BASE_URL"),
        model="text-embedding-v3",
        dimensions=1024,
        check_embedding_ctx_length=False,
        chunk_size=10   # 默认是 1000，但是 Qwen 最多接受 10 个待处理的字符串
    )

from semantic_chunker import SemanticChunker

chunker = SemanticChunker(
    embedding_model=e_m,
    mode='semantic',    # 可以利用 set_mode 方法更改，有 "semantic"、"recursive" 两种
    percentile_threshold=95,    # 只实现了百分位阈值划分
    length_threshold=500,   # 后续可以利用 et_lengththreshold 方法更改
    buffer_size=1   # 可以利用 set_buffer_size 更改
)


nomer_chunks = chunker.chunk(sentences=sentences)
# nomer_chunks 是
# [
# 'Hans Barthold Andresen Butenschøn( 27 December 1877 – 28 November 1971) was a Norwegian businessperson. He was born in Kristiania as a son of Nils August Andresen Butenschøn and Hanna Butenschøn, and grandson of Nicolay Andresen. Together with Mabel Anette Plahte( 1877 – 1973, a daughter of Frithjof M. Plahte) he had the son Hans Barthold Andresen Butenschøn Jr. and was through him the father- in- law of Ragnhild Butenschøn and grandfather of Peter Butenschøn. Through his daughter Marie Claudine he was the father- in- law of Joakim Lehmkuhl, through his daughter Mabel Anette he was the father- in- law of Harald Astrup( a son of Sigurd Astrup) and through his daughter Nini Augusta he was the father- in- law of Ernst Torp. He took commerce school and agricultural school.',
# "He was hired in the family company N. A. Andresen& Co, and became a co-owner in 1910. He eventually became chief executive officer. The bank changed its name to Andresens Bank in 1913 and merged with Bergens Kreditbank in 1920. The merger was dissolved later in the 1920s. He was also a landowner, owning Nedre Skøyen farm and a lot of land in Enebakk. He chaired the board of Nydalens Compagnie from 1926, having not been a board member before that. He also chaired the supervisory council of Forsikringsselskapet Viking and Nedre Glommen salgsforening, and was a supervisory council member of Filharmonisk Selskap. He was a member of the gentlemen's club SK Fram since 1890, and was proclaimed a lifetime member in 1964. He was buried in Enebakk."
# ]

chunker.get_mode(), chunker.set_mode(mode='recursive'), chunker.get_mode()  # ('semantic', None, 'recursive')

chunker.set_length_threshold(400)
recursive_chunks = chunker.chunk(sentences=sentences)
# recursive_chunks 是
# [
# 'Hans Barthold Andresen Butenschøn( 27 December 1877 – 28 November 1971) was a Norwegian businessperson. He was born in Kristiania as a son of Nils August Andresen Butenschøn and Hanna Butenschøn, and grandson of Nicolay Andresen.',
# 'Together with Mabel Anette Plahte( 1877 – 1973, a daughter of Frithjof M. Plahte) he had the son Hans Barthold Andresen Butenschøn Jr. and was through him the father- in- law of Ragnhild Butenschøn and grandfather of Peter Butenschøn. Through his daughter Marie Claudine he was the father- in- law of Joakim Lehmkuhl, through his daughter Mabel Anette he was the father- in- law of Harald Astrup( a son of Sigurd Astrup) and through his daughter Nini Augusta he was the father- in- law of Ernst Torp.',
# 'He took commerce school and agricultural school.',
# 'He was hired in the family company N. A. Andresen& Co, and became a co-owner in 1910. He eventually became chief executive officer. The bank changed its name to Andresens Bank in 1913 and merged with Bergens Kreditbank in 1920. The merger was dissolved later in the 1920s.',
# 'He was also a landowner, owning Nedre Skøyen farm and a lot of land in Enebakk. He chaired the board of Nydalens Compagnie from 1926, having not been a board member before that.',
# "He also chaired the supervisory council of Forsikringsselskapet Viking and Nedre Glommen salgsforening, and was a supervisory council member of Filharmonisk Selskap. He was a member of the gentlemen's club SK Fram since 1890, and was proclaimed a lifetime member in 1964. He was buried in Enebakk."
# ]

chunker.plot_chunk_differences(
    sentences=sentences,
    breakpoints=chunker.get_last_breakpoints(),
    distances=chunker.get_last_distances()
)

chunker.plot_distances(
    distances=chunker.get_last_distances(),
    breakpoints=chunker.get_last_breakpoints(),
    percentile_threshold=95
)
"""

from enum import Enum
from typing import List, Any, Optional
import numpy as np


class ChunkingMode(Enum):
    """分块模式枚举"""
    SEMANTIC = "semantic"           # 语义分块
    RECURSIVE_SEMANTIC = "recursive"  # 递归语义分块


class SemanticChunker:
    """
    语义分块器类，支持普通语义分块和递归语义分块
    """

    def __init__(
        self,
        embedding_model: Any,
        mode: str = "semantic",
        percentile_threshold: int = 95,
        length_threshold: int = 500,
        buffer_size: int = 1
    ):
        """
        初始化语义分块器

        Args:
            embedding_model: 嵌入模型
            mode: 分块模式，可选 "semantic" 或 "recursive"，默认为 "semantic"
            percentile_threshold: 百分位阈值，用于确定语义边界，默认为 95
            length_threshold: 递归分块时的长度阈值（按字符数计算），默认为 500
            buffer_size: 上下文窗口大小，控制用于计算嵌入的上下文句子数量，默认为 1
        """
        self.embedding_model = embedding_model
        self.chunking_mode = ChunkingMode(mode)
        self.percentile_threshold = percentile_threshold
        self.length_threshold = length_threshold
        self.buffer_size = buffer_size

    def get_mode(self) -> str:
        """返回当前分块模式"""
        return self.chunking_mode.value

    def set_mode(self, mode: str) -> None:
        """
        设置分块模式

        Args:
            mode: "semantic" 或 "recursive"
        """
        self.chunking_mode = ChunkingMode(mode)

    def set_length_threshold(self, length_threshold: int) -> None:
        """
        设置递归分块时的长度阈值

        Args:
            length_threshold: 长度阈值（按字符数计算）
        """
        self.length_threshold = length_threshold

    def set_buffer_size(self, buffer_size: int) -> None:
        """
        设置上下文窗口大小

        Args:
            buffer_size: 上下文窗口大小，控制用于计算嵌入的上下文句子数量
        """
        self.buffer_size = buffer_size

    def _combine_sentences(self, sentences: List[str], buffer_size: int) -> List[str]:
        """
        为每个句子创建包含上下文的组合句

        Args:
            sentences: 原始句子列表
            buffer_size: 上下文窗口大小

        Returns:
            List[str]: 组合句列表，每个元素是包含上下文的句子
        """
        combined_sentences = []

        for i in range(len(sentences)):
            combined_sentence = ""

            # 添加前缓冲区的句子
            for j in range(i - buffer_size, i):
                if j >= 0:
                    combined_sentence += sentences[j] + " "

            # 添加当前句子
            combined_sentence += sentences[i]

            # 添加后缓冲区的句子
            for j in range(i + 1, i + 1 + buffer_size):
                if j < len(sentences):
                    combined_sentence += " " + sentences[j]

            combined_sentences.append(combined_sentence)

        return combined_sentences

    def chunk(self, sentences: List[str]) -> List[str]:
        """
        对句子列表进行语义分块

        Args:
            sentences: 句子列表，每个元素是一个已分割好的句子

        Returns:
            分块后的文本列表，每个元素是一个语义完整的块
        """
        if not sentences:
            return []

        # 创建组合句（包含上下文）
        combined_sentences = self._combine_sentences(sentences, self.buffer_size)

        # 使用组合句计算嵌入
        embeddings = self.embedding_model.embed_documents(combined_sentences)

        # 计算相邻句子的余弦距离
        distances = self._calculate_cosine_distances(embeddings)

        # 根据模式选择分块算法
        if self.chunking_mode == ChunkingMode.SEMANTIC:
            breakpoints = self._find_breakpoints(distances, self.percentile_threshold)
        else:  # RECURSIVE_SEMANTIC
            breakpoints = self._recursive_chunk(
                distances,
                sentences,
                self.length_threshold,
                self.percentile_threshold
            )

        # 缓存断点，用于后续可视化
        self._last_breakpoints = breakpoints

        # 根据断点构建语义组
        return self._build_groups(sentences, breakpoints)

    def _calculate_cosine_distances(self, embeddings: List[List[float]]) -> np.ndarray:
        """
        计算相邻文本嵌入之间的余弦距离

        Args:
            embeddings: 嵌入向量列表

        Returns:
            numpy array: 相邻嵌入之间的余弦距离数组
        """
        len_embeddings = len(embeddings)
        cdists = np.empty(len_embeddings - 1)

        for i in range(1, len_embeddings):
            # 使用 numpy 计算余弦距离: 1 - cos_similarity
            v1 = np.array(embeddings[i])
            v2 = np.array(embeddings[i - 1])
            cdists[i - 1] = 1 - np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

        # 缓存距离，用于后续可视化
        self._last_distances = cdists

        return cdists

    def _find_breakpoints(
        self,
        distances: np.ndarray,
        percentile_threshold: int
    ) -> np.ndarray:
        """
        根据百分位阈值找到断点

        Args:
            distances: 余弦距离数组
            percentile_threshold: 百分位阈值（0-100）

        Returns:
            numpy array: 断点索引数组
        """
        if len(distances) == 0:
            return np.array([])

        # 计算百分位阈值
        breakpoint_distance_threshold = np.percentile(distances, percentile_threshold)

        # 找到所有超过阈值的点的索引位置
        threshold_indices = np.argwhere(distances >= breakpoint_distance_threshold).ravel()

        return threshold_indices

    def _build_groups(self, sentences: List[str], breakpoints: np.ndarray) -> List[str]:
        """
        根据断点构建语义组

        Args:
            sentences: 句子列表
            breakpoints: 断点索引数组

        Returns:
            List[str]: 语义组列表
        """
        start_index = 0
        grouped_texts = []

        # 添加结束标记
        breakpoints = np.append(breakpoints, [-1])

        for break_point in breakpoints:
            # 到达文本末尾
            if break_point == -1:
                grouped_texts.append(
                    " ".join([x for x in sentences[start_index:]])
                )
            else:
                grouped_texts.append(
                    " ".join([x for x in sentences[start_index: break_point + 1]])
                )

            start_index = break_point + 1

        return grouped_texts

    def _recursive_chunk(
        self,
        distances: np.ndarray,
        sentences: List[str],
        length_threshold: int,
        percentile_threshold: int
    ) -> np.ndarray:
        """
        递归语义分块算法（使用栈实现迭代版本）

        Args:
            distances: 余弦距离数组
            sentences: 句子列表（用于计算长度）
            length_threshold: 长度阈值（按字符数计算）
            percentile_threshold: 百分位阈值

        Returns:
            numpy array: 所有断点索引数组
        """
        # 初始化栈，存储待处理的距离范围 (start, end)
        S = [(0, len(distances))]
        all_breakpoints = set()

        while S:
            id_start, id_end = S.pop()

            # 提取当前范围的距离
            distance = distances[id_start:id_end]

            # 找到当前范围的断点
            updated_breakpoints = self._find_breakpoints(
                distance,
                percentile_threshold=percentile_threshold
            )

            # 如果没有断点，跳过
            if updated_breakpoints.size == 0:
                continue

            # 调整断点索引为全局索引
            updated_breakpoints += id_start

            # 添加范围边界作为虚拟断点
            updated_breakpoints = np.concatenate(
                (np.array([id_start - 1]), updated_breakpoints, np.array([id_end]))
            )

            # 检查每个子分块的长度，决定是否需要继续递归
            for index in updated_breakpoints:
                # 获取子分块对应的句子
                text_group = sentences[id_start: index + 1]

                # 计算子分块的字符数
                total_text = sum(len(text) for text in text_group)

                # 如果子分块超过长度阈值且包含多个句子，则继续递归
                if (len(text_group) > 2) and (total_text >= length_threshold):
                    S.append((id_start, index))

                id_start = index + 1

            # 添加断点到全局集合
            all_breakpoints.update(updated_breakpoints)

        # 返回排序后的断点数组（去掉虚拟边界）
        return np.array(sorted(all_breakpoints))[1:-1]

    def plot_distances(
        self,
        distances: np.ndarray,
        breakpoints: np.ndarray,
        percentile_threshold: int = None
    ) -> None:
        """
        可视化距离和断点

        Args:
            distances: 余弦距离数组
            breakpoints: 断点索引数组
            percentile_threshold: 百分位阈值（可选，用于显示阈值线）
        """
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 4))

        # 绘制所有距离点
        plt.plot(distances, "rx", markersize=5)

        # 绘制断点垂直线
        for index in breakpoints:
            plt.axvline(x=index, color='r', linestyle='--')

        # 绘制距离曲线
        plt.plot(distances)

        # 添加阈值线（如果提供了阈值）
        if percentile_threshold is not None:
            threshold_value = np.percentile(distances, percentile_threshold)
            plt.axhline(
                y=threshold_value,
                color='g',
                linestyle='--',
                label=f'{percentile_threshold}th percentile'
            )
            plt.legend()

        plt.xlabel("Index")
        plt.ylabel("Distance")
        plt.title("Cosine Distances with Breakpoints")
        plt.show()

    def plot_chunk_differences(
        self,
        sentences: List[str],
        breakpoints: np.ndarray,
        distances: np.ndarray
    ) -> None:
        """
        绘制分块差异和断点图

        Args:
            sentences: 句子列表
            breakpoints: 断点索引数组
            distances: 余弦距离数组
        """
        import matplotlib.pyplot as plt

        cumulative_len = np.cumsum([len(x) for x in sentences])
        fig = plt.figure(figsize=(20, 6))
        ax = fig.add_subplot(111)

        cosine_dist_min = 0
        cosine_dist_max = 1.1 * max(distances)

        ax.plot(cumulative_len[:-1], distances)
        ax.plot(cumulative_len[:-1], distances, "rx", markersize=5)
        ax.vlines(
            cumulative_len[breakpoints],
            ymin=cosine_dist_min,
            ymax=cosine_dist_max,
            colors="r",
            linestyles="--"
        )

        ax.set_xlabel("Cumulative characters")
        ax.set_ylabel("Cosine distance between splits")
        ax.set_title("Chunk Differences with Breakpoints")
        plt.tight_layout()
        plt.show()

    def get_last_distances(self) -> Optional[np.ndarray]:
        """
        获取上次分块计算的余弦距离

        Returns:
            Optional[np.ndarray]: 余弦距离数组，如果尚未调用 chunk() 则返回 None
        """
        return self._last_distances if hasattr(self, '_last_distances') else None

    def get_last_breakpoints(self) -> Optional[np.ndarray]:
        """
        获取上次分块计算的断点索引

        Returns:
            Optional[np.ndarray]: 断点索引数组，如果尚未调用 chunk() 则返回 None
        """
        return self._last_breakpoints if hasattr(self, '_last_breakpoints') else None
