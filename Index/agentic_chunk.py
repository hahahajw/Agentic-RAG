"""
AgenticChunk - LLM 驱动的命题级智能分块模块。

本模块实现智能分块功能：
1. 将相关命题分组为语义连贯的块
2. 为每个块动态生成人类可读的标题和摘要
3. 当新命题加入现有块时更新元数据
"""

import uuid
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
import logging

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate


# --- 日志记录器 ---
logger = logging.getLogger(__name__)


# --- 异常类 ---

class ChunkBatchError(Exception):
    """chunk_batch 处理异常的基类。"""
    pass


class PartialBatchError(ChunkBatchError):
    """
    部分任务失败时抛出的异常。

    包含成功处理的结果和失败任务的详细信息。

    属性：
        successful_results: 成功处理的结果字典，映射索引到结果
        failures: 失败任务列表，每项包含 (index, error_message, exception)
    """
    def __init__(
        self,
        successful_results: Dict[int, Any],
        failures: List[tuple]
    ):
        self.successful_results = successful_results
        self.failures = failures

        # 构建详细的错误消息
        failure_details = [
            f"索引 {idx}: {err_msg}"
            for idx, err_msg, _ in failures
        ]
        message = f"部分任务失败 ({len(failures)}/{len(successful_results) + len(failures)}):\n" + "\n".join(failure_details)

        super().__init__(message)

    def get_failures(self) -> List[tuple]:
        """获取失败任务列表：[(index, error_message, exception), ...]"""
        return self.failures

    def get_successful_results(self) -> Dict[int, Any]:
        """获取成功处理的结果：{index: result, ...}"""
        return self.successful_results


# --- Pydantic 模式用于结构化输出 ---

# 说明：
# 1. 为什么 with_structured_output 方法中需要设置 method="function_calling"？
#    - Qwen 模型对 function calling 格式支持最好，输出最稳定
#    - json_mode 在某些模型上可能不够稳定
#    - function_calling 会将 Pydantic 模型转换为 function definition 格式
# 2. 在 LangChain 中，ChatPromptTemplate 是提示词模板，不是可被直接使用的消息，
#    需要使用 .invoke 或 .format 方法向其中的占位符传递需要的信息后，才能转换成可以直接传递给 LLM 的消息



class PropositionExtraction(BaseModel):
    """从文本中提取命题列表的响应模型。"""
    propositions: List[str] = Field(
        ...,
        description="List of standalone factual statements extracted from the text"
    )


class ChunkAssignment(BaseModel):
    """块分配响应模型。"""
    chunk_id: Optional[str] = Field(
        None,
        description="ID of existing chunk to join, or None if no matching chunk found"
    )
    found: bool = Field(
        ...,
        description="Whether a matching chunk was found"
    )


class Summary(BaseModel):
    """块摘要响应模型。"""
    summary: str = Field(
        ...,
        description="One-sentence summary describing what the chunk is about"
    )


class Title(BaseModel):
    """块标题响应模型。"""
    title: str = Field(
        ...,
        description="Brief title (2-5 words) describing the chunk topic"
    )


# --- 默认 Prompt 模板（来自 https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/agentic_chunker.py） ---

# LangHub prompt 用于命题提取（wfh/proposal-indexing），来自 https://smith.langchain.com/hub/wfh/proposal-indexing?organizationId=97591f89-2916-48d3-804e-20cab23f91aa
DEFAULT_PROPOSITION_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    (
        "system",
        """Decompose the "Content" into clear and simple propositions, ensuring they are interpretable out of
context.
1. Split compound sentence into simple sentences. Maintain the original phrasing from the input
whenever possible.
2. For any named entity that is accompanied by additional descriptive information, separate this
information into its own distinct proposition.
3. Decontextualize the proposition by adding necessary modifier to nouns or entire sentences
and replacing pronouns (e.g., "it", "he", "she", "they", "this", "that") with the full name of the
entities they refer to.
4. Present the results as a list of strings, formatted in JSON.

Example:

Input: Title: ¯Eostre. Section: Theories and interpretations, Connection to Easter Hares. Content:
The earliest evidence for the Easter Hare (Osterhase) was recorded in south-west Germany in
1678 by the professor of medicine Georg Franck von Franckenau, but it remained unknown in
other parts of Germany until the 18th century. Scholar Richard Sermon writes that "hares were
frequently seen in gardens in spring, and thus may have served as a convenient explanation for the
origin of the colored eggs hidden there for children. Alternatively, there is a European tradition
that hares laid eggs, since a hare’s scratch or form and a lapwing’s nest look very similar, and
both occur on grassland and are first seen in the spring. In the nineteenth century the influence
of Easter cards, toys, and books was to make the Easter Hare/Rabbit popular throughout Europe.
German immigrants then exported the custom to Britain and America where it evolved into the
Easter Bunny."
Output: [ "The earliest evidence for the Easter Hare was recorded in south-west Germany in
1678 by Georg Franck von Franckenau.", "Georg Franck von Franckenau was a professor of
medicine.", "The evidence for the Easter Hare remained unknown in other parts of Germany until
the 18th century.", "Richard Sermon was a scholar.", "Richard Sermon writes a hypothesis about
the possible explanation for the connection between hares and the tradition during Easter", "Hares
were frequently seen in gardens in spring.", "Hares may have served as a convenient explanation
for the origin of the colored eggs hidden in gardens for children.", "There is a European tradition
that hares laid eggs.", "A hare’s scratch or form and a lapwing’s nest look very similar.", "Both
hares and lapwing’s nests occur on grassland and are first seen in the spring.", "In the nineteenth
century the influence of Easter cards, toys, and books was to make the Easter Hare/Rabbit popular
throughout Europe.", "German immigrants exported the custom of the Easter Hare/Rabbit to
Britain and America.", "The custom of the Easter Hare/Rabbit evolved into the Easter Bunny in
Britain and America."]""",
    ),
    ("user", "Decompose the following: \n{input}"),
])

DEFAULT_FIND_CHUNK_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        Determine whether or not the "Proposition" should belong to any of the existing chunks.

        A proposition should belong to a chunk of their meaning, direction, or intention are similar.
        The goal is to group similar propositions and chunks.

        If you think a proposition should be joined with a chunk, return the chunk id.
        If you do not think an item should be joined with an existing chunk, indicate that no chunk was found.

        Example:
        Input:
            - Proposition: "Greg really likes hamburgers"
            - Current Chunks:
                - Chunk ID: 43d3ef5e-bdfe-4ac6-b781-fc7b0d998f2f
                - Chunk Name: Places in San Francisco
                - Chunk Summary: Overview of the things to do with San Francisco Places

                - Chunk ID: 0d38e1d4-52d7-4b89-a46d-4a649dde8721
                - Chunk Name: Food Greg likes
                - Chunk Summary: Lists of the food and dishes that Greg likes
        Output: 0d38e1d4-52d7-4b89-a46d-4a649dde8721
        """,
    ),
    ("user", "Current Chunks:\n--Start of current chunks--\n{chunk_outline}\n--End of current chunks--"),
    ("user", "Determine if the following statement should belong to one of the chunks outlined:\n{proposition}"),
])

DEFAULT_NEW_CHUNK_SUMMARY_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic
        You should generate a very brief 1-sentence summary which will inform viewers what a chunk group is about.

        A good summary will say what the chunk is about, and give any clarifying instructions on what to add to the chunk.

        You will be given a proposition which will go into a new chunk. This new chunk needs a summary.

        Your summaries should anticipate generalization. If you get a proposition about apples, generalize it to food.
        Or month, generalize it to "date and times".

        Example:
        Input: Proposition: Greg likes to eat pizza
        Output: This chunk contains information about the types of food Greg likes to eat.

        Only respond with the chunk new summary, nothing else.
        """,
    ),
    ("user", "Determine the summary of the new chunk that this proposition will go into:\n{proposition}"),
])

DEFAULT_NEW_CHUNK_TITLE_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic
        You should generate a very brief few word chunk title which will inform viewers what a chunk group is about.

        A good chunk title is brief but encompasses what the chunk is about

        You will be given a summary of a chunk which needs a title

        Your titles should anticipate generalization. If you get a proposition about apples, generalize it to food.
        Or month, generalize it to "date and times".

        Example:
        Input: Summary: This chunk is about dates and times that the author talks about
        Output: Date & Times

        Only respond with the new chunk title, nothing else.
        """,
    ),
    ("user", "Determine the title of the chunk that this summary belongs to:\n{summary}"),
])

DEFAULT_UPDATE_SUMMARY_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic
        A new proposition was just added to one of your chunks, you should generate a very brief 1-sentence summary which will inform viewers what a chunk group is about.

        A good summary will say what the chunk is about, and give any clarifying instructions on what to add to the chunk.

        You will be given a group of propositions which are in the chunk and the chunks current summary.

        Your summaries should anticipate generalization. If you get a proposition about apples, generalize it to food.
        Or month, generalize it to "date and times".

        Example:
        Input: Proposition: Greg likes to eat pizza
        Output: This chunk contains information about the types of food Greg likes to eat.

        Only respond with the chunk new summary, nothing else.
        """,
    ),
    ("user", "Chunk's propositions:\n{propositions}\n\nCurrent chunk summary:\n{current_summary}"),
])

DEFAULT_UPDATE_TITLE_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic
        A new proposition was just added to one of your chunks, you should generate a very brief updated chunk title which will inform viewers what a chunk group is about.

        A good title will say what the chunk is about.

        You will be given a group of propositions which are in the chunk, chunk summary and the chunk title.

        Your title should anticipate generalization. If you get a proposition about apples, generalize it to food.
        Or month, generalize it to "date and times".

        Example:
        Input: Summary: This chunk is about dates and times that the author talks about
        Output: Date & Times

        Only respond with the new chunk title, nothing else.
        """,
    ),
    ("user", "Chunk's propositions:\n{propositions}\n\nChunk summary:\n{current_summary}\n\nCurrent chunk title:\n{current_title}"),
])


class AgenticChunk:
    """
    LLM 驱动的智能分块，基于语义相似性分组命题并生成人类可读的元数据（标题和摘要）。

    分块流程：
    1. 从文本中获得命题
    2. 对每个命题，找到现有的块加入
    3. 如果没有匹配的块，创建新块
    4. 添加到块时，更新其标题和摘要以反映新内容

    属性：
        llm: LangChain 兼容的聊天模型
        chunks: 映射 chunk_id 到块数据的字典
        generate_metadata: 是否生成/更新标题和摘要
        print_logging: 是否打印日志消息

    示例：
        >>> from langchain_openai import ChatOpenAI
        >>> llm = ChatOpenAI(model="qwen3-max", temperature=0)
        >>> chunker = AgenticChunk(llm=llm)
        >>> propositions = ["The year is 2023.", "The month is October."]
        >>> chunks = chunker.chunk(propositions)
        >>> print(chunker.get_chunks())
    """

    def __init__(
        self,
        llm: Optional[Any] = None,
    ):
        """
        初始化 AgenticChunker。

        参数：
            llm: LangChain 兼容的聊天模型。如果为 None，使用 qwen3-max
        """
        self.llm = llm or self._default_llm()
        self.chunks: Dict[str, dict] = {}
        self.generate_metadata = True
        self.print_logging = False

        # 初始化 prompt 模板属性（避免使用 hasattr 检查）
        self._find_prompt_template: Optional[ChatPromptTemplate] = None
        self._new_summary_prompt_template: Optional[ChatPromptTemplate] = None
        self._new_title_prompt_template: Optional[ChatPromptTemplate] = None
        self._update_summary_prompt_template: Optional[ChatPromptTemplate] = None
        self._update_title_prompt_template: Optional[ChatPromptTemplate] = None

        # 初始化结构化输出模型
        self._setup_structured_outputs()

    def _default_llm(self) -> Any:
        from langchain_openai import ChatOpenAI
        import os
        return ChatOpenAI(
            api_key=os.getenv("BL_API_KEY"),
            base_url=os.getenv("BL_BASE_URL"),
            model='qwen3-max',
            temperature=0.0
        )

    def _setup_structured_outputs(self):
        """为每个任务初始化结构化输出模型。"""
        self.chunk_assignment_model = self.llm.with_structured_output(ChunkAssignment, method="function_calling")
        self.summary_model = self.llm.with_structured_output(Summary, method="function_calling")
        self.title_model = self.llm.with_structured_output(Title, method="function_calling")
        self.proposition_model = self.llm.with_structured_output(PropositionExtraction, method="function_calling")

    # ==================== 命题提取 ====================

    def get_propositions(
        self,
        text: str,
        prompt_template: Optional[ChatPromptTemplate] = None
    ) -> List[str]:
        """
        从原始文本中提取命题。

        参数：
            text: 要提取命题的原始文本。输入的 text 应当包含标题和内容，见 https://smith.langchain.com/hub/wfh/proposal-indexing?organizationId=50995362-9ea0-4378-ad97-b4edae2f9f22
            prompt_template: 可选的自定义提示模板。如果提供，将覆盖默认提示。

        返回：
            提取的命题列表。
        """
        if prompt_template is None:
            prompt_template = DEFAULT_PROPOSITION_PROMPT_TEMPLATE

        model = self.llm.with_structured_output(PropositionExtraction, method="function_calling")
        response = model.invoke(prompt_template.format(input=text))

        return response.propositions

    def get_propositions_batch(
        self,
        texts: List[str],
        prompt_template: Optional[ChatPromptTemplate] = None
    ) -> List[List[str]]:
        """
        批量从多个文本中提取命题。

        参数：
            texts: 要提取命题的原始文本列表。
            prompt_template: 可选的自定义提示模板，应用于所有文本。

        返回：
            命题列表的列表，保持输入顺序。
            示例：[["prop1", "prop2"], ["prop3"], ["prop4", "prop5", "prop6"]]

        注意：
            使用 LangChain 的 batch() 方法进行高效的并行处理。
            输出顺序与输入顺序匹配。
        """
        if prompt_template is None:
            prompt_template = DEFAULT_PROPOSITION_PROMPT_TEMPLATE

        model = self.llm.with_structured_output(PropositionExtraction, method="function_calling")
        inputs = [prompt_template.format(input=text) for text in texts]

        # batch() 返回 List[PropositionExtraction]，顺序与 inputs 一致
        responses = model.batch(inputs)

        return [response.propositions for response in responses]

    # ==================== 核心分块方法 ====================

    def chunk(
        self,
        propositions: List[str],
        find_chunk_prompt_template: Optional[ChatPromptTemplate] = None,
        new_chunk_summary_prompt_template: Optional[ChatPromptTemplate] = None,
        new_chunk_title_prompt_template: Optional[ChatPromptTemplate] = None,
        update_summary_prompt_template: Optional[ChatPromptTemplate] = None,
        update_title_prompt_template: Optional[ChatPromptTemplate] = None
    ) -> Dict[str, dict]:
        """
        将命题列表分块为语义分组的块。

        参数：
            propositions: 要分块的命题列表。
            find_chunk_prompt_template: 用于查找相关块的可选提示模板。
            new_chunk_summary_prompt_template: 用于生成新块摘要的可选提示模板。
            new_chunk_title_prompt_template: 用于生成新块标题的可选提示模板。
            update_summary_prompt_template: 用于更新块摘要的可选提示模板。
            update_title_prompt_template: 用于更新块标题的可选提示模板。

        返回：
            映射 chunk_id 到块数据的字典。
            每个块包含：chunk_id, title, summary, propositions, chunk_index
        """
        # 存储 prompt template 供内部方法使用
        self._find_prompt_template = find_chunk_prompt_template
        self._new_summary_prompt_template = new_chunk_summary_prompt_template
        self._new_title_prompt_template = new_chunk_title_prompt_template
        self._update_summary_prompt_template = update_summary_prompt_template
        self._update_title_prompt_template = update_title_prompt_template

        for proposition in propositions:
            if not self.chunks:
                self._create_new_chunk(proposition)
                continue

            chunk_id = self._find_relevant_chunk(proposition)

            if chunk_id:
                self._add_proposition_to_chunk(chunk_id, proposition)
            else:
                self._create_new_chunk(proposition)

        return self.chunks

    def chunk_batch(
        self,
        propositions_from_different_texts: List[List[str]],
        find_chunk_prompt_template: Optional[ChatPromptTemplate] = None,
        new_chunk_summary_prompt_template: Optional[ChatPromptTemplate] = None,
        new_chunk_title_prompt_template: Optional[ChatPromptTemplate] = None,
        update_summary_prompt_template: Optional[ChatPromptTemplate] = None,
        update_title_prompt_template: Optional[ChatPromptTemplate] = None,
        max_workers: Optional[int] = None,
        timeout_per_task: Optional[float] = None,
        on_error: str = "raise"
    ) -> List[Dict[str, dict]]:
        """
        批量分块来自多个独立文本的命题（同步版本）。

        每个文本的命题由独立的 AgenticChunk 实例处理，
        确保不同文本之间不会交叉污染。

        使用 ThreadPoolExecutor 实现并行处理，显著加快 LLM 调用速度。

        参数：
            propositions_from_different_texts: 命题列表的列表，每个内部列表包含来自单个文本的命题。
            find_chunk_prompt_template: 用于查找相关块的可选提示模板。
            new_chunk_summary_prompt_template: 用于生成新块摘要的可选提示模板。
            new_chunk_title_prompt_template: 用于生成新块标题的可选提示模板。
            update_summary_prompt_template: 用于更新块摘要的可选提示模板。
            update_title_prompt_template: 用于更新块标题的可选提示模板。
            max_workers: 最大并行工作线程数。默认为 None，将使用 5 和 CPU 核心数 * 2 的较小值。
            timeout_per_task: 每个任务的超时时间（秒）。默认为 None（无超时）。
            on_error: 错误处理策略。"raise" (默认) 抛出 PartialBatchError，"ignore" 返回部分结果。

        返回：
            块字典列表，每个输入文本一个。
            每个字典映射 chunk_id 到该文本的块数据。

            示例输出结构：
            [
                {  # 文本 1 的块
                    "abc123": {"chunk_id": "abc123", "title": "...", ...},
                    "def456": {...},
                },
                {  # 文本 2 的块
                    "ghi789": {...},
                },
            ]

        异常：
            PartialBatchError: 当部分任务失败且 on_error="raise" 时抛出。

        注意：
            使用 ThreadPoolExecutor 实现并行处理。
            输出顺序与输入顺序保持一致。
            空命题列表会返回空字典。

        示例用法：
            >>> # 访问第一个文本的块
            >>> text1_chunks = results[0]
            >>> # 访问特定块
            >>> chunk = results[0]["abc123"]
            >>> # 获取所有块作为列表
            >>> chunk_list = list(results[0].values())
        """
        if not propositions_from_different_texts:
            return []

        def process_single_propositions_group(args):
            """处理单个命题组的辅助函数。"""
            index, propositions = args

            # 空命题列表直接返回空字典
            if not propositions:
                return (index, {})

            fresh_chunker = AgenticChunk(llm=self.llm)
            chunks = fresh_chunker.chunk(
                propositions,
                find_chunk_prompt_template,
                new_chunk_summary_prompt_template,
                new_chunk_title_prompt_template,
                update_summary_prompt_template,
                update_title_prompt_template
            )
            return (index, chunks)

        # 创建带索引的任务列表
        tasks = list(enumerate(propositions_from_different_texts))

        # 更保守的默认线程数，避免触发 API rate limit
        if max_workers is None:
            import os
            max_workers = min(5, (os.cpu_count() or 1) * 2)

        results_map = {}
        failures = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # submit 返回 Future 对象，用于跟踪异步执行结果
            future_to_index = {
                executor.submit(process_single_propositions_group, task): task[0]
                for task in tasks
            }

            # 收集结果
            for future in as_completed(future_to_index, timeout=timeout_per_task):
                index = future_to_index[future]
                try:
                    _, chunks = future.result(timeout=timeout_per_task)
                    results_map[index] = chunks
                except FuturesTimeoutError as e:
                    logger.error(f"任务超时：索引 {index}")
                    failures.append((index, "任务超时", e))
                except Exception as e:
                    logger.error(f"任务失败：索引 {index}, 错误：{e}")
                    failures.append((index, str(e), e))

        # 处理失败情况
        if failures:
            if on_error == "raise":
                raise PartialBatchError(results_map, failures)
            # on_error == "ignore": 填充空结果并记录警告日志
            for idx, err_msg, _ in failures:
                logger.warning(f"任务失败被忽略：索引 {idx}, 错误：{err_msg}")
                results_map[idx] = {}

        # 按原始顺序返回结果
        return [results_map.get(i, {}) for i in range(len(propositions_from_different_texts))]


    # ==================== 内部方法 ====================

    def _find_relevant_chunk(
        self,
        proposition: str
    ) -> Optional[str]:
        """查找命题应该加入的现有块。"""
        chunk_outline = self._get_chunk_outline()
        prompt_template = self._find_prompt_template
        if prompt_template is None:
            prompt_template = DEFAULT_FIND_CHUNK_PROMPT_TEMPLATE

        response = self.chunk_assignment_model.invoke(prompt_template.format(chunk_outline=chunk_outline, proposition=proposition))

        if response.found and response.chunk_id:
            return response.chunk_id

        return None

    def _create_new_chunk(self, proposition: str):
        """为命题创建新块。"""
        new_chunk_id = str(uuid.uuid4())  # 完整 UUID，不截断

        if self.generate_metadata:
            # 生成摘要
            summary_prompt_template = self._new_summary_prompt_template
            if summary_prompt_template is None:
                summary_prompt_template = DEFAULT_NEW_CHUNK_SUMMARY_PROMPT_TEMPLATE
            summary_response = self.summary_model.invoke(summary_prompt_template.format(proposition=proposition))
            if summary_response is None:
                raise ValueError("LLM 返回的摘要响应为空，可能是内容触发了 API 审核策略")
            new_summary = summary_response.summary

            # 从摘要生成标题
            title_prompt_template = self._new_title_prompt_template
            if title_prompt_template is None:
                title_prompt_template = DEFAULT_NEW_CHUNK_TITLE_PROMPT_TEMPLATE
            title_response = self.title_model.invoke(title_prompt_template.format(summary=new_summary))
            if title_response is None:
                raise ValueError("LLM 返回的标题响应为空，可能是内容触发了 API 审核策略")
            new_title = title_response.title
        else:
            new_summary = proposition
            new_title = f"Chunk {len(self.chunks) + 1}"

        self.chunks[new_chunk_id] = {
            'chunk_id': new_chunk_id,
            'propositions': [proposition],
            'title': new_title,
            'summary': new_summary,
            'chunk_index': len(self.chunks)
        }

        if self.print_logging:
            print(f"创建新块 ({new_chunk_id}): {new_title}")

    def _add_proposition_to_chunk(
        self,
        chunk_id: str,
        proposition: str
    ):
        """将命题添加到现有块并更新元数据。"""
        self.chunks[chunk_id]['propositions'].append(proposition)

        if self.generate_metadata:
            chunk = self.chunks[chunk_id]

            # 更新摘要
            summary_prompt_template = self._update_summary_prompt_template
            if summary_prompt_template is None:
                summary_prompt_template = DEFAULT_UPDATE_SUMMARY_PROMPT_TEMPLATE
            summary_response = self.summary_model.invoke(summary_prompt_template.format(
                propositions="\n".join(chunk['propositions']),
                current_summary=chunk['summary']
            ))
            if summary_response is None:
                raise ValueError("LLM 返回的摘要响应为空，可能是内容触发了 API 审核策略")
            chunk['summary'] = summary_response.summary

            # 更新标题
            title_prompt_template = self._update_title_prompt_template
            if title_prompt_template is None:
                title_prompt_template = DEFAULT_UPDATE_TITLE_PROMPT_TEMPLATE
            title_response = self.title_model.invoke(title_prompt_template.format(
                propositions="\n".join(chunk['propositions']),
                current_summary=chunk['summary'],
                current_title=chunk['title']
            ))
            if title_response is None:
                raise ValueError("LLM 返回的标题响应为空，可能是内容触发了 API 审核策略")
            chunk['title'] = title_response.title

        if self.print_logging:
            print(f"添加到块 ({chunk_id}): {self.chunks[chunk_id]['title']}")

    def _get_chunk_outline(self) -> str:
        """生成块的字符串表示。"""
        outline = ""
        for chunk_id, chunk in self.chunks.items():
            outline += f"Chunk ID: {chunk_id}\n"
            outline += f"Chunk Name: {chunk['title']}\n"
            outline += f"Chunk Summary: {chunk['summary']}\n\n"

        return outline

    # ==================== 工具方法 ====================

    def get_chunks(self, get_type: str = 'dict'):
        """
        以指定格式检索块。

        参数：
            get_type: 'dict' 返回 {chunk_id: chunk_dict},
                      'list_of_strings' 返回 ["proposition string", ...],
                      'list_of_dicts' 返回 [chunk_dict, ...]

        返回：
            请求格式的块。

        异常：
            ValueError: 当 get_type 不是有效值时抛出。
        """
        if get_type == 'dict':
            return self.chunks
        if get_type == 'list_of_strings':
            return [" ".join(chunk['propositions']) for chunk in self.chunks.values()]
        if get_type == 'list_of_dicts':
            return list(self.chunks.values())

        raise ValueError(f"get_type 必须是 'dict'、'list_of_strings' 或 'list_of_dicts'，得到 '{get_type}'")

    def pretty_print_chunks(self):
        """以人类可读的格式打印块。"""
        print(f"\n你有 {len(self.chunks)} 个块\n")
        for chunk_id, chunk in self.chunks.items():
            print(f"第 {chunk['chunk_index']} 号块")
            print(f"Chunk ID: {chunk_id}")
            print(f"标题：{chunk['title']}")
            print(f"摘要：{chunk['summary']}")
            print(f"命题：")
            for prop in chunk['propositions']:
                print(f"    - {prop}")
            print("\n")

    def pretty_print_chunk_outline(self):
        """打印块大纲。"""
        print("块大纲\n")
        print(self._get_chunk_outline())

    def reset(self):
        """清除所有块并重新开始。"""
        self.chunks = {}
