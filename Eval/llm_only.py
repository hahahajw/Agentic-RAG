"""LLM-Only 评估器 — 基线：不调用检索，直接让 LLM 回答问题。"""

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from Eval.base import BaseEvaluator, NormalizedQuestion
from Eval.prompts import LLM_ONLY_SYSTEM_PROMPT, LLM_ONLY_USER_PROMPT

logger = logging.getLogger(__name__)


class LLMOnlyEvaluator(BaseEvaluator):
    """仅使用 LLM 回答问题，不进行任何检索。

    作为对比 RAG 系统的基线。
    """

    def __init__(self, **kwargs):
        super().__init__(eval_mode="llm_only", **kwargs)

    def evaluate_single(self, question: NormalizedQuestion) -> dict:
        """构造 prompt → 调用 llm → 返回预测答案。"""
        system_msg = SystemMessage(content=LLM_ONLY_SYSTEM_PROMPT)
        human_msg = HumanMessage(
            content=LLM_ONLY_USER_PROMPT.format(question=question.question)
        )
        response = self.llm.invoke([system_msg, human_msg])

        logger.debug("LLM-only 回答 q%d: %s", question.index, response.content[:100])
        return {"prediction": response.content.strip(), "error": None, "chunks": []}
