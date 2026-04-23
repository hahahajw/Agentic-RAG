"""EM/F1 指标计算 — 纯函数，仅依赖 NLTK + 标准库。

从 experimental/RAG-with-Judge/eval/metrics/em_f1.py 提取的核心函数，
去除了 evaluation() CLI 封装（依赖 ujson/loguru/tqdm，Eval 模块不需要）。
"""

import re
import string
from collections import Counter
from typing import List

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def normalize_answer(s) -> str:
    """归一化答案文本：去括号、小写、去标点、去冠词、修复空白。"""

    def remove_brackets_content(text):
        return re.sub(r'\[.*?\]', '', text)

    def lower(text):
        return text.lower()

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    return white_space_fix(remove_articles(remove_punc(lower(remove_brackets_content(s)))))


def get_tokens(normalized_s: str) -> List[str]:
    """对归一化后的文本分词，移除停用词（保留 'no'）。"""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
        nltk.download('punkt_tab')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

    tokens = word_tokenize(normalized_s)
    stop_words = set(stopwords.words('english'))
    stop_words.remove('no')  # no 不属于停用词
    tokens = [token for token in tokens if token not in stop_words]
    return tokens


def exact_match_score(prediction, ground_truth) -> float:
    """Exact Match：归一化后完全匹配返回 1.0，否则 0.0。"""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction, ground_truth) -> tuple[float, float, float]:
    """Token 级别 F1。返回 (f1, precision, recall)，完全不匹配时返回 (0, 0, 0)。"""
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = get_tokens(normalized_prediction)
    ground_truth_tokens = get_tokens(normalized_ground_truth)
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


# ─── 检索质量指标 ──────────────────────────────────────────────

def extract_supporting_titles(question_raw: dict, dataset_type: str) -> set[str]:
    """从 benchmark 原始问题中提取支撑段落标题集合。

    HotpotQA / 2Wiki: supporting_facts = [["title", sent_idx], ...]
    MuSiQue: question_decomposition[i]["paragraph_support_idx"] → paragraphs 索引
    """
    titles: set[str] = set()

    if dataset_type in ("hotpotqa", "2wikimultihopqa"):
        supporting_facts = question_raw.get("supporting_facts", [])
        for item in supporting_facts:
            if isinstance(item, (list, tuple)) and len(item) >= 1:
                titles.add(item[0])
    elif dataset_type == "musique":
        paragraphs = question_raw.get("paragraphs", [])
        decomposition = question_raw.get("question_decomposition", [])
        for item in decomposition:
            idx = item.get("paragraph_support_idx")
            if idx is not None and 0 <= idx < len(paragraphs):
                title = paragraphs[idx].get("title", "")
                if title:
                    titles.add(title)

    return titles


def compute_context_recall(retrieved_chunk_titles: list[str], supporting_titles: set[str], top_k: int | None = None) -> float:
    """检索到的 chunk 中，有多少 distinct title 命中支撑段落。

    context_recall = |retrieved_titles ∩ supporting_titles| / |supporting_titles|

    Args:
        top_k: 只考虑前 K 个检索结果。None 表示使用全部结果（Recall@All）。
    """
    if top_k is not None:
        retrieved_chunk_titles = retrieved_chunk_titles[:top_k]
    if not supporting_titles:
        return 0.0
    matched = set(retrieved_chunk_titles) & supporting_titles
    return len(matched) / len(supporting_titles)


def compute_hit(retrieved_chunk_titles: list[str], supporting_titles: set[str], top_k: int | None = None) -> int:
    """是否至少有一个 chunk 的 title 命中支撑段落。返回 1 或 0。

    Args:
        top_k: 只考虑前 K 个检索结果。None 表示使用全部结果。
    """
    if top_k is not None:
        retrieved_chunk_titles = retrieved_chunk_titles[:top_k]
    return 1 if (set(retrieved_chunk_titles) & supporting_titles) else 0


def compute_mrr(retrieved_chunk_titles: list[str], supporting_titles: set[str], top_k: int | None = None) -> float:
    """第一个命中支撑段落的 chunk 排名的倒数。未命中返回 0。

    Args:
        top_k: 只考虑前 K 个检索结果。None 表示使用全部结果。
    """
    if top_k is not None:
        retrieved_chunk_titles = retrieved_chunk_titles[:top_k]
    for rank, title in enumerate(retrieved_chunk_titles, 1):
        if title in supporting_titles:
            return 1.0 / rank
    return 0.0


def compute_retrieval_precision(retrieved_chunk_titles: list[str], supporting_titles: set[str], top_k: int | None = None) -> float:
    """检索到的 chunk 中，命中支撑段落的比例。

    Args:
        top_k: 只考虑前 K 个检索结果。None 表示使用全部结果。
    """
    if top_k is not None:
        retrieved_chunk_titles = retrieved_chunk_titles[:top_k]
    if not retrieved_chunk_titles:
        return 0.0
    hit_count = sum(1 for t in retrieved_chunk_titles if t in supporting_titles)
    return hit_count / len(retrieved_chunk_titles)
