# Agentic RAG

三种多跳问答 RAG 架构的对比研究：Naive RAG、递归 RAG with Judge、Agentic RAG（规划-执行-反馈闭环）。基于 LangGraph 工作流编排 + Milvus 混合检索，在 HotpotQA、2WikiMultihopQA、MuSiQue 三个基准数据集上系统评估。

## 功能特性

### 1. Naive RAG（基线系统）
- 多查询检索：LLM 将原始查询重写为 4 个变体 + 原始查询 = 5 路并行检索
- 两种融合策略对比：
  - **Scheme A**：客户端 RRF 融合（RRF_K=60）
  - **Scheme B**：服务端 AnnSearchRequest 级融合（Milvus 内置 RRFRanker）
- 支持后续问题建议生成

### 2. RAG with Judge（递归探索）
- 全局 Judge 判断：每次检索后由 LLM 判断当前知识是否足以回答问题
- 不足时自动生成 follow-up queries，递归探索知识缺口
- SEARCH_PATH 数据结构记录完整探索树，可序列化供前端渲染
- 最大深度控制 + visited 集合防环

### 3. Agentic RAG v3（规划-执行-反馈闭环）
- 子问题分解 → 串行求解（已解决的子问题答案注入后续查询）
- 知识可见性：Planner 可看到已检索 chunk 标题，Synthesizer 可看到 chunk 摘要
- 跨轮次子问题去重 + 两阶段子问题评估（判断 + 提取）
- 连续 2 轮 stuck 时触发反思机制，诊断根因并生成 pivot 子问题

### 4. 共享检索层
- Milvus 混合检索：Dense（HNSW + COSINE）+ Sparse（BM25）
- 命题级索引 → chunk 级聚合 → 可选 Reranker 重排 → 截断返回
- 支持查询重写 + RRF 融合 + 优先级配额分配

### 5. 评估框架
- 统一评估引擎：断点续跑 + 批量并行 + 瞬态错误自动重试
- 原子写入 checkpoint，崩溃不丢数据
- 支持 4 种评估模式：LLM-Only / Naive RAG / RAG with Judge / Agentic RAG v3

## 技术栈

| 类别 | 技术 |
|------|------|
| 语言 | Python 3.14+ |
| 包管理 | uv |
| 工作流 | LangGraph + LangChain |
| 向量数据库 | Milvus（pymilvus + langchain-milvus） |
| 嵌入模型 | text-embedding-v4（1024 维，Dashscope API） |
| LLM | Qwen 系列（qwen-max / qwen-plus / qwen-turbo，Dashscope） |
| 前端 | Streamlit |
| 网页搜索 | DuckDuckGo (ddgs)、Firecrawl（可选） |

## 快速安装

### 前置条件
- Python 3.14+
- [uv](https://github.com/astral-sh/uv) 包管理器
- Milvus 服务（默认 `http://localhost:19530`）

### 安装依赖
```bash
uv sync
```

### 配置环境变量
```bash
cp .env.example .env
# 编辑 .env 填入 API 密钥
```

`.env` 需要配置：
```
BL_API_KEY=sk-your-key-here
BL_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
MILVUS_URI=http://localhost:19530
```

### 构建 Milvus 索引
```bash
# 构建全部数据集的索引
uv run python Index/milvus_cli.py build

# 仅构建测试用的小数据集（前 10 条）
uv run python Index/milvus_cli.py build --limit 10
```

## 快速开始

### 单条查询

**Naive RAG**：
```bash
uv run python -m naive_rag --query "Einstein 任教的大学是哪年创立的？" --scheme b --dataset hotpotqa
```

**RAG with Judge**（程序调用）：
```python
from rag_with_judge import rag_with_judge

search_path = {}
answer = rag_with_judge(
    query="你的问题",
    path=search_path,
    visited=set(),
    depth=0,
    max_depth=3,
    app=app,
    config=config,
)
```

**Agentic RAG v3**（程序调用）：
```python
from agentic_rag_v3.nodes import run_agentic_rag_v3

result = run_agentic_rag_v3(
    query="你的问题",
    app=app,
    config=config,
    max_rounds=5,
)
```

### 系统评估

```bash
# LLM 纯生成（无检索基线）
uv run python Eval/run_eval.py --mode llm-only --dataset hotpotqa

# Naive RAG
uv run python Eval/run_eval.py --mode naive-rag --dataset hotpotqa

# RAG with Judge
uv run python Eval/run_eval.py --mode rag-with-judge --dataset hotpotqa --max-depth 3

# Agentic RAG v3
uv run python Eval/run_eval.py --mode agentic-rag-v3 --dataset hotpotqa --max-rounds 5

# 计算汇总指标
uv run python Eval/compute_metrics.py --mode agentic-rag-v3 --dataset hotpotqa
```

所有评估模式均支持断点续跑（`--retry-failed`）和强制重跑（`--force`）。

### 前端界面

**在线 QA（基于网络搜索，无需 Milvus 索引）**：
```bash
uv run streamlit run web_qa.py
```

**Streamlit 多页应用（基于 Milvus 索引）**：
```bash
uv run streamlit run frontend/app.py
```

提供 6 个页面：在线查询、Naive RAG 演示、Judge RAG 演示、Agentic RAG 演示、结果对比、指纹热力图。

## 系统架构

```
query
  │
  ├── Naive RAG ───→ rewrite(5 queries) → fan-out retrieve → RRF fuse → generate
  │
  ├── RAG with Judge ───→ rewrite → retrieve → Judge? ──yes──→ generate
  │                                              │
  │                                             no
  │                                              ↓
  │                                       generate follow-ups → recurse
  │
  └── Agentic RAG v3 ───→ plan(sub-questions) → solve(each) → synthesize
                                                   │
                                          complete? ──no──→ reflect → pivot
                                                   │
                                                  yes
                                                   ↓
                                              generate answer
```

### 模块依赖关系

```
Index/ (离线索引构建)
  │
  ├── milvus_config.py   ← 共享配置（Milvus URI, 嵌入函数单例）
  ├── milvus_schema.py   ← Collection Schema 定义
  ├── milvus_ingest.py   ← 数据展平 → 嵌入计算 → Milvus 插入
  ├── semantic_chunker.py
  ├── agentic_chunk.py
  └── benchmark_chunker_v3.py
  │
  ↓
Retrieval/ (在线检索)
  │
  └── milvus_retriever.py  ← Hybrid Search → chunk 聚合 → Reranker → top-N
  │
  ↓
naive_rag/  rag_with_judge/  agentic_rag_v3/  ← 各 RAG 实现
  │
  ↓
Eval/ (统一评估)
  │
  └── run_eval.py  ← CLI 入口
```

## 评估设置

### 数据集

| 数据集 | 样本数 | Collection 名称 |
|--------|--------|-----------------|
| HotpotQA | 500 | `agentic_propositions_hotpotqa` |
| 2WikiMultihopQA | 500 | `agentic_propositions_2wikimultihopqa` |
| MuSiQue | 500 | `agentic_propositions_musique` |

原始 benchmark 源文件（`Data/2WikiMultihopQA_dev.json` 等）未包含在仓库中，可从官方渠道下载后用 `uv run python Data/create_benchmarks.py` 重新生成。

### 大文件说明

以下文件因体积过大未包含在仓库中，后续将通过外部链接提供：

| 文件 | 说明 | 大小 |
|------|------|------|
| 嵌入向量文件 | 三个数据集的命题嵌入（`*_embeddings.json`），用于 Milvus 索引构建 | ~5 GB |
| Agentic RAG v3 结果 | `Eval/agentic_rag_v3_data/result/musique.json`，MuSiQue 数据集上的完整评估结果 | ~134 MB |

下载链接：（待补充）

### 评估指标

- **准确率**：EM（Exact Match）、F1
- **检索质量**：Hit、MRR、Precision、Recall、Context Recall
- **效率**：检索轮次、总 chunk 数、搜索深度、每轮 chunk 数

## 项目结构

```
agentic-rag/
├── Eval/                  # 统一评估框架
│   ├── run_eval.py        # CLI 入口
│   ├── base.py            # BaseEvaluator（批处理 + 断点续跑）
│   ├── checkpoint.py      # 原子写入 Checkpoint 管理器
│   ├── metrics.py         # EM/F1 计算
│   ├── naive_rag.py       # Naive RAG 评估器
│   └── {mode}_data/       # 各评估模式的结果目录
│
├── Index/                 # 离线索引构建
│   ├── milvus_cli.py      # CLI：build/stats/drop
│   ├── milvus_config.py   # Milvus 连接 + 嵌入函数单例
│   ├── milvus_ingest.py   # 数据展平 → 嵌入 → upsert
│   ├── milvus_schema.py   # Collection Schema 定义
│   ├── semantic_chunker.py
│   ├── agentic_chunk.py
│   └── benchmark_chunker_v3.py
│
├── Retrieval/             # 在线检索层
│   ├── milvus_retriever.py  # Hybrid Search + RRF + Reranker
│   ├── web_retriever.py
│   └── query_cli.py
│
├── naive_rag/             # Naive RAG 实现
├── rag_with_judge/        # RAG with Judge 实现
├── agentic_rag/           # Agentic RAG v1（历史版本）
├── agentic_rag_v2/        # Agentic RAG v2（历史版本）
├── agentic_rag_v3/        # Agentic RAG v3（当前推荐版本）
│
├── frontend/              # Streamlit Web 界面
├── streaming/             # LangGraph 流式回调
├── paper/                 # 基线复现：IR-CoT、Iter-RetGen、GenGround
│
├── Data/benchmark/        # 500 样本基准数据集 + chunked 结果
├── pyproject.toml         # 项目配置与依赖
├── uv.lock                # 依赖锁定文件
└── .env.example           # 环境变量模板
```

## 开源协议

MIT License
