"""Agentic RAG v3 — 子问题解决链

核心理念：Plan 生成的 targets 是需要逐一解决的子问题，而非原始问题的"重写"变体。
通过逐一解决每个子问题（每个子问题自动重写→多路检索→RRF 融合），构建推理链来回答原始多跳问题。

控制流：
START → plan → solve_sub_questions → synthesize
  ├─ complete → generate_answer → END
  ├─ incomplete → plan (补充子问题)
  ├─ stuck × 1 → plan (带小提醒)
  └─ stuck × 2+ → reflect → solve_sub_questions (换角度直接执行)
"""
