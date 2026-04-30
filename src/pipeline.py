"""端到端编排:question → 检索 → 重排 → 父文档 → Qwen 推理 → 结构化答案。"""
from __future__ import annotations
import json
from typing import TypedDict
from json_repair import repair_json

from src import config
from src.llm_client import chat
from src.prompts import ANSWER_SYSTEM, ANSWER_USER
from src.reranker import llm_rerank
from src.retrieval import RetrievedChunk, assemble_parent_context, hybrid_search


class AnswerPayload(TypedDict):
    question: str
    step_by_step_analysis: str
    reasoning_summary: str
    references: list[dict]
    final_answer: str
    sources_used: list[dict]   # 实际拼入 context 的页(供前端渲染来源列表)


def answer(question: str) -> AnswerPayload:
    hits = hybrid_search(question)
    top = llm_rerank(question, hits)
    context, sources = assemble_parent_context(top)

    prompt = ANSWER_USER.format(context=context, question=question)
    raw = chat(prompt, system=ANSWER_SYSTEM, model=config.LLM_MODEL, temperature=config.LLM_TEMPERATURE)

    parsed = _parse_answer_json(raw)
    return {
        "question": question,
        "step_by_step_analysis": parsed.get("step_by_step_analysis", ""),
        "reasoning_summary": parsed.get("reasoning_summary", ""),
        "references": parsed.get("references", []),
        "final_answer": parsed.get("final_answer", ""),
        "sources_used": sources,
    }


def _parse_answer_json(raw: str) -> dict:
    try:
        # 先尝试 fence 内 JSON
        if "```" in raw:
            parts = raw.split("```")
            for p in parts:
                p = p.strip()
                if p.startswith("json"):
                    p = p[4:].strip()
                if p.startswith("{"):
                    return json.loads(repair_json(p))
        return json.loads(repair_json(raw))
    except Exception as e:
        print(f"[pipeline] JSON 解析失败: {e}")
        return {
            "step_by_step_analysis": "(解析失败)",
            "reasoning_summary": "(解析失败)",
            "references": [],
            "final_answer": raw,
        }
