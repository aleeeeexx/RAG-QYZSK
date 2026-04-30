"""LLM 重排:让 Qwen 给候选 chunk 打分,选 top-K。"""
from __future__ import annotations
import json
from json_repair import repair_json

from src import config
from src.llm_client import chat
from src.retrieval import RetrievedChunk

RERANK_PROMPT = """你是一个相关性评估器。给定一个用户问题与若干候选文档片段,请为每个片段给出 0~1 之间的相关性分数(1 表示完全相关,0 表示完全无关)。

【问题】
{query}

【候选片段】
{candidates}

请只输出 JSON,不要额外解释。格式:
{{"scores": [{{"id": 0, "score": 0.92}}, {{"id": 1, "score": 0.10}}, ...]}}
必须为每个片段输出一项。"""


def _format_candidates(chunks: list[RetrievedChunk], max_chars_per_chunk: int = 600) -> str:
    lines = []
    for i, rc in enumerate(chunks):
        snippet = rc.chunk["text"].strip()
        if len(snippet) > max_chars_per_chunk:
            snippet = snippet[:max_chars_per_chunk] + "..."
        lines.append(f"[{i}] 来源:《{rc.chunk['file_name']}》{rc.chunk['page_label']}\n{snippet}")
    return "\n\n".join(lines)


def llm_rerank(query: str, chunks: list[RetrievedChunk], top_k: int = config.RERANK_TOP_K) -> list[RetrievedChunk]:
    if not chunks:
        return []

    prompt = RERANK_PROMPT.format(query=query, candidates=_format_candidates(chunks))
    raw = chat(prompt, model=config.RERANK_MODEL, temperature=0.0)
    try:
        repaired = repair_json(raw)
        data = json.loads(repaired)
        scored: dict[int, float] = {int(item["id"]): float(item["score"]) for item in data.get("scores", [])}
    except Exception as e:
        print(f"[reranker] 解析失败,降级使用原始 RRF 排序: {e}")
        return chunks[:top_k]

    # 给每个 chunk 注入新分数,缺省给原 RRF 分数 *0.5(排到末尾)
    rescored: list[RetrievedChunk] = []
    for i, rc in enumerate(chunks):
        new_score = scored.get(i, rc.score * 0.5)
        rescored.append(RetrievedChunk(
            chunk=rc.chunk,
            score=new_score,
            rank_vec=rc.rank_vec,
            rank_bm25=rc.rank_bm25,
        ))
    rescored.sort(key=lambda r: -r.score)
    return rescored[:top_k]
