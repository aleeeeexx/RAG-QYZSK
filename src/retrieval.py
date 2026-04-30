"""混合检索:向量 top-K + BM25 top-K → RRF 融合 → 父文档拼接。

依赖 ingestion.load_indexes() 加载索引,首次调用后在模块级缓存。
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from src import config
from src.chunker import Chunk
from src.ingestion import load_indexes, tokenize_zh
from src.llm_client import embed


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    rank_vec: int | None = None
    rank_bm25: int | None = None


@lru_cache(maxsize=1)
def _state() -> tuple[list[Chunk], dict[str, str], faiss.Index, BM25Okapi, list[list[str]]]:
    return load_indexes()


def vector_search(query: str, top_k: int) -> list[tuple[int, float]]:
    chunks, _, index, _, _ = _state()
    qv = np.asarray(embed([query]), dtype="float32")
    faiss.normalize_L2(qv)
    scores, ids = index.search(qv, top_k)
    return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]


def bm25_search(query: str, top_k: int) -> list[tuple[int, float]]:
    _, _, _, bm25, _ = _state()
    tokens = tokenize_zh(query)
    if not tokens:
        return []
    scores = bm25.get_scores(tokens)
    top_idx = np.argsort(-scores)[:top_k]
    return [(int(i), float(scores[i])) for i in top_idx if scores[i] > 0]


def rrf_fuse(
    vec_hits: list[tuple[int, float]],
    bm25_hits: list[tuple[int, float]],
    k: int = 60,
) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion: score = Σ 1/(k + rank_i),rank 从 1 开始。"""
    rrf: dict[int, float] = {}
    for rank, (cid, _) in enumerate(vec_hits, start=1):
        rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (k + rank)
    for rank, (cid, _) in enumerate(bm25_hits, start=1):
        rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(rrf.items(), key=lambda x: -x[1])


def hybrid_search(query: str, top_k: int = config.RRF_TOP_K) -> list[RetrievedChunk]:
    chunks, _, _, _, _ = _state()
    vec_hits = vector_search(query, config.VECTOR_TOP_K)
    bm25_hits = bm25_search(query, config.BM25_TOP_K)

    vec_rank = {cid: r for r, (cid, _) in enumerate(vec_hits, start=1)}
    bm25_rank = {cid: r for r, (cid, _) in enumerate(bm25_hits, start=1)}

    fused = rrf_fuse(vec_hits, bm25_hits)[:top_k]
    out: list[RetrievedChunk] = []
    for cid, score in fused:
        out.append(RetrievedChunk(
            chunk=chunks[cid],
            score=score,
            rank_vec=vec_rank.get(cid),
            rank_bm25=bm25_rank.get(cid),
        ))
    return out


def assemble_parent_context(top_chunks: list[RetrievedChunk]) -> tuple[str, list[dict]]:
    """对 top chunks 做父文档检索:把所属整页拉出来,去重,按文件分组拼接。

    返回:
        context_text: 拼好的上下文(给 LLM 的)
        sources:      [{file_name, page, page_label}] 用于答案 references
    """
    _, pages, _, _, _ = _state()

    seen: set[tuple[str, int]] = set()
    ordered: list[tuple[str, int, str, str]] = []
    for rc in top_chunks:
        key = (rc.chunk["doc_id"], rc.chunk["page"])
        if key in seen:
            continue
        seen.add(key)
        full = pages.get(f"{rc.chunk['doc_id']}::{rc.chunk['page']}", rc.chunk["text"])
        ordered.append((rc.chunk["file_name"], rc.chunk["page"], rc.chunk["page_label"], full))

    parts: list[str] = []
    sources: list[dict] = []
    for file_name, page, page_label, full in ordered:
        parts.append(f"【来源:《{file_name}》— {page_label}】\n{full}")
        sources.append({"file_name": file_name, "page": page, "page_label": page_label})
    return "\n\n---\n\n".join(parts), sources
