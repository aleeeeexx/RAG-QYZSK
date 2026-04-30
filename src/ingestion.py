"""构建 FAISS 向量索引 + BM25 倒排索引,落盘到 data/。

落盘文件:
- data/chunked/chunks.json     全部 chunk(JSON list)
- data/chunked/pages.json      page 全文(用于父文档检索)
- data/vector_db/index.faiss   FAISS IndexFlatIP
- data/bm25_db/bm25.pkl        pickled (BM25Okapi, tokenized_corpus)
"""
from __future__ import annotations
import json
import pickle
from pathlib import Path

import faiss
import jieba
import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from src import config
from src.chunker import Chunk, chunk_documents
from src.llm_client import embed
from src.parsers.unified import parse_all

CHUNKS_PATH = config.CHUNKED_DIR / "chunks.json"
PAGES_PATH = config.CHUNKED_DIR / "pages.json"
FAISS_PATH = config.VECTOR_DB_DIR / "index.faiss"
BM25_PATH = config.BM25_DB_DIR / "bm25.pkl"


def tokenize_zh(text: str) -> list[str]:
    return [t for t in jieba.lcut(text) if t.strip()]


def build_indexes(force_parse: bool = False) -> None:
    print("[ingest] 解析文档 ...")
    docs = parse_all(force=force_parse)

    print("[ingest] 切分 ...")
    chunks, pages = chunk_documents(docs)
    print(f"[ingest]   {len(chunks)} chunks, {len(pages)} pages")

    print("[ingest] 持久化 chunks/pages ...")
    CHUNKS_PATH.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    PAGES_PATH.write_text(json.dumps(pages, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[ingest] 调用 DashScope embedding ...")
    texts = [c["text"] for c in chunks]
    vecs: list[list[float]] = []
    BATCH = 10
    for i in tqdm(range(0, len(texts), BATCH), desc="embed"):
        vecs.extend(embed(texts[i : i + BATCH]))
    arr = np.asarray(vecs, dtype="float32")
    faiss.normalize_L2(arr)  # 归一化 -> IndexFlatIP 等价于 cosine

    print(f"[ingest] 构建 FAISS 索引 (dim={arr.shape[1]}, n={arr.shape[0]}) ...")
    index = faiss.IndexFlatIP(arr.shape[1])
    index.add(arr)
    faiss.write_index(index, str(FAISS_PATH))

    print("[ingest] 构建 BM25 (jieba 分词) ...")
    tokenized = [tokenize_zh(t) for t in texts]
    bm25 = BM25Okapi(tokenized)
    with BM25_PATH.open("wb") as f:
        pickle.dump((bm25, tokenized), f)

    print("[ingest] 完成。")
    print(f"   chunks: {CHUNKS_PATH}")
    print(f"   pages:  {PAGES_PATH}")
    print(f"   faiss:  {FAISS_PATH}")
    print(f"   bm25:   {BM25_PATH}")


def load_indexes() -> tuple[list[Chunk], dict[str, str], faiss.Index, BM25Okapi, list[list[str]]]:
    chunks: list[Chunk] = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    pages: dict[str, str] = json.loads(PAGES_PATH.read_text(encoding="utf-8"))
    index = faiss.read_index(str(FAISS_PATH))
    with BM25_PATH.open("rb") as f:
        bm25, tokenized = pickle.load(f)
    return chunks, pages, index, bm25, tokenized
