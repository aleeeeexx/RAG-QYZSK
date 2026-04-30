"""把 ParsedDoc 切成 Chunk。

策略:
- 文本块: RecursiveCharacterTextSplitter,按"\\n\\n / \\n / 。/ ; / , / 空白"优先级递归切
- 表格块: 整张表作为一个 chunk,不切分(保留语义)
- 每个 chunk 携带 doc_id / file_name / page / page_label,便于召回后做父文档检索与溯源

输出:
- chunks: 全局列表,每个元素是 dict(便于 JSON 持久化)
- pages: dict{(doc_id, page) -> full_page_text} 用于父文档检索
"""
from __future__ import annotations
from typing import TypedDict
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src import config
from src.parsers.schema import ParsedDoc

_ENC = tiktoken.get_encoding("cl100k_base")


def _tok_len(s: str) -> int:
    return len(_ENC.encode(s))


_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=config.CHUNK_SIZE_TOKENS,
    chunk_overlap=config.CHUNK_OVERLAP_TOKENS,
    length_function=_tok_len,
    separators=["\n\n", "\n", "。", "；", ";", ",", "、", " ", ""],
)


class Chunk(TypedDict):
    id: int
    doc_id: str
    file_name: str
    file_type: str
    page: int
    page_label: str
    text: str
    is_table: bool


def chunk_documents(docs: list[ParsedDoc]) -> tuple[list[Chunk], dict[str, str]]:
    chunks: list[Chunk] = []
    pages: dict[str, str] = {}
    next_id = 0

    for doc in docs:
        meta = doc["metainfo"]
        doc_id = meta["doc_id"]
        file_name = meta["file_name"]
        file_type = meta["file_type"]

        for page_block in doc["content"]:
            page = page_block["page"]
            page_label = page_block["page_label"]
            page_text = page_block["text"]
            pages[f"{doc_id}::{page}"] = page_text

            # 表格 -> 独立 chunk
            for tb in page_block.get("tables", []):
                md = tb.get("markdown", "").strip()
                if md:
                    chunks.append({
                        "id": next_id,
                        "doc_id": doc_id,
                        "file_name": file_name,
                        "file_type": file_type,
                        "page": page,
                        "page_label": page_label,
                        "text": md,
                        "is_table": True,
                    })
                    next_id += 1

            # 文本 -> 递归切分
            # 注意:为避免表格被重复(text 已包含表格 md),只切非表格部分
            non_table_text = page_text
            for tb in page_block.get("tables", []):
                non_table_text = non_table_text.replace(tb.get("markdown", ""), "")
            non_table_text = non_table_text.strip()

            if non_table_text:
                for piece in _SPLITTER.split_text(non_table_text):
                    piece = piece.strip()
                    if not piece:
                        continue
                    chunks.append({
                        "id": next_id,
                        "doc_id": doc_id,
                        "file_name": file_name,
                        "file_type": file_type,
                        "page": page,
                        "page_label": page_label,
                        "text": piece,
                        "is_table": False,
                    })
                    next_id += 1

    return chunks, pages
