"""统一解析输出的 schema:
{
  "metainfo": {"file_name": str, "file_type": "pdf"|"xlsx"|"pptx", "doc_id": str},
  "content": [
    {"page": int, "page_label": str, "text": str, "tables": [{"markdown": str}]}
  ]
}

`page` 是从 1 开始的连续整数(用于检索),`page_label` 是给用户看的名称(如 "Sheet1"、"幻灯片 3")。
"""
from __future__ import annotations
from typing import TypedDict


class TableBlock(TypedDict):
    markdown: str


class PageBlock(TypedDict):
    page: int
    page_label: str
    text: str
    tables: list[TableBlock]


class MetaInfo(TypedDict):
    file_name: str
    file_type: str
    doc_id: str


class ParsedDoc(TypedDict):
    metainfo: MetaInfo
    content: list[PageBlock]
