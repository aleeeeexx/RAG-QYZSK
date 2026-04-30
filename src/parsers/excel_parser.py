"""Excel 解析:每个 sheet 转 markdown 表,作为一"页"。"""
from __future__ import annotations
import hashlib
from pathlib import Path
import pandas as pd

from src.parsers.schema import ParsedDoc, PageBlock


def parse_excel(path: Path) -> ParsedDoc:
    sheets: dict[str, pd.DataFrame] = pd.read_excel(path, sheet_name=None, header=None)

    pages: list[PageBlock] = []
    for idx, (sheet_name, df) in enumerate(sheets.items(), start=1):
        df = df.dropna(how="all").dropna(how="all", axis=1).fillna("")
        if df.empty:
            continue
        markdown = df.to_markdown(index=False, headers=[f"列{i+1}" for i in range(df.shape[1])])
        pages.append({
            "page": idx,
            "page_label": f"工作表「{sheet_name}」",
            "text": f"# {sheet_name}\n\n{markdown}",
            "tables": [{"markdown": markdown}],
        })

    return {
        "metainfo": {
            "file_name": path.name,
            "file_type": "xlsx",
            "doc_id": _file_md5(path),
        },
        "content": pages,
    }


def _file_md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
