"""按文件后缀分发到对应 parser,统一输出 ParsedDoc,并支持磁盘缓存。"""
from __future__ import annotations
import json
from pathlib import Path

from src import config
from src.parsers.schema import ParsedDoc
from src.parsers.pdf_parser import parse_pdf
from src.parsers.excel_parser import parse_excel
from src.parsers.pptx_parser import parse_pptx

SUPPORTED = {".pdf", ".xlsx", ".xls", ".pptx"}


def parse_one(path: Path, force: bool = False) -> ParsedDoc:
    cache_path = config.PARSED_DIR / f"{path.stem}.json"
    if cache_path.exists() and not force:
        return json.loads(cache_path.read_text(encoding="utf-8"))

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        doc = parse_pdf(path)
    elif suffix in (".xlsx", ".xls"):
        doc = parse_excel(path)
    elif suffix == ".pptx":
        doc = parse_pptx(path)
    else:
        raise ValueError(f"不支持的文件类型: {suffix}")

    cache_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return doc


def iter_input_files(raw_dir: Path = config.RAW_DIR) -> list[Path]:
    files: list[Path] = []
    for p in raw_dir.iterdir():
        if not p.is_file():
            continue
        if p.name.endswith(":Zone.Identifier"):
            continue
        if p.suffix.lower() in SUPPORTED:
            files.append(p)
    return sorted(files)


def parse_all(force: bool = False) -> list[ParsedDoc]:
    out: list[ParsedDoc] = []
    for f in iter_input_files():
        print(f"[parse] {f.name}")
        try:
            doc = parse_one(f, force=force)
            n_pages = len(doc["content"])
            print(f"        -> {n_pages} 页")
            out.append(doc)
        except Exception as e:
            print(f"        ✗ 失败: {e}")
    return out
