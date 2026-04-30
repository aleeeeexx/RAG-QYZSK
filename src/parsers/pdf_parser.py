"""MinerU 在线 API 解析 PDF。

流程:
1. POST /api/v4/file-urls/batch  申请预签名上传 URL
2. PUT 上传 PDF(无 Authorization 头)
3. GET /api/v4/extract-results/batch/{batch_id} 轮询直到 done
4. 下载 full_zip_url,解压
5. 解析 *_content_list.json -> 统一 schema

解析结果会缓存在 data/parsed/_mineru_cache/{md5}/,二次跑跳过 API 调用。
"""
from __future__ import annotations
import hashlib
import io
import json
import time
import zipfile
from pathlib import Path

import requests

from src import config
from src.parsers.schema import ParsedDoc, PageBlock

API_BASE = "https://mineru.net/api/v4"
CACHE_DIR = config.PARSED_DIR / "_mineru_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def parse_pdf(path: Path, language: str = "ch", enable_table: bool = True) -> ParsedDoc:
    md5 = _file_md5(path)
    cache_subdir = CACHE_DIR / md5
    content_list = _load_cached_content_list(cache_subdir)

    if content_list is None:
        print(f"[mineru] 提交解析任务: {path.name}")
        zip_bytes = _submit_and_download(path, language=language, enable_table=enable_table)
        cache_subdir.mkdir(parents=True, exist_ok=True)
        _extract_zip(zip_bytes, cache_subdir)
        content_list = _load_cached_content_list(cache_subdir)
        if content_list is None:
            raise RuntimeError(f"MinerU 返回 zip 中未找到 *_content_list.json: {cache_subdir}")

    pages = _content_list_to_pages(content_list)
    return {
        "metainfo": {"file_name": path.name, "file_type": "pdf", "doc_id": md5},
        "content": pages,
    }


def _submit_and_download(
    path: Path, language: str, enable_table: bool, poll_interval: int = 5, timeout: int = 600
) -> bytes:
    headers = {"Authorization": f"Bearer {config.MINERU_API_TOKEN}"}

    # 1. 申请上传 URL
    resp = requests.post(
        f"{API_BASE}/file-urls/batch",
        headers={**headers, "Content-Type": "application/json"},
        json={
            "enable_formula": False,
            "language": language,
            "enable_table": enable_table,
            "files": [{"name": path.name, "is_ocr": True, "data_id": path.stem}],
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("code") not in (0, 200):
        raise RuntimeError(f"MinerU 申请上传 URL 失败: {data}")
    payload = data["data"]
    batch_id = payload["batch_id"]
    upload_url = payload["file_urls"][0]
    print(f"[mineru] batch_id={batch_id}")

    # 2. 上传 PDF(注意:此 PUT 不能带 Authorization 头)
    with path.open("rb") as f:
        put_resp = requests.put(upload_url, data=f.read(), timeout=120)
    put_resp.raise_for_status()
    print(f"[mineru] 上传完成,等待解析...")

    # 3. 轮询
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(poll_interval)
        r = requests.get(
            f"{API_BASE}/extract-results/batch/{batch_id}",
            headers=headers,
            timeout=30,
        )
        r.raise_for_status()
        body = r.json()
        if body.get("code") not in (0, 200):
            raise RuntimeError(f"MinerU 轮询失败: {body}")
        results = body["data"]["extract_result"]
        item = results[0]
        state = item.get("state")
        print(f"[mineru]   state={state}")
        if state == "done":
            zip_url = item["full_zip_url"]
            print(f"[mineru] 下载结果 zip ...")
            zr = requests.get(zip_url, timeout=120)
            zr.raise_for_status()
            return zr.content
        if state == "failed":
            raise RuntimeError(f"MinerU 解析失败: {item.get('err_msg')}")
    raise TimeoutError(f"MinerU 解析超时({timeout}s)")


def _extract_zip(zip_bytes: bytes, dest: Path) -> None:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        zf.extractall(dest)


def _load_cached_content_list(cache_subdir: Path) -> list[dict] | None:
    if not cache_subdir.exists():
        return None
    for jf in cache_subdir.rglob("*_content_list.json"):
        return json.loads(jf.read_text(encoding="utf-8"))
    return None


def _content_list_to_pages(content_list: list[dict]) -> list[PageBlock]:
    """把 MinerU 输出的扁平 block 列表按 page_idx 聚合成页。"""
    by_page: dict[int, dict] = {}
    for block in content_list:
        page_idx = block.get("page_idx", 0)
        slot = by_page.setdefault(page_idx, {"text_parts": [], "tables": []})
        btype = block.get("type")
        if btype == "text":
            t = (block.get("text") or "").strip()
            if t:
                slot["text_parts"].append(t)
        elif btype == "table":
            html_or_md = block.get("table_body") or block.get("table_caption") or ""
            md = _table_to_markdown(html_or_md)
            if md:
                slot["tables"].append({"markdown": md})
                slot["text_parts"].append(md)
        elif btype == "equation":
            t = (block.get("text") or "").strip()
            if t:
                slot["text_parts"].append(t)
        # image 类型暂不处理(不做多模态)

    pages: list[PageBlock] = []
    for page_idx in sorted(by_page.keys()):
        slot = by_page[page_idx]
        text = "\n\n".join(slot["text_parts"]).strip()
        if not text:
            continue
        pages.append({
            "page": page_idx + 1,  # 0-based -> 1-based
            "page_label": f"第 {page_idx + 1} 页",
            "text": text,
            "tables": slot["tables"],
        })
    return pages


def _table_to_markdown(table_body: str | list) -> str:
    """MinerU 的 table_body 通常是 HTML 字符串。简单转 markdown(保留原 HTML 也可以,LLM 都能读)。"""
    if isinstance(table_body, list):
        return "\n".join(str(r) for r in table_body)
    return str(table_body).strip()


def _file_md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
