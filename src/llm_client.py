"""DashScope (Qwen) 调用封装:对话补全 + 文本嵌入。"""
from typing import Iterable
import dashscope
from dashscope import Generation, TextEmbedding

from src import config

dashscope.api_key = config.DASHSCOPE_API_KEY


def chat(
    prompt: str,
    system: str | None = None,
    model: str | None = None,
    temperature: float = config.LLM_TEMPERATURE,
    max_tokens: int = config.LLM_MAX_TOKENS,
) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    resp = Generation.call(
        model=model or config.LLM_MODEL,
        messages=messages,
        result_format="message",
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"DashScope chat failed: {resp.code} {resp.message}")
    return resp.output.choices[0].message.content


def embed(texts: str | list[str]) -> list[list[float]]:
    """一次最多 25 条,超过自动分批。返回与输入对齐的向量列表。"""
    if isinstance(texts, str):
        texts = [texts]

    out: list[list[float]] = []
    BATCH = 10
    for i in range(0, len(texts), BATCH):
        batch = texts[i : i + BATCH]
        resp = TextEmbedding.call(
            model=config.EMBED_MODEL,
            input=batch,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"DashScope embed failed: {resp.code} {resp.message}")
        for item in resp.output["embeddings"]:
            out.append(item["embedding"])
    return out
