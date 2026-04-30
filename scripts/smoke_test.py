"""P1 冒烟测试:验证 DashScope chat + embedding 能跑通。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm_client import chat, embed
from src import config


def main() -> None:
    assert config.DASHSCOPE_API_KEY, "DASHSCOPE_API_KEY 未配置"

    print("=== chat ===")
    answer = chat("用一句话介绍 RAG 是什么。")
    print(answer)

    print("\n=== embed ===")
    vecs = embed(["企业制度问答", "差旅费报销标准"])
    print(f"got {len(vecs)} vectors, dim={len(vecs[0])}")
    assert len(vecs) == 2
    assert len(vecs[0]) == config.EMBED_DIM, f"维度不匹配:{len(vecs[0])} != {config.EMBED_DIM}"
    print("OK")


if __name__ == "__main__":
    main()
