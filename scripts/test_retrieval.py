"""P4 验证脚本:跑 5 个真实问题,打印混合检索 + 重排结果,人工判断相关性。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval import hybrid_search, assemble_parent_context
from src.reranker import llm_rerank

QUESTIONS = [
    "差旅费报销的标准是什么?",
    "员工试用期多久?",
    "怎么在阿里商旅订机票?",
    "个税起征点是多少?",
    "财务单据怎么填写?",
]


def main() -> None:
    for q in QUESTIONS:
        print("=" * 60)
        print(f"Q: {q}")
        hits = hybrid_search(q)
        print(f"\n[hybrid 召回 top-{len(hits)}]")
        for rc in hits[:8]:
            print(f"  RRF={rc.score:.4f}  vec_rank={rc.rank_vec}  bm25_rank={rc.rank_bm25}  "
                  f"《{rc.chunk['file_name']}》{rc.chunk['page_label']}  "
                  f"{rc.chunk['text'][:60].replace(chr(10),' ')}...")

        top = llm_rerank(q, hits)
        print(f"\n[LLM 重排 top-{len(top)}]")
        for rc in top:
            print(f"  score={rc.score:.3f}  《{rc.chunk['file_name']}》{rc.chunk['page_label']}")

        ctx, sources = assemble_parent_context(top)
        print(f"\n[父文档拼接] {len(sources)} 个来源页, context 长度 = {len(ctx)} 字")
        print()


if __name__ == "__main__":
    main()
