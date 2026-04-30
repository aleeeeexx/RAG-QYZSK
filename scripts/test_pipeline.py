"""P5 端到端验证:跑 5 个真实问题,打印 Qwen 回答与引用。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import answer

QUESTIONS = [
    "差旅费报销的标准是什么?",
    "员工试用期多久?",
    "怎么在阿里商旅订机票?",
    "个税起征点是多少?",
    "财务单据怎么填写?",
]


def main() -> None:
    for q in QUESTIONS:
        print("=" * 70)
        print(f"Q: {q}")
        result = answer(q)
        print(f"\n[推理摘要] {result['reasoning_summary']}")
        print(f"\n[引用]")
        for r in result["references"]:
            print(f"  - 《{r.get('file_name')}》{r.get('page_label')}")
        print(f"\n[最终答案]\n{result['final_answer']}")
        print()


if __name__ == "__main__":
    main()
