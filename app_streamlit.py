"""企业知识库问答 Streamlit 前端。

启动:
    streamlit run app_streamlit.py
"""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path

import streamlit as st

from src import config
from src.ingestion import load_indexes
from src.pipeline import answer

st.set_page_config(page_title="企业知识库问答", layout="wide", page_icon="📚")


@st.cache_resource(show_spinner="加载索引...")
def _load_state():
    return load_indexes()


def _strip_quotes(s: str) -> str:
    if not s:
        return ""
    return s.strip().strip("《》").strip()


# ---------- 顶部标题 ----------
st.markdown(
    """
    <div style="background: linear-gradient(90deg, #1e3a8a 0%, #2563eb 100%);
                padding: 22px 28px; border-radius: 12px; margin-bottom: 16px;">
      <h2 style="color: white; margin: 0;">企业知识库问答</h2>
      <div style="color: #cbd5e1; font-size: 14px; margin-top: 4px;">
        混合检索(向量+BM25) · LLM 重排 · 父文档检索 · Qwen 推理 · 答案溯源
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- 侧边栏 ----------
chunks, pages, faiss_index, _bm25, _ = _load_state()
file_counts = Counter((c["file_name"], c["file_type"]) for c in chunks)

with st.sidebar:
    st.subheader("🔍 提问")
    user_question = st.text_area(
        "请输入问题",
        value="差旅费报销的标准是什么?",
        height=110,
        label_visibility="collapsed",
    )
    submit = st.button("生成答案", use_container_width=True, type="primary")

    st.divider()
    st.subheader("📂 已索引文档")
    st.caption(f"{len(file_counts)} 份文档 · {len(pages)} 页 · {len(chunks)} chunk")
    for (fname, ftype), n in sorted(file_counts.items(), key=lambda kv: -kv[1]):
        emoji = {"pdf": "📄", "xlsx": "📊", "pptx": "📽"}.get(ftype, "📁")
        st.markdown(f"{emoji} `{fname}` · {n} chunk")

    st.divider()
    with st.expander("⚙️ 当前配置"):
        st.markdown(
            f"""
            - LLM: `{config.LLM_MODEL}`
            - Reranker: `{config.RERANK_MODEL}`
            - Embedding: `{config.EMBED_MODEL}` ({config.EMBED_DIM} 维)
            - Vector top-K: {config.VECTOR_TOP_K}
            - BM25 top-K: {config.BM25_TOP_K}
            - RRF top-K: {config.RRF_TOP_K}
            - Rerank top-K: {config.RERANK_TOP_K}
            """
        )

# ---------- 主区 ----------
if submit and user_question.strip():
    q = user_question.strip()
    with st.spinner("检索 → 重排 → 生成 ..."):
        result = answer(q)

    st.markdown(f"### 问题")
    st.info(q)

    final = result.get("final_answer", "")
    st.markdown("### 最终答案")
    st.markdown(
        f"<div style='background:#f0f9ff;padding:18px;border-left:4px solid #2563eb;"
        f"border-radius:6px;font-size:15px;line-height:1.7;'>{final}</div>",
        unsafe_allow_html=True,
    )

    refs = result.get("references", []) or []
    sources_used = result.get("sources_used", []) or []

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("### 📎 引用来源(LLM 标注)")
        if refs:
            for r in refs:
                fname = _strip_quotes(r.get("file_name", ""))
                page_label = r.get("page_label", "")
                st.markdown(f"- 《{fname}》 — {page_label}")
        else:
            st.caption("(无引用)")
    with col_r:
        st.markdown("### 📑 实际拼入上下文的页")
        for s in sources_used:
            st.markdown(f"- 《{s['file_name']}》 — {s['page_label']}")

    with st.expander("🧠 分步推理"):
        st.markdown(result.get("step_by_step_analysis", "") or "(空)")
    with st.expander("📝 推理摘要"):
        st.markdown(result.get("reasoning_summary", "") or "(空)")

    with st.expander("📖 查看引用页原文"):
        for s in sources_used:
            st.markdown(f"**《{s['file_name']}》 — {s['page_label']}**")
            full_text = pages.get(f"{s.get('doc_id', '')}::{s['page']}", "")
            if not full_text:
                # sources_used 不一定带 doc_id,这里通过 file_name+page 反查
                for c in chunks:
                    if c["file_name"] == s["file_name"] and c["page"] == s["page"]:
                        full_text = pages.get(f"{c['doc_id']}::{c['page']}", c["text"])
                        break
            st.code(full_text or "(原文丢失)", language="markdown")

    with st.expander("🔧 调试 · 完整返回 JSON"):
        st.code(json.dumps(result, ensure_ascii=False, indent=2), language="json")

elif not submit:
    st.markdown(
        """
        ### 👋 欢迎使用
        在左侧输入框中提问,例如:

        - 差旅费报销的标准是什么?
        - 员工试用期多久?
        - 怎么在阿里商旅订机票?
        - 财务单据怎么填写?
        - 借款审批单怎么填?
        """
    )
