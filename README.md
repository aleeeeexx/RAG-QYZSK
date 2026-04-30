# RAG-QYZSk 企业知识库问答

参考 [RAG-Challenge-2](https://github.com/IlyaRice/RAG-Challenge-2) 架构,面向企业内部制度文档(员工手册、报销规定、操作手册等)的中文 RAG 问答系统。

## 技术栈

- **LLM / Embedding**: DashScope Qwen + text-embedding-v3
- **PDF 解析**: MinerU(在线 API)
- **Excel/PPT 解析**: pandas / python-pptx
- **检索**: FAISS 向量检索 + BM25(jieba 分词) 混合,RRF 融合
- **重排**: Qwen LLM 打分
- **父文档检索**: 召回块所在整页作上下文
- **前端**: Streamlit

## 快速开始

```bash
# 1. 创建虚拟环境并安装依赖
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. 配置 .env
cp .env.example .env  # 或直接编辑已有 .env
# 填入 DASHSCOPE_API_KEY 与 MINERU_API_TOKEN

# 3. 冒烟测试
python scripts/smoke_test.py

# 4. 灌库(P3 完成后)
python ingest.py

# 5. 启动 UI(P6 完成后)
streamlit run app_streamlit.py
```

## 目录结构

```
RAG-hxyl/
├── rag_data/             # 原始文档(PDF/xlsx/pptx)
├── data/
│   ├── parsed/           # 解析中间产物(JSON)
│   ├── chunked/          # 分块结果(JSON)
│   ├── vector_db/        # FAISS 索引
│   └── bm25_db/          # BM25 pickle
├── src/
│   ├── config.py         # 集中配置
│   ├── llm_client.py     # DashScope 客户端
│   ├── parsers/          # 多格式解析器
│   ├── chunker.py        # 切分
│   ├── ingestion.py      # 索引构建
│   ├── retrieval.py      # 混合检索
│   ├── reranker.py       # LLM 重排
│   ├── prompts.py        # Prompt 模板
│   └── pipeline.py       # 端到端流程
├── scripts/
│   └── smoke_test.py
├── app_streamlit.py
├── ingest.py
└── requirements.txt
```


## 产物总览
```
  RAG-hxyl/
  ├── .env                       DashScope + MinerU API key
  ├── requirements.txt           18 个依赖
  ├── README.md
  ├── ingest.py                  CLI:一键灌库
  ├── app_streamlit.py           前端
  ├── venv/                      独立虚拟环境
  ├── rag_data/                  原始 8 份文档
  ├── data/
  │   ├── parsed/                104 页解析结果(JSON 缓存)
  │   ├── chunked/               175 chunks + pages
  │   ├── vector_db/             FAISS 1024×175
  │   └── bm25_db/               jieba BM25
  ├── src/
  │   ├── config.py
  │   ├── llm_client.py          DashScope Qwen 客户端
  │   ├── chunker.py             递归切分(中文分隔符 + tiktoken)
  │   ├── ingestion.py           双索引构建
  │   ├── retrieval.py           向量+BM25+RRF+父文档
  │   ├── reranker.py            Qwen LLM 重排
  │   ├── prompts.py             中文 Prompt
  │   ├── pipeline.py            answer() 端到端
  │   └── parsers/
  │       ├── schema.py          统一 schema
  │       ├── pdf_parser.py      MinerU 在线 API + 缓存
  │       ├── excel_parser.py    pandas → markdown
  │       ├── pptx_parser.py     python-pptx
  │       └── unified.py         分发 + 缓存
  └── scripts/
      ├── smoke_test.py          P1
      ├── test_retrieval.py      P4
      └── test_pipeline.py       P5
```