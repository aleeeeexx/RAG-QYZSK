"""集中管理路径、模型名、检索参数。所有模块从这里读取常量。"""
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "rag_data"
DATA_DIR = PROJECT_ROOT / "data"
PARSED_DIR = DATA_DIR / "parsed"
CHUNKED_DIR = DATA_DIR / "chunked"
VECTOR_DB_DIR = DATA_DIR / "vector_db"
BM25_DB_DIR = DATA_DIR / "bm25_db"

for d in (PARSED_DIR, CHUNKED_DIR, VECTOR_DB_DIR, BM25_DB_DIR):
    d.mkdir(parents=True, exist_ok=True)

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
MINERU_API_TOKEN = os.getenv("MINERU_API_TOKEN", "")

LLM_MODEL = "qwen-plus"
RERANK_MODEL = "qwen-turbo"
EMBED_MODEL = "text-embedding-v3"
EMBED_DIM = 1024

CHUNK_SIZE_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 100

VECTOR_TOP_K = 30
BM25_TOP_K = 30
RRF_TOP_K = 20
RERANK_TOP_K = 5

LLM_MAX_TOKENS = 2048
LLM_TEMPERATURE = 0.1
