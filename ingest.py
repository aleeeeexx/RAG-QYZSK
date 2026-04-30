"""一键灌库 CLI:解析 → 切分 → 双索引。

用法:
    python ingest.py            # 利用已有解析缓存
    python ingest.py --force    # 强制重新解析(注意 MinerU 仍走缓存)
"""
import argparse
from src.ingestion import build_indexes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="强制重新解析(忽略 data/parsed 缓存)")
    args = ap.parse_args()
    build_indexes(force_parse=args.force)


if __name__ == "__main__":
    main()
