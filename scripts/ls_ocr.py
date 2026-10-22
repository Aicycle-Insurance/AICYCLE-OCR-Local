"""Benchmark bước 1: OCR file của các task trên Label Studio, lưu task_id -> OCR run_id ra file txt.

    uv run python scripts/ls_ocr.py --project-id 15 --out runs/project15.txt
    uv run python scripts/ls_ocr.py --project-id 4 --task-id 12 --engine dots_mocr --out runs/p4_dots.txt

Chạy lại cùng --out: task đã completed thì bỏ qua, task partial thì resume các trang lỗi.
Sau đó chạy scripts/ls_extract.py với file txt này (khi đã serve LLM).
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Cho phép chạy trực tiếp `python scripts/<file>.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from scripts.label_studio import (
    LSSession,
    download_task_file,
    get_api_key,
    get_tasks,
    read_runs,
    write_runs,
)
from src.logger import setup_logger
from src.ocr import get_ocr_engine
from src.tracing import shutdown_langfuse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Label Studio tasks -> OCR -> file txt danh sách run_id")
    p.add_argument("--ls-url", default="http://127.0.0.1:8001")
    p.add_argument("--project-id", type=int, required=True)
    p.add_argument("--task-id", type=int, help="Chỉ OCR 1 task; mặc định tất cả task của project")
    p.add_argument("--engine", help="chandra | dots_mocr (mặc định theo config)")
    p.add_argument("--out", type=Path, default=Path("ocr_runs.txt"), help="File txt lưu task_id -> run_id")
    return p.parse_args()


async def main(args: argparse.Namespace) -> None:
    base = args.ls_url.rstrip("/")
    ls = LSSession(base, get_api_key())
    tasks = get_tasks(ls, base, args.project_id, args.task_id)
    engine = get_ocr_engine(args.engine)
    runs = read_runs(args.out)
    logger.info(f"{len(tasks)} task(s), engine={engine.name}, {len(runs)} already in {args.out}")

    counts = {"completed": 0, "partial": 0, "skipped": 0, "error": 0}
    for task in tasks:
        tid = task["id"]
        if task.get("project") is not None and task["project"] != args.project_id:
            print(f"[SKIP] {tid}: other project")
            counts["skipped"] += 1
            continue
        previous = runs.get(tid)
        if previous and previous[1] == "completed":
            print(f"[SKIP] {tid}: already OCR'd (run {previous[0]})")
            counts["skipped"] += 1
            continue
        try:
            if previous:
                result = await engine.resume(previous[0])
            else:
                file_bytes, file_name = download_task_file(ls, base, task)
                result = await engine.run(file_bytes, file_name)
        except Exception as e:
            logger.exception(f"Task {tid}: OCR failed: {e}")
            print(f"[ERROR] {tid}: {e}")
            counts["error"] += 1
            continue

        runs[tid] = (result.run_id, result.status)
        write_runs(args.out, runs)  # ghi sau mỗi task để dừng giữa chừng vẫn giữ kết quả
        counts[result.status] += 1
        print(
            f"[{result.status.upper()}] task={tid} run={result.run_id} "
            f"pages={result.page_count} failed={result.failed_pages} {result.total_seconds:.1f}s"
        )

    print(f"\nDone: {counts} -> {args.out}")


if __name__ == "__main__":
    args = parse_args()
    setup_logger("ls_ocr")
    try:
        asyncio.run(main(args))
    finally:
        shutdown_langfuse()
