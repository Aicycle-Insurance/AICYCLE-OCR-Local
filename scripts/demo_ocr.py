"""Demo bước 1: OCR 1 file (PDF/ảnh) bằng engine trong config (env OCR_ENGINE).

Cách chạy (từ thư mục gốc project, hoặc thay bằng đường dẫn tuyệt đối):
    uv run python scripts/demo_ocr.py pdfs/HuynhMinhTue.pdf
    uv run python scripts/demo_ocr.py pdfs/HuynhMinhTue.pdf --engine dots_mocr
    uv run python scripts/demo_ocr.py --resume <run_id>   # OCR lại các trang lỗi

Kết quả lưu ở data/ocr/{run_id}/; dùng run_id này cho scripts/demo_extract.py.
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Cho phép chạy trực tiếp `python scripts/<file>.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.logger import setup_logger
from src.ocr import get_ocr_engine
from src.ocr.schemas import OCRResponse
from src.tracing import shutdown_langfuse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OCR 1 file bằng vLLM")
    parser.add_argument("file", type=Path, nargs="?", help="File PDF hoặc ảnh cần OCR")
    parser.add_argument("--resume", metavar="RUN_ID", help="OCR lại các trang lỗi của run")
    parser.add_argument("--engine", help="chandra | dots_mocr (mặc định theo config)")
    args = parser.parse_args()
    if not args.file and not args.resume:
        parser.error("cần truyền file hoặc --resume RUN_ID")
    return args


async def main(args: argparse.Namespace) -> OCRResponse:
    engine = get_ocr_engine(args.engine)
    if args.resume:
        return await engine.resume(args.resume)
    return await engine.run(args.file.read_bytes(), args.file.name)


if __name__ == "__main__":
    args = parse_args()
    setup_logger("demo_ocr")
    try:
        result = asyncio.run(main(args))
    finally:
        shutdown_langfuse()

    print(f"\nrun_id: {result.run_id}")
    print(f"status: {result.status} | pages: {result.page_count} | {result.total_seconds:.2f}s")
    if result.failed_pages:
        print(f"failed pages: {result.failed_pages} -> chạy lại: "
              f"uv run python scripts/demo_ocr.py --resume {result.run_id}")
    print(f"next: uv run python scripts/demo_extract.py {result.run_id} --schemas examples/schemas.json")
