"""Warmup vLLM OCR + LLM 1 lần bằng tay (API src.app đã tự warmup khi model vừa bật).

Gửi 1 request OCR (ảnh nhỏ có chữ) và 1 request extraction Phase 1 (structured output
với schema mặc định ở examples/schemas.json) song song. Không ghi gì vào data/.

Cách chạy (từ thư mục gốc project):
    uv run python scripts/warmup.py
Exit code khác 0 nếu có model warmup lỗi.
"""

import asyncio
import sys
from pathlib import Path

# Cho phép chạy trực tiếp `python scripts/<file>.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from src.extraction.service import ExtractionService
from src.logger import setup_logger
from src.ocr import get_ocr_engine
from src.tracing import shutdown_langfuse
from src.warmup import warmup_llm, warmup_ocr


async def main() -> int:
    results = await asyncio.gather(
        warmup_ocr(get_ocr_engine()), warmup_llm(ExtractionService()), return_exceptions=True
    )
    failed = 0
    for name, result in zip(("OCR", "LLM"), results):
        if isinstance(result, BaseException):
            logger.error(f"Warmup {name} failed: {type(result).__name__}: {result}")
            failed += 1
    shutdown_langfuse()
    return 1 if failed else 0


if __name__ == "__main__":
    setup_logger("warmup")
    sys.exit(asyncio.run(main()))
