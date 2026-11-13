"""Warmup vLLM OCR + LLM trước khi mở API, để request đầu tiên của người dùng không bị chậm.

Gửi 1 request OCR (ảnh nhỏ có chữ) và 1 request extraction Phase 1 (structured output
với schema mặc định ở examples/schemas.json) song song. Không ghi gì vào data/.

Cách chạy (từ thư mục gốc project):
    uv run python scripts/warmup.py
Exit code khác 0 nếu có model warmup lỗi.
"""

import asyncio
import json
import sys
import time
from pathlib import Path

# Cho phép chạy trực tiếp `python scripts/<file>.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger
from PIL import Image, ImageDraw

from src.extraction.json_schema import build_phase1_response_schema
from src.extraction.schemas import DEFAULT_SCHEMAS_PATH, ExtractionSchema
from src.extraction.service import ExtractionService, GenerationParams
from src.logger import setup_logger
from src.ocr import get_ocr_engine
from src.prompts import build_phase1_prompt
from src.tracing import shutdown_langfuse

WARMUP_MAX_TOKENS = 256
WARMUP_TEXT = "GIAY RA VIEN\nHo ten: NGUYEN VAN A\nNgay vao vien: 01/01/2026"


async def warmup_ocr() -> None:
    engine = get_ocr_engine()
    image = Image.new("RGB", (800, 400), "white")
    ImageDraw.Draw(image).multiline_text((40, 40), WARMUP_TEXT, fill="black", spacing=20)
    started = time.perf_counter()
    await engine.client.chat.completions.create(
        model=engine.config.model,
        messages=engine.build_messages(image),
        temperature=engine.config.temperature,
        max_tokens=WARMUP_MAX_TOKENS,
    )
    logger.info(f"Warmup OCR ({engine.name}, {engine.config.model}): {time.perf_counter() - started:.2f}s")


async def warmup_llm() -> None:
    service = ExtractionService()
    schemas = [
        ExtractionSchema.model_validate(item)
        for item in json.loads(DEFAULT_SCHEMAS_PATH.read_text(encoding="utf-8"))
    ]
    started = time.perf_counter()
    await service._call_model(
        build_phase1_prompt(schemas, False),
        [(1, WARMUP_TEXT)],
        build_phase1_response_schema(schemas, False),
        GenerationParams(max_output_tokens=WARMUP_MAX_TOKENS),
        trace_name="warmup.phase1",
    )
    logger.info(f"Warmup LLM ({service.config.model}): {time.perf_counter() - started:.2f}s")


async def main() -> int:
    results = await asyncio.gather(warmup_ocr(), warmup_llm(), return_exceptions=True)
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
