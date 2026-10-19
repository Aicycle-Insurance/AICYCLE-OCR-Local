"""Demo bước 2: extract thông tin từ kết quả OCR của scripts/demo_ocr.py.

Cách chạy (từ thư mục gốc project, hoặc thay bằng đường dẫn tuyệt đối):
    uv run python scripts/demo_extract.py <ocr_run_id> --schemas examples/schemas.json
    uv run python scripts/demo_extract.py data/ocr/<run_id>/result.json --schemas examples/schemas.json \
        --extract-all-documents

Kết quả lưu ở data/extraction/{run_id}/ (request.json, phase1.json, response.json).
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Cho phép chạy trực tiếp `python scripts/<file>.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from src.config import settings
from src.extraction.schemas import ExtractRequest, ExtractResponse
from src.extraction.service import ExtractionService
from src.logger import setup_logger
from src.ocr.schemas import OCRResponse
from src.tracing import shutdown_langfuse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract thông tin từ kết quả OCR")
    parser.add_argument("ocr", help="OCR run_id hoặc đường dẫn tới result.json")
    parser.add_argument("--schemas", type=Path, required=True, help="File JSON list extraction_schemas")
    parser.add_argument("--extract-all-fields", action="store_true")
    parser.add_argument("--extract-all-documents", action="store_true")
    return parser.parse_args()


def load_ocr_result(ocr: str) -> OCRResponse:
    path = Path(ocr)
    if not path.is_file():
        path = settings.app.data_dir / "ocr" / ocr / "result.json"
    return OCRResponse.model_validate_json(path.read_text(encoding="utf-8"))


def build_request(args: argparse.Namespace) -> ExtractRequest:
    ocr_result = load_ocr_result(args.ocr)
    if ocr_result.failed_pages:
        logger.warning(
            f"OCR run {ocr_result.run_id} has failed pages {ocr_result.failed_pages}, "
            "they are skipped (resume the OCR run to include them)"
        )
    return ExtractRequest(
        pages=[
            {"page": page.page, "markdown": page.markdown}
            for page in ocr_result.pages
            if page.error is None
        ],
        extraction_schemas=json.loads(args.schemas.read_text(encoding="utf-8")),
        extract_all_fields=args.extract_all_fields,
        extract_all_documents=args.extract_all_documents,
    )


if __name__ == "__main__":
    args = parse_args()
    setup_logger("demo_extract")
    request = build_request(args)
    try:
        response: ExtractResponse = asyncio.run(ExtractionService().run(request))
    finally:
        shutdown_langfuse()

    print(json.dumps(response.model_dump(exclude_none=True), ensure_ascii=False, indent=2))
    print(f"\nrun_id: {response.run_id} -> {settings.app.data_dir / 'extraction' / response.run_id}")
