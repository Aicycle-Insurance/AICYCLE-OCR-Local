"""Benchmark bước 2: extract (LLM) từ các OCR run trong file txt của scripts/ls_ocr.py
-> predictions_preview.json cùng định dạng run.py cũ (dùng được với evaluate.py).

    uv run python scripts/ls_extract.py --runs runs/project4_chandra.txt \
        --request ../test/ocr_labelstudio_runner/request.json --project-id 4 --model-version chandra_gemma4_v1
    # thêm --upload để đẩy prediction lên Label Studio

``request.json`` là template như luồng cũ: lấy ``extraction_schemas`` và các option
có trong ExtractRequest; các key khác (file_url, include_bbox, model_name...) bị bỏ qua.
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

# Cho phép chạy trực tiếp `python scripts/<file>.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from scripts.label_studio import LSSession, extract_one, get_api_key, read_runs
from src.config import settings
from src.extraction.schemas import ExtractRequest
from src.extraction.service import ExtractionService
from src.logger import setup_logger
from src.ocr.schemas import OCRResponse
from src.tracing import shutdown_langfuse

# Key của template cũ (Gemini) không áp dụng cho vLLM
IGNORED_TEMPLATE_KEYS = ("file_url", "include_bbox", "model_name", "api_key")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OCR runs -> LLM extract -> Label Studio predictions")
    p.add_argument("--runs", type=Path, required=True, help="File txt từ scripts/ls_ocr.py")
    p.add_argument("--request", type=Path, required=True, help="Template JSON có extraction_schemas")
    p.add_argument("--project-id", type=int, required=True)
    p.add_argument("--model-version", default="ocr_local_v1")
    p.add_argument("--task-id", type=int, help="Chỉ chạy 1 task trong file runs")
    p.add_argument("--ls-url", default="http://127.0.0.1:8001")
    p.add_argument("--upload", action="store_true", help="Đẩy prediction lên LS; mặc định chỉ preview")
    p.add_argument("--out", type=Path, default=Path("predictions_preview.json"))
    p.add_argument(
        "--concurrency", type=int, default=1,
        help="Số task extract cùng lúc (mỗi task tự gọi LLM song song tới LLM_MAX_CONCURRENT_EXTRACTIONS)",
    )
    return p.parse_args()


def load_ocr_run(run_id: str) -> tuple[list[dict], float, float]:
    """Trả về (pages để extract, thời gian OCR, thời gian thực của cả run theo result.json).

    Thời gian OCR = trang lâu nhất (các trang chạy song song), tính từ lúc trang được gửi tới
    server, không tính lúc chờ slot ``OCR_MAX_CONCURRENCY`` khi chạy nhiều task cùng lúc
    (``total_seconds`` có tính phần chờ đó nên chỉ ghi lại để tham khảo).
    """
    path = settings.app.data_dir / "ocr" / run_id / "result.json"
    result = OCRResponse.model_validate_json(path.read_text(encoding="utf-8"))
    if result.failed_pages:
        logger.warning(f"OCR run {run_id}: failed pages {result.failed_pages} are skipped")
    done = [p for p in result.pages if p.error is None]
    pages = [{"page": p.page, "markdown": p.markdown} for p in done]
    ocr_seconds = max((p.seconds for p in done), default=0.0)
    return pages, ocr_seconds, result.total_seconds


async def main(args: argparse.Namespace) -> None:
    template = json.loads(args.request.read_text(encoding="utf-8"))
    options = {k: v for k, v in template.items() if k not in IGNORED_TEMPLATE_KEYS}
    schemas = {s["document_code"]: s for s in template["extraction_schemas"]}
    runs = read_runs(args.runs)
    if args.task_id:
        runs = {args.task_id: runs[args.task_id]}

    ls = None
    if args.upload:
        base = args.ls_url.rstrip("/")
        ls = LSSession(base, get_api_key())

    service = ExtractionService()
    previews = []
    semaphore = asyncio.Semaphore(args.concurrency)

    async def process(tid: int, run_id: str, status: str) -> None:
        try:
            if status != "completed":
                logger.warning(f"Task {tid}: OCR run {run_id} is {status}")
            pages, ocr_seconds, ocr_wall_seconds = load_ocr_run(run_id)
            async with semaphore:
                t0 = time.perf_counter()
                response = await service.run(ExtractRequest.model_validate({**options, "pages": pages}))
                extract_seconds = time.perf_counter() - t0
            code, result = extract_one(response.model_dump(), schemas)
        except Exception as e:
            logger.exception(f"Task {tid}: extract failed: {e}")
            print(f"[SKIP] {tid}: {e}")
            return

        prediction = {
            "task": tid,
            "project": args.project_id,
            "model_version": args.model_version,
            "result": result,
        }
        # Latency = OCR + extract (chạy tách 2 bước); model local nên cost = 0.
        # Các trường này chỉ có trong preview, không upload lên LS (giống run.py cũ).
        latency = ocr_seconds + extract_seconds
        previews.append({
            **prediction,
            "latency_s": round(latency, 3),
            "cost_usd": 0.0,
            "ocr_s": round(ocr_seconds, 3),
            "extract_s": round(extract_seconds, 3),
            "ocr_wall_s": round(ocr_wall_seconds, 3),  # có tính thời gian chờ slot, chỉ để tham khảo
        })
        print(
            f"[OK] task={tid} document_code={code} fields={len(result) - 1} extraction_run={response.run_id}"
            f" latency={latency:.2f}s (ocr={ocr_seconds:.2f}s extract={extract_seconds:.2f}s"
            f" ocr_wall={ocr_wall_seconds:.2f}s)"
        )
        if ls is not None:
            try:
                async with upload_lock:  # LSSession (requests) dùng chung, upload lần lượt
                    resp = await asyncio.to_thread(
                        ls.post, f"{ls.base}/api/predictions/", json=prediction, timeout=60
                    )
                resp.raise_for_status()
            except Exception as e:
                # Không để 1 task upload lỗi làm mất preview của các task khác
                logger.exception(f"Task {tid}: upload failed: {e}")
                print(f"[UPLOAD ERROR] task={tid}: {e}")
                return
            print(f"[UPLOADED] task={tid} prediction_id={resp.json().get('id')}")

    upload_lock = asyncio.Lock()
    started = time.perf_counter()
    await asyncio.gather(*(process(tid, run_id, status) for tid, (run_id, status) in sorted(runs.items())))
    previews.sort(key=lambda p: p["task"])
    args.out.write_text(json.dumps(previews, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"Preview: {args.out}; predictions={len(previews)}/{len(runs)}; uploaded={args.upload}; "
        f"{time.perf_counter() - started:.1f}s"
    )


if __name__ == "__main__":
    args = parse_args()
    setup_logger("ls_extract")
    try:
        asyncio.run(main(args))
    finally:
        shutdown_langfuse()
