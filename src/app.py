"""Service end-to-end: tải ``file_url`` -> OCR -> extract, cùng định dạng request của Gemini-OCR.

Cần cả vLLM OCR (engine theo OCR_ENGINE) và vLLM LLM cùng chạy.
Kết quả lưu như chạy từng bước: data/ocr/{ocr_run_id}/ và data/extraction/{run_id}/.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from loguru import logger

from src.config import settings
from src.download import DownloadError, download_file
from src.extraction.schemas import ExtractRequest, OCRExtractRequest, OCRExtractResponse
from src.extraction.service import ExtractionService, InvalidModelResponseError
from src.logger import setup_logger
from src.ocr import get_ocr_engine
from src.tracing import shutdown_langfuse

setup_logger("pipeline_service")
engine = get_ocr_engine()
service = ExtractionService()


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info(
        f"Pipeline service started with OCR engine={engine.name}, model={engine.config.model}, "
        f"LLM model={settings.llm.model}"
    )
    try:
        yield
    finally:
        shutdown_langfuse()
        await logger.complete()


app = FastAPI(title="Local OCR + Extraction Service", lifespan=lifespan)


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "engine": engine.name,
        "ocr_model": engine.config.model,
        "llm_model": settings.llm.model,
    }


@app.post("/api/v2/ocr/extract_local", response_model=OCRExtractResponse, response_model_exclude_none=True)
async def ocr_extract(request: OCRExtractRequest):
    """Tải file từ ``file_url``, OCR từng trang rồi extract theo ``extraction_schemas``.

    Trang OCR lỗi (sau retry) bị bỏ qua và trả về trong ``failed_pages``.
    """
    try:
        file_bytes, file_name = await download_file(request.file_url)
    except DownloadError as e:
        raise HTTPException(status_code=400, detail=str(e))
    logger.info(f"Downloaded {file_name} ({len(file_bytes)} bytes)")

    try:
        ocr_result = await engine.run(file_bytes, file_name)
    except Exception as e:
        logger.exception(f"OCR failed for {file_name}: {e}")
        raise HTTPException(status_code=500, detail=f"OCR failed: {e}")

    pages = [
        {"page": page.page, "markdown": page.markdown}
        for page in ocr_result.pages
        if page.error is None
    ]
    if not pages:
        raise HTTPException(
            status_code=502, detail=f"OCR failed on all pages (ocr_run_id={ocr_result.run_id})"
        )
    if ocr_result.failed_pages:
        logger.warning(
            f"OCR run {ocr_result.run_id} has failed pages {ocr_result.failed_pages}, "
            "they are skipped in extraction"
        )

    extract_request = ExtractRequest(
        pages=pages, **request.model_dump(exclude={"file_url"})
    )
    try:
        result = await service.run(extract_request)
    except InvalidModelResponseError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        logger.exception(f"Extraction failed for OCR run {ocr_result.run_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return OCRExtractResponse(
        **result.model_dump(),
        ocr_run_id=ocr_result.run_id,
        failed_pages=ocr_result.failed_pages,
    )
