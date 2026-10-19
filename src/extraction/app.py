from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from loguru import logger

from src.config import settings
from src.extraction.schemas import ExtractRequest, ExtractResponse
from src.extraction.service import ExtractionService, InvalidModelResponseError
from src.logger import setup_logger
from src.tracing import shutdown_langfuse

setup_logger("extraction_service")
service = ExtractionService()


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info(f"Extraction service started with model={settings.llm.model}")
    try:
        yield
    finally:
        shutdown_langfuse()
        await logger.complete()


app = FastAPI(title="Local Extraction Service", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    """Return only the error messages of a validation failure."""
    errors = [
        f"{' -> '.join(str(x) for x in error['loc'])}: {error['msg']}"
        for error in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content={"detail": errors if len(errors) > 1 else errors[0] if errors else "Validation error"},
    )


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": settings.llm.model}


@app.post("/api/v1/extract", response_model=ExtractResponse, response_model_exclude_none=True)
async def extract(request: ExtractRequest):
    """Schema-driven multi-document extraction trên OCR markdown từng trang.

    Phase 1 phân đoạn/phân loại tài liệu (start_page/end_page), Phase 2 trích
    xuất field theo ``extraction_schemas``. Input/output lưu ở data/extraction/{run_id}/.
    """
    try:
        return await service.run(request)
    except InvalidModelResponseError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error in extract: {e}")
        raise HTTPException(status_code=500, detail=str(e))
