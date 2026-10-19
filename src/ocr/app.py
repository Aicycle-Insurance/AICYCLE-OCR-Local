import re
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from loguru import logger

from src.logger import setup_logger
from src.ocr import get_ocr_engine
from src.ocr.schemas import OCRResponse
from src.tracing import shutdown_langfuse

setup_logger("ocr_service")
engine = get_ocr_engine()


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info(f"OCR service started with engine={engine.name}, model={engine.config.model}")
    try:
        yield
    finally:
        shutdown_langfuse()
        await logger.complete()


app = FastAPI(title="Local OCR Service", lifespan=lifespan)


@app.get("/health")
async def health_check():
    return {"status": "healthy", "engine": engine.name, "model": engine.config.model}


@app.post("/api/v1/ocr", response_model=OCRResponse)
async def ocr(file: UploadFile = File(..., description="File PDF hoặc ảnh")):
    """OCR từng trang của file bằng engine đang cấu hình (env OCR_ENGINE)."""
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    try:
        return await engine.run(file_bytes, file.filename or "input")
    except Exception as e:
        logger.exception(f"OCR failed for {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/ocr/{run_id}/resume", response_model=OCRResponse)
async def resume_ocr(run_id: str):
    """OCR lại các trang lỗi/chưa xong của run ``run_id``, giữ nguyên các trang đã có."""
    if not re.fullmatch(r"[\w-]+", run_id):
        raise HTTPException(status_code=400, detail="Invalid run_id")
    try:
        return await engine.resume(run_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"OCR resume failed for {run_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
