from src.config import settings
from src.ocr.base import BaseOCR
from src.ocr.chandra import ChandraOCR
from src.ocr.dots_mocr import DotsMOCR

ENGINES: dict[str, type[BaseOCR]] = {
    ChandraOCR.name: ChandraOCR,
    DotsMOCR.name: DotsMOCR,
}


def get_ocr_engine(name: str | None = None) -> BaseOCR:
    """Khởi tạo engine theo ``name``, mặc định lấy ``ocr.engine`` (env OCR_ENGINE)."""
    name = name or settings.ocr.engine
    if name not in ENGINES:
        raise ValueError(f"Unknown OCR engine '{name}'. Available: {', '.join(ENGINES)}")
    if name not in settings.ocr.engines:
        raise ValueError(f"Missing config for OCR engine '{name}' in config.yaml")
    return ENGINES[name](
        settings.ocr.engines[name],
        dpi=settings.ocr.dpi,
        max_concurrency=settings.ocr.max_concurrency,
        retry=settings.ocr.retry,
    )
