from typing import List, Literal, Optional

from pydantic import BaseModel


class OCRPage(BaseModel):
    """Kết quả OCR của 1 trang (page đánh số từ 1); ``error`` khác null nếu trang lỗi sau khi retry."""

    page: int
    markdown: str = ""
    seconds: float = 0
    finish_reason: Optional[str] = None
    attempts: int = 1
    error: Optional[str] = None


class OCRResponse(BaseModel):
    """Response của OCR service; ``pages`` truyền thẳng sang extraction API được.

    ``status``: running (đang chạy, chỉ thấy trong result.json), completed (đủ trang),
    partial (còn trang lỗi trong ``failed_pages``, gọi resume để OCR lại).
    """

    run_id: str
    engine: str
    model: str
    file_name: str
    status: Literal["running", "completed", "partial"]
    page_count: int
    total_seconds: float
    failed_pages: List[int]
    pages: List[OCRPage]
