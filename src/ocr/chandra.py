from markdownify import markdownify

from src.ocr.base import BaseOCR
from src.prompts import CHANDRA_OCR_LAYOUT_PROMPT


class ChandraOCR(BaseOCR):
    """datalab-to/chandra-ocr-2: output là HTML chia theo layout block."""

    name = "chandra"
    prompt = CHANDRA_OCR_LAYOUT_PROMPT
    raw_suffix = ".html"

    def to_markdown(self, raw: str) -> str:
        return markdownify(raw, heading_style="ATX").strip()
