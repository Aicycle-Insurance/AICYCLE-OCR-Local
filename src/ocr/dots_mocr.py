import json
from typing import Any

from loguru import logger
from PIL import Image

from src.ocr.base import BaseOCR
from src.ocr.markdown import html_to_markdown
from src.prompts import DOTS_MOCR_LAYOUT_ALL_PROMPT


class DotsMOCR(BaseOCR):
    """rednote-hilab/dots.mocr: output là JSON list các layout cell
    ``{"bbox", "category", "text"}`` theo thứ tự đọc."""

    name = "dots_mocr"
    prompt = DOTS_MOCR_LAYOUT_ALL_PROMPT
    raw_suffix = ".json"

    def build_messages(self, image: Image.Image) -> list[dict[str, Any]]:
        messages = super().build_messages(image)
        # Không có prefix này thì vLLM v1 tự chèn "\n" sau ảnh (theo dots.mocr)
        messages[0]["content"][1]["text"] = (
            f"<|img|><|imgpad|><|endofimg|>{self.prompt}"
        )
        return messages

    def to_markdown(self, raw: str) -> str:
        try:
            cells = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(f"[{self.name}] output is not valid JSON, keeping raw text")
            return raw.strip()
        if not isinstance(cells, list):
            logger.warning(f"[{self.name}] output is not a list of cells, keeping raw text")
            return raw.strip()

        items = []
        for cell in cells:
            category = cell.get("category")
            text = (cell.get("text") or "").strip()
            if category == "Picture" or not text:
                continue
            if category == "Formula":
                items.append(f"$$\n{text}\n$$")
            elif category == "Table":
                items.append(html_to_markdown(text))
            else:
                items.append(text)
        return "\n\n".join(items)
