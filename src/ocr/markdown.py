"""Chuyển HTML output của model OCR sang markdown; bảng giữ dạng HTML, ảnh thay bằng nhãn ngắn."""

import re
from typing import Any

from bs4 import BeautifulSoup, Tag
from markdownify import MarkdownConverter


TABLE_ATTRS = {"rowspan", "colspan", "alt"}

IMAGE_BLOCK_LABELS = ["Image", "Figure"]
# (từ khoá trong mô tả ảnh của Chandra, nhãn thay thế); chọn từ khoá xuất hiện sớm nhất trong mô tả,
# VD "official seal ... with a signature over it" -> con dấu.
IMAGE_LABELS = [
    (("signature", "signed"), "[Chữ ký]"),
    (("stamp", "seal"), "[Con dấu]"),
    (("barcode",), "[Mã vạch]"),
    (("qr",), "[Mã QR]"),
    (("fingerprint",), "[Vân tay]"),
    (("portrait", "photo of"), "[Ảnh chân dung]"),
    (("logo", "emblem"), "[Logo]"),
]


def _image_label(el: Tag) -> str:
    alts = [img.get("alt", "") for img in el.find_all("img")] + [el.get("alt", "")]
    desc = " ".join(alts + [el.get_text(" ")]).lower()
    best = None
    for keys, label in IMAGE_LABELS:
        for key in keys:
            m = re.search(rf"\b{re.escape(key)}", desc)
            if m and (best is None or m.start() < best[0]):
                best = (m.start(), label)
    return best[1] if best else "[Hình ảnh]"


def _replace_images(soup: BeautifulSoup, image_mode: str) -> None:
    """Xử lý mô tả ảnh: "keep" giữ nguyên, "label" thay bằng nhãn ngắn (VD [Chữ ký]), "remove" bỏ hẳn.

    Làm trên cây HTML (còn data-label) để bắt cả đoạn mô tả dài trong khối ảnh và thẻ <img> nằm giữa dòng chữ.
    Khối ảnh có chữ thật (VD Chandra gắn nhãn Image cho cả thẻ CCCD nhưng vẫn đọc chữ bên trong) thì giữ
    chữ, chỉ xử lý các thẻ <img> trong đó; mô tả ảnh nằm ở alt nên không tính là chữ.
    """
    if image_mode == "keep":
        return
    if image_mode not in ("label", "remove"):
        raise ValueError(f"image_mode không hợp lệ: {image_mode}")
    blocks = [
        b for b in soup.find_all("div", attrs={"data-label": IMAGE_BLOCK_LABELS}) if not b.get_text(strip=True)
    ]
    targets = blocks + [img for img in soup.find_all("img") if not any(img in b.descendants for b in blocks)]
    for el in targets:
        if image_mode == "remove":
            el.decompose()
        else:
            el.replace_with(_image_label(el))


class _Converter(MarkdownConverter):
    """markdownify, nhưng bảng giữ nguyên HTML (đúng cấu trúc ô gộp rowspan/colspan) thay vì bảng markdown."""

    def convert_table(self, el: Tag, text: str, parent_tags: set[str]) -> str:
        for tag in [el, *el.find_all(True)]:
            tag.attrs = {k: v for k, v in tag.attrs.items() if k in TABLE_ATTRS}
        html = "\n".join(line for line in str(el).splitlines() if line.strip())
        return f"\n\n{html}\n\n"


def html_to_markdown(html: str, image_mode: str = "label", **options: Any) -> str:
    """Như ``markdownify(html, **options)`` nhưng bảng giữ dạng HTML và ảnh được xử lý theo ``image_mode``.

    image_mode: "label" thay ảnh bằng nhãn ngắn (VD [Chữ ký]), "remove" bỏ hẳn, "keep" giữ "![mô tả]()".
    """
    soup = BeautifulSoup(html, "html.parser")
    _replace_images(soup, image_mode)
    return _Converter(**options).convert(str(soup)).strip()
