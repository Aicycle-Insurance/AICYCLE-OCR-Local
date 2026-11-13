"""Tải file từ URL, port từ Gemini-OCR ``download_file_from_url``."""

from urllib.parse import unquote, urlparse

import httpx

from src.storage import safe_filename

MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024
DOWNLOAD_TIMEOUT_SECONDS = 60.0
CHUNK_SIZE_BYTES = 1024 * 1024


class DownloadError(Exception):
    """URL không hợp lệ, tải lỗi, file rỗng/quá lớn hoặc không phải PDF/ảnh."""


def _check_size(size: int) -> None:
    if size > MAX_FILE_SIZE_BYTES:
        raise DownloadError(f"File too large. Maximum size is {MAX_FILE_SIZE_BYTES // 1024 // 1024}MB")


async def download_file(url: str) -> tuple[bytes, str]:
    """Tải ``url`` về bộ nhớ, trả về (bytes, tên file lấy từ path của URL)."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise DownloadError("Only valid HTTP and HTTPS URLs are allowed")

    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=DOWNLOAD_TIMEOUT_SECONDS) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                # Chặn sớm theo header, rồi chặn tiếp khi stream (header có thể thiếu/sai)
                content_length = response.headers.get("content-length")
                if content_length and content_length.isdigit():
                    _check_size(int(content_length))
                buffer = bytearray()
                async for chunk in response.aiter_bytes(CHUNK_SIZE_BYTES):
                    buffer.extend(chunk)
                    _check_size(len(buffer))
    except httpx.HTTPStatusError as e:
        raise DownloadError(f"Error downloading file from URL: status {e.response.status_code}")
    except httpx.RequestError as e:
        raise DownloadError(f"Error downloading file from URL: {type(e).__name__}: {e}")

    content = bytes(buffer)
    if not content:
        raise DownloadError("Downloaded file is empty")
    # Engine OCR nhận diện PDF/ảnh theo magic bytes, ở đây chỉ chặn sớm file không đọc được
    if not content.startswith((b"%PDF-", b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff")):
        raise DownloadError("Unsupported file type, only PDF, PNG and JPEG are allowed")

    # Bỏ query (vd. chữ ký của signed URL) để tên file không quá dài
    file_name = safe_filename(unquote(parsed.path.rsplit("/", 1)[-1]))[:100]
    return content, file_name
