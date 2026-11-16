# API tổng /api/v2/ocr/extract_local (src.app). vLLM OCR + LLM chạy ở container/máy khác,
# trỏ tới bằng CHANDRA_BASE_URL / LLM_BASE_URL (xem docker-compose.yml).
FROM python:3.11-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Cài dependency trước để cache layer khi chỉ sửa code
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev

COPY config.yaml ./
COPY examples ./examples
COPY src ./src

# Kết quả run và log; mount volume để giữ lại khi container bị xoá
RUN mkdir -p data logs
VOLUME ["/app/data", "/app/logs"]

ENV PORT=9006
EXPOSE 9006

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health', timeout=4)"

CMD ["sh", "-c", "exec uvicorn src.app:app --host 0.0.0.0 --port ${PORT}"]
