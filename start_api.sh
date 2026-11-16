#!/usr/bin/env bash
# Mở API /api/v2/ocr/extract_local. Server tự check health vLLM OCR + LLM và warmup
# model nào vừa sẵn sàng (lúc khởi động hoặc sau khi model được tắt/bật lại).
cd "$(dirname "$0")"

exec uv run uvicorn src.app:app --host 0.0.0.0 --port "${PORT:-9006}"
