#!/usr/bin/env bash
# Warmup vLLM OCR + LLM 1 lần rồi mở API /api/v2/ocr/extract_local.
# Warmup lỗi (vd. vLLM chưa lên) vẫn khởi động server.
cd "$(dirname "$0")"

uv run python scripts/warmup.py || echo "Warmup lỗi, vẫn khởi động server"
exec uv run uvicorn src.app:app --host 0.0.0.0 --port "${PORT:-9002}"
