# Hướng dẫn chạy scripts

Mọi lệnh chạy từ thư mục gốc project `ocr-local/`.

## 1. Cài đặt (1 lần)

```bash
uv sync
source .venv/bin/activate      # để gõ `python` thay cho `uv run python`
cp .env.example .env           # rồi sửa các giá trị cần thiết
```

Trong `.env` cần chú ý:

| Biến | Ý nghĩa |
|---|---|
| `OCR_ENGINE` | `chandra` hoặc `dots_mocr` |
| `CHANDRA_BASE_URL`, `DOTS_MOCR_BASE_URL` | Địa chỉ vLLM của model OCR |
| `LLM_BASE_URL`, `LLM_MODEL` | Địa chỉ vLLM + tên model LLM extract |
| `LABEL_STUDIO_API_KEY` | Personal Access Token của Label Studio (chỉ cần cho `ls_*.py`) |

## 2. Chạy thử 1 file

**Bước 1 – OCR** (cần vLLM OCR đang chạy):

```bash
python scripts/demo_ocr.py duong/dan/file.pdf
python scripts/demo_ocr.py duong/dan/file.pdf --engine dots_mocr   # đổi engine
python scripts/demo_ocr.py --resume <run_id>                       # OCR lại trang lỗi
```

Script in ra `run_id` để dùng cho bước 2.

**Bước 2 – Extract** (cần vLLM LLM đang chạy):

```bash
python scripts/demo_extract.py <run_id> --schemas examples/schemas.json
```

## 3. Benchmark với Label Studio

Máy chỉ serve được 1 model mỗi lúc nên chạy 2 bước riêng.

**Bước 1 – OCR tất cả task** (serve model OCR):

```bash
python scripts/ls_ocr.py --project-id 5 --out runs/p5_chandra.txt
```

- Thêm `--task-id N` để chạy 1 task, `--engine dots_mocr` để đổi engine.
- Bị dừng giữa chừng → chạy lại **đúng lệnh cũ**: task xong bỏ qua, task lỗi trang tự resume.

**Bước 2 – Extract → prediction** (tắt OCR, serve LLM):

```bash
python scripts/ls_extract.py --runs runs/p5_chandra.txt \
  --request ../test/ocr_labelstudio_runner/request.json \
  --project-id 5 --model-version chandra_gemma4_v1
```

- Kết quả: `predictions_preview.json` (cùng định dạng `run.py` cũ).
- Thêm `--upload` để đẩy prediction lên Label Studio (upload 2 lần sẽ bị trùng).

**Bước 3 – Chấm điểm** (script cũ):

```bash
cd ../test/ocr_labelstudio_runner
python evaluate.py --predictions /home/dangdh/work/ocr-local/predictions_preview.json
```

## 4. Kết quả lưu ở đâu

| Thư mục | Nội dung |
|---|---|
| `data/ocr/<run_id>/` | `input/` file gốc, `pages/page_XXXX.md` OCR từng trang, `result.json` |
| `data/extraction/<run_id>/` | `request.json`, `phase1.json` (phân loại), `response.json` |
| `runs/*.txt` | `task_id → ocr_run_id → status` của `ls_ocr.py` |
| `logs/` | Log của từng script/service |

## 5. Lưu ý

- `--project-id` ở bước OCR và bước extract phải giống nhau; mỗi project dùng 1 file `--out` riêng.
- Không xóa `data/ocr/` khi file `runs/*.txt` còn dùng.
- Mỗi trang chandra mất ~20–30s; log `page N: sending request...` là đang chờ server, không phải treo.
- Request đầu tiên sau khi restart vLLM rất chậm (warmup).
- Warning `output truncated at max_tokens`: model thường bị lặp vòng (vd. dòng trống của bảng), trang
  đó mất nội dung nhưng vẫn báo `completed` → kiểm tra lại `pages/page_XXXX.md` của trang đó.

## 6. Chạy dạng API (không bắt buộc)

```bash
uvicorn src.ocr.app:app --port 9000          # POST /api/v1/ocr (upload file)
uvicorn src.extraction.app:app --port 9001   # POST /api/v1/extract
uvicorn src.app:app --port 9002              # POST /api/v2/ocr/extract_local (file_url -> OCR -> extract)
```

Hoặc warmup vLLM OCR + LLM 1 lần rồi mở API `src.app` (port mặc định 9002, đổi bằng `PORT=...`):

```bash
./start_api.sh                    # = python scripts/warmup.py rồi uvicorn src.app:app
python scripts/warmup.py          # chỉ warmup
```

Endpoint `/api/v2/ocr/extract_local` nhận body định dạng Gemini-OCR v2; cần **cả** vLLM OCR và vLLM LLM cùng chạy.
Chỉ bắt buộc `file_url`; không gửi `extraction_schemas` thì server đọc từ `examples/schemas.json`
(đọc lại mỗi request, sửa file không cần restart). `include_bbox` được bỏ qua.

```bash
curl -X POST localhost:9002/api/v2/ocr/extract -H 'Content-Type: application/json' \
  -d '{"file_url": "https://.../file.pdf", "extract_all_documents": false}'
```

Response giống `/api/v1/extract`, thêm `ocr_run_id` (thư mục `data/ocr/`) và `failed_pages` (trang OCR lỗi
bị bỏ qua khi extract — resume OCR run đó rồi gọi lại `/api/v1/extract` nếu cần).
