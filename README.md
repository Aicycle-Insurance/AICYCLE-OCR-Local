# OCR Local

Hệ thống OCR và trích xuất thông tin từ tài liệu (PDF/ảnh), chạy hoàn toàn bằng model tự host trên vLLM,
không gọi API bên ngoài. Dùng cho hồ sơ y tế/bảo hiểm tiếng Việt: CCCD, đơn thuốc, giấy ra viện, bảng kê viện phí...

## Luồng xử lý

```
PDF / ảnh
   │
   ▼
[1] OCR  ── mỗi trang 1 request tới model OCR (chandra hoặc dots.mocr) ──► markdown từng trang
   │
   ▼
[2] Extract ── LLM (Gemma) ──► Phase 1: phân loại + tách tài liệu theo trang
   │                           Phase 2: trích xuất trường theo schema của từng loại tài liệu
   ▼
JSON kết quả
```

- **OCR**: chuyển PDF thành ảnh từng trang rồi gửi song song tới model OCR. Output HTML được chuyển sang markdown:
  bảng giữ nguyên HTML, ảnh thay bằng nhãn ngắn như `[Chữ ký]` hoặc `[Con dấu]`; chữ nằm trong khối ảnh được giữ lại.
  Trang lỗi được retry. Run bị dừng giữa chừng có thể resume để OCR lại các trang lỗi.
- **Chống lặp (chandra)**: output được stream về. Khi phát hiện model lặp vòng (thường ở hàng bảng rỗng),
  hệ thống ngắt request, cắt phần lặp rồi cho model viết tiếp với temperature cao hơn (tối đa 3 lần).
- **Extract**: schema các loại tài liệu nằm trong `examples/schemas.json`. Một file có nhiều tài liệu thì được tách ra
  và trích xuất song song.

## Thành phần

| Thư mục / file | Vai trò |
|---|---|
| `src/ocr/` | Engine OCR (`chandra.py`, `dots_mocr.py`), chuyển HTML → markdown, API `/api/v1/ocr` |
| `src/extraction/` | Phân loại + trích xuất bằng LLM, API `/api/v1/extract` |
| `src/app.py` | API tổng `/api/v2/ocr/extract_local` (file_url → OCR → extract), tự health check + warmup model |
| `src/prompts.py` | Prompt cho OCR và LLM |
| `scripts/` | Chạy thử (`demo_*.py`), benchmark với Label Studio (`ls_*.py`), warmup |
| `config.yaml`, `.env` | Cấu hình; `config.yaml` đọc biến môi trường theo cú pháp `${TEN_BIEN:mac_dinh}` |

## 1. Cài đặt

Yêu cầu:

- Python ≥ 3.11, [uv](https://docs.astral.sh/uv/)
- vLLM serve model OCR (mặc định `datalab-to/chandra-ocr-2` ở cổng 8000) và model LLM (mặc định cổng 10100)
- Tùy chọn: Langfuse để trace, Label Studio để benchmark

Mọi lệnh chạy từ thư mục gốc project `ocr-local/`.

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

`max_tokens` của engine OCR trong `config.yaml` phải nhỏ hơn context (`--max-model-len`) của vLLM trừ đi số token
input (prompt + ảnh, khoảng 4–5k với chandra), nếu không server trả lỗi 400.

## 2. API

Có 3 service FastAPI. Thường chỉ cần chạy service tổng `src.app`. Hai service còn lại dùng khi muốn chạy riêng
OCR hoặc extract (ví dụ máy chỉ serve được 1 model mỗi lúc).

| Service | Lệnh chạy | Endpoint | Cần vLLM |
|---|---|---|---|
| Tổng (OCR + extract) | `./start_api.sh` (cổng mặc định 9006, đổi bằng `PORT=...`) | `POST /api/v2/ocr/extract_local`, `GET /health` | OCR **và** LLM |
| OCR | `uvicorn src.ocr.app:app --port 9000` | `POST /api/v1/ocr`, `POST /api/v1/ocr/{run_id}/resume`, `GET /health` | OCR |
| Extract | `uvicorn src.extraction.app:app --port 9001` | `POST /api/v1/extract`, `GET /health` | LLM |

Swagger UI có sẵn ở `http://localhost:<port>/docs` của từng service.

### 2.1. `POST /api/v2/ocr/extract_local` — tải file → OCR → extract

Body JSON theo định dạng Gemini-OCR v2:

| Trường | Kiểu | Mặc định | Ý nghĩa |
|---|---|---|---|
| `file_url` | string | **bắt buộc** | URL HTTP/HTTPS của file PDF/ảnh (tối đa 50MB) |
| `extraction_schemas` | list | `examples/schemas.json` | Schema các loại tài liệu, xem [mục 2.5](#25-định-dạng-extraction_schemas). Không gửi thì server đọc file mặc định, đọc lại ở mỗi request nên sửa file không cần restart |
| `extract_all_documents` | bool | `false` | `true`: trả về cả tài liệu không thuộc schema nào, kèm `suggested_document_code/name` |
| `extract_all_fields` | bool | `false` | `true`: trích xuất thêm các trường có trong tài liệu nhưng không khai báo trong schema |
| `model_name` | string | `LLM_MODEL` | Đổi model LLM (phải là tên đang serve trên vLLM LLM) |
| `temperature`, `top_p`, `top_k`, `max_output_tokens` | number | theo `config.yaml` | Tham số sinh của LLM |

Các key khác của Gemini (`include_bbox`, `api_key`...) bị bỏ qua.

```bash
curl -X POST localhost:9006/api/v2/ocr/extract_local -H 'Content-Type: application/json' \
  -d '{"file_url": "https://.../file.pdf", "extract_all_documents": false}'
```

Response:

```json
{
  "run_id": "20261102_144036_e77c1694",
  "ocr_run_id": "20261102_143950_1a2b3c4d",
  "failed_pages": [],
  "documents": [
    {
      "document_code": "claim_form",
      "document_name": "Giấy yêu cầu bồi thường",
      "start_page": 1,
      "end_page": 2,
      "extracted_data": [
        { "insured_name": "ĐÀO THỊ LIÊN", "date_of_birth": "19/06/1975", "total_claimed_amount": 2141468, "...": "..." }
      ]
    }
  ]
}
```

| Trường | Ý nghĩa |
|---|---|
| `run_id` | Thư mục `data/extraction/<run_id>/` (request, kết quả phase 1, response) |
| `ocr_run_id` | Thư mục `data/ocr/<ocr_run_id>/` (file gốc, OCR từng trang) |
| `failed_pages` | Trang OCR lỗi sau khi retry, **bị bỏ qua khi extract**. Có thể resume OCR run đó (mục 2.2) rồi gọi lại `/api/v1/extract` |
| `documents[]` | Mỗi tài liệu tìm thấy trong file. Một file có thể chứa nhiều tài liệu |
| `documents[].start_page`, `end_page` | Khoảng trang của tài liệu (đánh số từ 1) |
| `documents[].extracted_data` | List, mỗi phần tử là 1 bản thể của tài liệu (VD 2 CCCD trên cùng 1 trang thành 2 phần tử). Key là `field_key`, giá trị không đọc được là `null`; trường `array` là list object theo `child_schema` |
| `documents[].suggested_document_code/name` | Chỉ có với tài liệu không thuộc schema (khi `extract_all_documents=true`) |

Mã lỗi: `400` tải file lỗi hoặc URL sai, `422` body sai định dạng, `502` OCR lỗi tất cả các trang hoặc LLM trả JSON
không hợp lệ, `500` lỗi khác (chi tiết trong `detail` và `logs/pipeline_service.log`).

### 2.2. `POST /api/v1/ocr` và `POST /api/v1/ocr/{run_id}/resume` — chỉ OCR

Upload file dạng `multipart/form-data`, trường `file`. Engine theo `OCR_ENGINE`.

```bash
curl -X POST localhost:9000/api/v1/ocr -F 'file=@duong/dan/file.pdf'
curl -X POST localhost:9000/api/v1/ocr/<run_id>/resume      # OCR lại các trang lỗi, giữ trang đã xong
```

Response:

```json
{
  "run_id": "20261116_115158_41b956f8",
  "engine": "chandra",
  "model": "chandra",
  "file_name": "file.pdf",
  "status": "completed",
  "page_count": 2,
  "total_seconds": 45.3,
  "failed_pages": [],
  "pages": [
    { "page": 1, "markdown": "...", "seconds": 24.1, "finish_reason": "stop", "attempts": 1, "error": null }
  ]
}
```

- `status`: `completed` (đủ trang), `partial` (còn trang lỗi trong `failed_pages`, gọi resume).
- `pages[].finish_reason`: `stop` bình thường; `length` hết `max_tokens`; `repeat` model lặp vòng đã bị cắt.
  Trang `length`/`repeat` vẫn tính là thành công nhưng có thể thiếu phần cuối, nên kiểm tra lại.
- `pages[].error`: khác `null` khi trang lỗi sau khi retry.
- `pages` truyền thẳng sang `/api/v1/extract` được.

Mã lỗi: `400` file rỗng hoặc `run_id` sai, `404` không có run, `500` lỗi khác.

### 2.3. `POST /api/v1/extract` — chỉ extract từ markdown đã OCR

Body JSON gồm các trường như mục 2.1 (trừ `file_url`), thêm `pages`. Ở đây `extraction_schemas` là **bắt buộc**.

```json
{
  "pages": [{ "page": 1, "markdown": "..." }, { "page": 2, "markdown": "..." }],
  "extraction_schemas": [ ... ],
  "extract_all_documents": false
}
```

Response giống mục 2.1 nhưng không có `ocr_run_id`, `failed_pages`. Mã lỗi: `422` body sai, `502` LLM trả JSON
không hợp lệ, `500` lỗi khác.

### 2.4. `GET /health`

Service tổng luôn trả `healthy` khi còn chạy (model tắt không làm restart container). Trạng thái từng model nằm ở `models`:

```json
{
  "status": "healthy",
  "engine": "chandra",
  "ocr_model": "chandra",
  "llm_model": "nvidia/Gemma-4-26B-A4B-NVFP4",
  "models": {
    "ocr:chandra": { "status": "ready" },
    "llm": { "status": "down", "detail": "ConnectError: All connection attempts failed" }
  }
}
```

`status` của model: `unknown` (chưa check), `down` (kèm `detail` lý do), `warming`, `ready`.

Service tổng check `/health` của vLLM OCR và LLM mỗi `HEALTH_CHECK_INTERVAL` giây (mặc định 10), đồng thời kiểm tra
tên model có đúng với `--served-model-name` không. Model đang `ready` chỉ chuyển sang `down` sau
`HEALTH_CHECK_FAILURE_THRESHOLD` lượt check lỗi liên tiếp (mặc định 3), để 1 lượt timeout lẻ khi đường truyền nghẽn
không gây warmup thừa; mỗi lượt lỗi lẻ chỉ ghi warning `check failed (i/N)`. Model nào vừa sẵn sàng (lúc API khởi động, hoặc sau khi vLLM
được tắt/bật lại) thì được **tự warmup** bằng 1 request nhỏ. Hai model warmup lần lượt vì dùng chung GPU. Không cần
restart API khi bật lại vLLM. Warmup OCR ngay sau khi vLLM vừa khởi động có thể mất khoảng 1 phút; request gửi
vào lúc `warming` vẫn chạy được nhưng chậm. Warmup bằng tay: `python scripts/warmup.py`.

### 2.5. Định dạng `extraction_schemas`

```json
[
  {
    "document_code": "prescription",
    "document_name": "Đơn thuốc",
    "fields": [
      { "field_key": "patient_name", "data_type": "string", "field_name": "Họ tên bệnh nhân" },
      { "field_key": "follow_up_appointment", "data_type": "date", "description": "Ngày hẹn tái khám" },
      {
        "field_key": "prescribed_medications",
        "data_type": "array",
        "child_schema": [
          { "field_key": "medication_name", "data_type": "string" },
          { "field_key": "quantity", "data_type": "number" }
        ]
      }
    ]
  }
]
```

| Trường | Ý nghĩa |
|---|---|
| `document_code` | Mã loại tài liệu, chỉ gồm `a-z`, `0-9`, `_`; không trùng nhau |
| `document_name` | Tên tiếng Việt, giúp LLM phân loại |
| `fields[].field_key` | Key trong `extracted_data`; không trùng trong cùng schema |
| `fields[].data_type` | `string`, `number`, `boolean`, `date`, `array` (`array` bắt buộc có `child_schema`) |
| `fields[].field_name`, `description` | Gợi ý cho LLM về ý nghĩa trường (không bắt buộc) |
| `fields[].nullable`, `required` | Mặc định `true` |

`examples/schemas.json` có sẵn 12 loại: bảng kê chi phí, giấy ra viện, giấy yêu cầu bồi thường, báo cáo y tế,
hoá đơn GTGT, bảng kê chi phí của người yêu cầu, bảng kê viện phí, CCCD, bảng kê chứng từ, thẻ bảo hiểm,
phiếu khám bệnh, đơn thuốc.

### 2.6. Triển khai bằng Docker

Image chỉ chứa API tổng (`src.app`); vLLM OCR và LLM vẫn chạy riêng (container `vllm/vllm-openai` hoặc `vllm serve`).

```bash
docker compose up -d --build        # build + chạy, cổng 9006 (đổi bằng PORT=... trong shell)
docker compose logs -f api          # xem log
docker compose down                 # dừng
```

- Biến môi trường lấy từ `.env`. Riêng địa chỉ vLLM: trong container `localhost` là chính container, nên compose
  mặc định trỏ tới `host.docker.internal:8000` (chandra), `:8001` (dots.mocr), `:10100` (LLM) trên máy host.
  vLLM ở máy khác thì đặt `CHANDRA_BASE_URL_DOCKER`, `DOTS_MOCR_BASE_URL_DOCKER`, `LLM_BASE_URL_DOCKER`
  (VD `LLM_BASE_URL_DOCKER=http://10.0.0.5:10100/v1`).
- vLLM phải lắng nghe trên `0.0.0.0` (mặc định của `vllm serve` và `docker run -p`), không chỉ `127.0.0.1`.
  Nếu bật firewall (ufw) thì cho phép mạng docker truy cập các cổng đó.
- `data/`, `logs/` được mount ra thư mục project; `examples/` mount read-only để sửa schema mặc định không cần build lại.
  Sửa `config.yaml` hoặc code thì chạy lại `docker compose up -d --build`.

## 3. Chạy thử 1 file bằng script

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
python scripts/demo_extract.py data/ocr/<run_id>/result.json --schemas examples/schemas.json \
  --extract-all-documents --extract-all-fields                     # tùy chọn như API
```

Trang OCR lỗi được bỏ qua khi extract; resume OCR run đó trước nếu cần đủ trang.

## 4. Benchmark với Label Studio

Máy chỉ serve được 1 model mỗi lúc nên chạy 2 bước riêng. Mặc định các task chạy tuần tự (trong 1 task thì các
trang OCR song song), nên `latency_s` của mỗi task không bị task khác chen vào.

**Chạy nhiều task cùng lúc** để benchmark nhanh hơn: thêm `--concurrency N` cho `ls_ocr.py` và `ls_extract.py`.

- OCR: các trang của mọi task dùng chung `OCR_MAX_CONCURRENCY` slot (mặc định 8), nên server vLLM không bị gửi quá
  số request đó dù `N` lớn. File đa số 1–2 trang thì `--concurrency 8` là hợp lý.
- Extract: mỗi task tự gọi LLM song song tới `LLM_MAX_CONCURRENT_EXTRACTIONS` (3), nên tổng request LLM ≈ `N × 3`.
  Giữ `N` nhỏ (2–3) nếu vLLM LLM chạy với `--max-num-seqs` thấp.
- Đánh đổi: tổng thời gian giảm nhiều, nhưng `latency_s` của từng task **tăng** vì các task tranh GPU và chờ slot,
  nên không so sánh trực tiếp được với số đo khi chạy tuần tự. Cần đo latency chuẩn thì chạy `--concurrency 1`.

**Bước 1 – OCR tất cả task** (serve model OCR):

```bash
python scripts/ls_ocr.py --project-id 5 --out runs/p5_chandra.txt
python scripts/ls_ocr.py --project-id 5 --out runs/p5_chandra.txt --concurrency 8   # nhiều task cùng lúc
```

- Thêm `--task-id N` để chạy 1 task, `--engine dots_mocr` để đổi engine.
- Bị dừng giữa chừng → chạy lại **đúng lệnh cũ**: task xong bỏ qua, task lỗi trang tự resume.

**Bước 2 – Extract → prediction** (tắt OCR, serve LLM):

```bash
python scripts/ls_extract.py --runs runs/p5_chandra.txt \
  --request ../test/ocr_labelstudio_runner/request.json \
  --project-id 5 --model-version chandra_gemma4_v1
```

- Kết quả: `predictions_preview.json` (cùng định dạng `run.py` cũ), mỗi task có thêm
  `latency_s` = thời gian OCR (`total_seconds` trong `result.json`) + thời gian extract, và `cost_usd` = 0 (model local).
  Hai trường này chỉ có trong file preview, không upload lên Label Studio.
- Thêm `--upload` để đẩy prediction lên Label Studio (upload 2 lần sẽ bị trùng).

**Chạy 1 lệnh** (chỉ khi serve được **cả** vLLM OCR và vLLM LLM cùng lúc): nối bước 1 và 2,
bước 2 chỉ chạy khi bước 1 không lỗi.

```bash
python scripts/ls_ocr.py --project-id 5 --out runs/p5_chandra.txt && \
python scripts/ls_extract.py --runs runs/p5_chandra.txt \
  --request ../test/ocr_labelstudio_runner/request.json \
  --project-id 5 --model-version chandra_gemma4_v1
```

**Bước 3 – Chấm điểm** (script cũ):

```bash
cd ../test/ocr_labelstudio_runner
python evaluate.py --predictions /home/dangdh/work/ocr-local/predictions_preview.json
```

## 5. Kết quả lưu ở đâu

| Thư mục | Nội dung |
|---|---|
| `data/ocr/<run_id>/` | `input/` file gốc, `pages/page_XXXX.html` output thô + `page_XXXX.md` markdown từng trang, `result.json` |
| `data/extraction/<run_id>/` | `request.json`, `phase1.json` (phân loại), `response.json` |
| `runs/*.txt` | `task_id → ocr_run_id → status` của `ls_ocr.py` |
| `logs/` | Log của từng script/service |

## 6. Lưu ý

- `--project-id` ở bước OCR và bước extract phải giống nhau; mỗi project dùng 1 file `--out` riêng.
- Không xóa `data/ocr/` khi file `runs/*.txt` còn dùng.
- Sửa `config.yaml`/`.env` thì phải chạy lại script hoặc restart API; config chỉ được đọc lúc khởi động.
- Mỗi trang chandra mất ~20–30s; log `page N: sending request...` là đang chờ server, không phải treo.
- Request đầu tiên sau khi restart vLLM rất chậm (warmup).
- Warning `output truncated at max_tokens` hoặc `output still repeating`: model bị lặp vòng (vd. dòng trống của bảng,
  dòng `<<<<` của MRZ trên CCCD), trang đó có thể mất phần cuối nhưng vẫn báo `completed` → kiểm tra lại
  `pages/page_XXXX.md` của trang đó.

### Bảo vệ server vLLM dùng chung (chandra)

Client được thiết kế để không làm quá tải hay treo server vLLM chạy 24/7, kể cả khi ngắt kết nối không hủy được
request trên server (VD đi qua SSH tunnel):

| Cơ chế | Cấu hình | Tác dụng |
|---|---|---|
| Giới hạn song song toàn process | `OCR_MAX_CONCURRENCY` (8) | Tổng request gửi tới vLLM không vượt ngưỡng, kể cả khi API nhận nhiều file cùng lúc |
| Chia nhỏ request | `CHANDRA_SEGMENT_TOKENS` (2048) | Mỗi request xin tối đa 2048 token, trang dài thì viết tiếp ở request sau; không request nào chiếm GPU/KV cache quá lâu |
| Không ngắt kết nối khi lặp | — | Phát hiện lặp thì đọc nốt (bỏ phần thừa) tới hết segment rồi mới gửi request viết tiếp, nên không có request "mồ côi" chạy song song |
| Phát hiện treo | `CHANDRA_STALL_TIMEOUT` (120s) | Quá thời gian này không nhận được token thì ngắt request |
| Ngắt mạch | `OCR_CIRCUIT_COOLDOWN` (120s) | Sau 1 request bị treo: không retry, ngừng gửi request mới trong 120s; các trang trong lúc đó báo lỗi ngay (`OCR server paused...`) để resume sau, không dồn request vào server đang kẹt |

Log `no token from server for 120s` hoặc `OCR server paused` nghĩa là vLLM đang quá tải hoặc bị treo, cần kiểm tra
phía server (`/metrics`: `num_requests_running`, `num_requests_waiting`). Sau khi server ổn, chạy lại lệnh cũ
(`ls_ocr.py`) hoặc gọi `/api/v1/ocr/{run_id}/resume` để OCR lại các trang lỗi.
