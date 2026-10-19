import asyncio
import base64
import io
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

import pymupdf
from loguru import logger
from openai import AsyncOpenAI, BadRequestError
from PIL import Image

from src.config import OCREngineConfig, RetryConfig, settings
from src.ocr.schemas import OCRPage, OCRResponse
from src.storage import new_run_dir, safe_filename, write_json, write_text
from src.tracing import trace_observation, trace_pipeline, update_observation


class BaseOCR(ABC):
    """OCR một file (PDF/ảnh) qua model serve bằng vLLM (OpenAI-compatible API).

    Mỗi trang là 1 request, chạy song song tối đa ``max_concurrency`` request,
    lỗi thì retry theo ``retry``. Trang nào xong là lưu ngay vào thư mục run,
    nên run bị dừng giữa chừng hoặc còn trang lỗi có thể ``resume``.
    Engine con chỉ cần khai báo ``prompt`` và cách chuyển output thô sang markdown.
    """

    name: ClassVar[str]
    prompt: ClassVar[str]
    raw_suffix: ClassVar[str] = ".txt"

    def __init__(
        self,
        config: OCREngineConfig,
        dpi: int,
        max_concurrency: int,
        retry: RetryConfig,
    ):
        self.config = config
        self.dpi = dpi
        self.max_concurrency = max_concurrency
        self.retry = retry
        # Retry do BaseOCR tự xử lý theo từng trang
        self.client = AsyncOpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.timeout,
            max_retries=0,
        )

    @abstractmethod
    def to_markdown(self, raw: str) -> str:
        """Chuyển output thô của model sang markdown."""

    def build_messages(self, image: Image.Image) -> list[dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": self._image_to_data_url(image)}},
                    {"type": "text", "text": self.prompt},
                ],
            }
        ]

    @staticmethod
    def _image_to_data_url(image: Image.Image) -> str:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

    def load_images(self, file_bytes: bytes) -> list[Image.Image]:
        """PDF → ảnh từng trang theo ``dpi``; ảnh thường → 1 trang."""
        if not file_bytes.startswith(b"%PDF"):
            with Image.open(io.BytesIO(file_bytes)) as source:
                return [source.convert("RGB")]

        images = []
        matrix = pymupdf.Matrix(self.dpi / 72, self.dpi / 72)
        with pymupdf.open(stream=file_bytes, filetype="pdf") as doc:
            for page in doc:
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                images.append(
                    Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                )
        return images

    @property
    def _model_parameters(self) -> dict[str, Any]:
        return {
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": self.config.max_tokens,
        }

    async def ocr_page(self, page: int, image: Image.Image) -> tuple[str, OCRPage]:
        """OCR 1 trang, trả về (output thô, kết quả markdown)."""
        with trace_observation(
            "ocr.page",
            as_type="generation",
            model=self.config.model,
            model_parameters=self._model_parameters,
            input={"page": page, "image_size": list(image.size), "prompt": self.prompt},
            metadata={"engine": self.name, "page": page},
        ) as generation:
            logger.info(f"[{self.name}] page {page}: sending request...")
            started = time.perf_counter()
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=self.build_messages(image),
                **self._model_parameters,
            )
            seconds = time.perf_counter() - started
            raw = response.choices[0].message.content or ""
            finish_reason = response.choices[0].finish_reason
            usage = response.usage
            update_observation(
                generation,
                output={"characters": len(raw), "finish_reason": finish_reason},
                usage_details={
                    "input": usage.prompt_tokens if usage else 0,
                    "output": usage.completion_tokens if usage else 0,
                    "total": usage.total_tokens if usage else 0,
                },
                level="WARNING" if finish_reason == "length" else None,
            )

        if finish_reason == "length":
            logger.warning(f"[{self.name}] page {page}: output truncated at max_tokens")
        logger.info(f"[{self.name}] page {page}: {seconds:.2f}s, {len(raw)} chars")
        return raw, OCRPage(
            page=page,
            markdown=self.to_markdown(raw),
            seconds=round(seconds, 3),
            finish_reason=finish_reason,
        )

    async def ocr_page_with_retry(
        self, page: int, image: Image.Image
    ) -> tuple[str | None, OCRPage]:
        """OCR 1 trang có retry; hết lượt thì trả về OCRPage có ``error`` (raw = None)."""
        delay = self.retry.initial_delay
        for attempt in range(1, self.retry.max_attempts + 1):
            try:
                raw, result = await self.ocr_page(page, image)
                result.attempts = attempt
                return raw, result
            except BadRequestError as e:
                # Request sai (ảnh quá lớn, vượt context...) thì retry cũng không khỏi
                error = e
                break
            except Exception as e:
                error = e
                if attempt < self.retry.max_attempts:
                    logger.warning(
                        f"[{self.name}] page {page}: attempt {attempt} failed "
                        f"({type(e).__name__}: {e}), retrying in {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, self.retry.max_delay)

        logger.error(f"[{self.name}] page {page}: failed after {attempt} attempt(s): {error}")
        return None, OCRPage(
            page=page, attempts=attempt, error=f"{type(error).__name__}: {error}"
        )

    async def run(self, file_bytes: bytes, file_name: str) -> OCRResponse:
        """OCR cả file, lưu input/output vào data/ocr/{run_id}/."""
        run_id, run_dir = new_run_dir("ocr")
        file_name = safe_filename(file_name)
        (run_dir / "input").mkdir()
        (run_dir / "input" / file_name).write_bytes(file_bytes)
        logger.info(f"[{self.name}] run {run_id}: OCR {file_name}")
        return await self._process(run_id, run_dir, file_bytes, file_name, done={})

    async def resume(self, run_id: str) -> OCRResponse:
        """OCR lại các trang bị lỗi hoặc chưa chạy xong của 1 run đã có."""
        run_dir = settings.app.data_dir / "ocr" / run_id
        result_path = run_dir / "result.json"
        if not result_path.is_file():
            raise FileNotFoundError(f"OCR run not found: {run_id}")
        previous = OCRResponse.model_validate_json(result_path.read_text(encoding="utf-8"))
        if previous.engine != self.name:
            raise ValueError(
                f"Run {run_id} was created with engine '{previous.engine}', "
                f"current engine is '{self.name}'"
            )
        done = {p.page: p for p in previous.pages if p.error is None}
        file_bytes = (run_dir / "input" / previous.file_name).read_bytes()
        logger.info(f"[{self.name}] run {run_id}: resume, {len(done)} page(s) already done")
        return await self._process(run_id, run_dir, file_bytes, previous.file_name, done)

    async def _process(
        self,
        run_id: str,
        run_dir: Path,
        file_bytes: bytes,
        file_name: str,
        done: dict[int, OCRPage],
    ) -> OCRResponse:
        """OCR các trang chưa có trong ``done``; mỗi trang xong là ghi file và cập nhật result.json."""
        started = time.perf_counter()
        pages = dict(done)

        def save_result(status: str) -> OCRResponse:
            ordered = [pages[page] for page in sorted(pages)]
            response = OCRResponse(
                run_id=run_id,
                engine=self.name,
                model=self.config.model,
                file_name=file_name,
                status=status,
                page_count=len(ordered),
                total_seconds=round(time.perf_counter() - started, 3),
                failed_pages=[p.page for p in ordered if p.error is not None],
                pages=ordered,
            )
            write_json(run_dir / "result.json", response.model_dump())
            return response

        with trace_pipeline(
            "ocr.run",
            input={
                "run_id": run_id,
                "engine": self.name,
                "file_name": file_name,
                "file_size_bytes": len(file_bytes),
                "resumed_pages": sorted(done),
            },
            metadata={"engine": self.name, "model": self.config.model},
        ) as trace:
            images = await asyncio.to_thread(self.load_images, file_bytes)
            todo = [(i, image) for i, image in enumerate(images, 1) if i not in pages]
            save_result("running")
            semaphore = asyncio.Semaphore(self.max_concurrency)

            async def ocr_one(page: int, image: Image.Image) -> None:
                async with semaphore:
                    raw, result = await self.ocr_page_with_retry(page, image)
                if raw is not None:
                    write_text(run_dir / "pages" / f"page_{page:04d}{self.raw_suffix}", raw)
                    write_text(run_dir / "pages" / f"page_{page:04d}.md", result.markdown)
                pages[page] = result
                save_result("running")

            await asyncio.gather(*(ocr_one(page, image) for page, image in todo))
            failed = [page for page, result in pages.items() if result.error is not None]
            response = save_result("partial" if failed else "completed")
            update_observation(
                trace,
                output={
                    "status": response.status,
                    "page_count": response.page_count,
                    "ocr_pages": len(todo),
                    "failed_pages": response.failed_pages,
                    "total_seconds": response.total_seconds,
                    "truncated_pages": [
                        p.page for p in response.pages if p.finish_reason == "length"
                    ],
                },
                level="WARNING" if failed else None,
            )

        logger.info(
            f"[{self.name}] run {run_id}: {response.status}, {len(todo)} page(s) OCR'd "
            f"in {response.total_seconds:.2f}s, failed pages: {response.failed_pages} -> {run_dir}"
        )
        return response
