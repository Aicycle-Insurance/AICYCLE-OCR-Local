"""Warmup vLLM OCR + LLM và theo dõi health để tự warmup lại khi model vừa bật.

Request đầu tiên sau khi vLLM khởi động rất chậm (compile grammar JSON, cache...),
nên mỗi model được gửi 1 request nhỏ ngay khi nó sẵn sàng. ``ModelWatcher`` check
``/health`` định kỳ và warmup ở lượt OK đầu tiên: lúc server mới chạy hoặc ngay sau
một lượt check lỗi (model vừa được tắt/bật lại).
"""

import asyncio
import json
import time
from typing import Awaitable, Callable, Literal, Optional

import httpx
from loguru import logger
from PIL import Image, ImageDraw

from src.extraction.json_schema import build_phase1_response_schema
from src.extraction.schemas import DEFAULT_SCHEMAS_PATH, ExtractionSchema
from src.extraction.service import ExtractionService, GenerationParams
from src.ocr.base import BaseOCR
from src.prompts import build_phase1_prompt

WARMUP_MAX_TOKENS = 256
WARMUP_TEXT = "GIAY RA VIEN\nHo ten: NGUYEN VAN A\nNgay vao vien: 01/01/2026"

ModelStatus = Literal["unknown", "down", "warming", "ready"]


async def warmup_ocr(engine: BaseOCR) -> None:
    """Gửi 1 ảnh nhỏ có chữ tới vLLM OCR."""
    image = Image.new("RGB", (800, 400), "white")
    ImageDraw.Draw(image).multiline_text((40, 40), WARMUP_TEXT, fill="black", spacing=20)
    started = time.perf_counter()
    await engine.client.chat.completions.create(
        model=engine.config.model,
        messages=engine.build_messages(image),
        temperature=engine.config.temperature,
        max_tokens=WARMUP_MAX_TOKENS,
    )
    logger.info(f"Warmup OCR ({engine.name}, {engine.config.model}): {time.perf_counter() - started:.2f}s")


async def warmup_llm(service: ExtractionService) -> None:
    """Gửi 1 request Phase 1 với schema mặc định để vLLM compile sẵn grammar JSON."""
    schemas = [
        ExtractionSchema.model_validate(item)
        for item in json.loads(DEFAULT_SCHEMAS_PATH.read_text(encoding="utf-8"))
    ]
    started = time.perf_counter()
    await service._call_model(
        build_phase1_prompt(schemas, False),
        [(1, WARMUP_TEXT)],
        build_phase1_response_schema(schemas, False),
        GenerationParams(max_output_tokens=WARMUP_MAX_TOKENS),
        trace_name="warmup.phase1",
    )
    logger.info(f"Warmup LLM ({service.config.model}): {time.perf_counter() - started:.2f}s")


class ModelWatcher:
    """Check health 1 model vLLM định kỳ, warmup khi chuyển từ lỗi/chưa biết sang OK.

    Trạng thái: ``unknown`` (chưa check) -> ``down`` | ``warming`` -> ``ready``.
    Warmup lỗi thì về ``down`` để lượt check OK sau warmup lại.
    """

    def __init__(
        self,
        name: str,
        base_url: str,
        model: str,
        api_key: str,
        warmup: Callable[[], Awaitable[None]],
        http: httpx.AsyncClient,
        lock: asyncio.Lock,
    ):
        self.name = name
        # /health nằm ở gốc server, không nằm dưới /v1
        self.root_url = base_url.rstrip("/").removesuffix("/v1")
        self.models_url = base_url.rstrip("/") + "/models"
        self.model = model
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.warmup = warmup
        self.http = http
        self.lock = lock
        self.status: ModelStatus = "unknown"
        self.detail: Optional[str] = None

    async def check(self) -> Optional[str]:
        """Trả về None nếu model sẵn sàng, ngược lại là lý do lỗi."""
        try:
            response = await self.http.get(f"{self.root_url}/health")
            if response.status_code != 200:
                return f"/health returned {response.status_code}"
            # Bắt trường hợp bật nhầm model hoặc sai --served-model-name
            response = await self.http.get(self.models_url, headers=self.headers)
            if response.status_code != 200:
                return f"/v1/models returned {response.status_code}"
            served = [item.get("id") for item in response.json().get("data", [])]
            if self.model not in served:
                return f"model '{self.model}' not served (served: {served})"
        except (httpx.HTTPError, ValueError) as e:
            return f"{type(e).__name__}: {e}"
        return None

    def _set(self, status: ModelStatus, detail: Optional[str] = None) -> None:
        if status != self.status:
            message = f"[{self.name}] {self.status} -> {status}" + (f": {detail}" if detail else "")
            if status == "down":
                logger.warning(message)
            else:
                logger.info(message)
        self.status, self.detail = status, detail

    async def run(self, interval: float, failure_threshold: int = 1) -> None:
        failures = 0
        while True:
            error = await self.check()
            if error is not None:
                failures += 1
                # Đang ready thì chịu được vài lượt lỗi lẻ; chưa ready thì báo down ngay
                if self.status == "ready" and failures < failure_threshold:
                    logger.warning(f"[{self.name}] check failed ({failures}/{failure_threshold}): {error}")
                else:
                    self._set("down", error)
                await asyncio.sleep(interval)
                continue
            failures = 0
            if self.status in ("unknown", "down"):
                self._set("warming")
                try:
                    # Các model dùng chung GPU nên warmup lần lượt
                    async with self.lock:
                        await self.warmup()
                    self._set("ready")
                except Exception as e:
                    self._set("down", f"warmup failed: {type(e).__name__}: {e}")
            await asyncio.sleep(interval)
