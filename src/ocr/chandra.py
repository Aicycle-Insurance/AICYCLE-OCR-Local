import asyncio
from typing import Any

from loguru import logger
from PIL import Image

from src.ocr.base import BaseOCR
from src.ocr.markdown import html_to_markdown
from src.prompts import CHANDRA_OCR_LAYOUT_PROMPT


def scale_to_fit(
    img: Image.Image,
    max_size: tuple[int, int] = (3072, 2048),
    min_size: tuple[int, int] = (1792, 28),
    grid_size: int = 28,
) -> Image.Image:
    """Scale ảnh về grid 28px trong khoảng pixel cho phép (chép từ chandra.model.util)."""
    width, height = img.size
    if width <= 0 or height <= 0:
        return img

    original_ar = width / height
    current_pixels = width * height
    max_pixels = max_size[0] * max_size[1]
    min_pixels = min_size[0] * min_size[1]

    scale = 1.0
    if current_pixels > max_pixels:
        scale = (max_pixels / current_pixels) ** 0.5
    elif current_pixels < min_pixels:
        scale = (min_pixels / current_pixels) ** 0.5

    w_blocks = max(1, round((width * scale) / grid_size))
    h_blocks = max(1, round((height * scale) / grid_size))

    # Bớt từng block cho tới khi dưới max_pixels, chọn chiều giữ tỉ lệ ảnh tốt hơn
    while (w_blocks * h_blocks * grid_size * grid_size) > max_pixels:
        if w_blocks == 1 and h_blocks == 1:
            break
        if w_blocks == 1:
            h_blocks -= 1
            continue
        if h_blocks == 1:
            w_blocks -= 1
            continue
        ar_w_loss = abs(((w_blocks - 1) / h_blocks) - original_ar)
        ar_h_loss = abs((w_blocks / (h_blocks - 1)) - original_ar)
        if ar_w_loss < ar_h_loss:
            w_blocks -= 1
        else:
            h_blocks -= 1

    new_size = (w_blocks * grid_size, h_blocks * grid_size)
    if new_size == (width, height):
        return img
    return img.resize(new_size, resample=Image.Resampling.LANCZOS)


def _max_repeats(seq_len: int, base: int, scaling_factor: float = 3.0) -> int:
    # Chuỗi càng ngắn càng cần lặp nhiều lần mới tính là lặp
    return int(base * (1 + scaling_factor / seq_len))


def _repeat_start(text: str, base: int, window: int = 500) -> int | None:
    """Vị trí bắt đầu đoạn lặp ở cuối ``text`` (giữ lại 1 lần), None nếu không lặp.

    Cùng tiêu chí với ``chandra.model.util.detect_repeat_token``.
    """
    for seq_len in range(1, window // 2 + 1):
        unit = text[-seq_len:]
        pos, count = len(text) - seq_len, 0
        if pos < 0:
            continue
        while pos >= 0 and text[pos : pos + seq_len] == unit:
            count += 1
            pos -= seq_len
        if count > _max_repeats(seq_len, base):
            return pos + 2 * seq_len
    return None


class ChandraOCR(BaseOCR):
    """datalab-to/chandra-ocr-2: output là HTML chia theo layout block.

    Model hay lặp vô hạn ở các hàng bảng rỗng: output được stream, phát hiện lặp thì
    ngắt request, cắt phần lặp, đóng bảng rồi cho model viết tiếp từ đó.
    """

    name = "chandra"
    prompt = CHANDRA_OCR_LAYOUT_PROMPT
    raw_suffix = ".html"

    # Temperature của từng lần cắt lặp rồi viết tiếp (số phần tử = số lần tối đa). Lần đầu
    # dùng config (greedy); viết tiếp từ cùng prefix mà vẫn greedy thì sẽ lặp y hệt.
    continue_temperatures = (0.2, 0.4, 0.6)
    # top_p của config (0.1) gần như luôn chỉ còn 1 token -> temperature vô tác dụng
    continue_top_p = 0.9
    repeat_base = 40  # ngưỡng cao hơn mặc định 4 để không bắt nhầm vài hàng rỗng hợp lệ
    check_every = 200  # cứ mỗi 200 ký tự mới thì kiểm tra lặp 1 lần

    def build_messages(self, image: Image.Image) -> list[dict[str, Any]]:
        return super().build_messages(scale_to_fit(image))

    def to_markdown(self, raw: str) -> str:
        return html_to_markdown(raw, heading_style="ATX")

    async def generate(self, image: Image.Image) -> tuple[str, str | None, dict[str, int]]:
        messages = self.build_messages(image)
        usage = {"input": 0, "output": 0, "total": 0}
        text, finish_reason = "", None
        max_continues = len(self.continue_temperatures)
        repeats = 0
        while True:
            # max_tokens là tổng cho cả trang, để prefix + phần viết tiếp không vượt context.
            # Mỗi request chỉ xin tối đa segment_tokens: request bị ngắt mà vLLM không hủy
            # (VD đi qua SSH tunnel) thì cũng chỉ chạy thừa tối đa chừng đó token.
            remaining = self.config.max_tokens - usage["output"]
            if remaining <= 0:
                finish_reason = "length"
                break
            params = {**self._model_parameters, "max_tokens": min(remaining, self.config.segment_tokens)}
            if repeats > 0:
                params["temperature"] = self.continue_temperatures[repeats - 1]
                params["top_p"] = self.continue_top_p
            text, finish_reason, step_usage = await self._stream_once(messages, text, params)
            for key in usage:
                usage[key] += step_usage[key]
            if finish_reason == "length":
                # Hết segment (hoặc hết tổng max_tokens -> vòng sau dừng): viết tiếp nguyên văn
                logger.info(f"[{self.name}] segment done at {len(text)} chars, continue")
                continue
            if finish_reason != "repeat":
                break

            start = _repeat_start(text, self.repeat_base)
            if start is not None:
                text = text[:start]
            if repeats == max_continues:
                break
            repeats += 1
            logger.warning(
                f"[{self.name}] repeat after {len(text)} chars, cut and continue "
                f"({repeats}/{max_continues}, "
                f"temperature={self.continue_temperatures[repeats - 1]})"
            )
            last_row = text.rfind("</tr>")
            if text.count("<table") > text.count("</table>") and last_row != -1:
                # Mở sẵn block tiếp theo, nếu không model coi như trang đã hết và dừng luôn
                text = (
                    text[: last_row + len("</tr>")]
                    + '</tbody></table></div><div data-bbox="'
                )
        return text, finish_reason, usage

    async def _stream_once(
        self, messages: list[dict[str, Any]], prefix: str, params: dict[str, Any]
    ) -> tuple[str, str | None, dict[str, int]]:
        """Stream 1 request (viết tiếp từ ``prefix`` nếu có); finish_reason = "repeat" khi
        phát hiện lặp (phần sinh sau đó bị bỏ).

        Quá ``stall_timeout`` giây không nhận được chunk nào (kể cả lúc chờ chunk đầu) thì
        ngắt request và raise ``TimeoutError`` để retry, không chờ tới ``timeout`` của client.
        """
        extra_body = None
        if prefix:
            messages = [*messages, {"role": "assistant", "content": prefix}]
            extra_body = {"continue_final_message": True, "add_generation_prompt": False}

        text, last_check, finish_reason = prefix, len(prefix), None
        chunks, usage, stream = 0, None, None
        # Phát hiện lặp thì không ngắt kết nối mà đọc nốt (bỏ phần thừa) tới hết segment: ngắt kết nối
        # không chắc vLLM hủy request (VD qua SSH tunnel) và request mới gửi ngay sẽ chạy song song
        # với request cũ. Đọc nốt thì số request mình chiếm trên server không bao giờ vượt concurrency.
        draining = False
        stall = self.config.stall_timeout
        loop = asyncio.get_running_loop()
        try:
            async with asyncio.timeout(stall) as deadline:
                stream = await self.client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    **params,
                    stream=True,
                    stream_options={"include_usage": True},
                    extra_body=extra_body,
                )
                async for chunk in stream:
                    deadline.reschedule(loop.time() + stall)
                    if chunk.usage:
                        usage = chunk.usage
                    if not chunk.choices:
                        continue
                    choice = chunk.choices[0]
                    if draining:
                        continue
                    if choice.delta.content:
                        text += choice.delta.content
                        chunks += 1
                    finish_reason = choice.finish_reason or finish_reason
                    if len(text) - last_check >= self.check_every:
                        last_check = len(text)
                        if _repeat_start(text, self.repeat_base) is not None:
                            finish_reason = "repeat"
                            draining = True
        except TimeoutError as e:
            if stream is not None:
                await stream.close()
            raise TimeoutError(
                f"no token from server for {stall:.0f}s after {len(text) - len(prefix)} new chars "
                "(vLLM overloaded or stuck?)"
            ) from e

        if usage:
            return text, finish_reason, {
                "input": usage.prompt_tokens,
                "output": usage.completion_tokens,
                "total": usage.total_tokens,
            }
        # Request bị ngắt giữa chừng thì không có usage, ước lượng 1 chunk ~ 1 token
        return text, finish_reason, {"input": 0, "output": chunks, "total": chunks}
