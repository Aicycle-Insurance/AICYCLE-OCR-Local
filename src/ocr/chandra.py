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
        for attempt in range(max_continues + 1):
            # max_tokens là tổng cho cả trang, để prefix + phần viết tiếp không vượt context
            remaining = self.config.max_tokens - usage["output"]
            if remaining <= 0:
                finish_reason = "length"
                break
            params = {**self._model_parameters, "max_tokens": remaining}
            if attempt > 0:
                params["temperature"] = self.continue_temperatures[attempt - 1]
                params["top_p"] = self.continue_top_p
            text, finish_reason, step_usage = await self._stream_once(messages, text, params)
            for key in usage:
                usage[key] += step_usage[key]
            if finish_reason != "repeat":
                break

            start = _repeat_start(text, self.repeat_base)
            if start is not None:
                text = text[:start]
            if attempt == max_continues:
                break
            logger.warning(
                f"[{self.name}] repeat after {len(text)} chars, cut and continue "
                f"({attempt + 1}/{max_continues}, "
                f"temperature={self.continue_temperatures[attempt]})"
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
        phát hiện lặp và đã ngắt request."""
        extra_body = None
        if prefix:
            messages = [*messages, {"role": "assistant", "content": prefix}]
            extra_body = {"continue_final_message": True, "add_generation_prompt": False}
        stream = await self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            **params,
            stream=True,
            stream_options={"include_usage": True},
            extra_body=extra_body,
        )

        text, last_check, finish_reason = prefix, len(prefix), None
        chunks, usage = 0, None
        async for chunk in stream:
            if chunk.usage:
                usage = chunk.usage
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            if choice.delta.content:
                text += choice.delta.content
                chunks += 1
            finish_reason = choice.finish_reason or finish_reason
            if len(text) - last_check >= self.check_every:
                last_check = len(text)
                if _repeat_start(text, self.repeat_base) is not None:
                    await stream.close()  # ngắt kết nối -> vLLM hủy request, giải phóng GPU
                    finish_reason = "repeat"
                    break

        if usage:
            return text, finish_reason, {
                "input": usage.prompt_tokens,
                "output": usage.completion_tokens,
                "total": usage.total_tokens,
            }
        # Request bị ngắt giữa chừng thì không có usage, ước lượng 1 chunk ~ 1 token
        return text, finish_reason, {"input": 0, "output": chunks, "total": chunks}
