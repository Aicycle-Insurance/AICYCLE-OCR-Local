"""Extraction 2 phase trên OCR markdown, port từ Gemini-OCR ``parse_with_schema_v2``.

Phase 1: phân đoạn + phân loại tài liệu theo trang (start_page/end_page).
Phase 2: trích xuất field theo schema cho từng tài liệu, chạy song song.
Mọi lời gọi LLM đều dùng structured output (JSON Schema) của vLLM.
"""

import asyncio
import json
import re
import time
import unicodedata
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Optional

from loguru import logger
from openai import AsyncOpenAI, BadRequestError
from pydantic import ValidationError

from src.config import settings
from src.extraction.json_schema import (
    build_phase1_response_schema,
    build_phase2_response_schema,
)
from src.extraction.schemas import (
    ExtractionSchema,
    ExtractRequest,
    ExtractResponse,
    FieldSchema,
)
from src.prompts import build_document_input, build_phase1_prompt, build_phase2_prompt
from src.storage import new_run_dir, write_json
from src.tracing import trace_observation, trace_pipeline, update_observation

Page = tuple[int, str]  # (page number, markdown)


@dataclass(frozen=True)
class GenerationParams:
    """Model and sampling options shared by every LLM call of one request."""

    model_name: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_output_tokens: Optional[int] = None


class ResponseTruncatedError(Exception):
    """Raised when a response was cut off by the output token limit."""

    def __init__(self, usage: dict[str, Any]):
        super().__init__("Response truncated at max_tokens")
        self.usage = usage


class InvalidModelResponseError(Exception):
    """Raised when the final result does not match the response model."""


class PromptTooLongError(Exception):
    """Raised when vLLM rejects a request whose pages exceed the model context length."""

    def __init__(self, message: str):
        super().__init__(message)
        self.usage = {"calls": 1}


# Lỗi cần chia nhỏ số trang rồi gọi lại (input quá context hoặc output bị cắt)
SPLITTABLE_ERRORS = (ResponseTruncatedError, PromptTooLongError)


def _is_context_overflow(error: BadRequestError) -> bool:
    # vLLM: "This model's maximum context length is ..." /
    # "The decoder prompt (length ...) is longer than the maximum model length ..."
    message = str(error).lower()
    return "context length" in message or "model length" in message


def parse_json(text: str) -> Any:
    """Parse JSON, chấp nhận cả trường hợp bị bọc trong ```json ... ```."""
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass
    logger.warning("Failed to parse JSON from LLM response")
    return text


class ExtractionService:
    def __init__(self) -> None:
        self.config = settings.llm
        self.client = AsyncOpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            timeout=self.config.timeout,
        )

    @staticmethod
    def _empty_usage() -> dict[str, int]:
        return {"calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    @classmethod
    def _sum_usage(cls, usages: list[dict[str, Any]]) -> dict[str, int]:
        total = cls._empty_usage()
        for usage in usages:
            for key in total:
                total[key] += int(usage.get(key, 0) or 0)
        return total

    async def _call_model(
        self,
        system_prompt: str,
        pages: list[Page],
        response_json_schema: dict[str, Any],
        params: GenerationParams,
        trace_name: str,
        trace_metadata: Optional[dict[str, Any]] = None,
    ) -> tuple[str, dict[str, Any]]:
        model = params.model_name or self.config.model
        model_parameters = {
            "temperature": params.temperature if params.temperature is not None else self.config.temperature,
            "top_p": params.top_p if params.top_p is not None else self.config.top_p,
            "max_tokens": params.max_output_tokens or self.config.max_tokens,
        }
        model_parameters = {k: v for k, v in model_parameters.items() if v is not None}
        extra_body = dict(self.config.extra_body)
        top_k = params.top_k if params.top_k is not None else self.config.top_k
        if top_k is not None:
            extra_body["top_k"] = top_k

        document_input = build_document_input(pages)
        with trace_observation(
            trace_name,
            as_type="generation",
            model=model,
            model_parameters={**model_parameters, **extra_body},
            input={
                "prompt": system_prompt,
                "pages": [page for page, _ in pages],
                "document_characters": len(document_input),
                "response_json_schema": response_json_schema,
            },
            metadata=trace_metadata,
        ) as generation:
            try:
                response = await self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": document_input},
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "response",
                            "schema": response_json_schema,
                            "strict": True,
                        },
                    },
                    extra_body=extra_body or None,
                    **model_parameters,
                )
            except BadRequestError as e:
                if _is_context_overflow(e):
                    raise PromptTooLongError(str(e)) from e
                raise
            text = response.choices[0].message.content or ""
            finish_reason = response.choices[0].finish_reason
            usage = response.usage
            usage_info = {
                "calls": 1,
                "finish_reason": finish_reason,
                "input_tokens": usage.prompt_tokens if usage else 0,
                "output_tokens": usage.completion_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0,
            }
            update_observation(
                generation,
                output={"response_characters": len(text), "finish_reason": finish_reason},
                usage_details={
                    "input": usage_info["input_tokens"],
                    "output": usage_info["output_tokens"],
                    "total": usage_info["total_tokens"],
                },
            )
        logger.info(
            f"{trace_name}: finish_reason={finish_reason}, "
            f"input={usage_info['input_tokens']:,}, output={usage_info['output_tokens']:,} tokens"
        )
        return text, usage_info

    async def _classify_and_segment(
        self,
        pages: list[Page],
        extraction_schemas: list[ExtractionSchema],
        extract_all_documents: bool,
        params: GenerationParams,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        text, usage_info = await self._call_model(
            build_phase1_prompt(extraction_schemas, extract_all_documents),
            pages,
            build_phase1_response_schema(extraction_schemas, extract_all_documents),
            params,
            trace_name="llm.classify_segment",
            trace_metadata={"phase": "phase1"},
        )
        result = parse_json(text)
        if not isinstance(result, dict):
            if usage_info.get("finish_reason") == "length":
                raise ResponseTruncatedError(usage_info)
            logger.warning("Phase 1 returned non-object payload. Marking file as invalid.")
            return {"documents": []}, usage_info
        if not isinstance(result.get("documents"), list):
            result["documents"] = []
        return result, usage_info

    @staticmethod
    def _same_document_type(a: dict[str, Any], b: dict[str, Any]) -> bool:
        code = a.get("document_code")
        if code != b.get("document_code"):
            return False
        return code != "unknown" or (
            a.get("suggested_document_code") == b.get("suggested_document_code")
        )

    @classmethod
    def _merge_split_segments(
        cls,
        left: list[dict[str, Any]],
        right: list[dict[str, Any]],
        boundary: tuple[int, int],
    ) -> list[dict[str, Any]]:
        """Join Phase-1 results of two adjacent page halves.

        A document ending on the last page of ``left`` and one starting on the
        first page of ``right`` with the same type is treated as one document cut
        by the split and merged. Same-page instance entries are never merged.
        """
        left_page, right_page = boundary
        tail = next((d for d in left if d.get("end_page") == left_page), None)
        head = next((d for d in right if d.get("start_page") == right_page), None)
        if (
            tail is not None
            and head is not None
            and cls._same_document_type(tail, head)
            and not tail.get("instances")
            and not head.get("instances")
        ):
            logger.info(
                f"Phase 1: merging {tail.get('document_code')} across split "
                f"(pages {tail.get('start_page')}-{head.get('end_page')})"
            )
            tail["end_page"] = head.get("end_page", right_page)
            right = [d for d in right if d is not head]
        return left + right

    async def _segment_pages(
        self,
        pages: list[Page],
        extraction_schemas: list[ExtractionSchema],
        extract_all_documents: bool,
        params: GenerationParams,
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        """Phase 1 on ``pages``; if the request is too long, split pages in half and merge."""
        try:
            result, usage = await self._classify_and_segment(
                pages, extraction_schemas, extract_all_documents, params
            )
            return result["documents"], self._sum_usage([usage])
        except SPLITTABLE_ERRORS as exc:
            if len(pages) == 1:
                raise
            failed_usage = exc.usage

        middle = len(pages) // 2
        logger.warning(
            "Phase 1 request too long for pages {}-{}; retrying as {}-{} + {}-{}",
            pages[0][0], pages[-1][0],
            pages[0][0], pages[middle - 1][0], pages[middle][0], pages[-1][0],
        )
        (left, left_usage), (right, right_usage) = await asyncio.gather(
            self._segment_pages(pages[:middle], extraction_schemas, extract_all_documents, params),
            self._segment_pages(pages[middle:], extraction_schemas, extract_all_documents, params),
        )
        return (
            self._merge_split_segments(left, right, (pages[middle - 1][0], pages[middle][0])),
            self._sum_usage([failed_usage, left_usage, right_usage]),
        )

    async def _extract_segment(
        self,
        pages: list[Page],
        schema: Optional[ExtractionSchema],
        extract_all_fields: bool,
        page_chunk: Optional[tuple[int, int]],
        params: GenerationParams,
        instance_hint: Optional[dict[str, Any]] = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        prompt = build_phase2_prompt(schema, extract_all_fields, page_chunk, instance_hint)
        response_json_schema = build_phase2_response_schema(schema, extract_all_fields)
        trace_metadata = {
            "phase": "phase2",
            "document_code": schema.document_code if schema else "unknown",
            "source_page_numbers": [page for page, _ in pages],
            "schema_field_count": len(schema.fields) if schema else 0,
            "has_instance_hint": instance_hint is not None,
        }

        usages = []
        for attempt in (1, 2):
            text, usage_info = await self._call_model(
                prompt,
                pages,
                response_json_schema,
                params,
                trace_name=f"llm.extract.attempt_{attempt}",
                trace_metadata={**trace_metadata, "attempt": attempt},
            )
            usages.append(usage_info)
            result = parse_json(text)
            if isinstance(result, dict):
                return result, self._sum_usage(usages)
            # A cut-off response would be cut off again on retry; let the caller split it.
            if usage_info.get("finish_reason") == "length":
                raise ResponseTruncatedError(self._sum_usage(usages))
            if attempt == 1:
                logger.warning(
                    f"Extraction returned {type(result).__name__} instead of dict. Retrying once."
                )
        raise ValueError(f"Phase 2 response is not a JSON object (got {type(result).__name__})")

    @staticmethod
    def _needs_page_chunking(schema: Optional[ExtractionSchema], page_count: int) -> bool:
        """Long table-bearing documents overflow one response, so split them by page."""
        chunk_pages = settings.llm.table_chunk_pages
        if chunk_pages <= 0 or page_count <= chunk_pages:
            return False
        return schema is None or any(f.data_type == "array" for f in schema.fields)

    @staticmethod
    def _merge_chunk_results(results: list[dict[str, Any]]) -> dict[str, Any]:
        """Join per-chunk extractions in page order.

        Lists (table rows) are appended and scalar fields keep their first non-null
        value. Unknown-document tables are merged by ``table_key`` and fields are
        deduplicated by ``field_key``.
        """
        merged: dict[str, Any] = {}
        for result in results:
            if not isinstance(result, dict):
                continue
            for key, value in result.items():
                current = merged.get(key)
                if isinstance(value, list) and isinstance(current, list):
                    current.extend(value)
                elif current is None:
                    merged[key] = value

        tables: dict[Any, dict[str, Any]] = {}
        for table in merged.get("tables") or []:
            first = tables.setdefault(table.get("table_key"), table)
            if first is not table:
                first.setdefault("rows", []).extend(table.get("rows") or [])
        if "tables" in merged:
            merged["tables"] = list(tables.values())

        fields: dict[Any, dict[str, Any]] = {}
        for field in merged.get("fields") or []:
            fields.setdefault(field.get("field_key"), field)
        if "fields" in merged:
            merged["fields"] = list(fields.values())
        return merged

    async def _extract_document_pages(
        self,
        pages: list[Page],
        schema: Optional[ExtractionSchema],
        extract_all_fields: bool,
        semaphore: asyncio.Semaphore,
        params: GenerationParams,
        instance_hint: Optional[dict[str, Any]] = None,
    ) -> tuple[dict[str, Any], dict[str, int]]:
        """Extract one document, running long table documents as parallel page chunks.

        A request whose pages exceed the context length, or whose response is cut
        off by the output token limit, is split in half by page and retried; the
        halves are merged, so a long document converges instead of failing.
        """
        total = len(pages)

        async def extract_span(start: int, end: int) -> tuple[dict[str, Any], dict[str, int]]:
            partial_span = (start, end) != (1, total)
            async with semaphore:
                try:
                    return await self._extract_segment(
                        pages[start - 1 : end],
                        schema,
                        extract_all_fields,
                        page_chunk=(end - start + 1, total) if partial_span else None,
                        params=params,
                        instance_hint=instance_hint,
                    )
                except SPLITTABLE_ERRORS as exc:
                    if end == start:
                        raise
                    truncated_usage = exc.usage

            middle = (start + end) // 2
            logger.warning(
                "Phase 2 request too long for pages {}-{} of {}; retrying as {}-{} + {}-{}",
                start, end, total, start, middle, middle + 1, end,
            )
            halves = await asyncio.gather(
                extract_span(start, middle), extract_span(middle + 1, end)
            )
            return (
                self._merge_chunk_results([half for half, _ in halves]),
                self._sum_usage([truncated_usage, *(usage for _, usage in halves)]),
            )

        size = settings.llm.table_chunk_pages
        spans = (
            [(i, min(i + size - 1, total)) for i in range(1, total + 1, size)]
            if self._needs_page_chunking(schema, total)
            else [(1, total)]
        )
        if len(spans) > 1:
            logger.info(f"Phase 2: Splitting {total} pages into {len(spans)} chunks of up to {size} pages")
        results = await asyncio.gather(*(extract_span(*span) for span in spans))
        if len(results) == 1:
            return results[0]
        return (
            self._merge_chunk_results([result for result, _ in results]),
            self._sum_usage([usage for _, usage in results]),
        )

    @staticmethod
    def _document_pages(doc: dict[str, Any], pages_by_number: dict[int, str]) -> list[Page]:
        """OCR pages covered by one Phase-1 document's start_page..end_page."""
        start = int(doc.get("start_page", 1))
        end = int(doc.get("end_page", start))
        start, end = min(start, end), max(start, end)
        doc["start_page"], doc["end_page"] = start, end
        return [(page, md) for page, md in pages_by_number.items() if start <= page <= end]

    @staticmethod
    def _merge_same_page_instances(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Merge Phase-1 entries of the same code on the same page.

        Entries without ``instances`` (regular range-based documents) pass through.
        """
        merged: list[dict[str, Any]] = []
        positions: dict[tuple[str, int], int] = {}
        for doc in docs:
            instances = doc.get("instances")
            start_page = doc.get("start_page", 1)
            if (
                not isinstance(instances, list)
                or not instances
                or start_page != doc.get("end_page", start_page)
            ):
                merged.append(doc)
                continue
            key = (str(doc.get("document_code", "unknown")), int(start_page))
            if key in positions:
                merged[positions[key]]["instances"].extend(instances)
            else:
                positions[key] = len(merged)
                merged.append(doc)
        return merged

    @staticmethod
    def _normalize_card_name(name: Any) -> str:
        """Normalize a printed or MRZ holder name for front/back matching."""
        if not isinstance(name, str):
            return ""
        text = name.replace("Đ", "D").replace("đ", "d").replace("<", " ")
        text = unicodedata.normalize("NFD", text)
        text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
        return " ".join(text.upper().split())

    @staticmethod
    def _single_card_side(doc: dict[str, Any]) -> Optional[str]:
        """Return ``front``/``back`` for a one-page entry holding one one-sided card."""
        instances = doc.get("instances")
        start_page = doc.get("start_page", 1)
        if (
            not isinstance(instances, list)
            or len(instances) != 1
            or not isinstance(instances[0], dict)
            or start_page != doc.get("end_page", start_page)
        ):
            return None
        sides = set(instances[0].get("sides") or [])
        if len(sides) == 1 and sides <= {"front", "back"}:
            return sides.pop()
        return None

    @classmethod
    def _merge_adjacent_card_sides(cls, docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Merge a front-only card page with a back-only card page next to it.

        Both entries must share a code and hold exactly one instance each. A pair is
        skipped when both holder names are readable and share no word. Everything
        else passes through unchanged.
        """
        merged: list[dict[str, Any]] = []
        used: set[int] = set()
        for i, doc in enumerate(docs):
            if i in used:
                continue
            side = cls._single_card_side(doc)
            if side is None:
                merged.append(doc)
                continue
            for j, other in enumerate(docs):
                if j == i or j in used:
                    continue
                other_side = cls._single_card_side(other)
                if (
                    other_side is None
                    or other_side == side
                    or other.get("document_code") != doc.get("document_code")
                    or abs(int(other.get("start_page", 1)) - int(doc.get("start_page", 1))) != 1
                ):
                    continue
                front, back = (doc, other) if side == "front" else (other, doc)
                front_instance = front["instances"][0]
                back_instance = back["instances"][0]
                front_name = cls._normalize_card_name(front_instance.get("front_name"))
                mrz_name = cls._normalize_card_name(back_instance.get("mrz_name"))
                # MRZ lines are often misread or cut off, so only reject names
                # that share no word at all.
                if front_name and mrz_name and not set(front_name.split()) & set(mrz_name.split()):
                    continue
                used.add(j)
                pages = [int(doc.get("start_page", 1)), int(other.get("start_page", 1))]
                doc = {
                    **front,
                    "start_page": min(pages),
                    "end_page": max(pages),
                    "instances": [
                        {
                            "identifier": front_instance.get("identifier"),
                            "front_name": front_instance.get("front_name"),
                            "mrz_name": back_instance.get("mrz_name"),
                            "sides": ["front", "back"],
                        }
                    ],
                }
                logger.info(
                    f"Merged card front (page {front.get('start_page')}) with back "
                    f"(page {back.get('start_page')}) for {doc.get('document_code')}"
                )
                break
            merged.append(doc)
        return merged

    @classmethod
    def _wrap_unlocated(cls, value: Any, field: Optional[FieldSchema]) -> Any:
        """Wrap a plain extracted value as ``{"value": ..., "coordinates": None}``."""
        if field is not None and field.data_type == "array":
            if not isinstance(value, list):
                return value
            children = {child.field_key: child for child in field.child_schema or []}
            return [
                {key: cls._wrap_unlocated(item, children.get(key)) for key, item in row.items()}
                if isinstance(row, dict)
                else row
                for row in value
            ]
        return {"value": value, "coordinates": None}

    @classmethod
    def _add_null_coordinates(
        cls, result: dict[str, Any], schema: Optional[ExtractionSchema]
    ) -> dict[str, Any]:
        """Give values the same shape as Gemini-OCR located ones, with null coordinates."""
        if schema is None:
            for field in result.get("fields") or []:
                field["coordinates"] = None
            for table in result.get("tables") or []:
                for column in table.get("columns") or []:
                    column["coordinates"] = None
                table["rows"] = [
                    {key: cls._wrap_unlocated(cell, None) for key, cell in row.items()}
                    for row in table.get("rows") or []
                ]
            return result
        fields = {field.field_key: field for field in schema.fields}
        return {
            key: cls._wrap_unlocated(value, fields.get(key))
            for key, value in result.items()
        }

    async def _process_doc(
        self,
        doc: dict[str, Any],
        *,
        schema_map: dict[str, ExtractionSchema],
        pages_by_number: dict[int, str],
        extract_all_fields: bool,
        params: GenerationParams,
        semaphore: asyncio.Semaphore,
    ) -> dict[str, int]:
        """Phase 2 for one Phase-1 document, one extraction request per instance.

        ``extracted_data`` is a list with one extracted object per instance.
        """
        doc_code = doc.get("document_code", "unknown")
        with trace_observation(
            "document.extract", metadata={"document_code": doc_code}
        ) as document_observation:
            try:
                schema = schema_map.get(doc_code)
                instances = doc.get("instances") or []
                pages = self._document_pages(doc, pages_by_number)
                if not pages:
                    raise ValueError(
                        f"No OCR pages in range {doc['start_page']}-{doc['end_page']}"
                    )
                logger.info(
                    f"Phase 2: Extracting {doc_code} block (Pages {doc['start_page']}-{doc['end_page']}, "
                    f"{max(len(instances), 1)} instance(s))"
                )
                # Same-code documents sharing a page are extracted one request per
                # instance (in parallel); a lone document needs no instance hint.
                hints = instances if len(instances) > 1 else [None]
                update_observation(
                    document_observation,
                    input={
                        "document_code": doc_code,
                        "source_page_numbers": [page for page, _ in pages],
                        "instances": instances,
                        "extract_all_fields": extract_all_fields,
                    },
                )

                async def extract_one(
                    hint: Optional[dict[str, Any]],
                ) -> tuple[dict[str, Any], dict[str, int]]:
                    try:
                        result, usage = await self._extract_document_pages(
                            pages, schema, extract_all_fields, semaphore, params,
                            instance_hint=hint,
                        )
                        # Wrapped after chunk merging, which relies on plain null values.
                        return self._add_null_coordinates(result, schema), usage
                    except Exception as e:
                        logger.error(f"Phase 2 extraction failed for {doc_code}: {e}")
                        return (
                            {"error": str(e), "error_code": "EXTRACTION_FAILED"},
                            self._empty_usage(),
                        )

                results = await asyncio.gather(*(extract_one(hint) for hint in hints))
                doc["extracted_data"] = [item for item, _ in results]
                usage = self._sum_usage([item_usage for _, item_usage in results])
                errors = [item["error"] for item, _ in results if "error_code" in item]
                update_observation(
                    document_observation,
                    output={"success": not errors, "instances": len(hints), **usage},
                    level="ERROR" if errors else None,
                    status_message=errors[0] if errors else None,
                )
                return usage
            except Exception as e:
                logger.error(f"Phase 2 extraction failed for {doc_code}: {e}")
                update_observation(
                    document_observation,
                    output={"success": False, "error_type": type(e).__name__},
                    level="ERROR",
                    status_message=str(e),
                )
                doc["extracted_data"] = [{"error": str(e), "error_code": "EXTRACTION_FAILED"}]
                return self._empty_usage()

    async def parse_with_schema(self, request: ExtractRequest, run_dir: Path) -> dict[str, Any]:
        """Phase 1 classify/segment the pages, then Phase 2 extract each document concurrently."""
        params = GenerationParams(
            model_name=request.model_name,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            max_output_tokens=request.max_output_tokens,
        )
        pages = [(page.page, page.markdown) for page in request.pages]
        pages_by_number = dict(pages)
        schema_codes = [schema.document_code for schema in request.extraction_schemas]

        logger.info(f"Phase 1: Segmenting {len(pages)} page(s)")
        phase1_started = time.perf_counter()
        with trace_observation(
            "phase1.classify_segment",
            input={
                "page_count": len(pages),
                "schema_codes": schema_codes,
                "extract_all_documents": request.extract_all_documents,
            },
        ) as phase1:
            docs, phase1_usage = await self._segment_pages(
                pages, request.extraction_schemas, request.extract_all_documents, params
            )
            update_observation(phase1, output={"document_count": len(docs), **phase1_usage})
        write_json(run_dir / "phase1.json", {"documents": docs})
        phase1_seconds = time.perf_counter() - phase1_started
        logger.info(
            f"Phase 1 done: {phase1_seconds:.2f}s, {phase1_usage['calls']} LLM call(s), "
            f"{len(docs)} segment(s)"
        )

        docs = self._merge_adjacent_card_sides(self._merge_same_page_instances(docs))
        if not docs:
            return {"documents": []}

        semaphore = asyncio.Semaphore(settings.llm.max_concurrent_extractions)
        process_doc = partial(
            self._process_doc,
            schema_map={s.document_code: s for s in request.extraction_schemas},
            pages_by_number=pages_by_number,
            extract_all_fields=request.extract_all_fields,
            params=params,
            semaphore=semaphore,
        )

        logger.info(f"Phase 2: Starting concurrent extraction for {len(docs)} documents")
        phase2_started = time.perf_counter()
        with trace_observation(
            "phase2.extract_documents",
            input={
                "document_count": len(docs),
                "document_codes": [doc.get("document_code", "unknown") for doc in docs],
                "max_concurrency": settings.llm.max_concurrent_extractions,
            },
        ) as phase2:
            phase2_usage = self._sum_usage(
                await asyncio.gather(*[process_doc(doc) for doc in docs])
            )
            update_observation(phase2, output={"document_count": len(docs), **phase2_usage})

        phase2_seconds = time.perf_counter() - phase2_started
        pipeline_usage = self._sum_usage([phase1_usage, phase2_usage])
        logger.info(
            f"Phase 2 done: {phase2_seconds:.2f}s | Pipeline total: "
            f"{phase1_seconds + phase2_seconds:.2f}s, {pipeline_usage['calls']} LLM call(s), "
            f"Input: {pipeline_usage['input_tokens']:,}, Output: {pipeline_usage['output_tokens']:,} tokens"
        )
        for doc in docs:
            doc.pop("instances", None)
        return {"documents": docs}

    async def run(self, request: ExtractRequest) -> ExtractResponse:
        """Chạy extraction cho 1 request, lưu request/kết quả vào data/extraction/{run_id}/."""
        started_at = time.perf_counter()
        run_id, run_dir = new_run_dir("extraction")
        write_json(run_dir / "request.json", request.model_dump())
        logger.info(f"Extract request {run_id} started ({len(request.pages)} page(s))")

        with trace_pipeline(
            "extraction.extract",
            input={
                "run_id": run_id,
                "page_count": len(request.pages),
                "requested_model": request.model_name or settings.llm.model,
                "schema_codes": [s.document_code for s in request.extraction_schemas],
                "extraction_options": {
                    "extract_all_fields": request.extract_all_fields,
                    "extract_all_documents": request.extract_all_documents,
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                    "top_k": request.top_k,
                    "max_output_tokens": request.max_output_tokens,
                },
            },
        ) as trace:
            result = await self.parse_with_schema(request, run_dir)
            try:
                response = ExtractResponse.model_validate({"run_id": run_id, **result})
            except ValidationError as e:
                logger.error(f"Invalid response payload from model: {e}")
                write_json(run_dir / "invalid_result.json", result)
                raise InvalidModelResponseError(
                    "Model returned an invalid response payload. Please retry."
                ) from e
            update_observation(
                trace,
                output={
                    "document_count": len(response.documents),
                    "document_codes": [doc.document_code for doc in response.documents],
                },
            )

        write_json(run_dir / "response.json", response.model_dump(exclude_none=True))
        logger.info(
            "Extract request {} completed in {:.4f} seconds",
            run_id,
            time.perf_counter() - started_at,
        )
        return response
