"""Request/response models, port từ Gemini-OCR /api/v2/ocr/extract (bỏ bbox)."""

import json
import re
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from src.config import PROJECT_ROOT

# Schema mặc định của endpoint file_url khi body không gửi extraction_schemas
DEFAULT_SCHEMAS_PATH = PROJECT_ROOT / "examples" / "schemas.json"


# Request models
class FieldSchema(BaseModel):
    """Defines a single field to extract from a document.

    Attributes:
        field_key: Unique key (English) used in the extracted_data output.
        data_type: Data type of the field.
        field_name: Optional extraction hint (Vietnamese/English) to disambiguate field meaning.
        description: Detailed instruction for the LLM on how to extract this specific field.
        nullable: If True, the field value can be null in the output.
        required: If True, the LLM must include this field key in the output.
        child_schema: For array fields, defines the structure of each item.
    """
    field_key: str
    data_type: Literal["string", "number", "boolean", "date", "array"]
    field_name: Optional[str] = None
    description: Optional[str] = None
    nullable: bool = True
    required: bool = True
    child_schema: Optional[List["FieldSchema"]] = None

    @field_validator("field_key")
    @classmethod
    def validate_field_key(cls, v: str) -> str:
        field_key = v.strip()
        if not field_key:
            raise ValueError("field_key must not be empty")
        return field_key

    @field_validator("field_name", "description")
    @classmethod
    def validate_optional_strings(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        return v.strip() or None

    @model_validator(mode="after")
    def validate_array_shape(self):
        if self.data_type == "array":
            if not self.child_schema:
                raise ValueError("child_schema is required when data_type is 'array'")
        elif self.child_schema is not None:
            raise ValueError("child_schema is only allowed when data_type is 'array'")
        return self


class ExtractionSchema(BaseModel):
    """Defines the extraction schema for one document type.

    Attributes:
        document_code: Unique code identifier for the document type.
        document_name: Human-readable name (Vietnamese) of the document type.
        fields: List of fields to extract from this document type.
    """
    document_code: str
    document_name: str
    fields: List[FieldSchema]

    @field_validator("document_code")
    @classmethod
    def validate_document_code(cls, v: str) -> str:
        document_code = v.strip()
        if not document_code:
            raise ValueError("document_code must not be empty")
        if not re.fullmatch(r"[a-z0-9_]+", document_code):
            raise ValueError("document_code must match pattern ^[a-z0-9_]+$")
        return document_code

    @field_validator("document_name")
    @classmethod
    def validate_document_name(cls, v: str) -> str:
        document_name = v.strip()
        if not document_name:
            raise ValueError("document_name must not be empty")
        return document_name

    @field_validator("fields")
    @classmethod
    def validate_fields_not_empty(cls, v: List[FieldSchema]) -> List[FieldSchema]:
        if not v:
            raise ValueError("fields must contain at least one field")
        return v

    @model_validator(mode="after")
    def validate_unique_field_keys(self):
        seen = set()
        duplicates = set()
        for field in self.fields:
            if field.field_key in seen:
                duplicates.add(field.field_key)
            seen.add(field.field_key)

        if duplicates:
            duplicate_keys = ", ".join(sorted(duplicates))
            raise ValueError(f"Duplicate field_key values in fields: {duplicate_keys}")
        return self


class PageMarkdown(BaseModel):
    """OCR markdown of one physical page (1-indexed)."""
    page: int = Field(ge=1)
    markdown: str


class ExtractOptions(BaseModel):
    """Schemas and options shared by every extraction request.

    Attributes:
        extraction_schemas: List of document schemas that define what to extract.
        extract_all_fields: If true, include all fields found in each document beyond schema-defined ones.
        extract_all_documents: If true, extract all documents including unknown types not in schema.
        model_name: Optional served model name override.
        temperature: Controls randomness.
        top_p: Nucleus sampling threshold.
        top_k: Top-k sampling parameter.
        max_output_tokens: Maximum tokens to generate.
    """
    extraction_schemas: List[ExtractionSchema]
    extract_all_fields: bool = False
    extract_all_documents: bool = False
    model_name: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_output_tokens: Optional[int] = None

    @field_validator("extraction_schemas")
    @classmethod
    def validate_schemas_not_empty(cls, v: List[ExtractionSchema]) -> List[ExtractionSchema]:
        if not v:
            raise ValueError("extraction_schemas must contain at least one schema")
        return v

    @model_validator(mode="after")
    def validate_unique_document_codes(self):
        seen = set()
        duplicates = set()
        for schema in self.extraction_schemas:
            if schema.document_code in seen:
                duplicates.add(schema.document_code)
            seen.add(schema.document_code)

        if duplicates:
            duplicate_codes = ", ".join(sorted(duplicates))
            raise ValueError(f"Duplicate document_code values in extraction_schemas: {duplicate_codes}")
        return self


class ExtractRequest(ExtractOptions):
    """Request body for the schema-driven extraction endpoint.

    Attributes:
        pages: OCR markdown per page (``pages`` of the OCR service response can be passed as-is).
    """
    pages: List[PageMarkdown]

    @field_validator("pages")
    @classmethod
    def validate_pages(cls, v: List[PageMarkdown]) -> List[PageMarkdown]:
        if not v:
            raise ValueError("pages must contain at least one page")
        page_numbers = [page.page for page in v]
        if len(set(page_numbers)) != len(page_numbers):
            raise ValueError("pages must not contain duplicate page numbers")
        return sorted(v, key=lambda page: page.page)


class OCRExtractRequest(ExtractOptions):
    """Request body for the end-to-end endpoint: download ``file_url`` -> OCR -> extract.

    Attributes:
        file_url: Publicly accessible HTTP/HTTPS URL of the PDF/image.
        extraction_schemas: Optional; defaults to the schemas in examples/schemas.json
            (re-read on every request, so edits apply without restarting the server).
    """
    file_url: str
    extraction_schemas: Optional[List[ExtractionSchema]] = None

    @model_validator(mode="before")
    @classmethod
    def default_extraction_schemas(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("extraction_schemas") is None:
            schemas = json.loads(DEFAULT_SCHEMAS_PATH.read_text(encoding="utf-8"))
            data = {**data, "extraction_schemas": schemas}
        return data

    @field_validator("file_url")
    @classmethod
    def validate_file_url(cls, v: str) -> str:
        v = v.strip()
        if not v.startswith(("http://", "https://")):
            raise ValueError("file_url must be a valid HTTP/HTTPS URL")
        return v


# Response models
class ExtractedDocument(BaseModel):
    """Extraction result for a single identified document within the file.

    Attributes:
        document_code: The document_code from the matched ExtractionSchema.
        document_name: Human-readable name of the document type.
        suggested_document_code: AI-suggested document code (only for unknown/undefined documents).
        suggested_document_name: AI-suggested document name (only for unknown/undefined documents).
        start_page: 1-indexed first page of this document.
        end_page: 1-indexed last page of this document.
        extracted_data: One extracted object per document instance, values keyed by field_key.
    """
    document_code: str
    document_name: Optional[str] = None
    suggested_document_code: Optional[str] = None
    suggested_document_name: Optional[str] = None
    start_page: int
    end_page: int
    extracted_data: List[Dict[str, Any]]


class ExtractResponse(BaseModel):
    """Response from the schema-driven extraction endpoint.

    Attributes:
        run_id: Folder name under data/extraction/ holding request, phase results and response.
        documents: List of extracted documents found in the file.
    """
    run_id: str
    documents: List[ExtractedDocument]


class OCRExtractResponse(ExtractResponse):
    """Response from the end-to-end endpoint.

    Attributes:
        ocr_run_id: Folder name under data/ocr/ holding the downloaded file and OCR pages.
        failed_pages: Pages that failed OCR and were skipped in extraction (resume the OCR run to retry).
    """
    ocr_run_id: str
    failed_pages: List[int]
