"""Dựng JSON Schema cho structured output của vLLM (port từ Gemini-OCR, bỏ bbox)."""

from typing import Any, Dict, List, Optional

from src.extraction.schemas import ExtractionSchema, FieldSchema

DATA_TYPE_MAP: Dict[str, str] = {
    "string": "string",
    "number": "number",
    "boolean": "boolean",
    "date": "string",  # ISO 8601 date string
    "array": "array",
}


def _attach_description(
    schema_fragment: Dict[str, Any], description: Optional[str]
) -> Dict[str, Any]:
    if description:
        schema_fragment["description"] = description
    return schema_fragment


def _strip_schema_descriptions(schema_fragment: Any) -> Any:
    """Remove descriptions in-place to keep response schemas compact.

    Phase-2 field guidance is already included in the prompt; repeating it in
    every nested node only grows the structured-output grammar.
    """
    if isinstance(schema_fragment, dict):
        schema_fragment.pop("description", None)
        for value in schema_fragment.values():
            _strip_schema_descriptions(value)
    elif isinstance(schema_fragment, list):
        for value in schema_fragment:
            _strip_schema_descriptions(value)
    return schema_fragment


def build_field_schema(field: FieldSchema) -> Dict[str, Any]:
    """Recursively build a JSON Schema fragment for one field."""
    dtype = DATA_TYPE_MAP.get(field.data_type, "string")
    description = field.description or field.field_name

    if dtype == "array":
        child_props: Dict[str, Any] = {}
        required_child_keys: List[str] = []
        if field.child_schema:
            child_props = {f.field_key: build_field_schema(f) for f in field.child_schema}
            required_child_keys = [f.field_key for f in field.child_schema if f.required]

        schema: Dict[str, Any] = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": child_props,
                "required": required_child_keys,
                "additionalProperties": False,
            },
        }
        if field.nullable:
            schema["type"] = ["array", "null"]
        return _attach_description(schema, description)

    # Date formatting belongs in the prompt, so dates are plain strings here.
    value_schema: Dict[str, Any] = {"type": dtype}
    if field.nullable:
        value_schema["type"] = [dtype, "null"]
    return _attach_description(value_schema, description)


def _build_unknown_document_extracted_data_schema() -> Dict[str, Any]:
    """Build extracted_data schema for unknown documents with fields/tables structure.

    This schema allows the LLM to extract arbitrary fields and tables from documents
    that don't match any known document_code.
    """
    return {
        "type": "object",
        "properties": {
            "fields": {
                "type": "array",
                "items": {
                    "type": "object",
                    "description": "Each field is a JSON object with field_key, field_name, and value. Only extract content that are outside of any tables. For content inside tables, extract them in the tables structure.",
                    "properties": {
                        "field_key": {
                            "type": "string",
                            "description": "Unique key for the extracted field in English in snake_case",
                        },
                        "field_name": {
                            "type": "string",
                            "description": "Name of the extracted field in Vietnamese in normal language",
                        },
                        "value": {
                            "type": ["string", "null"],
                            "description": "Extracted value for this field as a string",
                        },
                    },
                    "required": ["field_key", "field_name", "value"],
                    "additionalProperties": False,
                },
            },
            "tables": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "table_key": {
                            "type": "string",
                            "description": "Unique key for the extracted table in English in snake_case",
                        },
                        "table_name": {
                            "type": "string",
                            "description": "Name of the extracted table in Vietnamese in normal language",
                        },
                        "columns": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "key": {
                                        "type": "string",
                                        "description": "Unique key for the column in English in snake_case",
                                    },
                                    "label": {
                                        "type": "string",
                                        "description": "Name of the column in Vietnamese in normal language",
                                    },
                                },
                                "required": ["key", "label"],
                                "additionalProperties": False,
                            },
                        },
                        "rows": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "description": "Each row is a JSON object whose column keys map to cell values.",
                                "additionalProperties": {"type": ["string", "null"]},
                            },
                        },
                    },
                    "required": ["table_key", "table_name", "columns", "rows"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["fields", "tables"],
        "additionalProperties": False,
    }


def build_phase1_response_schema(
    extraction_schemas: Optional[List[ExtractionSchema]] = None,
    extract_all_documents: bool = False,
) -> Dict[str, Any]:
    """Build a JSON Schema object for Phase 1: Segmentation."""
    valid_codes = [schema.document_code for schema in extraction_schemas or []]

    # Force extract_all_documents to True if no schemas are provided,
    # so the LLM can return 'unknown' and suggest types.
    if not extraction_schemas:
        extract_all_documents = True

    doc_code_schema: Dict[str, Any] = {"type": "string"}
    if extract_all_documents:
        doc_code_schema["enum"] = valid_codes + ["unknown"]
        doc_code_schema["description"] = "Document code. Output 'unknown' if it does not match any known schemas."
    elif valid_codes:
        doc_code_schema["enum"] = valid_codes

    properties: Dict[str, Any] = {
        # Đứng trước document_code: model chép tiêu đề trước rồi mới chọn loại (suy luận ngắn
        # thay cho thinking), tránh chọn loại chỉ theo ấn tượng ở token đầu tiên
        "title": {
            "type": "string",
            "description": "Main heading of the document copied verbatim from the OCR text (e.g. 'BẢNG KÊ CHI PHÍ'), empty string if there is none.",
        },
        "document_code": doc_code_schema,
        "document_name": {
            "type": "string",
            "description": "Human-readable name of the document type in Vietnamese. For known documents, use the name from the schema. For unknown documents (document_code='unknown'), leave this as an empty string.",
        },
        "start_page": {"type": "integer"},
        "end_page": {"type": "integer"},
    }
    required_fields = ["title", "document_code", "document_name", "start_page", "end_page"]

    # Several same-code documents on one page (e.g. 2 CCCD of 2 people).
    # Without bbox, each instance is identified by text taken from the OCR output.
    # ``sides`` replaces Gemini's region sides, used to merge a front-only page with an adjacent back-only page.
    properties["instances"] = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "identifier": {"type": "string"},
                "front_name": {"type": ["string", "null"]},
                "mrz_name": {"type": ["string", "null"]},
                "sides": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["front", "back", "single"]},
                    "minItems": 1,
                },
            },
            "required": ["identifier", "front_name", "mrz_name", "sides"],
            "additionalProperties": False,
        },
        "minItems": 1,
    }

    if extract_all_documents:
        properties["suggested_document_code"] = {
            "type": "string",
            "description": "ONLY output this field if document_code is 'unknown'. Suggested document code for the unknown document in English in snake_case.",
        }
        properties["suggested_document_name"] = {
            "type": "string",
            "description": "ONLY output this field if document_code is 'unknown'. Suggested document name for the unknown document in Vietnamese.",
        }

    return {
        "type": "object",
        "properties": {
            "documents": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": properties,
                    "required": required_fields,
                    "additionalProperties": False,
                },
            }
        },
        "required": ["documents"],
        "additionalProperties": False,
    }


def build_phase2_response_schema(
    schema: Optional[ExtractionSchema],
    extract_all_fields: bool = False,
) -> Dict[str, Any]:
    """Build a JSON Schema object for Phase 2: Extraction of a single document."""
    if schema is None:
        return _strip_schema_descriptions(_build_unknown_document_extracted_data_schema())

    response_schema = {
        "type": "object",
        "properties": {field.field_key: build_field_schema(field) for field in schema.fields},
        "required": [field.field_key for field in schema.fields if field.required],
        "additionalProperties": (
            {"type": ["string", "null"]} if extract_all_fields else False
        ),
    }
    return _strip_schema_descriptions(response_schema)
