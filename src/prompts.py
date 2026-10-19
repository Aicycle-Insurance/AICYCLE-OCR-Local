"""Toàn bộ prompt của OCR engine và extraction."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from src.extraction.schemas import ExtractionSchema, FieldSchema


# ── OCR: Chandra (datalab-to/chandra-ocr-2) ─────────────────────────────────

CHANDRA_ALLOWED_TAGS = [
    "math", "br", "i", "b", "u", "del", "sup", "sub", "table", "tr", "td", "p",
    "th", "div", "pre", "h1", "h2", "h3", "h4", "h5", "ul", "ol", "li", "input",
    "a", "span", "img", "hr", "tbody", "small", "caption", "strong", "thead",
    "big", "code", "chem",
]
CHANDRA_ALLOWED_ATTRIBUTES = [
    "class", "colspan", "rowspan", "display", "checked", "type", "border",
    "value", "style", "href", "alt", "align", "data-bbox", "data-label",
]

CHANDRA_OCR_LAYOUT_PROMPT = f"""
OCR this image to HTML, arranged as layout blocks.  Each layout block should be a div with the data-bbox attribute representing the bounding box of the block in x0 y0 x1 y1 format.  Bboxes are normalized 0-1000. The data-label attribute is the label for the block.

Use the following labels:
- Caption
- Footnote
- Equation-Block
- List-Group
- Page-Header
- Page-Footer
- Image
- Section-Header
- Table
- Text
- Complex-Block
- Code-Block
- Form
- Table-Of-Contents
- Figure
- Chemical-Block
- Diagram
- Bibliography
- Blank-Page

Only use these tags {CHANDRA_ALLOWED_TAGS}, and these attributes {CHANDRA_ALLOWED_ATTRIBUTES}.

Guidelines:
* Inline math: Surround math with <math>...</math> tags. Math expressions should be rendered in KaTeX-compatible LaTeX. Use display for block math.
* Tables: Use colspan and rowspan attributes to match table structure.
* Formatting: Maintain consistent formatting with the image, including spacing, indentation, subscripts/superscripts, and special characters.
* Images: Include a description of any images in the alt attribute of an <img> tag. Do not fill out the src property. Describe in detail inside the div tag. Also convert charts to high fidelity data, and convert diagrams to mermaid.
* Forms: Mark checkboxes and radio buttons properly.
* Text: join lines together properly into paragraphs using <p>...</p> tags.  Use <br> tags for line breaks within paragraphs, but only when absolutely necessary to maintain meaning.
* Chemistry: Use <chem>...</chem> tags for chemical formulas with reactive SMILES.
* Lists: Preserve indents and proper list markers.
* Use the simplest possible HTML structure that accurately represents the content of the block.
* Make sure the text is accurate and easy for a human to read and interpret.  Reading order should be correct and natural.
""".strip()


# ── OCR: dots.mocr (rednote-hilab/dots.mocr) ────────────────────────────────

DOTS_MOCR_LAYOUT_ALL_PROMPT = """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object.
"""


# ── Extraction: input document ──────────────────────────────────────────────

TEMPLATE_DOCUMENT_INPUT = """<document>
The document below is the OCR output (Markdown/HTML) of a scanned file. Each page is wrapped in a <page number="N"> tag, where N is the physical page number.

{pages}
</document>"""

TEMPLATE_PAGE = """<page number="{page}">
{markdown}
</page>"""


# ── Extraction: Phase 1 (segmentation & classification) ────────────────────

TEMPLATE_PHASE1_SEGMENTATION = """<role>You are an expert Document Segmentation System.</role>

<task>
Analyze the OCR text of the entire file, which may contain multiple distinct documents over multiple pages. Identify boundaries between documents and classify each one.
</task>

<document_schemas>
The following document types may be present in this file. Each entry includes the document_code.
{schema_description}
</document_schemas>

<guidelines>
1. MERGE pages into one document if: content is sequential/continuing, table rows continue from previous page, or pagination markers exist (e.g., "Trang 2/3").
2. SPLIT into separate documents if: a new document header appears (e.g., Hospital name, "HÓA ĐƠN", "CỘNG HÒA XÃ HỘI...") with no continuity markers.
3. `start_page` and `end_page` MUST be page numbers taken from the <page number="N"> tags.
</guidelines>

<critical_rules>
{all_documents_rule}
</critical_rules>

<output_requirements>
Do NOT extract any specific data fields. ONLY return the structured array of identified documents with `document_code`, `start_page`, and `end_page` (and suggested fields if unknown).

**Same-page instances (CRITICAL)**
- Vietnamese CCCD/CMND: analyze each page independently and return one document entry per document_code per page (start_page == end_page).
- Any other document_code: only when several distinct documents of the SAME code appear on one page, return one entry for that page (start_page == end_page). Otherwise use the normal segmentation rules (merge continuing pages into one start_page/end_page range) and OMIT `instances`.
- `instances` lists every distinct logical document of that code on the page. Each instance has an `identifier`: a short text copied verbatim from the OCR text that uniquely identifies that document (e.g. ID/document number, holder name, invoice number).
- For CCCD/CMND, the front side has the photo-side fields (name, ID number, date of birth) and the back side has the MRZ lines. If at least one front exists, the number of instances equals the number of fronts; two fronts are NEVER one instance.
- Match a back to a front only when the normalized printed front name equals the normalized MRZ name (uppercase, no Vietnamese accents, `Đ` -> `D`, `<` -> space). A matched front and back form one instance; when fronts exist, an unmatched back does not create another instance. If only backs exist, each back is one instance.
- `front_name` is the printed holder name from the front, otherwise null. `mrz_name` is the raw MRZ holder-name line from the matched back, otherwise null.
</output_requirements>
"""


# ── Extraction: Phase 2 (field extraction for one document) ────────────────

TEMPLATE_PHASE2_EXTRACTION = """<role>You are an expert Document Intelligence System.</role>

<task>
You have been provided the OCR text of a specific chunk of a document (`start_page` to `end_page`). Extract the required fields for the document type `{document_code}`.{instance_hint}
</task>

<document_schema>
{schema_description}
</document_schema>

<guidelines>
1. Match the document to the corresponding keys.
{field_extraction_rule}
3. If a field_key or its corresponding value is not found in the document, its value MUST be null.
4. If a field includes "Hint", treat it as guidance for what to find that value.
5. IGNORE OCR noise such as watermark text and repeated background patterns.
6. ONLY use stamps, seals, or digital signatures to verify info if printed text is missing.
    When checking for signatures or checkboxes, you MUST verify the actual presence of a mark/tick in the OCR text (e.g. checked input, ☑, [x]); if the area is empty or only contains a placeholder, return false.
7. MUST strictly verify the semantic label immediately preceding or associated with a value to ensure accurate field mapping.
    Never "borrow" values from neighboring rows if their specific label is missing.
    If a value lacks a matching label or the label is functionally blank, treat the field as null.
</guidelines>

<strict_type_coercion>
Convert all extracted values to the correct data type:
- string: Plain text, no conversion needed.
- number: A pure numeric value. Strip all currency symbols, units, and thousand separators. Example: "1.500.000 VNĐ" → 1500000, "2,5" → 2.5. Must be a JSON number, NOT a string.
- boolean: Must be JSON true or false. Convert: "Có"/"Yes"/"X" → true, "Không"/"No"/ empty → false.
- array: A JSON array of objects. Each object must follow the child_schema structure with exact field_keys.
</strict_type_coercion>
"""

UNKNOWN_DOCUMENT_DESCRIPTION = (
    "Undefined document (unknown). \n"
    "CRITICAL INSTRUCTIONS FOR UNKNOWN DOCUMENTS:\n"
    "1. Extract ALL texts, key-value pairs, and tables into the generic format.\n"
    "2. VERY IMPORTANT: All `field_key`, `table_key`, and `key` for columns MUST be thoughtfully translated into English and formatted strictly in snake_case.\n"
    "3. All `field_name`, `table_name`, and `label` for columns MUST remain in their original language (e.g., Vietnamese) as written on the document.\n"
    "4. For tables: extract ALL columns but LIMIT to maximum 50 rows per table. If more rows exist, extract only the first 50."
)


def build_document_input(pages: list[tuple[int, str]]) -> str:
    """Ghép markdown từng trang thành 1 khối <document> có đánh số trang."""
    return TEMPLATE_DOCUMENT_INPUT.format(
        pages="\n\n".join(
            TEMPLATE_PAGE.format(page=page, markdown=markdown.strip())
            for page, markdown in pages
        )
    )


def build_phase1_prompt(
    extraction_schemas: list[ExtractionSchema] | None = None,
    extract_all_documents: bool = False,
) -> str:
    """Generates a prompt for Phase 1: Segmentation and Classification."""
    if extraction_schemas:
        schema_description = "\n".join(
            f'- document_code: "{schema.document_code}" | Name: {schema.document_name}'
            for schema in extraction_schemas
        )
    else:
        schema_description = "None provided. Treat all documents as undefined."

    if extract_all_documents or not extraction_schemas:
        doc_rule = "- Extract EVERY document found in the file, including undefined document types that don't match document_code in the provided schema. For undefined documents, set document_code to 'unknown', leave document_name as empty string, and fill suggested_document_code and suggested_document_name with appropriate suggestions."
    else:
        doc_rule = "- Only return documents whose document_code exists in the provided schema list."

    return TEMPLATE_PHASE1_SEGMENTATION.format(
        schema_description=schema_description,
        all_documents_rule=doc_rule,
    )


def _format_field_descriptor(field: FieldSchema) -> str:
    descriptor = f"{field.field_key} ({field.data_type})"
    hint = field.description or field.field_name
    if hint:
        descriptor += f" - Hint: {hint}"
    return descriptor


def build_phase2_prompt(
    schema: Optional[ExtractionSchema],
    extract_all_fields: bool = False,
    page_chunk: tuple[int, int] | None = None,
    instance_hint: dict | None = None,
) -> str:
    """Generates a prompt for Phase 2: Extraction of one document (optionally one of several on the page)."""
    if schema:
        schema_lines = [
            f'- document_code: "{schema.document_code}" | Name: {schema.document_name}',
            "  Fields to extract:",
        ]
        for field in schema.fields:
            if field.data_type == "array" and field.child_schema:
                child_keys = ", ".join(
                    _format_field_descriptor(f) for f in field.child_schema
                )
                array_descriptor = (
                    f"{field.field_key} (array of objects, each with: {child_keys})"
                )
                if field.field_name:
                    array_descriptor += f" - Hint: {field.field_name}"
                schema_lines.append(f"    - {array_descriptor}")
            else:
                schema_lines.append(f"    - {_format_field_descriptor(field)}")
        schema_description = "\n".join(schema_lines)
        document_code = schema.document_code
    else:
        schema_description = UNKNOWN_DOCUMENT_DESCRIPTION
        document_code = "unknown"

    if extract_all_fields:
        field_rule = "2. Extract ALL fields (everything) from each document segment"
    else:
        field_rule = "2. Extract ONLY the fields defined in that document's schema."

    hint_block = ""
    if page_chunk is not None:
        hint_block += (
            f"\n\nThe supplied {page_chunk[0]} pages are only part of a longer "
            f"{page_chunk[1]}-page document. Extract ONLY what appears on these pages: "
            "list every table row on them (including a row continued from the previous "
            "page), and set fields that do not appear on them to null."
        )
    if instance_hint is not None:
        hint_block += (
            "\n\nThe supplied page contains several documents of this type. Extract ONLY "
            "the one identified by the instance below; never borrow values from the other "
            "documents. If both its front and back sides are present, merge their fields "
            "into one object.\n<instance_hint>\n"
            f"{json.dumps(instance_hint, ensure_ascii=False)}\n</instance_hint>"
        )

    return TEMPLATE_PHASE2_EXTRACTION.format(
        document_code=document_code,
        schema_description=schema_description,
        field_extraction_rule=field_rule,
        instance_hint=hint_block,
    )
