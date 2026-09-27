from __future__ import annotations

import re
from pathlib import Path

from pptx import Presentation
from pptx.enum.shapes import PP_PLACEHOLDER
from pypdf import PdfReader

from storage.models import DocumentFormat, DocumentSpec, ParsedBlock


class DocumentParseError(RuntimeError):
    pass


def _normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_pdf(document: DocumentSpec, path: Path) -> list[ParsedBlock]:
    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        raise DocumentParseError(f"Cannot open PDF {document.document_id}: {exc}") from exc
    if not reader.pages:
        raise DocumentParseError(f"PDF has no pages: {document.document_id}")

    blocks: list[ParsedBlock] = []
    order = 0
    for page_number, page in enumerate(reader.pages, 1):
        try:
            raw_text = page.extract_text() or ""
        except Exception as exc:
            raise DocumentParseError(
                f"Cannot extract page {page_number} of {document.document_id}: {exc}"
            ) from exc
        text = _normalize_text(raw_text)
        if not text:
            continue
        segments = [segment.strip() for segment in re.split(r"\n\s*\n", text) if segment.strip()]
        if len(segments) == 1:
            segments = [line.strip() for line in text.splitlines() if line.strip()]
        for index, segment in enumerate(segments):
            block_type = "heading" if index == 0 and len(segment.split()) <= 16 else "body"
            blocks.append(
                ParsedBlock(
                    block_id=f"{document.document_id}:page:{page_number}:block:{index + 1}",
                    document_id=document.document_id,
                    locator=f"page:{page_number}",
                    order=order,
                    block_type=block_type,
                    text=segment,
                )
            )
            order += 1
    if not blocks:
        raise DocumentParseError(
            f"PDF contains no extractable text: {document.document_id}. "
            "OCR-only documents are outside the text-first pilot."
        )
    return blocks


def _shape_text(shape) -> str:
    if not getattr(shape, "has_text_frame", False):
        return ""
    paragraphs: list[str] = []
    for paragraph in shape.text_frame.paragraphs:
        text = _normalize_text("".join(run.text for run in paragraph.runs) or paragraph.text)
        if text:
            paragraphs.append(text)
    return "\n".join(paragraphs)


def _shape_sort_key(shape) -> tuple[int, int]:
    """Return a stable position key even for malformed legacy slide shapes."""
    top = getattr(shape, "top", None)
    left = getattr(shape, "left", None)
    return int(top) if top is not None else 0, int(left) if left is not None else 0


def parse_pptx(document: DocumentSpec, path: Path, include_notes: bool = True) -> list[ParsedBlock]:
    try:
        presentation = Presentation(str(path))
    except Exception as exc:
        raise DocumentParseError(f"Cannot open PPTX {document.document_id}: {exc}") from exc
    if not presentation.slides:
        raise DocumentParseError(f"PPTX has no slides: {document.document_id}")

    blocks: list[ParsedBlock] = []
    order = 0
    for slide_number, slide in enumerate(presentation.slides, 1):
        shapes = sorted(
            slide.shapes,
            key=_shape_sort_key,
        )
        slide_block_count = 0
        for shape_index, shape in enumerate(shapes, 1):
            text = _shape_text(shape)
            if not text:
                continue
            block_type = "body"
            if getattr(shape, "is_placeholder", False):
                placeholder_type = shape.placeholder_format.type
                if placeholder_type in {PP_PLACEHOLDER.TITLE, PP_PLACEHOLDER.CENTER_TITLE}:
                    block_type = "title"
            blocks.append(
                ParsedBlock(
                    block_id=f"{document.document_id}:slide:{slide_number}:shape:{shape_index}",
                    document_id=document.document_id,
                    locator=f"slide:{slide_number}",
                    order=order,
                    block_type=block_type,
                    text=text,
                )
            )
            order += 1
            slide_block_count += 1

        if include_notes:
            try:
                notes_text = _normalize_text(slide.notes_slide.notes_text_frame.text)
            # python-pptx can raise multiple backend-specific exceptions when a
            # notes relationship is malformed; notes are optional for this parser.
            except Exception:  # noqa: BLE001
                notes_text = ""
            notes_text = re.sub(r"(?im)^\s*slide image\s*$", "", notes_text).strip()
            if notes_text:
                blocks.append(
                    ParsedBlock(
                        block_id=f"{document.document_id}:slide:{slide_number}:notes",
                        document_id=document.document_id,
                        locator=f"slide:{slide_number}",
                        order=order,
                        block_type="notes",
                        text=notes_text,
                    )
                )
                order += 1
                slide_block_count += 1

        # Empty slides are legal and intentionally skipped. The entire deck may not be empty.
        if slide_block_count == 0:
            continue
    if not blocks:
        raise DocumentParseError(f"PPTX contains no extractable text: {document.document_id}")
    return blocks


def parse_document(document: DocumentSpec, base_dir: Path) -> list[ParsedBlock]:
    path = document.resolved_path(base_dir)
    if document.format == DocumentFormat.PDF:
        return parse_pdf(document, path)
    if document.format == DocumentFormat.PPTX:
        return parse_pptx(document, path)
    raise DocumentParseError(f"Unsupported format for {document.document_id}: {document.format}")
