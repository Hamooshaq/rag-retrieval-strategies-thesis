"""Document ingestion with page/slide provenance."""

from .parsers import DocumentParseError, parse_document, parse_pdf, parse_pptx

__all__ = ["DocumentParseError", "parse_document", "parse_pdf", "parse_pptx"]
