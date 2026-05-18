"""Shared data types for the PDF extraction layer.

Extractors (pypdfium2 / marker) all return `PdfExtractionResult` with the same
shape so the downstream chunking / ingestion code stays extractor-agnostic.
"""
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class PdfAsset:
    """A figure or table extracted from the PDF, stored on disk as a PNG."""
    kind: Literal["figure", "table"]
    page_num: int
    path: str           # absolute path on disk (served via /pdf_asset?...)
    caption: str = ""   # surrounding caption text if found
    bbox: tuple[float, float, float, float] | None = None  # (x0, y0, x1, y1) in PDF points


@dataclass
class PdfPage:
    page_num: int        # 1-indexed
    text: str            # extracted text in reading order
    section: str = ""    # the section this page belongs to, filled in later


@dataclass
class PdfMetadata:
    title: str = ""
    authors: str = ""
    abstract: str = ""
    num_pages: int = 0
    extractor: str = ""


@dataclass
class PdfExtractionResult:
    metadata: PdfMetadata
    pages: list[PdfPage]
    assets: list[PdfAsset] = field(default_factory=list)
    full_text: str = ""
