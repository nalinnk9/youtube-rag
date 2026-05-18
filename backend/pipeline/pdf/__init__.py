"""Extractor factory + a single render entry point for citation visuals.

Public surface:
  - extract(pdf_path, extractor='pypdfium2'|'marker', doc_id=...) -> PdfExtractionResult
  - render_page(pdf_path, page_num, scale=1.5) -> bytes (PNG)
  - available_extractors() -> list[str]
"""
import io

from . import marker_extractor, pypdfium2_extractor
from .types import PdfAsset, PdfExtractionResult, PdfMetadata, PdfPage


def available_extractors() -> list[dict]:
    """Return a UI-friendly list of supported extractors with availability flags."""
    return [
        {
            "name": "pypdfium2",
            "label": "Fast (pypdfium2)",
            "description": "Chrome's PDFium engine. ~100 pages/sec. Text only — no figure or table extraction.",
            "available": True,
            "default": True,
        },
        {
            "name": "marker",
            "label": "High quality (marker)",
            "description": "LLM-based extraction. Preserves equations as LaTeX, tables as markdown, and saves figures/tables as images. Slow (~1 page/sec on CPU), downloads ~2GB of model weights on first run.",
            "available": marker_extractor.is_available(),
            "default": False,
        },
    ]


def extract(pdf_path: str, extractor: str = "pypdfium2", doc_id: str = "") -> PdfExtractionResult:
    extractor = (extractor or "pypdfium2").lower()
    if extractor == "pypdfium2":
        return pypdfium2_extractor.extract(pdf_path)
    if extractor == "marker":
        if not doc_id:
            raise ValueError("marker extractor requires a doc_id (used as the assets subdir)")
        return marker_extractor.extract(pdf_path, doc_id=doc_id)
    raise ValueError(f"Unknown extractor: {extractor!r}")


def render_page(pdf_path: str, page_num: int, scale: float = 1.5) -> bytes:
    """Render a single PDF page as PNG bytes via pypdfium2 (no system deps)."""
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument(pdf_path)
    page = pdf[max(0, page_num - 1)]
    pil = page.render(scale=scale).to_pil()
    buf = io.BytesIO()
    pil.save(buf, "PNG")
    page.close()
    return buf.getvalue()


__all__ = [
    "available_extractors", "extract", "render_page",
    "PdfAsset", "PdfExtractionResult", "PdfMetadata", "PdfPage",
]
