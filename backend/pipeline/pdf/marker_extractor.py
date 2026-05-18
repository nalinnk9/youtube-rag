"""marker-pdf backed extractor — slow but highest quality.

Marker (https://github.com/VikParuchuri/marker) produces clean markdown with
preserved equations (LaTeX), tables (markdown tables), and image references for
figures. We parse its output to:
  - rebuild per-page text (marker emits a single markdown blob; we split on its
    "{N}------------------------------------------------" page-break markers when
    --paginate is used, else fall back to whole-doc as page 1).
  - capture each extracted image (figures + tables) as a PdfAsset on disk.

Marker is optional. The function raises ImportError with an actionable message
if the package isn't installed, so the API can return a clean 400.
"""
import os
import re
import shutil

from ...config import settings
from .sections import detect_sections, find_title_and_authors
from .types import PdfAsset, PdfExtractionResult, PdfMetadata, PdfPage


def is_available() -> bool:
    try:
        import importlib.util
        return importlib.util.find_spec("marker") is not None
    except Exception:
        return False


def extract(pdf_path: str, doc_id: str) -> PdfExtractionResult:
    if not is_available():
        raise ImportError(
            "marker is not installed. Run `pip install marker-pdf` (downloads model "
            "weights on first run, ~2GB; CPU works but GPU is much faster)."
        )

    # marker's public API has churned; we import lazily to tolerate version differences.
    try:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        from marker.output import text_from_rendered
    except ImportError as e:
        raise ImportError(f"marker is installed but its API changed: {e!r}") from e

    converter = PdfConverter(artifact_dict=create_model_dict())
    rendered = converter(pdf_path)
    md_text, _stats, images = text_from_rendered(rendered)

    # Persist any extracted images (figures / tables) into pdf_assets_dir/<doc_id>/
    assets: list[PdfAsset] = []
    assets_dir = os.path.join(settings.pdf_assets_dir, doc_id)
    os.makedirs(assets_dir, exist_ok=True)
    for img_name, pil_image in (images or {}).items():
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", img_name)
        out_path = os.path.join(assets_dir, safe_name)
        try:
            pil_image.save(out_path)
        except Exception:
            continue
        # marker names images like "fig_3.png", "table_1.png" — sniff kind by name
        kind = "table" if "table" in safe_name.lower() else "figure"
        # Best-effort page: marker doesn't always tag pages on images; default to 0 (unknown)
        page_num = _guess_image_page(img_name)
        assets.append(PdfAsset(kind=kind, page_num=page_num, path=out_path))

    # Page splitting: if marker emitted page break markers, split on them; else single page.
    page_chunks = _split_into_pages(md_text)
    pages: list[PdfPage] = [PdfPage(page_num=i + 1, text=t.strip()) for i, t in enumerate(page_chunks)]

    sections = detect_sections([{"page_num": p.page_num, "text": p.text} for p in pages])
    page_to_section: dict[int, str] = {}
    for s in sections:
        for pn in range(s.page_start, s.page_end + 1):
            page_to_section.setdefault(pn, s.title)
    for p in pages:
        p.section = page_to_section.get(p.page_num, "")

    title, authors = find_title_and_authors(pages[0].text if pages else "")
    meta = PdfMetadata(
        title=title or "Untitled document",
        authors=authors,
        num_pages=len(pages),
        extractor="marker",
    )
    return PdfExtractionResult(
        metadata=meta,
        pages=pages,
        assets=assets,
        full_text=md_text,
    )


def _split_into_pages(md: str) -> list[str]:
    """Split marker's markdown by its page-break sentinel; fall back to single page."""
    # marker may emit "\n\n{N}\n------------------------------------------------\n" between pages
    parts = re.split(r"\n\s*\{\d+\}\s*\n-{10,}\n", md)
    parts = [p for p in parts if p.strip()]
    return parts or [md]


def _guess_image_page(name: str) -> int:
    m = re.search(r"page[_-]?(\d+)", name, re.IGNORECASE)
    return int(m.group(1)) if m else 0
