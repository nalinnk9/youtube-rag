"""pypdfium2-backed PDF extractor.

Fast text extraction via Chrome's PDFium engine. No layout-detection model, so
figures/tables are NOT extracted as separate assets — only text + per-page renders
for citation visuals. This is the right tradeoff for arXiv-style PDFs where the
text layer is clean; if a paper has critical figures, use the marker extractor.
"""
from .sections import detect_sections, find_title_and_authors
from .types import PdfExtractionResult, PdfMetadata, PdfPage


def extract(pdf_path: str) -> PdfExtractionResult:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(pdf_path)
    pages: list[PdfPage] = []
    full_text_parts: list[str] = []

    for i in range(len(pdf)):
        page = pdf[i]
        textpage = page.get_textpage()
        text = textpage.get_text_range().strip()
        textpage.close()
        page.close()
        pages.append(PdfPage(page_num=i + 1, text=text))
        full_text_parts.append(text)

    full_text = "\n\n".join(full_text_parts)

    # Detect sections and attach the section title back onto each page
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
        extractor="pypdfium2",
    )

    return PdfExtractionResult(
        metadata=meta,
        pages=pages,
        assets=[],   # pypdfium2 path doesn't extract figures/tables
        full_text=full_text,
    )
