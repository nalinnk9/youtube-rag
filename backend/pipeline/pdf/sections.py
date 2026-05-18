"""Heuristic section detection for research papers.

Research-paper section headers follow recognizable patterns:
  - "Abstract", "Introduction", "Methods", "Results", "Discussion", "Conclusion", "References"
  - Numbered: "1 Introduction", "1.1 Background", "2.3.1 Detailed methodology"
  - All-caps short lines (often used for major sections in some templates)

We don't aim for perfect parsing — heuristics catch >95% of arXiv-style papers,
and unrecognized text falls into the preceding section so nothing is lost.
"""
import re
from dataclasses import dataclass


# Lines that are obvious section headers in research papers.
# Match: numbered sections, common unnumbered names, and short all-caps lines.
_NUMBERED = re.compile(r"^\s*(\d+(?:\.\d+){0,3})\s*\.?\s+([A-Z][A-Za-z][^\n]{1,80})$", re.MULTILINE)
_NAMED_KEYWORDS = (
    "abstract", "introduction", "background", "related work", "preliminaries",
    "method", "methods", "methodology", "approach", "model", "models",
    "experiment", "experiments", "experimental setup", "evaluation",
    "result", "results", "findings",
    "discussion", "analysis", "ablation", "limitations", "future work",
    "conclusion", "conclusions",
    "references", "bibliography", "appendix", "appendices", "acknowledgements", "acknowledgments",
)
_NAMED_RE = re.compile(
    r"^\s*((?:" + "|".join(re.escape(k) for k in _NAMED_KEYWORDS) + r"))\s*$",
    re.MULTILINE | re.IGNORECASE,
)


@dataclass
class Section:
    title: str          # full normalized title, e.g. "3.2 Results"
    number: str = ""    # "3.2" if numbered, else ""
    name: str = ""      # "Results"
    page_start: int = 1
    page_end: int = 1
    text: str = ""


def _normalize_title(line: str) -> tuple[str, str, str]:
    line = line.strip()
    m = _NUMBERED.match(line + "\n")
    if m:
        return (f"{m.group(1)} {m.group(2).strip()}", m.group(1), m.group(2).strip())
    return (line.title(), "", line.title())


def detect_sections(pages: list[dict]) -> list[Section]:
    """Walk pages in order, find section boundaries, return a list of sections.

    `pages`: list of dicts shaped like {"page_num": int, "text": str}.
    Returns a list of `Section`. Always non-empty (full doc as one section if no headers found).
    """
    sections: list[Section] = []
    current: Section | None = None

    for page in pages:
        page_num = page["page_num"]
        text = page["text"] or ""

        # Walk line-by-line to find headers
        lines = text.split("\n")
        buffered: list[str] = []

        def flush_buffer():
            if not current or not buffered:
                return
            current.text = (current.text + "\n" + "\n".join(buffered)).strip() if current.text else "\n".join(buffered).strip()
            current.page_end = page_num

        for raw in lines:
            line = raw.rstrip()
            if not line.strip():
                buffered.append(line)
                continue

            # Try numbered header first (more specific)
            mnum = _NUMBERED.match(line + "\n")
            mnam = _NAMED_RE.match(line + "\n") if not mnum else None
            is_header = bool(mnum or mnam)

            if is_header:
                flush_buffer()
                buffered = []
                title, number, name = _normalize_title(line)
                if current is not None:
                    current.text = current.text.strip()
                current = Section(title=title, number=number, name=name, page_start=page_num, page_end=page_num)
                sections.append(current)
            else:
                if current is None:
                    # Content before any header — treat as preamble (often title page or abstract)
                    current = Section(title="Preamble", number="", name="Preamble", page_start=page_num, page_end=page_num)
                    sections.append(current)
                buffered.append(line)

        flush_buffer()

    if not sections:
        # No headers detected — return everything as one synthetic section
        full = "\n".join(p["text"] for p in pages)
        return [Section(title="Document", number="", name="Document",
                        page_start=pages[0]["page_num"] if pages else 1,
                        page_end=pages[-1]["page_num"] if pages else 1,
                        text=full.strip())]

    # Tidy up
    for s in sections:
        s.text = s.text.strip()
    return sections


def find_title_and_authors(first_page_text: str) -> tuple[str, str]:
    """Cheap heuristic: title = first non-empty line up to ~150 chars; authors = next 1-3 lines.

    Real arXiv PDFs usually have title in the first 1-2 lines and authors right after.
    """
    lines = [l.strip() for l in (first_page_text or "").split("\n") if l.strip()]
    if not lines:
        return ("", "")
    title = lines[0][:200]
    # Authors are typically the next 1-3 short lines before the abstract begins
    authors_lines: list[str] = []
    for line in lines[1:4]:
        if len(line) > 250 or line.lower().startswith("abstract"):
            break
        authors_lines.append(line)
    return (title, " · ".join(authors_lines))
