"""Read and normalize source PDF data with page-level traceability."""
import re
import statistics
import unicodedata
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from .models import DocumentPage

HEADING_RE = re.compile(
    r"^(?:\d+(?:\.\d+)*\s+[A-Z]|[IVX]+\.\s+[A-Z]|[A-D]\.\s+[A-Z]|"
    r"(?:Abstract|Introduction|Related Work|Methodology|Evaluation|Results|"
    r"Limitations|Conclusion|Conclusions|Acknowledgments|References|Appendix)\b|"
    r"(?:Table|Fig(?:ure)?|Algorithm)\s*\d+\b)",
)

SECTION_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("appendix", re.compile(r"^(?:B\s+DBSCAN|C\s+Marker|D\s+Llama3|Table\s+[45]:\s+All)")),
    (
        "evaluation",
        re.compile(
            r"^(?:(?:4|5|6|7)\s+(?:Experiments|Evaluation|Results|Limitations|Conclusion|Conclusions)\b|"
            r"(?:IV|V|VI)\.\s+(?:EXPERIMENTAL\s*RESULTS|LIMITATION|CONCLUSION))"
        ),
    ),
    (
        "methodology",
        re.compile(
            r"^(?:(?:3|4|5)\s+(?:CBR-RAG|Representation|Embedding|Synthetic|The RAGentA|Methodology)\b|"
            r"(?:II|III)\.\s+(?:METHODOLOGY|EXPERIMENTAL\s*SETUP))"
        ),
    ),
    ("intro_related", re.compile(r"^(?:1\s+Introduction\b|2\s+Related Work\b|I\.\s+INTRODUCTION\b|Abstract\b)")),
]


def normalize_text(text: str) -> str:
    """Normalize Unicode, de-hyphenate line wraps, and reconstruct paragraphs."""
    cleaned = unicodedata.normalize("NFKC", text).replace("\u00ad", "")
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in cleaned.splitlines()]
    lines = [ln for ln in lines if ln]
    if not lines:
        return ""

    long_lengths = [len(ln) for ln in lines if len(ln) > 20]
    typical_len = statistics.median(long_lengths) if long_lengths else 60
    paragraphs: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if not current:
            return
        joined = current[0]
        for nxt in current[1:]:
            if joined.endswith("-") and len(joined) > 1 and joined[-2].isalpha() and nxt[0].islower():
                joined = joined[:-1] + nxt
            else:
                joined = f"{joined} {nxt}"
        paragraphs.append(joined)
        current.clear()

    for ln in lines:
        is_heading = bool(HEADING_RE.match(ln)) and len(ln) < 120
        is_bullet = ln.startswith(("•", "– ", "- ", "* "))
        if (is_heading or is_bullet) and current:
            flush()
        current.append(ln)
        if is_heading:
            flush()
            continue
        if ln.endswith((".", "!", "?", ":")) and len(ln) < 0.82 * typical_len:
            flush()

    flush()
    return "\n\n".join(paragraphs)


def infer_page_section(norm_text: str, active_section: str) -> tuple[str, str]:
    """Deterministically infer the dominant section_type for a page and next active state."""
    if not norm_text:
        return active_section, active_section

    matches: list[tuple[float, str]] = []
    offset = 0
    total_len = max(1, len(norm_text))
    for para in norm_text.split("\n\n"):
        for sec_name, pattern in SECTION_RULES:
            if pattern.match(para):
                matches.append((offset / total_len, sec_name))
                break
        offset += len(para) + 2

    if not matches:
        return active_section, active_section

    first_rel, first_sec = matches[0]
    if first_rel <= 0.45:
        active_section = first_sec
    return active_section, matches[-1][1]


def load_pdf(
    path: Path,
    document_id: str,
    source: str,
    *,
    content_pages: list[int] | None = None,
    base_metadata: dict[str, Any] | None = None,
    page_logs: list[dict[str, Any]] | None = None,
) -> list[DocumentPage]:
    """Extract and normalize physical pages, logging empty or excluded pages."""
    reader = PdfReader(str(path))
    allowed = set(content_pages) if content_pages is not None else None
    active_section = "intro_related"
    pages: list[DocumentPage] = []

    for page_num, raw_page in enumerate(reader.pages, start=1):
        try:
            raw_text = raw_page.extract_text() or ""
        except Exception as exc:  # pragma: no cover - defensive PDF error capture
            if page_logs is not None:
                page_logs.append(
                    {"document_id": document_id, "page": page_num, "status": "extract_error", "error": str(exc)}
                )
            continue

        norm = normalize_text(raw_text)
        if not norm:
            if page_logs is not None:
                page_logs.append({"document_id": document_id, "page": page_num, "status": "empty_no_text"})
            continue

        if allowed is not None and page_num not in allowed:
            if page_logs is not None:
                page_logs.append(
                    {
                        "document_id": document_id,
                        "page": page_num,
                        "status": "excluded_by_manifest",
                        "chars": len(norm),
                    }
                )
            continue

        page_section, active_section = infer_page_section(norm, active_section)
        meta = dict(base_metadata or {})
        meta.setdefault("section_type", page_section)
        meta["normalized_chars"] = len(norm)

        if page_logs is not None:
            page_logs.append(
                {
                    "document_id": document_id,
                    "page": page_num,
                    "status": "included",
                    "section_type": page_section,
                    "chars": len(norm),
                }
            )
        pages.append(
            DocumentPage(
                document_id=document_id,
                source=source,
                page=page_num,
                text=norm,
                metadata=meta,
            )
        )
    return pages
