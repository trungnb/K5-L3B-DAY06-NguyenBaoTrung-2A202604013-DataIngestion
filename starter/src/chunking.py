"""Fixed-size and paragraph-aware chunking preserving page metadata."""
import re

from .models import Chunk, DocumentPage

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _split_long_unit(text: str, max_size: int) -> list[str]:
    """Split an oversized text block on sentence boundaries, then words if needed."""
    if len(text) <= max_size:
        return [text]

    parts: list[str] = []
    buf = ""
    for sent in SENTENCE_SPLIT_RE.split(text):
        sent = sent.strip()
        if not sent:
            continue
        if len(sent) > max_size:
            if buf:
                parts.append(buf)
                buf = ""
            # Single sentence exceeds max_size -> split cleanly on word boundaries
            words = sent.split()
            wbuf = ""
            for w in words:
                cand = f"{wbuf} {w}".strip() if wbuf else w
                if len(cand) <= max_size:
                    wbuf = cand
                else:
                    if wbuf:
                        parts.append(wbuf)
                    wbuf = w[:max_size] if len(w) > max_size else w
            if wbuf:
                parts.append(wbuf)
            continue

        cand = f"{buf} {sent}".strip() if buf else sent
        if len(cand) <= max_size:
            buf = cand
        else:
            if buf:
                parts.append(buf)
            buf = sent

    if buf:
        parts.append(buf)
    return parts


def fixed_size(page: DocumentPage, size: int, overlap: int) -> list[Chunk]:
    """Split normalized page text into overlapping character windows."""
    if size <= 0:
        raise ValueError("size must be > 0")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must satisfy 0 <= overlap < size")
    if not page.text:
        return []

    step = size - overlap
    chunks: list[Chunk] = []
    start = 0
    idx = 0
    text_len = len(page.text)

    while start < text_len:
        end = min(start + size, text_len)
        window = page.text[start:end].strip()
        if window:
            meta = {**page.metadata, "document_id": page.document_id, "source": page.source, "page": page.page, "chunk_index": idx}
            chunks.append(
                Chunk(
                    chunk_id=f"{page.document_id}:fixed:p{page.page:02d}:{idx}",
                    document_id=page.document_id,
                    source=page.source,
                    page=page.page,
                    text=window,
                    metadata=meta,
                )
            )
            idx += 1
        if end == text_len:
            break
        start += step

    return chunks


def by_paragraph(page: DocumentPage, max_size: int) -> list[Chunk]:
    """Group paragraphs up to max_size and split oversized blocks on sentences."""
    if max_size <= 0:
        raise ValueError("max_size must be > 0")
    if not page.text:
        return []

    raw_paras = [p.strip() for p in page.text.split("\n\n") if p.strip()]
    units: list[str] = []
    for para in raw_paras:
        units.extend(_split_long_unit(para, max_size))

    chunks: list[Chunk] = []
    buf = ""
    idx = 0

    def emit(body: str) -> None:
        nonlocal idx
        meta = {**page.metadata, "document_id": page.document_id, "source": page.source, "page": page.page, "chunk_index": idx}
        chunks.append(
            Chunk(
                chunk_id=f"{page.document_id}:para:p{page.page:02d}:{idx}",
                document_id=page.document_id,
                source=page.source,
                page=page.page,
                text=body,
                metadata=meta,
            )
        )
        idx += 1

    for unit in units:
        cand = f"{buf}\n\n{unit}" if buf else unit
        if len(cand) <= max_size:
            buf = cand
        else:
            if buf:
                emit(buf)
            buf = unit

    if buf:
        emit(buf)
    return chunks
