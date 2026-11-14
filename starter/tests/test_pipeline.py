"""Focused tests for Day 06 ingestion, chunking, vector store, and evaluation."""
from pathlib import Path

import pytest

from src.chunking import by_paragraph, fixed_size
from src.evaluation import evaluate_query, summarize
from src.ingestion import normalize_text
from src.models import Chunk, DocumentPage
from src.store import QdrantVectorStore


def test_normalize_text_dehyphenates_and_preserves_paragraphs():
    raw = (
        "1 Introduction\n"
        "Retrieval-Augmented Genera-\n"
        "tion reduces hallucina-\n"
        "tions.\n"
        "2 Methodology\n"
        "Short line end.\n"
        "Next paragraph starts here."
    )
    out = normalize_text(raw)
    assert "Generation" in out
    assert "hallucinations." in out
    paras = out.split("\n\n")
    assert paras[0] == "1 Introduction"
    assert "2 Methodology" in paras
    assert "Next paragraph starts here." in paras


def test_fixed_size_overlap_and_metadata():
    page = DocumentPage(
        document_id="doc1",
        source="starter/data/doc1.pdf",
        page=3,
        text="A" * 120,
        metadata={"year": 2025, "section_type": "methodology"},
    )
    chunks = fixed_size(page, size=50, overlap=10)
    assert [len(c.text) for c in chunks] == [50, 50, 40]
    assert all(c.document_id == "doc1" and c.page == 3 for c in chunks)
    assert all(c.metadata["year"] == 2025 for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)

    with pytest.raises(ValueError):
        fixed_size(page, size=50, overlap=50)


def test_by_paragraph_groups_and_splits_oversized_on_sentences():
    text = (
        "3.1 Overview\n\n"
        "Short intro.\n\n"
        "Sentence one is here. Sentence two is also here. Sentence three finishes the block."
    )
    page = DocumentPage(
        document_id="doc2",
        source="starter/data/doc2.pdf",
        page=2,
        text=text,
        metadata={"year": 2026, "section_type": "methodology"},
    )
    chunks = by_paragraph(page, max_size=55)
    assert all(len(c.text) <= 55 for c in chunks)
    joined = " ".join(c.text for c in chunks)
    assert "3.1 Overview" in joined
    assert "Sentence two is also here." in joined
    assert all(c.metadata["section_type"] == "methodology" for c in chunks)


def test_qdrant_store_upsert_cleanup_filter_and_validation(tmp_path: Path):
    store = QdrantVectorStore(storage_path=str(tmp_path / "qdrant"), dimensions=4)
    store.create_schema()

    c1 = Chunk("doc1:B:p1:0", "doc1", "doc1.pdf", 1, "alpha text", {"year": 2025, "section_type": "methodology"})
    c2 = Chunk("doc1:B:p1:1", "doc1", "doc1.pdf", 1, "beta text", {"year": 2026, "section_type": "evaluation"})
    v1 = [1.0, 0.0, 0.0, 0.0]
    v2 = [0.0, 1.0, 0.0, 0.0]

    store.upsert("B", [c1, c2], [v1, v2])
    assert store.count("B") == 2

    # Re-ingest identical document -> no duplicates
    store.upsert("B", [c1, c2], [v1, v2])
    assert store.count("B") == 2

    # Filter search
    hits = store.search("B", [0.9, 0.1, 0.0, 0.0], top_k=3, filters={"year": 2026, "section_type": "evaluation"})
    assert len(hits) == 1
    assert hits[0]["chunk_id"] == "doc1:B:p1:1"
    assert hits[0]["metadata"]["year"] == 2026

    # Re-ingest changed doc with obsolete chunk removed
    store.upsert("B", [c1], [v1])
    assert store.count("B") == 1

    # Trust-boundary validation
    with pytest.raises(ValueError):
        store.search("B", v1, top_k=3, filters={"unknown_field": "x"})
    with pytest.raises(ValueError):
        store.upsert("B", [c1], [[1.0, 0.0]])


def test_evaluate_query_requires_sufficient_evidence_and_detects_splits():
    q = {
        "id": "q01",
        "split": "dev",
        "question": "What are A and B?",
        "gold_source_id": "doc1",
        "gold_page": 5,
        "evidence_quote": "Alpha is 10 and Beta is 20.",
        "required_evidence_clauses": ["Alpha is 10", "Beta is 20"],
        "alternative_gold": [{"gold_source_id": "doc2", "gold_page": 2}],
    }
    # Split across two chunks -> neither chunk alone is a hit, flag split_across_chunks
    split_hits = [
        {"chunk_id": "c1", "text": "Alpha is 10 only.", "score": 0.9, "metadata": {"document_id": "doc1", "page": 5}},
        {"chunk_id": "c2", "text": "Beta is 20 only.", "score": 0.8, "metadata": {"document_id": "doc1", "page": 5}},
    ]
    tr_split = evaluate_query(q, split_hits)
    assert tr_split["hit_at_1"] is False
    assert tr_split["hit_at_3"] is False
    assert tr_split["split_across_chunks"] is True

    # Alternative gold source with all clauses in rank 2 -> hit@3 true, hit@1 false
    alt_hits = [
        {"chunk_id": "c0", "text": "Irrelevant", "score": 0.95, "metadata": {"document_id": "doc9", "page": 1}},
        {"chunk_id": "c3", "text": "Alpha is 10 and Beta is 20.", "score": 0.85, "metadata": {"document_id": "doc2", "page": 2}},
    ]
    tr_alt = evaluate_query(q, alt_hits)
    assert tr_alt["hit_at_1"] is False
    assert tr_alt["hit_at_3"] is True

    summary = summarize([{"strategy": "A", **tr_split}, {"strategy": "A", **tr_alt}])
    assert summary["A"]["dev"]["hit_at_1"] == {"hits": 0, "total": 2, "rate": 0.0}
    assert summary["A"]["dev"]["hit_at_3"] == {"hits": 1, "total": 2, "rate": 0.5}
