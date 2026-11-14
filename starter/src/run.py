"""Day 06 Data Ingestion pipeline runner: ingest, chunk, embed, index, evaluate, and export audit samples."""
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .chunking import by_paragraph, fixed_size
from .embeddings import MyEmbedder
from .evaluation import evaluate_query, summarize
from .ingestion import load_pdf
from .models import Chunk, DocumentPage
from .store import QdrantVectorStore

REPO_ROOT = Path(__file__).resolve().parents[2]
STARTER_DIR = Path(__file__).resolve().parents[1]


def _chunk_stats(chunks: list[Chunk]) -> dict[str, Any]:
    lengths = sorted(len(c.text) for c in chunks)
    n = len(lengths)
    if n == 0:
        return {"count": 0, "total_chars": 0}
    p95_idx = min(n - 1, int(0.95 * (n - 1)))
    return {
        "count": n,
        "total_chars": sum(lengths),
        "min_chars": lengths[0],
        "median_chars": int(statistics.median(lengths)),
        "mean_chars": round(statistics.mean(lengths), 1),
        "p95_chars": lengths[p95_idx],
        "max_chars": lengths[-1],
    }


def _select_coherence_positions(pages: list[DocumentPage]) -> list[dict[str, Any]]:
    """Deterministically pick 10 pre-defined source positions across all 4 papers before viewing scores."""
    per_doc_quota = {"2404.04302": 3, "2506.16988": 2, "2506.18027": 3, "2603.05698": 2}
    picked: list[dict[str, Any]] = []
    counts: dict[str, int] = {}

    for dp in sorted(pages, key=lambda p: (p.document_id, p.page)):
        quota = per_doc_quota.get(dp.document_id, 0)
        if counts.get(dp.document_id, 0) >= quota:
            continue
        paras = [p.strip() for p in dp.text.split("\n\n") if len(p.strip()) >= 120]
        if not paras:
            continue
        # Pick the first substantive body paragraph on the page (avoid title/author header block if longer body exists)
        target = paras[1] if len(paras) > 1 and dp.page == 1 else paras[0]
        anchor = target[:60]
        picked.append(
            {
                "position_id": f"coh_{len(picked) + 1:02d}",
                "document_id": dp.document_id,
                "page": dp.page,
                "anchor_start": anchor,
                "source_paragraph": target,
            }
        )
        counts[dp.document_id] = counts.get(dp.document_id, 0) + 1
    return picked


def _chunks_covering_position(chunks: list[Chunk], pos: dict[str, Any]) -> list[dict[str, Any]]:
    doc_id = pos["document_id"]
    page = pos["page"]
    anchor_head = pos["source_paragraph"][:40]
    anchor_mid = pos["source_paragraph"][40:80]
    matched = [
        {
            "chunk_id": c.chunk_id,
            "chars": len(c.text),
            "text": c.text,
        }
        for c in chunks
        if c.document_id == doc_id
        and c.page == page
        and (anchor_head in c.text or (anchor_mid and anchor_mid in c.text))
    ]
    return matched


def main() -> None:
    load_dotenv(STARTER_DIR / ".env")
    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("EMBEDDING_MODEL") or "gemini-embedding-001"

    manifest = json.loads((REPO_ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))
    benchmark = json.loads((REPO_ROOT / "benchmark" / "questions.json").read_text(encoding="utf-8"))
    questions = benchmark["questions"]

    # 1. Load & normalize corpus per manifest
    t0 = time.perf_counter()
    page_logs: list[dict[str, Any]] = []
    all_pages: list[DocumentPage] = []
    for doc in manifest["documents"]:
        pdf_path = REPO_ROOT / doc["local_path"]
        loaded = load_pdf(
            pdf_path,
            doc["source_id"],
            doc["local_path"],
            content_pages=doc["content_pages"],
            base_metadata={"year": doc["year"], "topic": doc["topic"]},
            page_logs=page_logs,
        )
        all_pages.extend(loaded)
    ingest_parse_seconds = round(time.perf_counter() - t0, 3)

    if len(all_pages) < 30:
        raise RuntimeError(f"Need at least 30 content pages, loaded only {len(all_pages)}")

    # 2. Build chunks for A (fixed_size 500/50), B (by_paragraph 650), C (identical to B)
    chunks_a = [c for dp in all_pages for c in fixed_size(dp, size=500, overlap=50)]
    chunks_b = [c for dp in all_pages for c in by_paragraph(dp, max_size=650)]
    chunks_c = chunks_b  # B and C share identical chunks/vectors; only query-time filter differs

    # 3. Real Gemini embeddings (no mock fallback)
    embedder = MyEmbedder(model_name=model_name, api_key=api_key, batch_size=32)

    t_emb_a = time.perf_counter()
    vecs_a = embedder.embed_documents([c.text for c in chunks_a])
    calls_a = embedder.api_calls
    chars_a = embedder.embedded_chars
    embed_a_seconds = round(time.perf_counter() - t_emb_a, 3)

    t_emb_b = time.perf_counter()
    vecs_b = embedder.embed_documents([c.text for c in chunks_b])
    calls_b = embedder.api_calls - calls_a
    chars_b = embedder.embedded_chars - chars_a
    embed_b_seconds = round(time.perf_counter() - t_emb_b, 3)

    # 4. Persistent local Qdrant vector store
    qdrant_path = STARTER_DIR / "results" / "qdrant_db"
    store = QdrantVectorStore(storage_path=str(qdrant_path), dimensions=embedder.dimensions, strategies=("A", "B", "C"))
    try:
        store.create_schema()

        t_up_a = time.perf_counter()
        store.upsert("A", chunks_a, vecs_a)
        upsert_a_seconds = round(time.perf_counter() - t_up_a, 3)

        t_up_b = time.perf_counter()
        store.upsert("B", chunks_b, vecs_b)
        upsert_b_seconds = round(time.perf_counter() - t_up_b, 3)

        t_up_c = time.perf_counter()
        store.upsert("C", chunks_c, vecs_b)
        upsert_c_seconds = round(time.perf_counter() - t_up_c, 3)

        # Verify re-ingest idempotency (same version reloaded -> zero duplicate rows)
        before_reload = {s: store.count(s) for s in ("A", "B", "C")}
        first_doc_id = manifest["documents"][0]["source_id"]
        doc_chunks_a = [c for c in chunks_a if c.document_id == first_doc_id]
        doc_vecs_a = [v for c, v in zip(chunks_a, vecs_a) if c.document_id == first_doc_id]
        store.upsert("A", doc_chunks_a, doc_vecs_a)
        after_reload_a = store.count("A")

        # 5. Embed all 20 benchmark queries once with RETRIEVAL_QUERY
        t_qemb = time.perf_counter()
        query_vecs = [embedder.embed_query(q["question"]) for q in questions]
        query_embed_seconds = round(time.perf_counter() - t_qemb, 3)

        # 6. Run top-3 search & evaluate per strategy
        all_traces: list[dict[str, Any]] = []
        query_timings: dict[str, float] = {}
        strategy_configs = {
            "A": {"use_filter": False, "chunking": "fixed_size(500, 50)"},
            "B": {"use_filter": False, "chunking": "by_paragraph(650)"},
            "C": {"use_filter": True, "chunking": "by_paragraph(650) + metadata_filter"},
        }

        for strategy, cfg in strategy_configs.items():
            t_search = time.perf_counter()
            for q, qvec in zip(questions, query_vecs):
                q_filter = q["filter"] if (cfg["use_filter"] and q.get("filter")) else None
                hits = store.search(strategy, qvec, top_k=3, filters=q_filter)
                tr = evaluate_query(q, hits)
                tr["strategy"] = strategy
                tr["applied_filter"] = q_filter
                all_traces.append(tr)
            query_timings[strategy] = round(time.perf_counter() - t_search, 3)

        summary = summarize(all_traces)

        # 7. Export results & audit artifacts
        results_dir = REPO_ROOT / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        ingestion_report = {
            "corpus_name": manifest["corpus_name"],
            "loaded_content_pages": len(all_pages),
            "parse_seconds": ingest_parse_seconds,
            "page_logs": page_logs,
            "reload_idempotency_check": {
                "before_reload_counts": before_reload,
                "reloaded_document_id": first_doc_id,
                "after_reload_count_A": after_reload_a,
                "no_duplicates": after_reload_a == before_reload["A"],
            },
            "chunk_stats": {
                "A": _chunk_stats(chunks_a),
                "B": _chunk_stats(chunks_b),
                "C": _chunk_stats(chunks_c),
            },
            "embedding": {
                "provider": "google-genai",
                "model_name": embedder.model_name,
                "dimensions": embedder.dimensions,
                "document_task_type": "RETRIEVAL_DOCUMENT",
                "query_task_type": "RETRIEVAL_QUERY",
                "api_calls_total": embedder.api_calls,
                "embedded_chars_total": embedder.embedded_chars,
                "embed_seconds_A": embed_a_seconds,
                "embed_seconds_B_shared_with_C": embed_b_seconds,
                "query_embed_seconds": query_embed_seconds,
                "api_calls_A": calls_a,
                "api_calls_B_shared_with_C": calls_b,
                "cost_usd_note": "Gemini API free tier / studio key; exact monetary cost depends on billing tier (0 USD if within free tier quota).",
            },
            "vector_store": {
                "engine": "qdrant-client (local persistent disk storage)",
                "storage_path": str(qdrant_path.relative_to(REPO_ROOT)),
                "metric": "COSINE",
                "search_mode": "exact brute-force scan in Qdrant local Python mode",
                "upsert_seconds": {"A": upsert_a_seconds, "B": upsert_b_seconds, "C": upsert_c_seconds},
                "query_seconds": query_timings,
            },
        }
        (results_dir / "ingestion_log.json").write_text(
            json.dumps(ingestion_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

        traces_payload = {
            "summary": summary,
            "strategy_configs": strategy_configs,
            "traces": all_traces,
        }
        (results_dir / "traces_all.json").write_text(
            json.dumps(traces_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

        # Pre-selected 10 source positions for manual chunk coherence scoring
        positions = _select_coherence_positions(all_pages)
        coherence_samples = []
        for pos in positions:
            coherence_samples.append(
                {
                    **pos,
                    "A_chunks": _chunks_covering_position(chunks_a, pos),
                    "B_C_chunks": _chunks_covering_position(chunks_b, pos),
                }
            )
        (results_dir / "coherence_samples.json").write_text(
            json.dumps(coherence_samples, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

        # Extract the 5 test questions x top-3 hits per strategy for manual metadata & grounding audit
        test_audit = [tr for tr in all_traces if tr["split"] == "test"]
        (results_dir / "test_audit_samples.json").write_text(
            json.dumps(test_audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

        print("=== SUMMARY ===")
        print(json.dumps(summary, indent=2))
        print("Reload check A:", before_reload["A"], "->", after_reload_a)
        print("Total API calls:", embedder.api_calls, "| Total embedded chars:", embedder.embedded_chars)
    finally:
        store.close()


if __name__ == "__main__":
    main()
