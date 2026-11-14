"""Evaluate retrieved evidence against gold source/page and preserve full traces."""
import re
from collections import defaultdict
from typing import Any


def _norm_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _accepted_golds(question: dict[str, Any]) -> list[tuple[str, int]]:
    golds = [(str(question["gold_source_id"]), int(question["gold_page"]))]
    for alt in question.get("alternative_gold", []):
        golds.append((str(alt["gold_source_id"]), int(alt["gold_page"])))
    return golds


def _clause_sets(question: dict[str, Any]) -> list[list[str]]:
    sets: list[list[str]] = []
    primary = question.get("required_evidence_clauses")
    if primary:
        sets.append([_norm_ws(c) for c in primary])
    else:
        sets.append([_norm_ws(question["evidence_quote"])])
    for alt in question.get("alternative_evidence_clauses", []):
        if alt:
            sets.append([_norm_ws(c) for c in alt])
    return sets


def evaluate_query(question: dict[str, Any], hits: list[dict[str, Any]]) -> dict[str, Any]:
    """Check gold source/page and sufficient evidence clauses per hit without guessing manual labels."""
    accepted_golds = set(_accepted_golds(question))
    clause_sets = _clause_sets(question)

    hit_details: list[dict[str, Any]] = []
    gold_page_texts: list[str] = []
    first_hit_rank: int | None = None

    for rank, hit in enumerate(hits, start=1):
        meta = hit.get("metadata", {})
        doc_id = str(meta.get("document_id", ""))
        page = meta.get("page")
        norm_chunk = _norm_ws(hit.get("text", ""))
        on_gold_page = (doc_id, int(page)) in accepted_golds if page is not None else False

        is_evidence_hit = False
        if on_gold_page:
            gold_page_texts.append(norm_chunk)
            for reqs in clause_sets:
                if all(clause in norm_chunk for clause in reqs):
                    is_evidence_hit = True
                    break

        if is_evidence_hit and first_hit_rank is None:
            first_hit_rank = rank

        hit_details.append(
            {
                "rank": rank,
                "chunk_id": hit.get("chunk_id"),
                "score": hit.get("score"),
                "metadata": meta,
                "on_gold_page": on_gold_page,
                "is_evidence_hit": is_evidence_hit,
                "text": hit.get("text", ""),
            }
        )

    hit_at_1 = first_hit_rank == 1
    hit_at_3 = first_hit_rank is not None and first_hit_rank <= 3

    combined_gold = " ".join(gold_page_texts)
    split_across_chunks = False
    if not hit_at_3 and combined_gold:
        for reqs in clause_sets:
            if all(clause in combined_gold for clause in reqs):
                split_across_chunks = True
                break

    return {
        "question_id": question["id"],
        "split": question["split"],
        "type": question.get("type"),
        "filter_used": question.get("filter"),
        "gold_source_id": question["gold_source_id"],
        "gold_page": question["gold_page"],
        "hit_at_1": hit_at_1,
        "hit_at_3": hit_at_3,
        "first_hit_rank": first_hit_rank,
        "split_across_chunks": split_across_chunks,
        "top_hits": hit_details,
    }


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate exact Hit@1 and Evidence Hit@3 fractions by strategy and dev/test split."""
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for tr in traces:
        grouped[tr["strategy"]][tr["split"]].append(tr)

    summary: dict[str, Any] = {}
    for strategy, splits in grouped.items():
        summary[strategy] = {}
        for split, rows in splits.items():
            n = len(rows)
            h1 = sum(1 for r in rows if r["hit_at_1"])
            h3 = sum(1 for r in rows if r["hit_at_3"])
            splits_err = sum(1 for r in rows if r.get("split_across_chunks"))
            summary[strategy][split] = {
                "hit_at_1": {"hits": h1, "total": n, "rate": round(h1 / n, 4) if n else 0.0},
                "hit_at_3": {"hits": h3, "total": n, "rate": round(h3 / n, 4) if n else 0.0},
                "split_across_chunks_count": splits_err,
            }
    return summary
