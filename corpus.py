"""Corpus-level facts for meta questions about the paper collection.

Content-RAG (retrieve top-k chunks by embedding similarity) can't answer
questions about the collection itself ("how many papers?", "what's the most
popular?") because the answer lives in per-paper metadata (`upvotes`, `date`),
not in any single chunk's text. This module computes a small plain-text
summary of that metadata once at startup; app.py hands it to the generator
as the sole context for the "overview" intent, with no retrieval and no
similarity gate.
"""

import json


def compute_overview_context(chunks: list[dict], top_n: int = 10) -> str:
    """Build a plain-text summary of the corpus: paper count, date span, and
    the top `top_n` papers by upvotes. `chunks` is the full list of per-chunk
    records; multiple chunks share an `arxiv_id`, so we dedup by it first to
    get one record per paper.
    """
    papers_by_id = {}
    for chunk in chunks:
        arxiv_id = chunk.get("arxiv_id")
        if arxiv_id not in papers_by_id:
            papers_by_id[arxiv_id] = chunk

    papers = list(papers_by_id.values())

    def _coerce_upvotes(record):
        try:
            return int(str(record.get("upvotes") or 0))
        except (ValueError, TypeError):
            return 0

    dates = sorted(d for d in (p.get("date") for p in papers) if d)
    date_span = f"{dates[0]} to {dates[-1]}" if dates else "unknown"

    top_papers = sorted(papers, key=_coerce_upvotes, reverse=True)[:top_n]

    lines = [
        f"This corpus contains {len(papers)} recent arXiv ML papers "
        f"(dated {date_span}). The most upvoted papers are:"
    ]
    for paper in top_papers:
        lines.append(
            f"- {paper.get('title', 'Untitled')} "
            f"({_coerce_upvotes(paper)} upvotes, {paper.get('date', 'unknown date')})"
        )

    return "\n".join(lines)


if __name__ == "__main__":
    with open("data/chunks.json") as f:
        chunks = json.load(f)
    print(compute_overview_context(chunks))
