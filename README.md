# arXiv RAG Chatbot

A retrieval-augmented chatbot over the full text of recent arXiv machine learning papers
(`cs.CL` / `cs.LG`), built to answer questions about research published *after* a general-purpose
LLM's training cutoff — with citations, and with a calibrated refusal when retrieval comes back weak.

That refusal is the point as much as the answers are. A RAG system that confidently hallucinates over
a bad match is worse than one that says "I don't know," so the retrieval gate is measured and tuned
rather than guessed at.

## Status

🚧 Under active reconstruction. The project is being rebuilt from an earlier Wikipedia-backed
prototype; see [the design spec](docs/superpowers/specs/2026-07-12-portfolio-rag-upgrade-design.md)
for the plan, the evidence behind it, and the decisions taken.

Where it's going:

- **Corpus** — full-text arXiv papers, scraped and cleaned, instead of Wikipedia articles.
- **Retrieval** — hybrid dense (FAISS) + sparse (BM25) candidate generation fused with reciprocal
  rank fusion, then reordered by a cross-encoder reranker.
- **Generation** — an API-backed LLM behind a thin provider-agnostic interface, streaming, with
  answers attributed to their source papers.
- **Evaluation** — a labeled question set (including deliberately unanswerable questions) measuring
  hit@k, MRR, and the gate's false-refusal / false-answer rates, so every retrieval change ships with
  a measured before/after rather than a claim.

Setup and usage instructions will land once the pipeline is rebuilt.
