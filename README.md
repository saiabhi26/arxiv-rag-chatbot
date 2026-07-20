# arXiv ML Chatbot

A retrieval-augmented chatbot over the full text of **recent arXiv machine-learning papers**
(`cs.CL` / `cs.LG`), built to answer questions about research published *after* a general-purpose
LLM's training cutoff — grounded in the papers, and with a **calibrated refusal when retrieval comes
back weak**.

> **The pitch:** ask it about ML research from the last few months. A general model answers those from
> memory or not at all; this answers from the actual papers — and when the corpus doesn't cover your
> question, it says so instead of making something up.

That refusal is the point as much as the answers are. A RAG system that confidently hallucinates over
a weak match is worse than one that says "I don't have enough information," so the retrieval gate is a
deliberate, tunable component rather than an afterthought.

## Live demo

**[→ live demo](https://arxiv-rag-chatbot.streamlit.app/)**

> ⏳ Hosted on a free tier that sleeps when idle — the first load after a quiet period may take ~30s
> to wake and load the embedding model + index. Subsequent questions are fast.

## How it works

A three-stage pipeline; `app.py` wires the stages together and each module owns one artifact.

```
scraper.py + pdf_parser.py ─► data/documents.json      (198 papers, full text, cleaned)
retriever.py               ─► data/faiss_index.bin + data/chunks.json

query ─► router.py   (LLM intent: chit-chat · overview · knowledge)
              │
              ├─ chit-chat ─► LLM replies directly
              ├─ overview  ─► answered from corpus facts (top papers by upvotes, counts, date span)
              └─ knowledge ─► dense retrieval (FAISS, cosine) top-5
                                   │
                                   ▼  gate: max similarity < threshold ─► "I don't have enough information"
                                   ▼
                              generator.py (Claude) ─► grounded answer + sources
```

- **Retrieval** — MiniLM (`all-MiniLM-L6-v2`) embeddings, normalized, in a FAISS `IndexFlatIP` index,
  so scores are cosine similarity (bounded 0–1, higher is better). Each chunk is prefixed with its
  `"{title} — {section}"` before embedding.
- **The gate** — the app compares the best similarity against a threshold and refuses rather than
  answering over weak matches. *The current threshold is provisional* — it gets calibrated against an
  eval set in Part 2 (see Status).
- **Generation** — Claude (`claude-haiku-4-5`) behind a thin, provider-agnostic `Generator` protocol,
  instructed to answer **only** from the retrieved context. Missing-key, rate-limit (429), and
  server (5xx) conditions degrade to a polite message instead of a stack trace.
- **Cost guard** — because generation runs on a paid API, a public deploy carries per-session and
  per-day request caps so the demo can't be drained by a burst or a bot.

## Run it locally

```bash
pip install -r requirements.txt

# Add your Anthropic API key (get one at https://console.anthropic.com/settings/keys)
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# then edit .streamlit/secrets.toml and paste your key

streamlit run app.py
```

The index artifacts (`data/faiss_index.bin`, `data/chunks.json`) are committed, so the app runs
without rebuilding anything. To rebuild the corpus and index from scratch (offline, on a laptop —
the app never scrapes at serve time), install the extra scraper dependencies first:

```bash
pip install -r requirements-dev.txt   # adds pymupdf for PDF parsing
python scraper.py     # re-download + parse papers -> data/documents.json  (needs network; ~500MB of PDFs, gitignored)
python retriever.py   # re-embed + build the index  -> data/faiss_index.bin + data/chunks.json
```

> `data/faiss_index.bin` and `data/chunks.json` are a coupled pair — rerun `retriever.py` as a unit;
> regenerating one without the other returns the wrong text for a vector.

## Status

**Part 1 — a working baseline — is done:** arXiv corpus, cosine retrieval + gate, Claude generation,
LLM routing, corpus-overview answers, and the deploy cost guard.

**Part 2 — measured improvements — is next**, and is where the interesting work lives:

- an **eval harness** (labeled questions, including deliberately unanswerable ones) reporting hit@k,
  MRR, and the gate's false-refusal / false-answer rates;
- **cross-encoder reranking** and **hybrid dense+BM25 retrieval** (fused with hand-rolled reciprocal
  rank fusion), each shipped with a measured before/after;
- **threshold calibration** against the eval set (replacing today's provisional value);
- **citations** and **streaming**.

The ablation table lands here once the harness exists — every retrieval change reported as a measured
before/after rather than a claim. See the
[design spec](docs/superpowers/specs/2026-07-12-portfolio-rag-upgrade-design.md) for the full plan,
the evidence behind each decision, and the trade-offs taken.
