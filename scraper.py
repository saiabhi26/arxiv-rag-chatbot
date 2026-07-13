"""Build the corpus: notable recent ML papers → data/documents.json

The corpus is deliberately *recent* (last ~month) so that it sits after a
general-purpose LLM's training cutoff — that is the whole premise of the demo,
and it's verifiable by date rather than by guesswork.

**Selection.** arXiv publishes hundreds of ML papers a day and exposes no
popularity signal whatsoever — only date and relevance. Taking the newest N
therefore returns whatever was posted in the last 48 hours: a lumpy, arbitrary
slice. And the obvious popularity proxy, citation count, is useless here, since
a three-week-old paper has essentially none — recency and citations are in
direct conflict.

So papers are selected from **Hugging Face's daily papers feed**, which carries
community upvotes and is the one popularity signal that exists for brand-new ML
work. We take the top ~PAPERS_PER_WEEK of each week's ~300 candidates. The IDs
are arXiv IDs, so the PDFs still come from arXiv.

Full text, not abstracts: abstracts make retrieval near-trivial, which would
leave no headroom for reranking and hybrid search to show up in the eval.

This runs **offline**, on a laptop — never at serve time. The live app loads the
prebuilt artifacts and never touches arXiv. PDF → text is pdf_parser.py's job.
"""

import json
import os
import time
import urllib.request
from datetime import date, timedelta

from pdf_parser import extract_sections

WEEKS_BACK = 4
PAPERS_PER_WEEK = 50

# The corpus is rewritten wholesale and the PDF cache pruned to match, so a
# failed upstream fetch must never be mistaken for "there are no papers" — that
# would overwrite documents.json with [] and delete every cached PDF. Refuse to
# publish a corpus this far below expectation.
MIN_PAPERS = PAPERS_PER_WEEK * WEEKS_BACK // 2

PDF_DIR = "data/pdfs"
DOCS_PATH = "data/documents.json"

DAILY_PAPERS_URL = "https://huggingface.co/api/daily_papers?date={date}"
PDF_URL = "https://arxiv.org/pdf/{arxiv_id}"

USER_AGENT = "arxiv-rag-chatbot/1.0 (https://github.com/saiabhi26/arxiv-rag-chatbot)"
DOWNLOAD_DELAY = 1.0  # seconds between PDF fetches


def fetch_daily_papers(day):
    """Papers featured on Hugging Face's daily feed for one day."""
    request = urllib.request.Request(
        DAILY_PAPERS_URL.format(date=day.isoformat()),
        headers={"User-Agent": USER_AGENT},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        entries = json.load(response)

    papers = []
    for entry in entries:
        paper = entry.get("paper", {})
        # Skip a malformed record rather than letting it raise: the caller
        # catches per-day, so one bad entry would otherwise discard every paper
        # featured that day.
        if not paper.get("id") or not paper.get("publishedAt"):
            continue
        papers.append({
            "arxiv_id": paper["id"],
            "title": paper.get("title", "").strip().replace("\n", " "),
            "date": paper["publishedAt"][:10],
            "upvotes": paper.get("upvotes", 0),
        })
    return papers


def select_papers():
    """The most-upvoted papers from each of the last WEEKS_BACK weeks.

    Ranking within each week rather than across the whole window keeps the corpus
    spread evenly — otherwise one viral week would crowd out the other three.
    """
    selected = {}
    today = date.today()

    for week in range(WEEKS_BACK):
        week_end = today - timedelta(days=7 * week)
        candidates = {}
        for offset in range(7):
            day = week_end - timedelta(days=offset)
            try:
                for paper in fetch_daily_papers(day):
                    candidates[paper["arxiv_id"]] = paper
            except Exception as exc:  # a quiet day or a hiccup — not fatal
                print(f"  ⚠️  {day}: {exc}")

        ranked = sorted(candidates.values(), key=lambda p: -p["upvotes"])
        keep = ranked[:PAPERS_PER_WEEK]
        for paper in keep:
            selected.setdefault(paper["arxiv_id"], paper)

        if keep:
            print(f"  week to {week_end}: kept {len(keep)} of {len(candidates)} "
                  f"({keep[-1]['upvotes']}–{keep[0]['upvotes']} upvotes)")

    print(f"  → {len(selected)} unique papers across {WEEKS_BACK} weeks")
    return list(selected.values())


def download_pdf(arxiv_id):
    """Download the paper's PDF from arXiv, or reuse the cached copy.

    arXiv asks unidentified bulk clients to back off, hence the User-Agent and
    the delay between fetches.
    """
    path = os.path.join(PDF_DIR, f"{arxiv_id}.pdf")
    if os.path.exists(path):
        return path

    request = urllib.request.Request(
        PDF_URL.format(arxiv_id=arxiv_id), headers={"User-Agent": USER_AGENT}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        pdf_bytes = response.read()

    # arXiv answers 200 with an HTML page for papers whose PDF isn't built yet.
    # Caching those bytes under a .pdf name would poison the cache permanently:
    # the exists() check above would keep short-circuiting to a file that can
    # never be parsed, so the paper would silently vanish from every future run.
    if not pdf_bytes.startswith(b"%PDF"):
        raise ValueError("response is not a PDF (arXiv may still be generating it)")

    # Write via a temp name so an interrupted run can't leave a truncated PDF in
    # the cache that later runs would happily reuse.
    tmp_path = path + ".part"
    with open(tmp_path, "wb") as f:
        f.write(pdf_bytes)
    os.replace(tmp_path, path)

    time.sleep(DOWNLOAD_DELAY)
    return path


def prune_pdf_cache(keep):
    """Delete cached PDFs for papers that have aged out of the window.

    Also sweeps .part files, which an interrupted download leaves behind and
    nothing else would ever clean up.
    """
    pruned = 0
    for filename in os.listdir(PDF_DIR):
        path = os.path.join(PDF_DIR, filename)
        if filename.endswith(".part"):
            os.remove(path)
        elif filename.endswith(".pdf") and filename.removesuffix(".pdf") not in keep:
            os.remove(path)
            pruned += 1
    return pruned


def run_scraper():
    os.makedirs(PDF_DIR, exist_ok=True)

    print(f"Selecting top {PAPERS_PER_WEEK} papers/week from the last {WEEKS_BACK} weeks...")
    papers = select_papers()

    # If Hugging Face is unreachable, every per-day fetch fails, select_papers()
    # swallows the errors, and we arrive here with an empty list that looks
    # exactly like "there were no papers this month". Publishing it would
    # overwrite the corpus with [] and prune the entire PDF cache — silently,
    # with exit code 0. Fail loudly and leave the existing corpus intact.
    if len(papers) < MIN_PAPERS:
        raise SystemExit(
            f"❌ Only {len(papers)} papers selected (expected ≥{MIN_PAPERS}). "
            f"Upstream feed is probably down — refusing to overwrite "
            f"{DOCS_PATH} and prune the PDF cache."
        )

    documents = []
    seen = set()
    failed = 0
    for i, paper in enumerate(papers, 1):
        print(f"[{i}/{len(papers)}] {paper['arxiv_id']} "
              f"({paper['upvotes']}▲) {paper['title'][:55]}...")
        try:
            pdf_path = download_pdf(paper["arxiv_id"])
            sections = extract_sections(pdf_path)
        except Exception as exc:  # some PDFs are malformed, withdrawn, or scans
            print(f"  ⚠️  skipped: {exc}")
            failed += 1
            continue

        # A paper whose headings are never detected yields nothing at all. Say so
        # — otherwise it is neither counted as failed nor present in the corpus,
        # and simply disappears.
        if not sections:
            print("  ⚠️  no sections extracted (heading detection found nothing)")
            failed += 1
            continue

        for section in sections:
            # A paper can repeat a section verbatim (a boxed summary reprinted
            # in an appendix); an exact duplicate chunk is dead weight in the
            # index and can crowd a real match out of the top-k.
            fingerprint = (paper["arxiv_id"], section["text"])
            if fingerprint in seen:
                continue
            seen.add(fingerprint)

            documents.append({
                "title": paper["title"],
                "section": section["section"],
                "text": section["text"],
                "arxiv_id": paper["arxiv_id"],
                "date": paper["date"],
                "upvotes": paper["upvotes"],
            })

    # Rewritten wholesale, never appended to: the corpus is a rolling window, not
    # an archive. Growth would cost repo size, build time, and — since
    # IndexFlat* is a brute-force scan — query latency, linearly.
    with open(DOCS_PATH, "w") as f:
        json.dump(documents, f, indent=2)

    pruned = prune_pdf_cache(keep={p["arxiv_id"] for p in papers})

    papers_kept = len({d["arxiv_id"] for d in documents})
    print(f"\n✅ {len(documents)} sections from {papers_kept} papers → {DOCS_PATH}")
    if failed:
        print(f"   ({failed} papers skipped on extraction failure)")
    if pruned:
        print(f"   ({pruned} cached PDFs pruned — papers no longer in the window)")


if __name__ == "__main__":
    run_scraper()
