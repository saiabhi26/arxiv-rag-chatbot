import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


# Chunking config: word-based, with overlap so context isn't lost at
# chunk boundaries. Replaces the old approach of truncating each
# document to its first 512 characters (which discarded everything
# past that point and could cut off mid-sentence).
CHUNK_SIZE_WORDS = 150
CHUNK_OVERLAP_WORDS = 30

# Embedding model. Must be identical between index build and query time — the
# committed index holds vectors in THIS model's embedding space.
EMBED_MODEL = "all-MiniLM-L6-v2"
INDEX_PATH = "data/faiss_index.bin"
DOCS_PATH = "data/documents.json"
# Per-chunk records, positionally coupled to the FAISS index: chunks[i] is the
# text+metadata for index vector i. Replaces the old texts.npy (which stored
# only bare strings and duplicated 44 MB). Rebuilt as a UNIT with the index —
# regenerating one without the other silently returns the wrong text for a
# vector (Trap #8).
CHUNKS_PATH = "data/chunks.json"


def chunk_text(text, chunk_size=CHUNK_SIZE_WORDS, overlap=CHUNK_OVERLAP_WORDS):
    """Split text into overlapping word-based chunks."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end >= len(words):
            break
        start += chunk_size - overlap
    return chunks


def embed_text(chunk):
    """The string actually embedded for a chunk: its title/section prefix
    followed by the chunk body. Prefixing gives an otherwise context-free
    chunk its paper and section, which improves retrieval and gives step 12's
    citations something to display. The prefix is derived here (not stored) so
    the index and the on-disk records can never disagree about it."""
    return f"{chunk['title']} — {chunk['section']}\n\n{chunk['text']}"


def build_index():
    """Chunk, encode all documents and save the FAISS index + chunk records."""
    print("Loading documents...")
    with open(DOCS_PATH) as f:
        docs = json.load(f)

    print("Chunking documents...")
    chunks = []
    for d in docs:
        for piece in chunk_text(d["text"]):
            chunks.append({
                "title": d["title"],
                "section": d["section"],
                "text": piece,
                "arxiv_id": d["arxiv_id"],
                "date": d["date"],
                "upvotes": d["upvotes"],
            })
    print(f"  → {len(docs)} documents split into {len(chunks)} chunks")

    print("Encoding documents (this takes a while)...")
    model = SentenceTransformer(EMBED_MODEL)
    # normalize_embeddings=True so that inner product == cosine similarity.
    # The query side (retrieve()) must normalize identically, or the scores are
    # meaningless.
    embeddings = model.encode(
        [embed_text(c) for c in chunks],
        show_progress_bar=True,
        batch_size=64,
        normalize_embeddings=True,
    )
    embeddings = np.array(embeddings).astype("float32")

    print("Building FAISS index...")
    # IndexFlatIP over normalized vectors => cosine similarity: bounded 0–1,
    # HIGHER is better (was IndexFlatL2, where lower squared-distance was better).
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "w") as f:
        json.dump(chunks, f)

    print(f"✅ Index built with {index.ntotal} vectors")


def load_retriever():
    """Load model, index, and chunk records for querying."""
    model = SentenceTransformer(EMBED_MODEL)
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH) as f:
        chunks = json.load(f)
    return model, index, chunks


def retrieve(query, model, index, chunks, top_k=5):
    """Return the top-k most relevant chunk records and their similarity scores.

    Scores are cosine similarity (inner product over normalized vectors),
    bounded 0–1 where HIGHER is more relevant. The query must be encoded with
    the same normalization used at build time (Trap #2).
    """
    query_vec = model.encode([query], normalize_embeddings=True).astype("float32")
    scores, indices = index.search(query_vec, top_k)
    results = [chunks[i] for i in indices[0]]
    return results, scores[0]


if __name__ == "__main__":
    build_index()
