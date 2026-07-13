# THIS PROJECT IS NOW BEING REBUILT INTO SOMETHING BETTER: SEE BRANCH REBUILD/ARXIV-RAG-PART1

# Wiki Chatbot

A Streamlit app that answers user questions with a Wikipedia-backed retrieval and answer-generation pipeline.

## What it does

- Classifies each prompt as either chit-chat or knowledge-seeking, using sentence embeddings.
- Uses sentence embeddings and a FAISS index to retrieve relevant Wikipedia text chunks.
- Generates a focused answer to the question using the retrieved context.
- Falls back to an "I don't have enough information" response when no retrieved chunk is a close enough match.
- Keeps a lightweight chat history inside the app.

## Project structure

- `app.py` - Streamlit UI and request routing.
- `scraper.py` - Wikipedia documents scraper.
- `classifier.py` - Intent classifier for chit-chat vs. knowledge questions (sentence embeddings + Logistic Regression).
- `retriever.py` - Document chunking, embedding, and FAISS retrieval.
- `generator.py` - Answer generation and simple conversational replies.
- `data/` - Cached documents, embeddings, and FAISS index files.
- `models/` - Saved classifier artifact.

## Setup

Setup virtual environment (optional):
```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Scrape Wikipedia documents, build the retriever and classifier:

```bash
python scraper.py
python retriever.py
python classifier.py
```

## Run the app

Start the Streamlit interface with:

```bash
streamlit run app.py
```

If the FAISS index is missing, the app will show an error and ask you to run `python retriever.py` first.

## How it works

1. The classifier embeds the prompt and decides whether it's casual conversation or a factual question.
2. Knowledge questions are embedded and compared against the FAISS index, which stores overlapping 150-word chunks of each Wikipedia document.
3. If the closest retrieved chunk isn't a confident match, the app responds that it doesn't have enough information rather than guessing.
4. Otherwise, the most relevant chunks are passed to an instruction-tuned model (`flan-t5-base`) along with the original question, which generates a focused answer rather than a generic summary.
5. The generated answer is shown in the chat UI, with source text available in an expander.

## Notes

- The app includes a small macOS OpenMP workaround in `app.py` for FAISS and PyTorch.
- Documents are split into overlapping chunks (150 words, 30-word overlap) before embedding, so retrieval isn't limited to the first few hundred characters of long articles.
- The retrieval and model files are generated locally, so the first setup may take a few minutes.
