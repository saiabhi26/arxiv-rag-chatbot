import os

# Fix macOS OpenMP thread safety issue with FAISS + PyTorch
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_MAX_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '1'

import streamlit as st
from retriever import load_retriever, retrieve
from generator import ClaudeGenerator
from router import Router
from corpus import compute_overview_context

st.set_page_config(page_title="arXiv ML Chatbot", page_icon="🤖", layout="centered")
st.title("🤖 arXiv ML Chatbot")

# Cosine-similarity gate (higher = more relevant). PROVISIONAL — the honest,
# calibrated value comes from the eval harness in step 11. Do not present this
# number in the README as though it were principled.
SIMILARITY_THRESHOLD = 0.35

# The API key lives in Streamlit Cloud secrets on deploy, or a gitignored
# .streamlit/secrets.toml (or an ANTHROPIC_API_KEY env var) locally. Never committed.
def get_api_key():
    try:
        if "ANTHROPIC_API_KEY" in st.secrets:
            return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        pass  # no secrets.toml present at all
    return os.environ.get("ANTHROPIC_API_KEY")

# Load embedding/retrieval models and the Claude client once, and cache them.
@st.cache_resource
def load_all_models(api_key):
    retrieval_model, index, chunks = load_retriever()
    overview_context = compute_overview_context(chunks)
    router = Router(api_key)
    generator = ClaudeGenerator(api_key)
    return retrieval_model, index, chunks, overview_context, router, generator

# Check index exists before loading
if not os.path.exists("data/faiss_index.bin"):
    st.error("⚠️ FAISS index not found. Run `python retriever.py` first to build it.")
    st.stop()

api_key = get_api_key()
if not api_key:
    st.error(
        "⚠️ No Anthropic API key found. Add `ANTHROPIC_API_KEY` to "
        "`.streamlit/secrets.toml` (or the Streamlit Cloud secrets dashboard). "
        "Get a key at https://console.anthropic.com/settings/keys."
    )
    st.stop()

with st.spinner("Loading models..."):
    retrieval_model, index, chunks, overview_context, router, generator = load_all_models(api_key)

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            intent = router.classify(prompt)

            if intent == "chit-chat":
                response = generator.chit_chat(prompt)
            elif intent == "overview":
                response = generator.generate(prompt, [overview_context])
            else:
                docs, scores = retrieve(prompt, retrieval_model, index, chunks, top_k=5)
                if max(scores) < SIMILARITY_THRESHOLD:
                    response = "I don't have enough information on that in my knowledge base. Try rephrasing, or ask about a different topic."
                else:
                    response = generator.generate(prompt, [d["text"] for d in docs])

        st.markdown(response)

        # Show retrieved docs in expander for knowledge queries
        if intent == "knowledge" and max(scores) >= SIMILARITY_THRESHOLD:
            with st.expander("📄 Source documents used"):
                for i, doc in enumerate(docs, 1):
                    st.markdown(f"**{i}. {doc['title']} — {doc['section']}**\n\n{doc['text'][:300]}...")

    st.session_state.messages.append({"role": "assistant", "content": response})