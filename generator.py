"""Answer generation over retrieved context.

A thin `Generator` protocol with one implementation (`ClaudeGenerator`). The
protocol is the seam: swapping providers means writing another ~20-line class
behind this interface, untouched by app.py. The generator is framework-agnostic
— it takes an API key, not a Streamlit secrets object — so it can be exercised
from a plain script as well as from the app.
"""

from typing import Protocol

import anthropic


# Answers use cheap, fast Haiku by default; overridable for a higher-quality
# final demo (e.g. Sonnet). "-latest"-style pinned ids keep behaviour stable.
ANSWER_MODEL = "claude-haiku-4-5"
MAX_TOKENS = 1024

_SYSTEM_INSTRUCTION = (
    "You are a question-answering assistant for a corpus of recent arXiv "
    "machine-learning papers. Answer the user's question using ONLY the "
    "provided context. If the context does not contain the answer, say you "
    "don't have enough information — do not use outside knowledge and do not "
    "guess. Be concise."
)

_CHIT_CHAT_SYSTEM_INSTRUCTION = (
    "You are a friendly chatbot that answers questions about recent arXiv "
    "machine-learning papers. The user just sent a conversational message "
    "(a greeting, thanks, small talk, or a question about you) rather than a "
    "research question. Reply warmly and briefly — one or two sentences — "
    "and, where natural, remind them they can ask about recent ML research."
)

# Shown for transient API failures instead of a stack trace mid-demo (Trap #5).
# 429 = rate limit; 5xx / overloaded = provider-side capacity spikes.
_RATE_LIMIT_MESSAGE = (
    "I'm getting more requests than the rate limit allows right now. "
    "Please wait a few seconds and try again."
)
_BUSY_MESSAGE = (
    "The model is experiencing high demand right now. "
    "Please wait a few seconds and try again."
)


class Generator(Protocol):
    def generate(self, query: str, contexts: list[str]) -> str:
        """Answer `query` grounded in `contexts` (retrieved chunk texts)."""
        ...


class ClaudeGenerator:
    def __init__(self, api_key: str, model: str = ANSWER_MODEL):
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def generate(self, query: str, contexts: list[str]) -> str:
        # All retrieved chunks fit comfortably in Claude's context window —
        # nothing is silently dropped (unlike the old flan-t5 path, which
        # sliced context to 3000 chars then truncated the prompt to 512 tokens).
        context_block = "\n\n---\n\n".join(contexts)
        prompt = (
            f"Context:\n{context_block}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=MAX_TOKENS,
                system=_SYSTEM_INSTRUCTION,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except anthropic.RateLimitError:
            return _RATE_LIMIT_MESSAGE
        except (anthropic.InternalServerError, anthropic.APIStatusError) as e:
            # 529 overloaded / any 5xx: degrade politely rather than throw.
            if getattr(e, "status_code", 500) >= 500:
                return _BUSY_MESSAGE
            raise

    def chit_chat(self, query: str) -> str:
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=MAX_TOKENS,
                system=_CHIT_CHAT_SYSTEM_INSTRUCTION,
                messages=[{"role": "user", "content": query}],
            )
            return response.content[0].text
        except anthropic.RateLimitError:
            return _RATE_LIMIT_MESSAGE
        except (anthropic.InternalServerError, anthropic.APIStatusError) as e:
            if getattr(e, "status_code", 500) >= 500:
                return _BUSY_MESSAGE
            raise
