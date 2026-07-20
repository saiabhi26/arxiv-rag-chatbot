"""Intent routing over Claude.

Replaces the pickled sklearn classifier: a single cheap Claude call decides
whether a query is "chit-chat", "overview", or "knowledge". Routing is
internal plumbing — it must never crash the app or surface an error string,
so any API failure (and any ambiguous/unparseable response) defaults to
"knowledge". We would rather retrieve-and-gate on a borderline query than
wrongly small-talk (or wrongly overview-answer) past a real question.
"""

import anthropic

ROUTER_MODEL = "claude-haiku-4-5"
MAX_TOKENS = 10

_SYSTEM_INSTRUCTION = (
    "You are an intent router for a chatbot over recent arXiv machine-learning "
    "papers. Classify the user's message into exactly one of three intents:\n\n"
    "\"knowledge\" - anything asking about the CONTENT of the research: "
    "ML facts, how something works, definitions, math, methods, benchmarks, "
    "or requesting information about a topic. This is also the default for "
    "vague or open-ended requests that don't specifically ask about the "
    "collection's metadata. Examples: \"explain machine learning\", \"what "
    "benchmark evaluates the method\", \"tell me about anything\".\n"
    "\"overview\" - questions specifically ABOUT the paper collection as a "
    "whole, not about any research content: how many papers there are, which "
    "papers are most popular/upvoted, what date range the collection covers, "
    "or \"what papers do you have\". Examples: \"what papers are in your "
    "corpus\", \"what are the most popular papers\", \"how many papers do you "
    "have\".\n"
    "\"chit-chat\" - greetings, thanks, small talk, or questions about the bot "
    "itself (who/what are you, what can you do).\n\n"
    "When in doubt between \"overview\" and \"knowledge\", prefer \"knowledge\". "
    "Respond with only the single word \"knowledge\", \"overview\", or "
    "\"chit-chat\" and nothing else."
)


class Router:
    def __init__(self, api_key: str, model: str = ROUTER_MODEL):
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def classify(self, query: str) -> str:
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=MAX_TOKENS,
                system=_SYSTEM_INSTRUCTION,
                messages=[{"role": "user", "content": query}],
            )
            label = response.content[0].text.lower().strip()
        except (anthropic.APIError, IndexError, AttributeError):
            # Rate limit, 5xx, empty/malformed response, or any other failure:
            # fall back to the safer default rather than let routing crash the app.
            return "knowledge"

        if "chit" in label:
            return "chit-chat"
        elif "overview" in label:
            return "overview"
        return "knowledge"


if __name__ == "__main__":
    import os

    def _load_api_key():
        try:
            import tomllib
        except ImportError:  # Python < 3.11
            import tomli as tomllib
        secrets_path = os.path.join(".streamlit", "secrets.toml")
        if os.path.exists(secrets_path):
            with open(secrets_path, "rb") as f:
                secrets = tomllib.load(f)
            if "ANTHROPIC_API_KEY" in secrets:
                return secrets["ANTHROPIC_API_KEY"]
        return os.environ.get("ANTHROPIC_API_KEY")

    api_key = _load_api_key()
    if not api_key:
        raise SystemExit(
            "No ANTHROPIC_API_KEY found in .streamlit/secrets.toml or the "
            "environment."
        )

    router = Router(api_key)
    tests = [
        ("what papers are in your corpus", "overview"),
        ("what are the most popular papers", "overview"),
        ("how many papers do you have", "overview"),
        ("tell me about anything", "knowledge"),
        ("hello!", "chit-chat"),
        ("explain machine learning", "knowledge"),
        ("how are you?", "chit-chat"),
        ("what is a transformer", "knowledge"),
        ("thanks, that helps", "chit-chat"),
        ("what benchmark evaluates the method", "knowledge"),
    ]
    for query, expected in tests:
        actual = router.classify(query)
        status = "OK" if actual == expected else "MISMATCH"
        print(f"  [{status}] '{query}' -> {actual} (expected {expected})")
