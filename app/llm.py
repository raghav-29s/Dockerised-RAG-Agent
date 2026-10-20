import os

from openai import OpenAI

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

PLACEHOLDER_KEYS = {"", "your_api_key_here"}

_api_key = os.getenv("OPENAI_API_KEY", "").strip().strip("\"'")

# OPENAI_BASE_URL lets the same client talk to any OpenAI-compatible server,
# e.g. the bundled Ollama container at http://ollama:11434/v1 (zero cost).
# The long timeout covers CPU-only local inference.
_client = None
if _api_key not in PLACEHOLDER_KEYS:
    _client = OpenAI(
        api_key=_api_key,
        base_url=os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1",
        timeout=180,
        max_retries=1,
    )


def enabled():
    return _client is not None


def chat(messages, json_mode=False):
    kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
    response = _client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0,
        **kwargs
    )
    return response.choices[0].message.content.strip()
