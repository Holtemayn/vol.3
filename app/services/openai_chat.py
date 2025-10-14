from __future__ import annotations

from typing import List, Dict

from openai import OpenAI

from app.core.config import settings

_CLIENT: OpenAI | None = None

SYSTEM_PROMPT = (
    "Du er en hjælpsom café-rådgiver. Du kender til forecasting, bemanding "
    "og daglig drift hos en specialkaffe-butik. Svar kort og handlingsorienteret."
)


def _client() -> OpenAI:
    global _CLIENT
    if _CLIENT is None:
        if not settings.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY mangler")
        _CLIENT = OpenAI(api_key=settings.OPENAI_API_KEY)
    return _CLIENT


def get_advice(history: List[Dict[str, str]]) -> str:
    client = _client()
    response = client.responses.create(
        model="gpt-4o-mini",
        input=[{"role": "system", "content": SYSTEM_PROMPT}, *history],
        max_output_tokens=600,
    )
    return response.output_text.strip()
