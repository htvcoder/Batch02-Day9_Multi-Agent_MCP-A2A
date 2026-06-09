"""Shared LLM factory for all agents.

Uses OpenRouter as an OpenAI-compatible API, so any provider's model
can be selected via the OPENROUTER_MODEL env var.
"""

import os

from langchain_openai import ChatOpenAI


def get_llm(
    *, temperature: float = 0.3, max_tokens: int | None = None
) -> ChatOpenAI:
    """Return a ChatOpenAI client pointed at OpenRouter."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "Missing OPENROUTER_API_KEY. Add it to your environment or .env file "
            "before running the demo."
        )

    client_kwargs = {
        "model": os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4-5"),
        "temperature": temperature,
        "openai_api_key": api_key,
        "openai_api_base": "https://openrouter.ai/api/v1",
    }

    if max_tokens is not None:
        client_kwargs["max_tokens"] = max_tokens

    return ChatOpenAI(
        **client_kwargs,
    )
