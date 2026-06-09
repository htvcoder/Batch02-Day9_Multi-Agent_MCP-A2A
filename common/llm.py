"""Shared LLM factory for all stages and agents."""

import os

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI


def get_llm(
    *, temperature: float = 0.3, max_tokens: int | None = None
) -> ChatOpenAI | ChatGoogleGenerativeAI:
    """Return an LLM client based on the configured provider."""
    provider = os.getenv("LLM_PROVIDER", "openrouter").lower()

    if provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "Missing GEMINI_API_KEY or GOOGLE_API_KEY. Add one of them to "
                "your environment or .env file before running the demo."
            )

        client_kwargs = {
            "model": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            "temperature": temperature,
            "api_key": api_key,
            "convert_system_message_to_human": True,
        }

        if max_tokens is not None:
            client_kwargs["max_tokens"] = max_tokens

        return ChatGoogleGenerativeAI(**client_kwargs)

    if provider == "openrouter":
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

        return ChatOpenAI(**client_kwargs)

    raise ValueError("Unsupported LLM_PROVIDER. Use 'openrouter' or 'gemini'.")
