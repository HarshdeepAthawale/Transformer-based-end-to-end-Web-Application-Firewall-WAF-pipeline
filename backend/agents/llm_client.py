"""Factory for AsyncOpenAI client using environment variables."""

import os
from openai import AsyncOpenAI

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_DEFAULT_MODEL = "llama-3.1-8b-instant"


def _get_api_key() -> str:
    """Resolve API key: GROQ_API_KEY > AGENT_API_KEY > OPENAI_API_KEY (for Groq/OpenAI-compatible)."""
    return (
        os.getenv("GROQ_API_KEY", "").strip()
        or os.getenv("AGENT_API_KEY", "").strip()
        or os.getenv("OPENAI_API_KEY", "").strip()
    )


def get_llm_client() -> AsyncOpenAI:
    """Create an AsyncOpenAI client from env vars.

    Env vars (for Groq):
        GROQ_API_KEY      — Groq API key (used with Groq base URL)
        AGENT_BASE_URL    — optional override (defaults to Groq if GROQ_API_KEY set)
        AGENT_MODEL       — model name (default: llama-3.1-8b-instant for Groq)

    Env vars (for other providers: Together, vLLM, etc.):
        AGENT_API_KEY     — API key for the provider
        AGENT_BASE_URL    — base URL
        AGENT_MODEL       — model name
    """
    api_key = _get_api_key()
    base_url = os.getenv("AGENT_BASE_URL", "").strip()

    # If GROQ_API_KEY set but no base URL, default to Groq
    if not base_url and os.getenv("GROQ_API_KEY", "").strip():
        base_url = GROQ_BASE_URL

    return AsyncOpenAI(
        api_key=api_key,
        base_url=base_url if base_url else None,
    )


def get_model() -> str:
    """Return the model name from env or default."""
    model = os.getenv("AGENT_MODEL", "").strip()
    if model:
        return model
    # Default to Groq model when using Groq
    if os.getenv("GROQ_API_KEY", "").strip():
        return GROQ_DEFAULT_MODEL
    return "gpt-4o-mini"
