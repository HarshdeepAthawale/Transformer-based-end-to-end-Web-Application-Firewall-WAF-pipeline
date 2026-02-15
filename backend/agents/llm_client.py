"""Factory for AsyncOpenAI client using environment variables."""

import os
from openai import AsyncOpenAI


def get_llm_client() -> AsyncOpenAI:
    """Create an AsyncOpenAI client from env vars.

    Env vars:
        OPENAI_API_KEY  — API key (required)
        AGENT_BASE_URL  — optional base URL override (Groq, Together, local vLLM)
    """
    base_url = os.getenv("AGENT_BASE_URL") or None
    return AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY", ""),
        base_url=base_url,
    )


def get_model() -> str:
    """Return the model name from env or default."""
    return os.getenv("AGENT_MODEL", "gpt-4o-mini")
