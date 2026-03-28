"""
LLM Provider Abstraction for Clinical Note Classifier
=======================================================
Routes LLM completion requests to the appropriate provider (OpenAI or Anthropic).
Keeps provider-specific details (message format, client initialization, response
parsing) isolated so the classifier engine remains provider-agnostic.

Supported providers:
  - "openai"    — OpenAI Chat Completions API (default)
  - "anthropic" — Anthropic Messages API

Usage:
    from src.providers import get_completion
    content = get_completion(
        system_message="You are a helpful assistant.",
        user_message="Hello!",
        model="gpt-4o-mini",
        temperature=0.0,
        provider="openai",
    )
"""

from src.logging_config import get_logger

logger = get_logger(__name__)

# Valid provider names
SUPPORTED_PROVIDERS = ("openai", "anthropic")

# Default models per provider
DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-4-20250514",
}


def get_completion(
    system_message: str,
    user_message: str,
    model: str,
    temperature: float = 0.0,
    provider: str = "openai",
) -> str:
    """
    Get a completion from the specified LLM provider.

    Args:
        system_message: The system prompt content.
        user_message: The user message content.
        model: Model identifier (e.g., "gpt-4o-mini", "claude-sonnet-4-20250514").
        temperature: Sampling temperature.
        provider: LLM provider — "openai" or "anthropic".

    Returns:
        The raw text content of the model's response.

    Raises:
        ValueError: If the provider is not supported.
        Various API errors: Propagated from the underlying SDK.
    """
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unsupported provider: '{provider}'. "
            f"Supported providers: {SUPPORTED_PROVIDERS}"
        )

    if provider == "openai":
        return _openai_completion(system_message, user_message, model, temperature)
    elif provider == "anthropic":
        return _anthropic_completion(system_message, user_message, model, temperature)


def _openai_completion(
    system_message: str,
    user_message: str,
    model: str,
    temperature: float,
) -> str:
    """Call OpenAI Chat Completions API."""
    from openai import OpenAI

    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content


def _anthropic_completion(
    system_message: str,
    user_message: str,
    model: str,
    temperature: float,
) -> str:
    """
    Call Anthropic Messages API.

    Note: Anthropic uses a separate `system` parameter (not part of the messages
    list), and does not have a native JSON mode. The system prompt already
    instructs the model to return JSON, so we rely on prompt-level enforcement.
    """
    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        temperature=temperature,
        system=system_message,
        messages=[
            {"role": "user", "content": user_message},
        ],
    )
    return response.content[0].text
