"""
core/client.py

Thin HTTP wrapper around any OpenAI-compatible chat completion endpoint.
Accepts a message list, returns a raw response string.
All retry/reflection logic lives in reliability/retry.py — not here.
"""

from __future__ import annotations

import httpx
from dataclasses import dataclass


class ClientError(Exception):
    """Raised on HTTP failure, timeout, or empty response."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class ClientConfig:
    base_url: str = "http://localhost:11434/v1"
    model: str = "llama3.2"
    temperature: float = 0.1
    timeout: float = 120.0
    max_tokens: int = 1024


def call(messages: list[dict], config: ClientConfig) -> str:
    """
    POST to /v1/chat/completions and return the content string of the first choice.

    Raises ClientError on:
    - HTTP error status
    - Network timeout
    - Missing or empty response content
    - Unexpected response shape
    """
    url = config.base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": config.model,
        "messages": messages,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "stream": False,
    }

    try:
        response = httpx.post(url, json=payload, timeout=config.timeout)
    except httpx.TimeoutException:
        raise ClientError(
            f"Request timed out after {config.timeout}s "
            f"(model={config.model}, url={url})"
        )
    except httpx.RequestError as exc:
        raise ClientError(f"Network error: {exc} (url={url})")

    if response.status_code != 200:
        raise ClientError(
            f"HTTP {response.status_code} from {url}: {response.text[:200]}",
            status_code=response.status_code,
        )

    try:
        data = response.json()
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, ValueError) as exc:
        raise ClientError(
            f"Unexpected response shape from {url}: {exc}. Body: {response.text[:200]}"
        )

    if not content or not content.strip():
        raise ClientError(f"Model returned empty content (model={config.model})")

    return content.strip()


def config_from_task(task) -> ClientConfig:
    """Build a ClientConfig from a Task's fields."""
    return ClientConfig(
        base_url=task.base_url,
        model=task.model,
        temperature=task.temperature,
        timeout=task.timeout,
        max_tokens=task.max_tokens,
    )
