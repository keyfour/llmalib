"""
core/client.py

Thin adapter over `llmalib._rust.call` (reqwest HTTP transport).
Accepts a message list, returns a raw response string.
All retry/reflection logic lives in reliability/retry.py — not here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from llmalib import _rust


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


def call(messages: list[dict[str, Any]], config: ClientConfig) -> str:
    """
    POST to /v1/chat/completions and return the content string of the first choice.

    Raises ClientError on:
    - HTTP error status
    - Network timeout
    - Missing or empty response content
    - Unexpected response shape
    """
    config_dict = {
        "base_url": config.base_url,
        "model": config.model,
        "temperature": config.temperature,
        "timeout": config.timeout,
        "max_tokens": config.max_tokens,
    }
    try:
        return _rust.call(messages, config_dict)
    except _rust.ClientError as exc:
        message = str(exc)
        match = re.search(r"\[status (\d+)\]\s*$", message)
        if match:
            message = message[: match.start()].rstrip()
            raise ClientError(message, status_code=int(match.group(1))) from None
        raise ClientError(message) from None


def config_from_task(task: Any) -> ClientConfig:
    """Build a ClientConfig from a Task's fields."""
    return ClientConfig(
        base_url=task.base_url,
        model=task.model,
        temperature=task.temperature,
        timeout=task.timeout,
        max_tokens=task.max_tokens,
    )
