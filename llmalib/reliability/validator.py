"""
reliability/validator.py

Extract JSON from raw model output and validate against a Pydantic schema.
Three-strategy extraction handles models that wrap JSON in prose or fences.
Never raises — all outcomes are returned as ParseResult.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pydantic import BaseModel, ValidationError


@dataclass(frozen=True)
class ParseResult:
    ok: bool
    value: BaseModel | None
    error: str | None


def parse_response(raw: str, schema: type[BaseModel]) -> ParseResult:
    """
    Attempt to extract and validate JSON from raw model output.

    Strategies tried in order:
    1. Direct json.loads() — clean JSON response
    2. Extract from markdown code fences  ```json ... ```
    3. Extract first {...} or [...] block via regex

    Returns ParseResult — never raises.
    """
    json_str = _extract_json(raw)
    if json_str is None:
        return ParseResult(
            ok=False,
            value=None,
            error=(
                f"Could not find valid JSON in response. "
                f"Response started with: {raw[:120]!r}"
            ),
        )

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        return ParseResult(
            ok=False,
            value=None,
            error=f"JSON parse error: {exc}. Extracted: {json_str[:120]!r}",
        )

    try:
        instance = schema.model_validate(data)
        return ParseResult(ok=True, value=instance, error=None)
    except ValidationError as exc:
        # Produce a concise, actionable error message for the reflection loop
        errors = exc.errors()
        messages = []
        for e in errors:
            loc = ".".join(str(p) for p in e["loc"]) if e["loc"] else "root"
            messages.append(f"  - Field '{loc}': {e['msg']}")
        present_keys = list(data.keys()) if isinstance(data, dict) else "non-object"
        error_str = (
            f"Schema validation failed. "
            f"Got keys: {present_keys}. Errors:\n" + "\n".join(messages)
        )
        return ParseResult(ok=False, value=None, error=error_str)


def _extract_json(raw: str) -> str | None:
    """Try three strategies to find a JSON string in raw model output."""
    raw = raw.strip()

    # Strategy 1: direct parse
    try:
        json.loads(raw)
        return raw
    except json.JSONDecodeError:
        pass

    # Strategy 2: markdown code fence  ```json ... ```  or  ``` ... ```
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if fence_match:
        candidate = fence_match.group(1).strip()
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass

    # Strategy 3: first {...} or [...] block (greedy from outermost brace)
    for pattern in (r"\{[\s\S]*\}", r"\[[\s\S]*\]"):
        match = re.search(pattern, raw)
        if match:
            candidate = match.group(0)
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                pass

    return None


def format_schema_hint(schema: type[BaseModel]) -> str:
    """
    Produce a compact JSON example of the expected output shape.
    Injected into every system prompt so the model knows exactly what to produce.

    Example output:
      {"label": "<str>", "confidence": "<float>", "reason": "<str>"}
    """
    hints = {}
    for name, field_info in schema.model_fields.items():
        annotation = field_info.annotation
        type_name = getattr(annotation, "__name__", str(annotation))
        hints[name] = f"<{type_name}>"
    return json.dumps(hints)
