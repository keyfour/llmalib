"""
reliability/validator.py

Extract JSON from raw model output and validate against a Pydantic schema.
Thin adapter over `llmalib._rust` (extraction strategies and JSON-Schema
validation run in Rust); the parsed value is re-hydrated into the pydantic
model class so callers keep working with typed instances.
Never raises — all outcomes are returned as ParseResult.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel

from llmalib import _rust
from llmalib._bridge import hydrate_value, schema_to_json


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
    out = _rust.parse_response(raw, schema_to_json(schema))
    if not out["ok"]:
        return ParseResult(ok=False, value=None, error=out["error"])
    return ParseResult(
        ok=True, value=hydrate_value(out["value"], schema), error=None
    )


def _extract_json(raw: str) -> str | None:
    """Try three strategies to find a JSON string in raw model output."""
    return _rust.extract_json(raw)


def format_schema_hint(schema: type[BaseModel]) -> str:
    """
    Produce a compact JSON example of the expected output shape.
    Injected into every system prompt so the model knows exactly what to produce.

    Example output:
      {"label": "<str>", "confidence": "<float>", "reason": "<str>"}
    """
    import json

    hints: dict[str, str] = {}
    for name, field_info in schema.model_fields.items():
        annotation = field_info.annotation
        type_name = getattr(annotation, "__name__", str(annotation))
        hints[name] = f"<{type_name}>"
    return json.dumps(hints)
