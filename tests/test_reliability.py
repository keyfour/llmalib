"""
tests/reliability/test_reliability.py
Tests for validator, guards, and retry modules.
"""

import pytest
import json
from unittest.mock import patch, MagicMock
from pydantic import BaseModel

from llmalib.reliability.validator import (
    parse_response,
    format_schema_hint,
    _extract_json,
)
from llmalib.reliability.guards import (
    field_in_set,
    float_in_range,
    max_length,
    no_content_from_outside_context,
)
from llmalib.reliability.retry import build_reflection_message, run_with_retry
from llmalib.core.client import ClientConfig, ClientError
from llmalib.core.task import Task
from llmalib.core.context import make_context


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class AnalysisResult(BaseModel):
    label: str
    confidence: float
    reason: str


def make_task(**overrides) -> Task:
    defaults = dict(
        name="test",
        prompt_template="Analyse: {{ text }}",
        output_schema=AnalysisResult,
        max_retries=3,
    )
    defaults.update(overrides)
    return Task(**defaults)


VALID_JSON = json.dumps({"label": "positive", "confidence": 0.9, "reason": "good"})


# ---------------------------------------------------------------------------
# validator.py
# ---------------------------------------------------------------------------


class TestExtractJson:
    def test_direct_json(self):
        result = _extract_json(VALID_JSON)
        assert result == VALID_JSON

    def test_markdown_fence_json(self):
        raw = f"Here is the result:\n```json\n{VALID_JSON}\n```"
        result = _extract_json(raw)
        assert json.loads(result) == json.loads(VALID_JSON)

    def test_markdown_fence_no_lang(self):
        raw = f"```\n{VALID_JSON}\n```"
        result = _extract_json(raw)
        assert json.loads(result) == json.loads(VALID_JSON)

    def test_json_embedded_in_prose(self):
        raw = f"The model thinks: {VALID_JSON} and nothing else."
        result = _extract_json(raw)
        assert json.loads(result) == json.loads(VALID_JSON)

    def test_returns_none_for_no_json(self):
        result = _extract_json("This is just plain text with no JSON.")
        assert result is None

    def test_returns_none_for_malformed(self):
        result = _extract_json("{not valid json at all")
        assert result is None


class TestParseResponse:
    def test_valid_response(self):
        r = parse_response(VALID_JSON, AnalysisResult)
        assert r.ok is True
        assert r.value.label == "positive"
        assert r.value.confidence == 0.9
        assert r.error is None

    def test_missing_required_field(self):
        raw = json.dumps({"label": "positive"})  # missing confidence, reason
        r = parse_response(raw, AnalysisResult)
        assert r.ok is False
        assert r.value is None
        assert "confidence" in r.error or "reason" in r.error

    def test_no_json_in_response(self):
        r = parse_response("I cannot help with that.", AnalysisResult)
        assert r.ok is False
        assert "Could not find valid JSON" in r.error

    def test_wrong_type(self):
        raw = json.dumps(
            {"label": "positive", "confidence": "not_a_float", "reason": "ok"}
        )
        r = parse_response(raw, AnalysisResult)
        # Pydantic will coerce "not_a_float" → ValidationError
        assert r.ok is False

    def test_fenced_response_parsed(self):
        raw = f"```json\n{VALID_JSON}\n```"
        r = parse_response(raw, AnalysisResult)
        assert r.ok is True


class TestFormatSchemaHint:
    def test_produces_valid_json(self):
        hint = format_schema_hint(AnalysisResult)
        data = json.loads(hint)
        assert set(data.keys()) == {"label", "confidence", "reason"}

    def test_type_hints_present(self):
        hint = format_schema_hint(AnalysisResult)
        assert "<str>" in hint or "<float>" in hint


# ---------------------------------------------------------------------------
# guards.py
# ---------------------------------------------------------------------------


class LabelledResult(BaseModel):
    label: str
    score: float
    text: str


def make_value(**kwargs) -> LabelledResult:
    defaults = dict(label="positive", score=0.8, text="Good product overall.")
    defaults.update(kwargs)
    return LabelledResult(**defaults)


class TestFieldInSet:
    def test_passes_valid_value(self):
        guard = field_in_set("label", {"positive", "negative", "neutral"})
        assert guard(make_value(label="positive"), None) == []

    def test_fails_invalid_value(self):
        guard = field_in_set("label", {"positive", "negative", "neutral"})
        errors = guard(make_value(label="POSITIVE"), None)
        assert len(errors) == 1
        assert "POSITIVE" in errors[0]

    def test_fails_invented_value(self):
        guard = field_in_set("label", {"positive", "negative"})
        errors = guard(make_value(label="mixed"), None)
        assert len(errors) == 1


class TestFloatInRange:
    def test_passes_in_range(self):
        guard = float_in_range("score", 0.0, 1.0)
        assert guard(make_value(score=0.5), None) == []

    def test_passes_boundary(self):
        guard = float_in_range("score", 0.0, 1.0)
        assert guard(make_value(score=0.0), None) == []
        assert guard(make_value(score=1.0), None) == []

    def test_fails_above(self):
        guard = float_in_range("score", 0.0, 1.0)
        errors = guard(make_value(score=1.5), None)
        assert len(errors) == 1
        assert "1.5" in errors[0]

    def test_fails_below(self):
        guard = float_in_range("score", 0.0, 1.0)
        errors = guard(make_value(score=-0.1), None)
        assert len(errors) == 1


class TestMaxLength:
    def test_passes_under_limit(self):
        guard = max_length("text", 100)
        assert guard(make_value(text="Short text."), None) == []

    def test_fails_over_limit(self):
        guard = max_length("text", 5)
        errors = guard(make_value(text="This is way too long"), None)
        assert len(errors) == 1
        assert "too long" in errors[0]

    def test_passes_none_field(self):
        # max_length guard should skip None values — test via a model that allows None
        from pydantic import BaseModel as BM
        from typing import Optional

        class NullableResult(BM):
            label: str = "x"
            score: float = 0.5
            text: Optional[str] = None

        guard = max_length("text", 10)
        assert guard(NullableResult(), None) == []


class TestGroundingGuard:
    def test_passes_overlapping_content(self):
        guard = no_content_from_outside_context("text", "source", threshold=0.2)
        ctx = make_context(source="Good product overall, works well.")
        result = make_value(text="product works well")
        assert guard(result, ctx) == []

    def test_fails_no_overlap(self):
        guard = no_content_from_outside_context("text", "source", threshold=0.5)
        ctx = make_context(source="xyzzyx blorph quux frobble wizzle")
        result = make_value(text="completely different alphazeta betamega gammadelta")
        errors = guard(result, ctx)
        assert len(errors) == 1
        assert "grounded" in errors[0]

    def test_skips_when_no_context(self):
        guard = no_content_from_outside_context("text", "source")
        ctx = make_context()  # no 'source' key
        assert guard(make_value(), ctx) == []


# ---------------------------------------------------------------------------
# retry.py
# ---------------------------------------------------------------------------


class TestBuildReflectionMessage:
    def test_includes_parse_error(self):
        msg = build_reflection_message("Missing field 'confidence'", [])
        assert msg["role"] == "user"
        assert "confidence" in msg["content"]

    def test_includes_guard_errors(self):
        msg = build_reflection_message(None, ["Score out of range", "Label invalid"])
        assert "Score out of range" in msg["content"]
        assert "Label invalid" in msg["content"]

    def test_always_has_json_directive(self):
        msg = build_reflection_message("some error", [])
        assert "JSON" in msg["content"]

    def test_both_errors(self):
        msg = build_reflection_message("parse err", ["guard err"])
        assert "parse err" in msg["content"]
        assert "guard err" in msg["content"]


class TestRunWithRetry:
    def _make_config(self):
        return ClientConfig()

    def test_success_on_first_attempt(self):
        task = make_task(max_retries=3)
        messages = [{"role": "user", "content": "test"}]
        config = self._make_config()

        with patch("llmalib.reliability.retry.call", return_value=VALID_JSON):
            result = run_with_retry(task, messages, config)

        assert result.ok is True
        assert result.attempts == 1
        assert result.value.label == "positive"

    def test_succeeds_on_second_attempt(self):
        task = make_task(max_retries=3)
        messages = [{"role": "user", "content": "test"}]
        config = self._make_config()

        bad = '{"label": "positive"}'  # missing fields → parse fails
        responses = [bad, VALID_JSON]

        with patch("llmalib.reliability.retry.call", side_effect=responses):
            result = run_with_retry(task, messages, config)

        assert result.ok is True
        assert result.attempts == 2

    def test_fails_after_max_retries(self):
        task = make_task(max_retries=2)
        messages = [{"role": "user", "content": "test"}]
        config = self._make_config()

        with patch("llmalib.reliability.retry.call", return_value="not json at all"):
            result = run_with_retry(task, messages, config)

        assert result.ok is False
        assert result.attempts == 2
        assert "Failed after 2 attempts" in result.error

    def test_trace_records_all_attempts(self):
        task = make_task(max_retries=3)
        messages = [{"role": "user", "content": "test"}]
        config = self._make_config()

        bad = "not json"
        responses = [bad, bad, VALID_JSON]

        with patch("llmalib.reliability.retry.call", side_effect=responses):
            result = run_with_retry(task, messages, config)

        assert len(result.trace) == 3
        assert result.trace[0].parse_error is not None
        assert result.trace[2].parse_error is None

    def test_guard_failure_triggers_retry(self):
        guard = field_in_set("label", {"negative"})  # will fail for "positive"
        task = make_task(max_retries=3, guards=(guard,))
        messages = [{"role": "user", "content": "test"}]
        config = self._make_config()

        # First call: valid JSON but fails guard; second: also fails; third: succeeds
        negative_json = json.dumps(
            {"label": "negative", "confidence": 0.9, "reason": "bad"}
        )
        responses = [VALID_JSON, VALID_JSON, negative_json]

        with patch("llmalib.reliability.retry.call", side_effect=responses):
            result = run_with_retry(task, messages, config)

        assert result.ok is True
        assert result.value.label == "negative"
        assert result.attempts == 3

    def test_client_error_propagates(self):
        task = make_task()
        messages = [{"role": "user", "content": "test"}]
        config = self._make_config()

        with patch(
            "llmalib.reliability.retry.call", side_effect=ClientError("network down")
        ):
            with pytest.raises(ClientError, match="network down"):
                run_with_retry(task, messages, config)
