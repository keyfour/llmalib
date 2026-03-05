"""
tests/core/test_core.py
Tests for result, context, task, and client modules.
"""

import pytest
import time
from unittest.mock import MagicMock, patch
from pydantic import BaseModel

from llmalib.core.result import (
    Attempt,
    Result,
    make_ok_result,
    make_error_result,
    make_attempt,
)
from llmalib.core.context import Context, make_context, update_context, get_result
from llmalib.core.task import Task
from llmalib.core.client import ClientConfig, ClientError, call


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class SentimentResult(BaseModel):
    label: str
    confidence: float


def make_sample_task(**overrides) -> Task:
    defaults = dict(
        name="test_task",
        prompt_template="Classify: {{ text }}",
        output_schema=SentimentResult,
    )
    defaults.update(overrides)
    return Task(**defaults)


# ---------------------------------------------------------------------------
# result.py
# ---------------------------------------------------------------------------


class TestResult:
    def test_make_ok_result(self):
        value = SentimentResult(label="positive", confidence=0.9)
        attempts = [
            make_attempt(1, "prompt", "response", None, [], time.monotonic() - 0.1)
        ]
        r = make_ok_result("my_task", value, attempts)
        assert r.ok is True
        assert r.value == value
        assert r.error is None
        assert r.attempts == 1
        assert r.task_name == "my_task"

    def test_make_error_result(self):
        r = make_error_result("my_task", "Something went wrong", [])
        assert r.ok is False
        assert r.value is None
        assert r.error == "Something went wrong"
        assert r.attempts == 0

    def test_attempt_duration(self):
        started = time.monotonic() - 0.05
        a = make_attempt(1, "p", "r", None, [], started)
        assert a.duration_ms >= 40  # at least 40ms elapsed
        assert a.attempt_number == 1
        assert a.parse_error is None
        assert a.guard_errors == []

    def test_attempt_with_errors(self):
        a = make_attempt(
            2, "p", "r", "missing field", ["out of range"], time.monotonic()
        )
        assert a.parse_error == "missing field"
        assert a.guard_errors == ["out of range"]


# ---------------------------------------------------------------------------
# context.py
# ---------------------------------------------------------------------------


class TestContext:
    def test_make_context(self):
        ctx = make_context(text="hello", lang="en")
        assert ctx.vars["text"] == "hello"
        assert ctx.vars["lang"] == "en"
        assert isinstance(ctx.run_id, str)
        assert len(ctx.run_id) > 0

    def test_update_context_ok_merges_fields(self):
        ctx = make_context(text="hello")
        value = SentimentResult(label="positive", confidence=0.9)
        result = make_ok_result("task_a", value, [])
        update_context(ctx, "task_a", result)

        assert ctx.results["task_a"] == result
        # Fields from the Pydantic model should be merged into vars
        assert ctx.vars["label"] == "positive"
        assert ctx.vars["confidence"] == 0.9

    def test_update_context_error_does_not_merge(self):
        ctx = make_context(text="hello")
        result = make_error_result("task_a", "failed", [])
        update_context(ctx, "task_a", result)

        assert ctx.results["task_a"] == result
        assert "label" not in ctx.vars  # nothing merged on failure

    def test_get_result(self):
        ctx = make_context()
        assert get_result(ctx, "missing") is None
        result = make_error_result("task_a", "err", [])
        update_context(ctx, "task_a", result)
        assert get_result(ctx, "task_a") is result

    def test_run_id_unique(self):
        ctx1 = make_context()
        ctx2 = make_context()
        assert ctx1.run_id != ctx2.run_id


# ---------------------------------------------------------------------------
# task.py
# ---------------------------------------------------------------------------


class TestTask:
    def test_task_defaults(self):
        task = make_sample_task()
        assert task.model == "llama3.2"
        assert task.temperature == 0.1
        assert task.max_retries == 3
        assert task.token_budget == 2048
        assert task.guards == ()
        assert task.examples == ()

    def test_task_is_frozen(self):
        task = make_sample_task()
        with pytest.raises((AttributeError, TypeError)):
            task.name = "other"  # type: ignore

    def test_task_with_overrides(self):
        task = make_sample_task(model="mistral", temperature=0.5, max_retries=5)
        assert task.model == "mistral"
        assert task.temperature == 0.5
        assert task.max_retries == 5


# ---------------------------------------------------------------------------
# client.py
# ---------------------------------------------------------------------------


class TestClient:
    def test_call_success(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "hello world"}}]
        }

        config = ClientConfig()
        with patch("httpx.post", return_value=mock_response):
            result = call([{"role": "user", "content": "hi"}], config)

        assert result == "hello world"

    def test_call_http_error(self):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        config = ClientConfig()
        with patch("httpx.post", return_value=mock_response):
            with pytest.raises(ClientError) as exc_info:
                call([{"role": "user", "content": "hi"}], config)

        assert "500" in str(exc_info.value)

    def test_call_empty_content(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "   "}}]}

        config = ClientConfig()
        with patch("httpx.post", return_value=mock_response):
            with pytest.raises(ClientError, match="empty content"):
                call([{"role": "user", "content": "hi"}], config)

    def test_call_timeout(self):
        import httpx as _httpx

        config = ClientConfig(timeout=1.0)
        with patch("httpx.post", side_effect=_httpx.TimeoutException("timed out")):
            with pytest.raises(ClientError, match="timed out"):
                call([{"role": "user", "content": "hi"}], config)

    def test_call_missing_choices(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": []}
        mock_response.text = "{}"

        config = ClientConfig()
        with patch("httpx.post", return_value=mock_response):
            with pytest.raises(ClientError):
                call([{"role": "user", "content": "hi"}], config)

    def test_url_construction(self):
        """Ensure trailing slash on base_url is handled."""
        captured = {}

        def fake_post(url, **kwargs):
            captured["url"] = url
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "ok"}}]
            }
            return mock_response

        config = ClientConfig(base_url="http://localhost:11434/v1/")
        with patch("httpx.post", side_effect=fake_post):
            call([{"role": "user", "content": "hi"}], config)

        assert captured["url"] == "http://localhost:11434/v1/chat/completions"
