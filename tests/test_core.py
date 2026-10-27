"""
tests/core/test_core.py
Tests for result, context, task, and client modules.
"""

import http.server
import json
import threading
import time

import pytest
from pydantic import BaseModel

from llmalib.core.client import ClientConfig, ClientError, call
from llmalib.core.context import get_result, make_context, update_context
from llmalib.core.result import (
    make_attempt,
    make_error_result,
    make_ok_result,
)
from llmalib.core.task import Task

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class SentimentResult(BaseModel):
    label: str
    confidence: float


def make_sample_task(**overrides) -> Task:
    defaults = {
        "name": "test_task",
        "prompt_template": "Classify: {{ text }}",
        "output_schema": SentimentResult,
    }
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
# client.py — exercised against a real local HTTP server (reqwest transport)
# ---------------------------------------------------------------------------


class _ChatHandler(http.server.BaseHTTPRequestHandler):
    """Canned /v1/chat/completions responses driven by server.mode."""

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        self.server.captured["path"] = self.path

        mode = self.server.mode
        if mode == "slow":
            time.sleep(1.0)

        if mode == "http500":
            status, body = 500, b"Internal Server Error"
        elif mode == "empty":
            status = 200
            body = json.dumps(
                {"choices": [{"message": {"content": "   "}}]}
            ).encode()
        elif mode == "missing_choices":
            status, body = 200, json.dumps({"choices": []}).encode()
        else:  # "ok" and "slow"
            status = 200
            body = json.dumps(
                {"choices": [{"message": {"content": "hello world"}}]}
            ).encode()

        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def model_server():
    servers: list[http.server.HTTPServer] = []

    def make(mode: str):
        srv = http.server.HTTPServer(("127.0.0.1", 0), _ChatHandler)
        srv.mode = mode
        srv.captured = {}
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        servers.append(srv)
        base_url = f"http://127.0.0.1:{srv.server_port}/v1"
        return srv, base_url

    yield make
    for srv in servers:
        srv.shutdown()
        srv.server_close()


class TestClient:
    def test_call_success(self, model_server):
        _srv, base_url = model_server("ok")
        result = call([{"role": "user", "content": "hi"}], ClientConfig(base_url=base_url))
        assert result == "hello world"

    def test_call_http_error(self, model_server):
        _srv, base_url = model_server("http500")
        with pytest.raises(ClientError) as exc_info:
            call([{"role": "user", "content": "hi"}], ClientConfig(base_url=base_url))
        assert "500" in str(exc_info.value)
        assert exc_info.value.status_code == 500

    def test_call_empty_content(self, model_server):
        _srv, base_url = model_server("empty")
        with pytest.raises(ClientError, match="empty content"):
            call([{"role": "user", "content": "hi"}], ClientConfig(base_url=base_url))

    def test_call_timeout(self, model_server):
        _srv, base_url = model_server("slow")
        config = ClientConfig(base_url=base_url, timeout=0.2)
        with pytest.raises(ClientError, match="timed out"):
            call([{"role": "user", "content": "hi"}], config)

    def test_call_missing_choices(self, model_server):
        _srv, base_url = model_server("missing_choices")
        with pytest.raises(ClientError, match="Unexpected response shape"):
            call([{"role": "user", "content": "hi"}], ClientConfig(base_url=base_url))

    def test_url_construction(self, model_server):
        """Ensure trailing slash on base_url is handled; posts to /chat/completions."""
        srv, base_url = model_server("ok")
        call([{"role": "user", "content": "hi"}], ClientConfig(base_url=base_url))
        assert srv.captured["path"] == "/v1/chat/completions"
