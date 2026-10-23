"""
tests/test_rust_binding.py
Smoke coverage for the llmalib._rust pyo3 extension (Phase 3).
The Phase 4 adapters will sit on exactly these entry points.
"""

import pytest

from llmalib import _rust


@pytest.fixture(autouse=True)
def _reset_caller():
    yield
    _rust.set_caller(None)


SCHEMA = {
    "properties": {"label": {"type": "string"}, "confidence": {"type": "number"}},
    "required": ["label", "confidence"],
}


def make_spec(**overrides):
    spec = {
        "name": "classify",
        "prompt_template": "Classify: {{ text }}",
        "output_schema": SCHEMA,
        "max_retries": 2,
    }
    spec.update(overrides)
    return spec


class TestValidator:
    def test_parse_response_ok(self):
        out = _rust.parse_response('{"label": "pos", "confidence": 0.9}', SCHEMA)
        assert out["ok"] is True
        assert out["value"]["label"] == "pos"
        assert out["error"] is None

    def test_parse_response_missing_field(self):
        out = _rust.parse_response('{"label": "pos"}', SCHEMA)
        assert out["ok"] is False
        assert "confidence" in out["error"]

    def test_extract_json_fenced(self):
        assert _rust.extract_json('```json\n{"a": 1}\n```') == '{"a": 1}'

    def test_format_schema_hint(self):
        hint = _rust.format_schema_hint(SCHEMA)
        assert "<str>" in hint or "<float>" in hint


class TestContextWindow:
    def test_count_tokens_default_and_custom(self):
        assert _rust.count_tokens("hello world", None) == 2
        assert _rust.count_tokens("anything", lambda text: 7) == 7

    def test_trim_to_budget_raises(self):
        messages = [
            {"role": "system", "content": "x" * 1000},
            {"role": "user", "content": "y" * 1000},
        ]
        with pytest.raises(_rust.BudgetExceededError):
            _rust.trim_to_budget(messages, 10, None)


class TestPipeline:
    def test_run_pipeline_with_caller_seam(self):
        _rust.set_caller(lambda messages, config: '{"label": "positive", "confidence": 0.9}')
        out = _rust.run_pipeline([make_spec()], vars={"text": "hello"})
        assert out["results"][0]["ok"] is True
        assert out["vars"]["label"] == "positive"
        assert out["run_id"]

    def test_guard_failure_triggers_reflection(self):
        _rust.set_caller(lambda m, c: '{"label": "positive", "confidence": 0.9}')
        spec = make_spec(
            guards=[lambda value, ctx: [] if value["label"] == "negative" else ["must be negative"]]
        )
        out = _rust.run_with_retry(spec, [{"role": "user", "content": "x"}], {})
        assert out["ok"] is False
        assert out["attempts"] == 2
        assert "must be negative" in out["trace"][0]["guard_errors"][0]

    def test_python_router_stops_on_failure(self):
        _rust.set_caller(lambda m, c: "not json")
        router = lambda result, remaining: remaining if result["ok"] else []
        out = _rust.run_pipeline(
            [make_spec(), make_spec(name="second")],
            vars={"text": "hello"},
            router=router,
        )
        assert len(out["results"]) == 1
        assert out["results"][0]["ok"] is False


class TestStoreAndTracer:
    def test_store_roundtrip(self):
        store = _rust.make_memory_store()
        store.set("k", {"v": 1}, score=2.0)
        entry = store.get("k")
        assert entry["value"] == {"v": 1}
        assert entry["score"] == 2.0
        assert store.list("") == ["k"]
        assert store.get_relevant("v", 1)[0]["key"] == "k"
        store.delete("k")
        assert store.get("k") is None

    def test_tracer_record_and_summary(self):
        _rust.set_caller(lambda m, c: '{"label": "positive", "confidence": 0.9}')
        tracer = _rust.make_tracer("runpy")
        _rust.run_pipeline([make_spec()], vars={"text": "hi"}, tracer=tracer)
        assert tracer.run_id == "runpy"
        assert len(tracer.task_traces) == 1
        assert tracer.task_traces[0]["ok"] is True
        assert "runpy" in tracer.summary()
