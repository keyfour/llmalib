"""
tests/pipeline/test_pipeline.py
Tests for pipeline runner, router, decomposer, tracer, and inspector.
"""

import json
import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock
from pydantic import BaseModel

from llmalib.core.context import make_context
from llmalib.core.task import Task
from llmalib.core.client import ClientConfig
from llmalib.pipeline.pipeline import run_pipeline
from llmalib.pipeline.router import (
    stop_on_failure,
    branch_on_field,
    compose_routers,
)
from llmalib.pipeline.decomposer import decompose, DecompositionError
from llmalib.debug.tracer import Tracer, make_tracer, load_trace
from llmalib.debug.inspector import print_result, print_trace


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class ClassifyResult(BaseModel):
    label: str
    confidence: float


class SummaryResult(BaseModel):
    summary: str


def make_task(name: str = "task", schema=ClassifyResult, **overrides) -> Task:
    defaults = dict(
        name=name,
        prompt_template="Process: {{ text }}",
        output_schema=schema,
        max_retries=1,
    )
    defaults.update(overrides)
    return Task(**defaults)


CLASSIFY_JSON = json.dumps({"label": "positive", "confidence": 0.9})
SUMMARY_JSON = json.dumps({"summary": "Good product."})
URGENT_JSON = json.dumps({"label": "urgent", "confidence": 0.95})
NORMAL_JSON = json.dumps({"label": "normal", "confidence": 0.8})


# ---------------------------------------------------------------------------
# pipeline.py
# ---------------------------------------------------------------------------


class TestRunPipeline:
    def test_single_task_success(self):
        task = make_task()
        ctx = make_context(text="hello")
        with patch("llmalib.reliability.retry.call", return_value=CLASSIFY_JSON):
            results = run_pipeline([task], ctx)
        assert len(results) == 1
        assert results[0].ok is True
        assert results[0].value.label == "positive"

    def test_multiple_tasks_sequential(self):
        task1 = make_task("classify", ClassifyResult)
        task2 = make_task(
            "summarise", SummaryResult, prompt_template="Summarise: {{ text }}"
        )
        ctx = make_context(text="hello")
        responses = [CLASSIFY_JSON, SUMMARY_JSON]
        with patch("llmalib.reliability.retry.call", side_effect=responses):
            results = run_pipeline([task1, task2], ctx)
        assert len(results) == 2
        assert results[0].ok is True
        assert results[1].ok is True

    def test_upstream_result_available_in_downstream_template(self):
        """Fields from task1 output should be in ctx.vars for task2 template."""
        task1 = make_task("classify", ClassifyResult)
        task2 = make_task(
            "followup",
            SummaryResult,
            prompt_template="The label was {{ label }}. Summarise.",
        )
        ctx = make_context(text="hello")
        responses = [CLASSIFY_JSON, SUMMARY_JSON]
        with patch(
            "llmalib.reliability.retry.call", side_effect=responses
        ) as mock_call:
            results = run_pipeline([task1, task2], ctx)
        assert results[1].ok is True

    def test_failed_task_still_returns_result(self):
        task = make_task(max_retries=1)
        ctx = make_context(text="hello")
        with patch("llmalib.reliability.retry.call", return_value="not json"):
            results = run_pipeline([task], ctx)
        assert len(results) == 1
        assert results[0].ok is False

    def test_debug_mode_does_not_raise(self, capsys):
        task = make_task()
        ctx = make_context(text="hello")
        with patch("llmalib.reliability.retry.call", return_value=CLASSIFY_JSON):
            results = run_pipeline([task], ctx, debug=True)
        assert results[0].ok is True

    def test_template_error_returns_error_result(self):
        task = make_task(prompt_template="{{ undefined_var | required }}")
        ctx = make_context()  # no vars
        results = run_pipeline([task], ctx)
        assert results[0].ok is False
        assert "Template" in results[0].error or "error" in results[0].error.lower()

    def test_tracer_populated(self):
        task = make_task()
        ctx = make_context(text="hello")
        tracer = make_tracer(ctx.run_id)
        with patch("llmalib.reliability.retry.call", return_value=CLASSIFY_JSON):
            run_pipeline([task], ctx, tracer=tracer)
        assert len(tracer.task_traces) == 1
        assert tracer.task_traces[0].ok is True


# ---------------------------------------------------------------------------
# router.py
# ---------------------------------------------------------------------------


class TestStopOnFailure:
    def test_continues_on_success(self):
        router = stop_on_failure()
        from llmalib.core.result import make_ok_result
        from pydantic import BaseModel

        class M(BaseModel):
            x: int = 1

        result = make_ok_result("t", M(), [])
        remaining = [make_task("next")]
        assert router(result, remaining) == remaining

    def test_stops_on_failure(self):
        router = stop_on_failure()
        from llmalib.core.result import make_error_result

        result = make_error_result("t", "err", [])
        remaining = [make_task("next")]
        assert router(result, remaining) == []


class TestBranchOnField:
    def test_branches_to_correct_tasks(self):
        urgent_task = make_task("urgent_handler")
        normal_task = make_task("normal_handler")
        router = branch_on_field(
            "label",
            {
                "urgent": [urgent_task],
                "normal": [normal_task],
            },
        )

        from llmalib.core.result import make_ok_result

        result = make_ok_result(
            "classify", ClassifyResult(label="urgent", confidence=0.9), []
        )
        remaining = [make_task("default")]
        new_remaining = router(result, remaining)
        assert new_remaining == [urgent_task]

    def test_unknown_value_unchanged(self):
        router = branch_on_field("label", {"urgent": []})
        from llmalib.core.result import make_ok_result

        result = make_ok_result("t", ClassifyResult(label="other", confidence=0.5), [])
        remaining = [make_task("next")]
        assert router(result, remaining) == remaining

    def test_failed_result_unchanged(self):
        router = branch_on_field("label", {"urgent": []})
        from llmalib.core.result import make_error_result

        result = make_error_result("t", "err", [])
        remaining = [make_task("next")]
        assert router(result, remaining) == remaining


class TestComposeRouters:
    def test_composed_stop_on_failure(self):
        r1 = stop_on_failure()
        r2 = branch_on_field("label", {})
        composed = compose_routers(r1, r2)

        from llmalib.core.result import make_error_result

        result = make_error_result("t", "err", [])
        remaining = [make_task("next")]
        # r1 stops the pipeline → r2 sees empty list → returns empty
        assert composed(result, remaining) == []


# ---------------------------------------------------------------------------
# decomposer.py
# ---------------------------------------------------------------------------


class TestDecompose:
    def test_valid_decomposition(self):
        task_registry = {
            "classify": make_task("classify"),
            "summarise": make_task("summarise", SummaryResult),
        }
        plan_json = json.dumps(
            {
                "tasks": [
                    {"task_name": "classify", "parameter_overrides": {}},
                    {"task_name": "summarise", "parameter_overrides": {}},
                ],
                "reasoning": "First classify, then summarise.",
            }
        )
        config = ClientConfig()
        with patch("llmalib.pipeline.decomposer.call", return_value=plan_json):
            tasks = decompose("Analyse this text", task_registry, config)
        assert len(tasks) == 2
        assert tasks[0].name == "classify"
        assert tasks[1].name == "summarise"

    def test_unknown_task_triggers_retry_then_fails(self):
        task_registry = {"classify": make_task("classify")}
        bad_plan = json.dumps(
            {
                "tasks": [{"task_name": "nonexistent", "parameter_overrides": {}}],
                "reasoning": "blah",
            }
        )
        config = ClientConfig()
        with patch("llmalib.pipeline.decomposer.call", return_value=bad_plan):
            with pytest.raises(DecompositionError, match="nonexistent"):
                decompose("Do something", task_registry, config)

    def test_invalid_json_triggers_retry_then_fails(self):
        config = ClientConfig()
        with patch("llmalib.pipeline.decomposer.call", return_value="not json"):
            with pytest.raises(DecompositionError):
                decompose("Do something", {"t": make_task()}, config)

    def test_empty_registry_raises(self):
        config = ClientConfig()
        with pytest.raises(DecompositionError, match="empty"):
            decompose("Do something", {}, config)

    def test_parameter_overrides_applied(self):
        task_registry = {"classify": make_task("classify")}
        plan_json = json.dumps(
            {
                "tasks": [
                    {"task_name": "classify", "parameter_overrides": {"max_retries": 5}}
                ],
                "reasoning": "",
            }
        )
        config = ClientConfig()
        with patch("llmalib.pipeline.decomposer.call", return_value=plan_json):
            tasks = decompose("Classify this", task_registry, config)
        assert tasks[0].max_retries == 5


# ---------------------------------------------------------------------------
# tracer.py
# ---------------------------------------------------------------------------


class TestTracer:
    def _make_result(self, task_name: str, ok: bool):
        from llmalib.core.result import make_ok_result, make_error_result, make_attempt
        import time

        attempt = make_attempt(1, "prompt", "response", None, [], time.monotonic())
        if ok:
            return make_ok_result(
                task_name, ClassifyResult(label="pos", confidence=0.9), [attempt]
            )
        else:
            return make_error_result(task_name, "failed", [attempt])

    def test_record_and_summary(self):
        tracer = make_tracer("test123")
        tracer.record(self._make_result("task_a", True))
        tracer.record(self._make_result("task_b", False))
        summary = tracer.summary()
        assert "test123" in summary
        assert "task_a" in summary
        assert "✓" in summary
        assert "✗" in summary

    def test_to_dict(self):
        tracer = make_tracer("abc")
        tracer.record(self._make_result("t", True))
        d = tracer.to_dict()
        assert d["run_id"] == "abc"
        assert len(d["tasks"]) == 1
        assert d["tasks"][0]["ok"] is True

    def test_save_and_load(self):
        tracer = make_tracer("xyz")
        tracer.record(self._make_result("task_a", True))

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            tracer.to_file(path)
            loaded = load_trace(path)
            assert loaded.run_id == "xyz"
            assert len(loaded.task_traces) == 1
            assert loaded.task_traces[0].task_name == "task_a"
            assert loaded.task_traces[0].ok is True
        finally:
            os.unlink(path)
