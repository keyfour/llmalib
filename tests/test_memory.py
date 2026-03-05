"""
tests/memory/test_memory.py
Tests for context_window, store, and examples modules.
"""

import json
import os
import time
import tempfile
import pytest
from pydantic import BaseModel

from llmalib.memory.context_window import (
    count_tokens,
    trim_to_budget,
    BudgetExceededError,
    _keep_first_and_last_sentence,
)
from llmalib.memory.store import (
    make_memory_store,
    make_file_store,
    StoreEntry,
    _bm25_rank,
    _tokenize,
)
from llmalib.memory.examples import (
    select_examples,
    format_examples_block,
    FormattedExample,
)
from llmalib.core.task import Task, Example
from llmalib.core.context import make_context


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class SentimentResult(BaseModel):
    label: str
    confidence: float


def make_task(**overrides) -> Task:
    defaults = dict(
        name="test",
        prompt_template="Classify: {{ text }}",
        output_schema=SentimentResult,
        token_budget=512,
    )
    defaults.update(overrides)
    return Task(**defaults)


def msg(role: str, content: str) -> dict:
    return {"role": role, "content": content}


# ---------------------------------------------------------------------------
# context_window.py
# ---------------------------------------------------------------------------


class TestCountTokens:
    def test_default_tokenizer(self):
        assert count_tokens("hello world") == 2  # 11 chars // 4
        assert count_tokens("a" * 400) == 100

    def test_custom_tokenizer(self):
        assert count_tokens("hello", tokenizer=lambda t: 99) == 99

    def test_minimum_one(self):
        assert count_tokens("a") == 1


class TestTrimToBudget:
    def test_no_trim_needed(self):
        messages = [
            msg("system", "You are helpful."),
            msg("user", "Hello!"),
        ]
        result = trim_to_budget(messages, budget=1000)
        assert result == messages

    def test_trims_middle_assistant(self):
        long_assistant = (
            "First sentence. " + ("middle content. " * 50) + "Last sentence."
        )
        messages = [
            msg("system", "System."),
            msg("user", "Q1"),
            msg("assistant", long_assistant),
            msg("user", "Q2"),  # final user — must be preserved
        ]
        budget = 50
        result = trim_to_budget(messages, budget=budget)
        # Final user turn must be intact
        assert result[-1]["content"] == "Q2"
        # System must be intact
        assert result[0]["content"] == "System."
        # Middle assistant should be shorter
        assistant_content = result[2]["content"]
        assert len(assistant_content) < len(long_assistant)

    def test_raises_if_minimum_exceeds_budget(self):
        messages = [
            msg("system", "A" * 1000),  # ~250 tokens
            msg("user", "B" * 1000),  # ~250 tokens — final user
        ]
        with pytest.raises(BudgetExceededError):
            trim_to_budget(messages, budget=100)

    def test_empty_messages(self):
        assert trim_to_budget([], budget=100) == []

    def test_does_not_mutate_input(self):
        original = [msg("system", "S"), msg("user", "U")]
        original_copy = [dict(m) for m in original]
        trim_to_budget(original, budget=1000)
        assert original == original_copy


class TestKeepFirstAndLastSentence:
    def test_short_text_unchanged(self):
        text = "Hello. World."
        assert _keep_first_and_last_sentence(text) == text

    def test_long_text_shortened(self):
        text = "First. Second. Third. Fourth. Fifth."
        result = _keep_first_and_last_sentence(text)
        assert "First" in result
        assert "Fifth" in result
        assert "Second" not in result
        assert "[...]" in result


# ---------------------------------------------------------------------------
# store.py
# ---------------------------------------------------------------------------


class TestInMemoryStore:
    def test_set_and_get(self):
        store = make_memory_store()
        store.set("key1", {"data": "value"})
        entry = store.get("key1")
        assert entry is not None
        assert entry.value == {"data": "value"}

    def test_get_missing_returns_none(self):
        store = make_memory_store()
        assert store.get("nonexistent") is None

    def test_delete(self):
        store = make_memory_store()
        store.set("key1", "val")
        store.delete("key1")
        assert store.get("key1") is None

    def test_list_with_prefix(self):
        store = make_memory_store()
        store.set("task:a", 1)
        store.set("task:b", 2)
        store.set("other:c", 3)
        keys = store.list("task:")
        assert set(keys) == {"task:a", "task:b"}

    def test_ttl_expiry(self):
        store = make_memory_store()
        store.set("ephemeral", "gone soon", ttl=0.01)
        assert store.get("ephemeral") is not None
        time.sleep(0.05)
        assert store.get("ephemeral") is None

    def test_score_stored(self):
        store = make_memory_store()
        store.set("key", "val", score=2.5)
        entry = store.get("key")
        assert entry.score == 2.5

    def test_get_relevant_returns_results(self):
        store = make_memory_store()
        store.set("sentiment:1", "positive happy joy good")
        store.set("sentiment:2", "negative sad bad terrible")
        store.set("unrelated:1", "the quick brown fox jumps")
        results = store.get_relevant("positive good", top_k=2)
        assert len(results) <= 2
        # Most relevant should be the first one
        assert results[0].key == "sentiment:1"


class TestFileStore:
    def test_persist_and_reload(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            store = make_file_store(path)
            store.set("key1", {"value": 42})

            # Reload from disk
            store2 = make_file_store(path)
            entry = store2.get("key1")
            assert entry is not None
            assert entry.value == {"value": 42}
        finally:
            os.unlink(path)

    def test_corrupted_file_starts_fresh(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not valid json {{{")
            path = f.name

        try:
            store = make_file_store(path)  # should not raise
            assert store.list() == []
        finally:
            os.unlink(path)


class TestBM25:
    def test_tokenize(self):
        assert _tokenize("Hello, World!") == ["hello", "world"]
        assert _tokenize("cat123") == ["cat123"]

    def test_ranking_order(self):
        entries = [
            StoreEntry(key="a", value="python machine learning models"),
            StoreEntry(key="b", value="cooking recipes pasta sauce"),
            StoreEntry(key="c", value="python deep learning neural"),
        ]
        results = _bm25_rank("python learning", entries, top_k=2)
        keys = [e.key for e in results]
        assert "a" in keys or "c" in keys
        assert "b" not in keys

    def test_empty_entries(self):
        assert _bm25_rank("query", [], top_k=5) == []

    def test_empty_query_returns_first_n(self):
        entries = [StoreEntry(key=f"k{i}", value=f"val{i}") for i in range(5)]
        result = _bm25_rank("", entries, top_k=3)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# examples.py
# ---------------------------------------------------------------------------


class TestSelectExamples:
    def _make_example(self, text: str, label: str) -> Example:
        return Example(
            input_text=text,
            output=SentimentResult(label=label, confidence=0.9),
        )

    def test_inline_examples_included(self):
        ex = self._make_example("Great product!", "positive")
        task = make_task(examples=(ex,))
        ctx = make_context(text="hello")
        results = select_examples(task, ctx.vars)
        assert len(results) == 1
        assert results[0].source == "inline"

    def test_store_examples_fill_remaining(self):
        store = make_memory_store()
        ex = self._make_example("Terrible experience!", "negative")
        store.set("ex:1", ex, score=1.0)

        task = make_task()
        ctx = make_context(text="Terrible service")
        results = select_examples(task, ctx.vars, store=store, max_examples=3)
        assert any(r.source == "store" for r in results)

    def test_max_examples_respected(self):
        examples = tuple(self._make_example(f"text {i}", "positive") for i in range(10))
        task = make_task(examples=examples)
        ctx = make_context(text="hi")
        results = select_examples(task, ctx.vars, max_examples=4)
        assert len(results) <= 4

    def test_no_examples_returns_empty(self):
        task = make_task()
        ctx = make_context(text="hi")
        results = select_examples(task, ctx.vars, store=None)
        assert results == []


class TestFormatExamplesBlock:
    def test_empty_returns_empty_string(self):
        assert format_examples_block([]) == ""

    def test_formats_correctly(self):
        examples = [FormattedExample(text="Input: hi\nOutput: {}", source="inline")]
        block = format_examples_block(examples)
        assert "Example 1:" in block
        assert "Input: hi" in block
        assert "--- Examples ---" in block
