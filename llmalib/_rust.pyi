"""
llmalib/_rust.pyi

Type stubs for the `llmalib._rust` pyo3 extension module so mypy sees the
flat FFI surface the Python adapters call into.
"""

import builtins
from typing import Any, Callable

CallerFn = Callable[[list[dict[str, Any]], dict[str, Any]], str]
RouterFn = Callable[[dict[str, Any], list[Any]], list[Any]]
TokenizerFn = Callable[[str], int]
GuardFn = Callable[[dict[str, Any], ContextView], list[str]]

def parse_response(raw: str, schema: dict[str, Any]) -> dict[str, Any]: ...
def extract_json(raw: str) -> str | None: ...
def format_schema_hint(schema: dict[str, Any]) -> str: ...
def build_reflection_message(
    parse_error: str | None = ...,
    guard_errors: list[str] | None = ...,
) -> dict[str, Any]: ...
def count_tokens(text: str, tokenizer: TokenizerFn | None = ...) -> int: ...
def trim_to_budget(
    messages: list[dict[str, Any]],
    budget: int,
    tokenizer: TokenizerFn | None = ...,
) -> list[dict[str, Any]]: ...  # raises BudgetExceededError
def keep_first_and_last_sentence(text: str) -> str: ...
def tokenize(text: str) -> list[str]: ...
def bm25_rank(query: str, entries: list[Any], top_k: int) -> list[Any]: ...
def select_examples(
    task_spec: dict[str, Any],
    vars: dict[str, Any],
    store: PyStore | None = ...,
    max_examples: int = ...,
) -> list[dict[str, Any]]: ...
def format_examples_block(examples: list[dict[str, Any]]) -> str: ...
def call(
    messages: list[dict[str, Any]], config: dict[str, Any]
) -> str: ...  # raises ClientError
def run_with_retry(
    task_spec: dict[str, Any],
    messages: list[dict[str, Any]],
    config: dict[str, Any],
    ctx: dict[str, Any] | None = ...,
    caller: CallerFn | None = ...,
) -> dict[str, Any]: ...  # raises ClientError
def run_pipeline(
    task_specs: list[dict[str, Any]],
    vars: dict[str, Any] | None = ...,
    run_id: str | None = ...,
    router: RouterFn | None = ...,
    store: PyStore | None = ...,
    tracer: PyTracer | None = ...,
    caller: CallerFn | None = ...,
    debug: bool = ...,
) -> dict[str, Any]: ...  # raises ClientError, BudgetExceededError
def decompose(
    prompt: str,
    registry: dict[str, dict[str, Any]],
    config: dict[str, Any],
    max_tasks: int = ...,
    caller: CallerFn | None = ...,
) -> list[dict[str, Any]]: ...  # raises DecompositionError, ClientError
def replay_task(
    tracer: PyTracer,
    task_name: str,
    task_spec: dict[str, Any],
    attempt_number: int = ...,
    model: str | None = ...,
    base_url: str | None = ...,
    prompt_override: str | None = ...,
    schema_override: dict[str, Any] | None = ...,
    diff: bool = ...,
    caller: CallerFn | None = ...,
) -> dict[str, Any]: ...  # raises RuntimeError
def set_caller(f: CallerFn | None = ...) -> None: ...

# native guards / router decisions / context merge ---------------------------
def guard_field_in_set(field: str, allowed: list[str]) -> NativeGuard: ...
def guard_float_in_range(field: str, min_val: float, max_val: float) -> NativeGuard: ...
def guard_max_length(field: str, max_chars: int) -> NativeGuard: ...
def guard_no_content_from_outside_context(
    response_field: str, context_field: str, threshold: float
) -> NativeGuard: ...
def route_stop_on_failure(ok: bool) -> bool: ...
def route_branch_key(
    field: str, ok: bool, value: Any, branch_keys: list[str]
) -> str | None: ...
def merge_value_fields(vars: dict[str, Any], value: Any) -> dict[str, Any]: ...

class NativeGuard:
    def __call__(self, value: Any, ctx: Any = ...) -> list[str]: ...

def make_memory_store() -> PyStore: ...
def make_file_store(path: str) -> PyStore: ...  # raises StoreError
def make_tracer(run_id: str) -> PyTracer: ...
def load_trace(path: str) -> PyTracer: ...  # raises RuntimeError

class PyStore:
    def set(
        self,
        key: str,
        value: Any,
        score: float = ...,
        ttl: float | None = ...,
    ) -> None: ...
    def get(self, key: str) -> dict[str, Any] | None: ...
    def delete(self, key: str) -> None: ...
    def list(self, prefix: str) -> builtins.list[str]: ...
    def get_relevant(self, query: str, top_k: int) -> builtins.list[dict[str, Any]]: ...

class PyTracer:
    run_id: str
    task_traces: list[dict[str, Any]]
    def record(self, result: dict[str, Any]) -> None: ...
    def summary(self) -> str: ...
    def to_dict(self) -> dict[str, Any]: ...
    def to_file(self, path: str) -> None: ...

class ContextView:
    vars: dict[str, Any]
    results: dict[str, Any]
    run_id: str

class ClientError(Exception): ...
class BudgetExceededError(Exception): ...
class DecompositionError(Exception): ...
class StoreError(Exception): ...
